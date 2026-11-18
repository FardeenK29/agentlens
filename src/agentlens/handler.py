"""LangChain/LangGraph callback handler that emits OpenTelemetry spans and metrics.

Span tree for one agent run:

    invoke_agent <agent>                 (root: one per graph invocation)
    ├── <node>                           (one per LangGraph node execution)
    │   ├── chat <model>                 (each LLM call: tokens, cost, finish reason)
    │   ├── retrieval <retriever>        (each retriever call: query, doc count)
    │   └── execute_tool <tool>          (each tool call: args, result, errors)
    └── ...

LangChain's internal plumbing runnables (RunnableSequence, ChannelWrite, branch
functions...) do not get spans: their children are attached to the nearest
real ancestor so traces stay readable.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Sequence
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from opentelemetry import metrics, trace
from opentelemetry.trace import Span, SpanKind, Status, StatusCode

from . import pricing
from . import semconv as sc

_TRACER_NAME = "agentlens"
_MAX_ATTR_CHARS = 8_000
# LangGraph uses exceptions for control flow (interrupts, Command bubbling).
_CONTROL_FLOW_ERRORS = {"GraphInterrupt", "NodeInterrupt", "GraphBubbleUp", "ParentCommand"}


@dataclass
class _RootStats:
    agent: str
    cost: float = 0.0
    llm_calls: int = 0
    tool_calls: int = 0
    tool_errors: int = 0


@dataclass
class _Run:
    kind: str                      # agent | node | chat | tool | retrieval | passthrough
    span: Span | None              # owned span, or the inherited parent span for passthrough
    root_id: UUID | None
    start: float = field(default_factory=time.perf_counter)
    attrs: dict[str, Any] = field(default_factory=dict)  # metric attributes

    @property
    def owns_span(self) -> bool:
        return self.kind != "passthrough"


def _truncate(value: Any) -> str:
    text = value if isinstance(value, str) else json.dumps(value, default=str, ensure_ascii=False)
    return text if len(text) <= _MAX_ATTR_CHARS else text[:_MAX_ATTR_CHARS] + "…[truncated]"


def _message_role(msg: Any) -> str:
    return {"human": "user", "ai": "assistant", "tool": "tool", "system": "system"}.get(
        getattr(msg, "type", ""), getattr(msg, "type", "unknown"))


def _serialize_messages(messages: Sequence[Any]) -> list[dict[str, Any]]:
    out = []
    for m in messages:
        item: dict[str, Any] = {"role": _message_role(m), "content": getattr(m, "content", str(m))}
        tool_calls = getattr(m, "tool_calls", None)
        if tool_calls:
            item["tool_calls"] = [{"name": tc.get("name"), "args": tc.get("args")} for tc in tool_calls]
        out.append(item)
    return out


class AgentLensCallbackHandler(BaseCallbackHandler):
    """Turns LangChain callback events into OpenTelemetry spans and metrics.

    Args:
        agent_name: value for ``gen_ai.agent.name`` and the root span name.
        capture_content: record prompts, completions and tool args/results on
            spans. Off by default because they can contain user data.
        tracer_provider / meter_provider: override the global providers (tests).
    """

    raise_error = False  # never let telemetry break the agent

    def __init__(self, agent_name: str = "agent", capture_content: bool = False,
                 tracer_provider: trace.TracerProvider | None = None,
                 meter_provider: metrics.MeterProvider | None = None) -> None:
        super().__init__()
        self.agent_name = agent_name
        self.capture_content = capture_content
        self._tracer = trace.get_tracer(_TRACER_NAME, tracer_provider=tracer_provider)
        meter = metrics.get_meter(_TRACER_NAME, meter_provider=meter_provider)
        self._runs: dict[UUID, _Run] = {}
        self._roots: dict[UUID, _RootStats] = {}
        self._lock = threading.Lock()

        self._m_op_duration = meter.create_histogram(
            sc.M_OP_DURATION, unit="s", description="Duration of GenAI operations (LLM calls)")
        self._m_tokens = meter.create_histogram(
            sc.M_TOKEN_USAGE, unit="{token}", description="Tokens used per LLM call")
        self._m_llm_cost = meter.create_counter(
            sc.M_LLM_COST, unit="usd", description="LLM spend in USD")
        self._m_node_duration = meter.create_histogram(
            sc.M_NODE_DURATION, unit="s", description="Duration of each graph node execution")
        self._m_tool_calls = meter.create_counter(
            sc.M_TOOL_CALLS, unit="{call}", description="Tool calls by tool and status")
        self._m_tool_duration = meter.create_histogram(
            sc.M_TOOL_DURATION, unit="s", description="Duration of tool calls")
        self._m_runs = meter.create_counter(
            sc.M_RUNS, unit="{run}", description="Agent runs by status")
        self._m_run_duration = meter.create_histogram(
            sc.M_RUN_DURATION, unit="s", description="End-to-end agent run duration")
        self._m_run_cost = meter.create_histogram(
            sc.M_RUN_COST, unit="usd", description="LLM cost per agent run")

    # ------------------------------------------------------------------ helpers

    def _parent(self, parent_run_id: UUID | None) -> _Run | None:
        if parent_run_id is None:
            return None
        with self._lock:
            return self._runs.get(parent_run_id)

    def _start(self, run_id: UUID, parent: _Run | None, kind: str, name: str,
               attributes: dict[str, Any], span_kind: SpanKind = SpanKind.INTERNAL,
               metric_attrs: dict[str, Any] | None = None) -> _Run:
        ctx = trace.set_span_in_context(parent.span) if parent and parent.span else None
        span = self._tracer.start_span(name, context=ctx, kind=span_kind,
                                       attributes={k: v for k, v in attributes.items() if v is not None})
        run = _Run(kind=kind, span=span, root_id=parent.root_id if parent else run_id,
                   attrs=metric_attrs or {})
        with self._lock:
            self._runs[run_id] = run
        return run

    def _passthrough(self, run_id: UUID, parent: _Run | None) -> None:
        with self._lock:
            self._runs[run_id] = _Run(kind="passthrough", span=parent.span if parent else None,
                                      root_id=parent.root_id if parent else None)

    def _pop(self, run_id: UUID) -> _Run | None:
        with self._lock:
            return self._runs.pop(run_id, None)

    def _root(self, run: _Run) -> _RootStats | None:
        with self._lock:
            return self._roots.get(run.root_id) if run.root_id else None

    @staticmethod
    def _fail(span: Span, error: BaseException) -> str:
        etype = type(error).__name__
        span.record_exception(error)
        span.set_status(Status(StatusCode.ERROR, str(error)[:500]))
        span.set_attribute(sc.ERROR_TYPE, etype)
        return etype

    # ------------------------------------------------------------- chains/nodes

    def on_chain_start(self, serialized: dict[str, Any] | None, inputs: Any, *, run_id: UUID,
                       parent_run_id: UUID | None = None, tags: list[str] | None = None,
                       metadata: dict[str, Any] | None = None, **kwargs: Any) -> None:
        name = kwargs.get("name") or (serialized or {}).get("name") or "chain"
        meta = metadata or {}
        parent = self._parent(parent_run_id)

        if parent is None:  # graph invocation = agent run
            attrs = {sc.OPERATION_NAME: sc.OP_INVOKE_AGENT, sc.AGENT_NAME: self.agent_name,
                     sc.CONVERSATION_ID: meta.get("thread_id")}
            if self.capture_content:
                attrs["agentlens.input"] = _truncate(inputs)
            self._start(run_id, None, "agent", f"{sc.OP_INVOKE_AGENT} {self.agent_name}", attrs)
            with self._lock:
                self._roots[run_id] = _RootStats(agent=self.agent_name)
            return

        node = meta.get("langgraph_node")
        is_node = node is not None and name == node and not (parent.kind == "node" and parent.attrs.get(sc.NODE) == node)
        if is_node:
            self._start(run_id, parent, "node", node,
                        {sc.NODE: node, sc.AGENT_NAME: self.agent_name,
                         "agentlens.step": meta.get("langgraph_step")},
                        metric_attrs={sc.NODE: node, sc.AGENT_NAME: self.agent_name})
        else:
            self._passthrough(run_id, parent)

    def on_chain_end(self, outputs: Any, *, run_id: UUID, **kwargs: Any) -> None:
        run = self._pop(run_id)
        if not run or not run.owns_span:
            return
        elapsed = time.perf_counter() - run.start
        if run.kind == "node":
            self._m_node_duration.record(elapsed, {**run.attrs, "status": "ok"})
        elif run.kind == "agent":
            self._finish_root(run_id, run, elapsed, "ok", outputs)
        run.span.set_status(Status(StatusCode.OK))
        run.span.end()

    def on_chain_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        run = self._pop(run_id)
        if not run or not run.owns_span:
            return
        elapsed = time.perf_counter() - run.start
        if type(error).__name__ in _CONTROL_FLOW_ERRORS:
            run.span.set_attribute("agentlens.interrupted", True)
            status = "interrupted"
        else:
            self._fail(run.span, error)
            status = "error"
        if run.kind == "node":
            self._m_node_duration.record(elapsed, {**run.attrs, "status": status})
        elif run.kind == "agent":
            self._finish_root(run_id, run, elapsed, status, None)
        run.span.end()

    def _finish_root(self, run_id: UUID, run: _Run, elapsed: float, status: str, outputs: Any) -> None:
        with self._lock:
            stats = self._roots.pop(run_id, None)
        if stats is None:
            return
        attrs = {sc.AGENT_NAME: stats.agent, "status": status}
        run.span.set_attribute(sc.RUN_COST_USD, round(stats.cost, 8))
        run.span.set_attribute(sc.RUN_LLM_CALLS, stats.llm_calls)
        run.span.set_attribute(sc.RUN_TOOL_CALLS, stats.tool_calls)
        if self.capture_content and outputs is not None:
            run.span.set_attribute("agentlens.output", _truncate(outputs))
        self._m_runs.add(1, attrs)
        self._m_run_duration.record(elapsed, attrs)
        self._m_run_cost.record(stats.cost, {sc.AGENT_NAME: stats.agent})

    # --------------------------------------------------------------------- LLMs

    def on_chat_model_start(self, serialized: dict[str, Any] | None, messages: list[list[Any]], *,
                            run_id: UUID, parent_run_id: UUID | None = None,
                            metadata: dict[str, Any] | None = None, **kwargs: Any) -> None:
        self._llm_start(run_id, parent_run_id, metadata, kwargs,
                        _serialize_messages(messages[0]) if messages else None)

    def on_llm_start(self, serialized: dict[str, Any] | None, prompts: list[str], *, run_id: UUID,
                     parent_run_id: UUID | None = None, metadata: dict[str, Any] | None = None,
                     **kwargs: Any) -> None:
        self._llm_start(run_id, parent_run_id, metadata, kwargs,
                        [{"role": "user", "content": p} for p in prompts])

    def _llm_start(self, run_id: UUID, parent_run_id: UUID | None, metadata: dict[str, Any] | None,
                   kwargs: dict[str, Any], input_messages: list[dict[str, Any]] | None) -> None:
        meta = metadata or {}
        params = kwargs.get("invocation_params") or {}
        model = params.get("model") or params.get("model_name") or meta.get("ls_model_name") or "unknown"
        provider = meta.get("ls_provider") or params.get("_type") or "unknown"
        attrs = {
            sc.OPERATION_NAME: sc.OP_CHAT,
            sc.PROVIDER_NAME: provider,
            sc.REQUEST_MODEL: model,
            sc.REQUEST_TEMPERATURE: meta.get("ls_temperature", params.get("temperature")),
            sc.REQUEST_MAX_TOKENS: meta.get("ls_max_tokens", params.get("max_tokens")),
            sc.NODE: meta.get("langgraph_node"),
        }
        if self.capture_content and input_messages is not None:
            attrs[sc.INPUT_MESSAGES] = _truncate(input_messages)
        self._start(run_id, self._parent(parent_run_id), "chat", f"{sc.OP_CHAT} {model}", attrs,
                    span_kind=SpanKind.CLIENT,
                    metric_attrs={sc.OPERATION_NAME: sc.OP_CHAT, sc.PROVIDER_NAME: provider,
                                  sc.REQUEST_MODEL: model, sc.NODE: meta.get("langgraph_node") or "none",
                                  sc.AGENT_NAME: self.agent_name})

    def on_llm_end(self, response: Any, *, run_id: UUID, **kwargs: Any) -> None:
        run = self._pop(run_id)
        if not run:
            return
        span, attrs = run.span, run.attrs
        elapsed = time.perf_counter() - run.start
        gen = response.generations[0][0] if response.generations and response.generations[0] else None
        message = getattr(gen, "message", None)
        meta = getattr(message, "response_metadata", None) or {}
        usage = getattr(message, "usage_metadata", None) or {}
        if not usage:  # legacy LLMs report usage in llm_output
            tu = (response.llm_output or {}).get("token_usage") or {}
            usage = {"input_tokens": tu.get("prompt_tokens", 0), "output_tokens": tu.get("completion_tokens", 0)}

        in_tok = int(usage.get("input_tokens") or 0)
        out_tok = int(usage.get("output_tokens") or 0)
        details = usage.get("input_token_details") or {}
        cache_read = int(details.get("cache_read") or 0)
        cache_write = int(details.get("cache_creation") or 0)
        resp_model = meta.get("model_name") or meta.get("model") or attrs.get(sc.REQUEST_MODEL)
        finish = meta.get("finish_reason") or meta.get("stop_reason")
        cost = pricing.cost_usd(resp_model, in_tok, out_tok, cache_read, cache_write)
        if cost is None:
            cost = pricing.cost_usd(attrs.get(sc.REQUEST_MODEL), in_tok, out_tok, cache_read, cache_write)

        if attrs.get(sc.REQUEST_MODEL) == "unknown" and resp_model and resp_model != "unknown":
            span.update_name(f"{sc.OP_CHAT} {resp_model}")
        span.set_attribute(sc.RESPONSE_MODEL, resp_model)
        span.set_attribute(sc.USAGE_INPUT_TOKENS, in_tok)
        span.set_attribute(sc.USAGE_OUTPUT_TOKENS, out_tok)
        if cache_read:
            span.set_attribute(sc.USAGE_CACHE_READ_TOKENS, cache_read)
        if cache_write:
            span.set_attribute(sc.USAGE_CACHE_WRITE_TOKENS, cache_write)
        if finish:
            span.set_attribute(sc.RESPONSE_FINISH_REASONS, [str(finish)])
        if getattr(message, "id", None):
            span.set_attribute(sc.RESPONSE_ID, message.id)
        if cost is not None:
            span.set_attribute(sc.COST_USD, round(cost, 8))
        if self.capture_content and message is not None:
            span.set_attribute(sc.OUTPUT_MESSAGES, _truncate(_serialize_messages([message])))
        span.set_status(Status(StatusCode.OK))
        span.end()

        m_attrs = {**attrs, sc.RESPONSE_MODEL: resp_model}
        self._m_op_duration.record(elapsed, m_attrs)
        self._m_tokens.record(in_tok, {**m_attrs, sc.TOKEN_TYPE: "input"})
        self._m_tokens.record(out_tok, {**m_attrs, sc.TOKEN_TYPE: "output"})
        if cost is not None:
            self._m_llm_cost.add(cost, m_attrs)
        stats = self._root(run)
        if stats:
            with self._lock:
                stats.llm_calls += 1
                stats.cost += cost or 0.0

    def on_llm_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        run = self._pop(run_id)
        if not run:
            return
        etype = self._fail(run.span, error)
        run.span.end()
        self._m_op_duration.record(time.perf_counter() - run.start, {**run.attrs, sc.ERROR_TYPE: etype})

    # -------------------------------------------------------------------- tools

    def on_tool_start(self, serialized: dict[str, Any] | None, input_str: str, *, run_id: UUID,
                      parent_run_id: UUID | None = None, metadata: dict[str, Any] | None = None,
                      inputs: dict[str, Any] | None = None, **kwargs: Any) -> None:
        name = kwargs.get("name") or (serialized or {}).get("name") or "tool"
        meta = metadata or {}
        attrs = {sc.OPERATION_NAME: sc.OP_EXECUTE_TOOL, sc.TOOL_NAME: name,
                 sc.TOOL_CALL_ID: kwargs.get("tool_call_id"), sc.NODE: meta.get("langgraph_node")}
        if self.capture_content:
            attrs[sc.TOOL_CALL_ARGUMENTS] = _truncate(inputs if inputs is not None else input_str)
        self._start(run_id, self._parent(parent_run_id), "tool", f"{sc.OP_EXECUTE_TOOL} {name}", attrs,
                    metric_attrs={sc.TOOL_NAME: name, sc.AGENT_NAME: self.agent_name})

    def on_tool_end(self, output: Any, *, run_id: UUID, **kwargs: Any) -> None:
        run = self._pop(run_id)
        if not run:
            return
        # ToolNode can turn exceptions into ToolMessage(status="error") instead of raising.
        errored = getattr(output, "status", None) == "error"
        if self.capture_content:
            run.span.set_attribute(sc.TOOL_CALL_RESULT, _truncate(getattr(output, "content", output)))
        if errored:
            run.span.set_status(Status(StatusCode.ERROR, _truncate(getattr(output, "content", ""))[:500]))
            run.span.set_attribute(sc.ERROR_TYPE, "ToolError")
        else:
            run.span.set_status(Status(StatusCode.OK))
        run.span.end()
        self._record_tool(run, "error" if errored else "ok")

    def on_tool_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        run = self._pop(run_id)
        if not run:
            return
        self._fail(run.span, error)
        run.span.end()
        self._record_tool(run, "error")

    def _record_tool(self, run: _Run, status: str) -> None:
        attrs = {**run.attrs, "status": status}
        self._m_tool_calls.add(1, attrs)
        self._m_tool_duration.record(time.perf_counter() - run.start, attrs)
        stats = self._root(run)
        if stats:
            with self._lock:
                stats.tool_calls += 1
                stats.tool_errors += status == "error"

    # --------------------------------------------------------------- retrievers

    def on_retriever_start(self, serialized: dict[str, Any] | None, query: str, *, run_id: UUID,
                           parent_run_id: UUID | None = None, metadata: dict[str, Any] | None = None,
                           **kwargs: Any) -> None:
        name = kwargs.get("name") or (serialized or {}).get("name") or "retriever"
        attrs = {sc.OPERATION_NAME: sc.OP_RETRIEVE, sc.DATA_SOURCE_ID: name,
                 sc.NODE: (metadata or {}).get("langgraph_node")}
        if self.capture_content:
            attrs["agentlens.retrieval.query"] = _truncate(query)
        self._start(run_id, self._parent(parent_run_id), "retrieval", f"{sc.OP_RETRIEVE} {name}", attrs)

    def on_retriever_end(self, documents: Sequence[Any], *, run_id: UUID, **kwargs: Any) -> None:
        run = self._pop(run_id)
        if not run:
            return
        run.span.set_attribute(sc.RETRIEVED_DOCS, len(documents))
        run.span.set_status(Status(StatusCode.OK))
        run.span.end()

    def on_retriever_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        run = self._pop(run_id)
        if run:
            self._fail(run.span, error)
            run.span.end()
