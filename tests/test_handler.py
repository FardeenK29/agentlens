"""End-to-end tests: a real LangGraph graph with a fake chat model, captured in memory."""

from typing import Annotated, TypedDict

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode

import agentlens
from agentlens import semconv as sc


class FakeModel(GenericFakeChatModel):
    def bind_tools(self, tools, **kwargs):
        return self

    def _llm_type(self) -> str:  # type: ignore[override]
        return "fake"


@tool
def get_weather(city: str) -> str:
    """Weather for a city."""
    return f"Sunny in {city}"


@tool
def flaky_lookup(query: str) -> str:
    """Always fails."""
    raise RuntimeError("upstream timeout")


class State(TypedDict):
    messages: Annotated[list, add_messages]


def _usage(i, o):
    return {"input_tokens": i, "output_tokens": o, "total_tokens": i + o}


def build_graph(script):
    model = FakeModel(messages=iter(script))

    def agent(state: State):
        return {"messages": [model.invoke(state["messages"])]}

    g = StateGraph(State)
    g.add_node("agent", agent)
    g.add_node("tools", ToolNode([get_weather, flaky_lookup], handle_tool_errors=True))
    g.add_edge(START, "agent")
    g.add_conditional_edges("agent", tools_condition)
    g.add_edge("tools", "agent")
    return g.compile()


@pytest.fixture
def otel():
    exporter = InMemorySpanExporter()
    tp = TracerProvider()
    tp.add_span_processor(SimpleSpanProcessor(exporter))
    reader = InMemoryMetricReader()
    mp = MeterProvider(metric_readers=[reader])
    return exporter, reader, {"tracer_provider": tp, "meter_provider": mp}


def _metrics(reader):
    out = {}
    for rm in reader.get_metrics_data().resource_metrics:
        for sm in rm.scope_metrics:
            for m in sm.metrics:
                out[m.name] = list(m.data.data_points)
    return out


SCRIPT = [
    AIMessage(content="", tool_calls=[
        {"name": "get_weather", "args": {"city": "Mumbai"}, "id": "c1"},
        {"name": "flaky_lookup", "args": {"query": "x"}, "id": "c2"},
    ], usage_metadata=_usage(1000, 50), response_metadata={"model_name": "gpt-4o-mini", "finish_reason": "tool_calls"}),
    AIMessage(content="It's sunny in Mumbai.", usage_metadata=_usage(1200, 20),
              response_metadata={"model_name": "gpt-4o-mini", "finish_reason": "stop"}),
]


def test_span_tree_and_attributes(otel):
    exporter, _, providers = otel
    app = agentlens.instrument(build_graph(SCRIPT), agent_name="weather-bot", capture_content=True, **providers)
    result = app.invoke({"messages": [HumanMessage("Weather in Mumbai?")]})
    assert result["messages"][-1].content == "It's sunny in Mumbai."

    spans = exporter.get_finished_spans()
    by_name = {}
    for s in spans:
        by_name.setdefault(s.name, []).append(s)

    root = by_name["invoke_agent weather-bot"][0]
    assert root.parent is None
    assert root.attributes[sc.RUN_LLM_CALLS] == 2
    assert root.attributes[sc.RUN_TOOL_CALLS] == 2
    # 2200 in * 0.15/M + 70 out * 0.60/M
    assert root.attributes[sc.RUN_COST_USD] == pytest.approx((2200 * 0.15 + 70 * 0.60) / 1e6)

    # Node spans are children of the root; LLM spans are children of the agent node.
    assert len(by_name["agent"]) == 2 and len(by_name["tools"]) == 1
    for node in by_name["agent"] + by_name["tools"]:
        assert node.parent.span_id == root.context.span_id
    chat = by_name["chat gpt-4o-mini"]
    assert len(chat) == 2
    agent_ids = {s.context.span_id for s in by_name["agent"]}
    assert all(c.parent.span_id in agent_ids for c in chat)
    assert chat[0].attributes[sc.USAGE_INPUT_TOKENS] in (1000, 1200)
    assert sc.INPUT_MESSAGES in chat[0].attributes

    # Tool spans hang off the tools node; the failing tool is marked as an error.
    tools_id = by_name["tools"][0].context.span_id
    ok = by_name["execute_tool get_weather"][0]
    bad = by_name["execute_tool flaky_lookup"][0]
    assert ok.parent.span_id == tools_id and bad.parent.span_id == tools_id
    assert ok.status.status_code == StatusCode.OK
    assert bad.status.status_code == StatusCode.ERROR
    assert bad.attributes[sc.ERROR_TYPE] == "RuntimeError"

    # No spans leak for LangChain plumbing runnables.
    assert not any(n.startswith(("Runnable", "ChannelWrite", "Branch")) for n in by_name)


def test_metrics(otel):
    _, reader, providers = otel
    app = agentlens.instrument(build_graph(list(SCRIPT)), agent_name="weather-bot", **providers)
    app.invoke({"messages": [HumanMessage("Weather in Mumbai?")]})
    m = _metrics(reader)

    tokens = {dp.attributes[sc.TOKEN_TYPE]: dp.sum for dp in m[sc.M_TOKEN_USAGE]}
    assert tokens == {"input": 2200, "output": 70}
    tool_calls = {(dp.attributes[sc.TOOL_NAME], dp.attributes["status"]): dp.value for dp in m[sc.M_TOOL_CALLS]}
    assert tool_calls == {("get_weather", "ok"): 1, ("flaky_lookup", "error"): 1}
    assert sum(dp.value for dp in m[sc.M_RUNS]) == 1
    nodes = {dp.attributes[sc.NODE]: dp.count for dp in m[sc.M_NODE_DURATION]}
    assert nodes == {"agent": 2, "tools": 1}
    assert m[sc.M_LLM_COST][0].value == pytest.approx((2200 * 0.15 + 70 * 0.60) / 1e6)


def test_content_not_captured_by_default(otel):
    exporter, _, providers = otel
    app = agentlens.instrument(build_graph(list(SCRIPT)), agent_name="weather-bot", **providers)
    app.invoke({"messages": [HumanMessage("secret question")]})
    for s in exporter.get_finished_spans():
        assert sc.INPUT_MESSAGES not in s.attributes
        assert sc.TOOL_CALL_ARGUMENTS not in s.attributes
        assert "agentlens.input" not in s.attributes


def test_unknown_model_has_no_cost(otel):
    exporter, _, providers = otel
    script = [AIMessage(content="hi", usage_metadata=_usage(10, 2), response_metadata={"model_name": "mystery-1"})]
    app = agentlens.instrument(build_graph(script), agent_name="a", **providers)
    app.invoke({"messages": [HumanMessage("hi")]})
    chat = [s for s in exporter.get_finished_spans() if s.name.startswith("chat")][0]
    assert sc.COST_USD not in chat.attributes


def test_pricing_prefix_and_cache():
    from agentlens import pricing
    assert pricing.lookup("gpt-4o-mini-2024-07-18").input == 0.15
    # 1000 prompt tokens of which 800 cached reads on Claude Haiku 5.5
    c = pricing.cost_usd("claude-haiku-5-5", 1000, 100, cache_read=800)
    assert c == pytest.approx((200 * 0.10 + 800 * 0.01 + 100 * 0.50) / 1e6)
