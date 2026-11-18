"""Send traffic through the instrumented support agent.

    python examples/run_demo.py --runs 30            # local model via Ollama (default)
    MODEL=openai:gpt-4o-mini python examples/run_demo.py   # any LangChain provider works

Then open Grafana (localhost:3000) and Jaeger (localhost:16686).
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import time
import uuid
from pathlib import Path

from langchain_core.messages import HumanMessage

sys.path.insert(0, str(Path(__file__).parent))
import agentlens  # noqa: E402
from support_agent import QUESTIONS, build_graph  # noqa: E402


def make_model(spec: str):
    provider, _, name = spec.partition(":")
    if provider == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(model=name, temperature=0)
    from langchain.chat_models import init_chat_model
    return init_chat_model(spec)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=20)
    ap.add_argument("--model", default=os.getenv("MODEL", "ollama:qwen2.5:3b"))
    args = ap.parse_args()

    agentlens.configure(service_name="support-agent", metric_interval_ms=5000,
                        resource_attributes={"deployment.environment": "demo"})
    app = agentlens.instrument(build_graph(make_model(args.model)), agent_name="support-agent",
                               capture_content=True)

    for i in range(args.runs):
        q = random.choice(QUESTIONS)
        t = time.perf_counter()
        try:
            out = app.invoke({"messages": [HumanMessage(q)]},
                             config={"configurable": {"thread_id": str(uuid.uuid4())}})
            answer = out["messages"][-1].content.replace("\n", " ")
        except Exception as e:  # keep the load going; the error is in the trace
            answer = f"ERROR {type(e).__name__}: {e}"
        print(f"[{i + 1}/{args.runs}] {time.perf_counter() - t:5.1f}s  {q}\n         -> {answer[:140]}")

    agentlens.shutdown()


if __name__ == "__main__":
    main()
