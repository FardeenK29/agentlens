"""Demo: an e-commerce support agent built with LangGraph.

Tools hit a small in-memory "database". To make the dashboards interesting,
two tools have realistic behaviour injected on purpose:
  * get_shipping_status sleeps 0.2-1.5 s (a slow carrier API)
  * get_shipping_status fails ~12% of the time (an unreliable upstream)
Both are clearly synthetic and documented in the README.
"""

from __future__ import annotations

import os
import random
import time
from typing import Annotated, TypedDict

from langchain_core.messages import SystemMessage
from langchain_core.tools import tool
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

ORDERS = {
    "A1001": {"item": "Wireless earbuds", "price": 2499, "status": "shipped", "carrier": "BlueDart", "days_since_order": 3},
    "A1002": {"item": "Running shoes", "price": 4299, "status": "delivered", "carrier": "Delhivery", "days_since_order": 12},
    "A1003": {"item": "Mechanical keyboard", "price": 6999, "status": "processing", "carrier": None, "days_since_order": 1},
    "A1004": {"item": "Water bottle", "price": 799, "status": "delivered", "carrier": "Ekart", "days_since_order": 40},
    "A1005": {"item": "Laptop stand", "price": 1899, "status": "shipped", "carrier": "BlueDart", "days_since_order": 5},
}
RETURN_WINDOW_DAYS = 30
FAILURE_RATE = float(os.getenv("DEMO_TOOL_FAILURE_RATE", "0.12"))


@tool
def lookup_order(order_id: str) -> dict:
    """Look up an order by its ID (e.g. A1001). Returns item, price in INR, status and age."""
    order = ORDERS.get(order_id.strip().upper())
    if not order:
        return {"error": f"No order found with id {order_id}"}
    return {"order_id": order_id.upper(), **order}


@tool
def get_shipping_status(order_id: str) -> str:
    """Get live tracking status from the carrier for a shipped order."""
    time.sleep(random.uniform(0.2, 1.5))  # slow third-party API
    if random.random() < FAILURE_RATE:
        raise TimeoutError("carrier API timed out")
    order = ORDERS.get(order_id.strip().upper())
    if not order or order["status"] != "shipped":
        return f"Order {order_id} is not in transit."
    return f"{order['carrier']}: out for delivery, expected within {random.randint(1, 2)} day(s)."


@tool
def check_refund_eligibility(order_id: str) -> dict:
    """Check whether an order can be returned for a refund under the 30-day policy."""
    order = ORDERS.get(order_id.strip().upper())
    if not order:
        return {"eligible": False, "reason": "order not found"}
    if order["status"] != "delivered":
        return {"eligible": False, "reason": "order not delivered yet"}
    if order["days_since_order"] > RETURN_WINDOW_DAYS:
        return {"eligible": False, "reason": f"outside the {RETURN_WINDOW_DAYS}-day window"}
    return {"eligible": True, "refund_amount_inr": order["price"]}


TOOLS = [lookup_order, get_shipping_status, check_refund_eligibility]

SYSTEM_PROMPT = (
    "You are a support agent for an online store. Use the tools to look up real order data; "
    "never invent order details. If a tool fails, say so briefly and offer to retry. "
    "Answer in 1-3 short sentences."
)


class State(TypedDict):
    messages: Annotated[list, add_messages]


def build_graph(model):
    """Classic ReAct loop: agent -> tools -> agent ... -> answer."""
    llm = model.bind_tools(TOOLS)

    def agent(state: State):
        return {"messages": [llm.invoke([SystemMessage(SYSTEM_PROMPT), *state["messages"]])]}

    g = StateGraph(State)
    g.add_node("agent", agent)
    g.add_node("tools", ToolNode(TOOLS, handle_tool_errors=True))
    g.add_edge(START, "agent")
    g.add_conditional_edges("agent", tools_condition)
    g.add_edge("tools", "agent")
    return g.compile(name="support-agent")


QUESTIONS = [
    "Where is my order A1001?",
    "Can I get a refund for order A1002?",
    "I want to return A1004, is that possible?",
    "What did I buy in order A1003 and when will it ship?",
    "Track A1005 for me please",
    "Is order A1002 delivered? If yes can I return it?",
    "My order A9999 hasn't arrived",
    "How much did I pay for A1005 and where is it now?",
    "Refund status for A1003?",
    "What's the status of A1001 and A1005?",
]
