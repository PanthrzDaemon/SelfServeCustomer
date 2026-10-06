from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition

from app.tools import (
    lookup_account,
    get_usage,
    get_invoices,
    get_plan_limits,
    check_platform_status,
)


# --------------------------------------------------
# Trusted authenticated account
# --------------------------------------------------

AUTHENTICATED_ACCOUNT_ID = "A1001"


# --------------------------------------------------
# Account tool
# --------------------------------------------------

@tool
def get_current_account():
    """
    Get the currently authenticated customer's account information.

    Use this tool when the customer asks:
    - What plan am I currently on?
    - What subscription plan do I have?
    - What is my account status?
    - What is my account information?
    - What is my name?

    IMPORTANT:
    Use this tool to identify the customer's current plan.
    Do NOT use get_current_plan_limits to identify the plan.
    """
    return lookup_account(AUTHENTICATED_ACCOUNT_ID)


# --------------------------------------------------
# Storage usage tool
# --------------------------------------------------

@tool
def get_current_usage():
    """
    Get the current storage usage of the authenticated customer.

    Use this tool when the customer asks:
    - How much storage have I used?
    - What is my storage usage?
    - How much storage is left?
    """
    return get_usage(AUTHENTICATED_ACCOUNT_ID)


# --------------------------------------------------
# Invoice tool
# --------------------------------------------------

@tool
def get_current_invoices():
    """
    Get invoices for the currently authenticated customer.

    Use this tool when the customer asks about:
    - invoices
    - billing history
    - recent payments
    - invoice status
    """
    return get_invoices(AUTHENTICATED_ACCOUNT_ID)


# --------------------------------------------------
# Plan limits tool
# --------------------------------------------------

@tool
def get_current_plan_limits():
    """
    Get the storage limit associated with the customer's
    current subscription plan.

    Use this tool when the customer asks:
    - How much storage does my plan provide?
    - What is my storage limit?
    - How much storage is included in my plan?

    Do NOT use this tool to identify the customer's plan.
    Use get_current_account for that.
    """

    account = lookup_account(AUTHENTICATED_ACCOUNT_ID)

    if not account["success"]:
        return account

    return get_plan_limits(account["plan"])


# --------------------------------------------------
# Platform status tool
# --------------------------------------------------

@tool
def get_platform_status():
    """
    Check the current CloudFlow platform status.

    Use this tool when the customer asks:
    - Is CloudFlow working?
    - Is the platform operational?
    - Is there an outage?
    - Is CloudFlow down?
    """
    return check_platform_status()


# --------------------------------------------------
# Tools available to the AI agent
# --------------------------------------------------

AGENT_TOOLS = [
    get_current_account,
    get_current_usage,
    get_current_invoices,
    get_current_plan_limits,
    get_platform_status,
]


# --------------------------------------------------
# Local Llama model
# --------------------------------------------------

llm = ChatOllama(
    model="llama3.2:3b"
)

llm_with_tools = llm.bind_tools(AGENT_TOOLS)


# --------------------------------------------------
# Agent node
# --------------------------------------------------

def call_model(state: MessagesState):
    response = llm_with_tools.invoke(state["messages"])

    print("\n--- MODEL RESPONSE ---")
    print(response)

    print("\n--- TOOL CALLS ---")
    print(response.tool_calls)

    return {
        "messages": [response]
    }


# --------------------------------------------------
# LangGraph workflow
# --------------------------------------------------

builder = StateGraph(MessagesState)

builder.add_node("agent", call_model)
builder.add_node("tools", ToolNode(AGENT_TOOLS))

builder.add_edge(START, "agent")

builder.add_conditional_edges(
    "agent",
    tools_condition
)

builder.add_edge("tools", "agent")

graph = builder.compile()


# --------------------------------------------------
# Test the agent
# --------------------------------------------------

if __name__ == "__main__":

    questions = [
        "What plan am I currently on?",
        "How much storage have I used?",
        "Is the CloudFlow platform currently operational?",
    ]

    for question in questions:

        print("\n" + "=" * 60)
        print("QUESTION:", question)

        result = graph.invoke({
            "messages": [
                {
                    "role": "user",
                    "content": question
                }
            ]
        })

        print("\nFINAL ANSWER:")
        print(result["messages"][-1].content)