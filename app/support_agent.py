from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition

from app.rag import search_knowledge
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
# RAG Knowledge Base Tool
# --------------------------------------------------

@tool
def search_knowledge_base(question: str):
    """
    Search the CloudFlow knowledge base for help-center
    and documentation information.

    Use this tool for questions about:
    - CloudFlow policies
    - product features
    - storage limits
    - password reset
    - billing policies
    - general CloudFlow documentation
    """
    results = search_knowledge(question, top_k=3)

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]

    knowledge = []

    for i in range(len(documents)):
        knowledge.append({
            "source": metadatas[i]["source"],
            "content": documents[i],
        })

    return knowledge


# --------------------------------------------------
# Account Tool
# --------------------------------------------------

@tool
def get_current_account():
    """
    Get information about the currently authenticated
    CloudFlow customer.

    Use this tool for questions such as:
    - What plan am I on?
    - What is my account status?
    - What is my account information?
    """
    return lookup_account(AUTHENTICATED_ACCOUNT_ID)


# --------------------------------------------------
# Usage Tool
# --------------------------------------------------

@tool
def get_current_usage():
    """
    Get storage usage for the currently authenticated
    CloudFlow customer.
    """
    return get_usage(AUTHENTICATED_ACCOUNT_ID)


# --------------------------------------------------
# Invoice Tool
# --------------------------------------------------

@tool
def get_current_invoices():
    """
    Get invoices for the currently authenticated
    CloudFlow customer.
    """
    return get_invoices(AUTHENTICATED_ACCOUNT_ID)


# --------------------------------------------------
# Plan Limits Tool
# --------------------------------------------------

@tool
def get_current_plan_limits():
    """
    Get the storage limit associated with the
    customer's current plan.

    Use this when the customer asks how much storage
    their plan provides.

    Do not use this to identify the customer's plan.
    """
    account = lookup_account(AUTHENTICATED_ACCOUNT_ID)

    if not account["success"]:
        return account

    return get_plan_limits(account["plan"])


# --------------------------------------------------
# Platform Status Tool
# --------------------------------------------------

@tool
def get_platform_status():
    """
    Check whether the CloudFlow platform is operational.
    """
    return check_platform_status()


# --------------------------------------------------
# All Agent Tools
# --------------------------------------------------

AGENT_TOOLS = [
    search_knowledge_base,
    get_current_account,
    get_current_usage,
    get_current_invoices,
    get_current_plan_limits,
    get_platform_status,
]


# --------------------------------------------------
# Local Llama Model
# --------------------------------------------------

llm = ChatOllama(
    model="llama3.2:3b"
)

llm_with_tools = llm.bind_tools(AGENT_TOOLS)


# --------------------------------------------------
# Agent Node
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
# LangGraph Workflow
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
# Test
# --------------------------------------------------

if __name__ == "__main__":

    questions = [
        "How much storage does the Pro plan provide?",
        "How much storage have I used?",
        "What plan am I currently on?",
        "Can I reset my CloudFlow password?",
        "Is the CloudFlow platform operational?",
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