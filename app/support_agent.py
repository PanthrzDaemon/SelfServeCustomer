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
from app.critic import critique_answer


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
    Search CloudFlow documentation and help-center articles.

    IMPORTANT:
    Use this tool for general product/documentation questions,
    including:
    - storage limits
    - Pro plan features
    - Business plan features
    - Enterprise plan features
    - password reset policies
    - billing policies

    For example, use this tool when the customer asks:
    "How much storage does the Pro plan provide?"

    Do NOT use customer-specific backend tools for general
    documentation questions.
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

    Use this for questions about:
    - current plan
    - account status
    - account information
    """
    return lookup_account(AUTHENTICATED_ACCOUNT_ID)


# --------------------------------------------------
# Usage Tool
# --------------------------------------------------
@tool
def get_current_usage():
    """
    Get ONLY the amount of storage currently USED by the
    authenticated customer.

    IMPORTANT:
    This tool does NOT tell how much storage the customer's
    plan PROVIDES.

    Use this tool ONLY for questions like:
    - How much storage have I used?
    - How much storage am I currently using?
    - What percentage of storage have I used?

    Do NOT use this tool for:
    - How much storage does my plan provide?
    - What is my storage limit?
    - How much storage is included in Pro?
    """
    return get_usage(AUTHENTICATED_ACCOUNT_ID)

# --------------------------------------------------
# Invoice Tool
# --------------------------------------------------

@tool
def get_current_invoices():
    """
    Get invoices for the currently authenticated customer.
    """
    return get_invoices(AUTHENTICATED_ACCOUNT_ID)


# --------------------------------------------------
# Plan Limits Tool
# --------------------------------------------------

@tool
def get_current_plan_limits():
    """
    Get the storage limit associated with the customer's
    current subscription plan.

    Use this when asking how much storage the plan provides.
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
# Agent Tools
# --------------------------------------------------

AGENT_TOOLS = [
    search_knowledge_base,
    get_current_account,
    get_current_usage,
    get_current_invoices,
    # get_current_plan_limits,
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
# Run Support Question
# --------------------------------------------------

def run_support_question(question: str):

    result = graph.invoke({
        "messages": [
            {
                "role": "user",
                "content": question
            }
        ]
    })

    answer = result["messages"][-1].content

    # Collect tool results as critic context
    context_parts = []

    for message in result["messages"]:
        if message.type == "tool":
            context_parts.append(message.content)

    context = "\n\n".join(context_parts)

    # Run critic
    critic_result = critique_answer(
        question=question,
        answer=answer,
        context=context
    )

    return answer, critic_result


# --------------------------------------------------
# Test
# --------------------------------------------------

if __name__ == "__main__":

    question = "How much storage does the Pro plan provide?"

    answer, critic_result = run_support_question(question)

    print("\n" + "=" * 60)

    print("\nQUESTION:")
    print(question)

    print("\nFINAL ANSWER:")
    print(answer)

    print("\nCRITIC RESULT:")
    print(critic_result)