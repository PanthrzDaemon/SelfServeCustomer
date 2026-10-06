from typing import TypedDict

from langgraph.graph import StateGraph, START, END

from app.rag import search_knowledge
from app.llm import generate_answer


# Graph ke andar jo information travel karegi
class SupportState(TypedDict):
    question: str
    context: str
    sources: list[str]
    answer: str


# RAG + Llama node
def support_node(state: SupportState):
    question = state["question"]

    # Step 1: Knowledge base se relevant documents retrieve karo
    results = search_knowledge(question, top_k=3)

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]

    # Step 2: Retrieved documents ko LLM context mein convert karo
    context = "\n\n".join(
        f"Source: {metadatas[i]['source']}\n{documents[i]}"
        for i in range(len(documents))
    )

    # Step 3: Llama ke liye grounded prompt
    prompt = f"""
You are CloudFlow's customer support assistant.

Answer the customer's question using ONLY the information
provided in the knowledge base.

Do not invent facts.

If the information is not available, say:
"I don't have enough information in the knowledge base to answer that."

Knowledge Base:
{context}

Customer Question:
{question}

Give a clear and concise answer.
"""

    # Step 4: Llama se answer generate karo
    answer = generate_answer(prompt)

    # Step 5: Graph state update karo
    return {
        "context": context,
        "sources": [metadata["source"] for metadata in metadatas],
        "answer": answer,
    }


# Graph create karo
builder = StateGraph(SupportState)

# Node add karo
builder.add_node("support", support_node)

# Workflow define karo
builder.add_edge(START, "support")
builder.add_edge("support", END)

# Graph compile karo
graph = builder.compile()


if __name__ == "__main__":
    question = "How much storage does the Pro plan provide?"

    result = graph.invoke({
        "question": question,
        "context": "",
        "sources": [],
        "answer": "",
    })

    print("\n--- QUESTION ---")
    print(result["question"])

    print("\n--- AI ANSWER ---")
    print(result["answer"])

    print("\n--- SOURCES ---")
    for source in result["sources"]:
        print("-", source)