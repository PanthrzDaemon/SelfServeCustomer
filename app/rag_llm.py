from app.rag import search_knowledge
from app.llm import generate_answer


def ask_question(question):
    results = search_knowledge(question, top_k=3)

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]

    context = "\n\n".join(
        f"Source: {metadatas[i]['source']}\n{documents[i]}"
        for i in range(len(documents))
    )

    prompt = f"""
You are CloudFlow's customer support assistant.

Answer the customer's question using ONLY the information provided
in the knowledge base below.

If the answer is not available in the knowledge base, say:
"I don't have enough information in the knowledge base to answer that."

Do not invent facts.

Knowledge Base:
{context}

Customer Question:
{question}

Give a clear and concise answer.
"""

    answer = generate_answer(prompt)

    return answer, metadatas


if __name__ == "__main__":
    question = "How much storage does the Pro plan provide?"

    answer, sources = ask_question(question)

    print("\n--- AI ANSWER ---")
    print(answer)

    print("\n--- SOURCES ---")
    for source in sources:
        print("-", source["source"])