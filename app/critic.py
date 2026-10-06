from langchain_ollama import ChatOllama


llm = ChatOllama(
    model="llama3.2:3b"
)


def critique_answer(question: str, answer: str, context: str = ""):
    """
    Evaluate whether an AI-generated support answer is
    correct, grounded, and safe.
    """

    prompt = f"""
You are a strict quality-control critic for CloudFlow
customer support.

Customer Question:
{question}

Available Context:
{context}

Generated Answer:
{answer}

Evaluate the generated answer.

Check:

1. Groundedness:
   Is every important factual claim supported by the context?

2. Correctness:
   Does the answer directly and correctly answer the question?

3. Hallucination:
   Does the answer contain unsupported facts?

4. Safety:
   Does the answer expose passwords, tokens, credentials,
   or other sensitive information?

Use EXACTLY one decision:

PASS
REVISE
ESCALATE

Decision rules:

PASS:
Use PASS when the answer is factually correct, supported
by the context, directly answers the question, and is safe.

REVISE:
Use REVISE ONLY when there is an actual factual,
grounding, relevance, or safety problem that can be
fixed using the available context.

Do NOT choose REVISE just because the answer could be
longer, more detailed, or more polite.

ESCALATE:
Use ESCALATE when the question cannot be safely or
reliably answered using the available information.

Return exactly this format:

Decision: PASS
Reason: The answer is correct, grounded, and safe.
"""

    response = llm.invoke(prompt)

    return response.content


if __name__ == "__main__":

    question = "How much storage does the Pro plan provide?"

    context = """
    The Pro plan includes 100 GB of storage.
    """

    answer = "The Pro plan includes 100 GB of storage."

    result = critique_answer(
        question,
        answer,
        context
    )

    print("\n--- CRITIC RESULT ---")
    print(result)