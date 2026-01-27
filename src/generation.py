from __future__ import annotations


def build_prompt(context: str, question: str) -> str:
    instruction = (
        "You are a helpful research assistant.\n"
        "Task: Summarize the retrieved context to answer the user's question.\n"
        "Constraints:\n"
        "1. Start directly with bullet points. Do NOT write an introduction paragraph.\n"
        "2. Provide exactly 3-5 distinct bullet points.\n"
        "3. Stop writing immediately after the last bullet point.\n"
        "4. Do not repeat the same point in different words.\n"
        "Terminate response immediately after the last point."
    )

    return (
        f"System: {instruction}\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\nAnswer:"
    )
