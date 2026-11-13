"""
Grounded Prompt Templates for Telugu and Multilingual RAG Generation.
"""

from typing import List
from app.schemas import RetrievedChunk

RAG_SYSTEM_PROMPT = """You are a precise, grounded AI assistant for a research and enterprise knowledge platform.
Your primary role is to answer questions strictly and faithfully based on the retrieved document context provided below.

Strict Operational Guidelines:
1. Grounding: Rely ONLY on the information explicitly stated in the context. Do NOT extrapolate, speculate, or incorporate outside facts.
2. Missing Information: If the provided context does not contain sufficient information to answer the question, explicitly state:
   - In English: "The provided document does not contain information to answer this question."
   - In Telugu (if query is in Telugu): "అందించిన పత్రంలో ఈ ప్రశ్నకు సమాధానం ఇవ్వడానికి తగిన సమాచారం లేదు."
3. Language & Telugu Support:
   - If the user asks in Telugu, respond in fluent, grammatically correct Telugu.
   - If the user asks in English or Telugu-English mixed text (e.g. "ఈ document ప్రకారం minimum attendance ఎంత?"), respond helpfully in the appropriate language (Telugu or English as queried).
4. Citations & Attribution: Cite the source document name and page number when referring to specific facts or rules from the context.
5. Tone: Maintain a professional, concise, and academic tone.
"""


def format_context_prompt(query: str, retrieved_chunks: List[RetrievedChunk]) -> str:
    """
    Format retrieved document chunks into a structured context block for the LLM.
    """
    if not retrieved_chunks:
        return f"""User Question:
{query}

Retrieved Context:
[No relevant document chunks were found in the knowledge base.]

Please formulate your response based strictly on the above context instruction."""

    context_blocks = []
    for idx, chunk in enumerate(retrieved_chunks, start=1):
        block = (
            f"[Source {idx}: {chunk.source} | Page: {chunk.page} | Relevance Score: {chunk.score:.2f}]\n"
            f"{chunk.text}"
        )
        context_blocks.append(block)

    formatted_context = "\n\n".join(context_blocks)

    user_prompt = f"""Retrieved Document Context:
========================================
{formatted_context}
========================================

User Question:
{query}

Please provide a well-grounded answer based strictly on the context above, citing the relevant source(s) and page number(s)."""

    return user_prompt
