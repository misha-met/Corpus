from __future__ import annotations

from typing import Optional

from .intent import Intent


# Intent-specific instructions (v1.2 - with meta-commentary suppression)
# NOTE: Bullet points are NOT enforced by default. Format is flexible and intent-driven.
INTENT_INSTRUCTIONS: dict[Intent, dict[str, str]] = {
    # OVERVIEW: Bird's-eye view - concise, purpose-driven
    Intent.OVERVIEW: {
        "task": "Provide a brief, high-level description of this document.",
        "format": (
            "Your first sentence MUST state what type of document this is and its primary purpose "
            "(e.g., 'This is a critical review of...', 'This is a research paper that examines...'). "
            "Keep your response to 1 short paragraph OR a maximum of 3 concise bullet points. "
            "Do NOT go into detailed arguments or methodology. "
            "Do NOT include page numbers, document headers, or citation markers. "
            "Prefer no more than 3 bullet points."
        ),
        "tone": "Neutral and concise. Think 'back-of-book blurb' level.",
    },
    # SUMMARIZE: Detailed summary with key points
    Intent.SUMMARIZE: {
        "task": "Provide a structured summary of the key points from the context.",
        "format": (
            "Start with one sentence stating what the document is. "
            "Then provide 3-5 distinct key points. Use bullet points if listing multiple items. "
            "Each point should be unique - do not repeat the same idea in different words. "
            "Do NOT include page numbers or citation markers."
        ),
        "tone": "Academic but accessible.",
    },
    # EXPLAIN: Simple language for non-experts (analogy-focused)
    Intent.EXPLAIN: {
        "task": "Explain the content in simple, non-technical language.",
        "format": (
            "Use short paragraphs. Avoid jargon. Use at least one analogy to clarify the main concept. "
            "The analogy should be the core of your explanation. "
            "Stop immediately after your final clarifying sentence. Do NOT include page numbers."
        ),
        "tone": "Conversational, as if explaining to a curious friend.",
    },
    # ANALYZE: Critique, controversy, evaluation (synthesis-focused)
    Intent.ANALYZE: {
        "task": "Analyze and synthesize the specific aspect the user is asking about.",
        "format": (
            "Present a synthesized analysis with evidence and reasoning - do NOT use bullet points. "
            "Write in flowing paragraphs that connect ideas. "
            "Acknowledge multiple perspectives if relevant."
        ),
        "tone": "Thoughtful and balanced.",
    },
}

# System message with grounding rules and meta-commentary suppression
_SYSTEM_MESSAGE = """You are a helpful research assistant. Follow these rules strictly:

1. Base your answer ONLY on the provided context below.
2. Answer the user's SPECIFIC question - do not default to a generic summary.
3. If the context doesn't contain relevant information, say so.
4. Stop generating after completing your answer.

CRITICAL: Provide the response directly. Do NOT include:
- Meta-commentary about your response
- Self-evaluations or descriptions of how you addressed the prompt
- Phrases like "Answer ends here", "This response reflects...", "Note:", "Overall,"
- Any text explaining what you did or why"""

# Citation rules for Academic Mode
_CITATION_RULES = """
CITATION REQUIREMENTS (Academic Mode):
When citing information from the context, use the following format:
- With page number: [SourceID, p. X] where X is the page number from the chunk header
- Without page number: [SourceID] as fallback when no page is available

Rules:
1. Every factual claim from the context MUST include a citation
2. Use the SOURCE and PAGE values from the [CHUNK START] markers
3. Multiple citations can be combined: [Source1, p. 5], [Source2, p. 12]
4. Place citations immediately after the relevant statement
5. The Source Legend at the end maps SourceIDs to document names"""


def build_prompt(
    context: str,
    question: str,
    intent: Optional[Intent] = None,
    extra_instructions: Optional[str] = None,
    citations_enabled: bool = False,
    source_legend: Optional[str] = None,
) -> str:
    """
    Build an intent-aware prompt for the LLM.
    
    Structure: System instructions -> Task/Format/Tone -> Context -> Question
    All instructions are placed BEFORE the context to prevent model continuation.
    
    Args:
        context: Retrieved document context
        question: User's original question
        intent: Classified intent (defaults to OVERVIEW if None)
        extra_instructions: Additional constraints to include
        citations_enabled: Whether to include citation rules (Academic Mode)
        source_legend: Optional source ID to document name mapping
    
    Returns:
        Formatted prompt string
    """
    if intent is None:
        intent = Intent.OVERVIEW
    
    cfg = INTENT_INSTRUCTIONS.get(intent, INTENT_INSTRUCTIONS[Intent.OVERVIEW])
    
    # Build system block with all instructions BEFORE context
    extra_block = ""
    if extra_instructions and extra_instructions.strip():
        extra_block = f"\nAdditional constraints: {extra_instructions.strip()}"
    
    # Add citation rules if enabled
    citation_block = ""
    if citations_enabled:
        citation_block = f"\n{_CITATION_RULES}"
        # Override format instructions to include citation requirement
        format_instructions = cfg['format']
        if "Do NOT include page numbers" in format_instructions:
            # Remove the "do not include citations" instruction for citation mode
            format_instructions = format_instructions.replace(
                "Do NOT include page numbers, document headers, or citation markers. ", ""
            ).replace(
                "Do NOT include page numbers or citation markers.", ""
            ).replace(
                "Do NOT include page numbers.", ""
            )
        format_instructions += " Include inline citations [SourceID, p. X] for factual claims."
        cfg = {**cfg, 'format': format_instructions}
    
    # Build system block - only add citation_block if citations are enabled
    # to avoid unnecessary newlines when in casual mode
    if citation_block:
        system_block = (
            f"{_SYSTEM_MESSAGE}"
            f"{citation_block}\n\n"
            f"Task: {cfg['task']}\n"
            f"Format: {cfg['format']}\n"
            f"Tone: {cfg['tone']}"
            f"{extra_block}"
        )
    else:
        system_block = (
            f"{_SYSTEM_MESSAGE}\n\n"
            f"Task: {cfg['task']}\n"
            f"Format: {cfg['format']}\n"
            f"Tone: {cfg['tone']}"
            f"{extra_block}"
        )

    # Add source legend if provided and citations enabled
    legend_block = ""
    if citations_enabled and source_legend:
        legend_block = f"\n\n{source_legend}"

    # User block contains only context and question - no trailing instructions
    return (
        f"System: {system_block}\n\n"
        f"Context:\n{context}{legend_block}\n\n"
        f"Question: {question}\n\n"
        f"Answer:"
    )


def build_messages(
    context: str,
    question: str,
    intent: Optional[Intent] = None,
    extra_instructions: Optional[str] = None,
    citations_enabled: bool = False,
    source_legend: Optional[str] = None,
) -> list[dict[str, str]]:
    """
    Build intent-aware chat messages for the LLM.

    Returns a list of role/content dicts suitable for chat templates.
    """
    if intent is None:
        intent = Intent.OVERVIEW

    cfg = INTENT_INSTRUCTIONS.get(intent, INTENT_INSTRUCTIONS[Intent.OVERVIEW])

    extra_block = ""
    if extra_instructions and extra_instructions.strip():
        extra_block = f"\nAdditional constraints: {extra_instructions.strip()}"

    citation_block = ""
    if citations_enabled:
        citation_block = f"\n{_CITATION_RULES}"
        format_instructions = cfg["format"]
        if "Do NOT include page numbers" in format_instructions:
            format_instructions = format_instructions.replace(
                "Do NOT include page numbers, document headers, or citation markers. ", ""
            ).replace(
                "Do NOT include page numbers or citation markers.", ""
            ).replace(
                "Do NOT include page numbers.", ""
            )
        format_instructions += " Include inline citations [SourceID, p. X] for factual claims."
        cfg = {**cfg, "format": format_instructions}

    if citation_block:
        system_block = (
            f"{_SYSTEM_MESSAGE}"
            f"{citation_block}\n\n"
            f"Task: {cfg['task']}\n"
            f"Format: {cfg['format']}\n"
            f"Tone: {cfg['tone']}"
            f"{extra_block}"
        )
    else:
        system_block = (
            f"{_SYSTEM_MESSAGE}\n\n"
            f"Task: {cfg['task']}\n"
            f"Format: {cfg['format']}\n"
            f"Tone: {cfg['tone']}"
            f"{extra_block}"
        )

    legend_block = ""
    if citations_enabled and source_legend:
        legend_block = f"\n\n{source_legend}"

    user_block = (
        f"Context:\n{context}{legend_block}\n\n"
        f"Question: {question}\n\n"
        f"Answer:"
    )

    return [
        {"role": "system", "content": system_block},
        {"role": "user", "content": user_block},
    ]


# Legacy function for backward compatibility
def build_prompt_legacy(context: str, question: str) -> str:
    """Original prompt builder (deprecated, kept for reference)."""
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
