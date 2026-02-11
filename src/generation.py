from __future__ import annotations

from typing import Optional

from .intent import Intent


# ---------------------------------------------------------------------------
# Intent prompts for REGULAR mode (30B model)
# ---------------------------------------------------------------------------
# NOTE: These prompts must NOT contain any citation-related instructions.
# Citations are handled separately by _CITATION_RULES and injected only
# when citations_enabled=True. Mixing them here creates conflicts.
# ---------------------------------------------------------------------------

INTENT_INSTRUCTIONS_REGULAR: dict[Intent, dict[str, str]] = {
    Intent.OVERVIEW: {
        "task": (
            "Provide a brief, high-level description of this document's type and purpose. "
            "Do NOT describe detailed findings, specific arguments, or examples."
        ),
        "format": (
            "Your first sentence MUST state what type of document this is and its primary purpose. "
            "Limit your response to a maximum of 3 sentences."
        ),
        "tone": "Neutral and concise.",
    },
    Intent.SUMMARIZE: {
        "task": (
            "Extract the main claims and findings from the context. "
            "Merge overlapping points. Report only what the document states."
        ),
        "format": (
            "Start with one sentence identifying the document. "
            "Then list 3-5 key points as bullet points. "
            "Each bullet should be one direct sentence capturing one core idea."
        ),
        "tone": "Academic but accessible.",
    },
    Intent.EXPLAIN: {
        "task": (
            "Explain the content as if teaching someone curious but with no background in this field. "
            "Use everyday language. Avoid all jargon and technical terms. "
            "Do NOT introduce facts, definitions, or topics not present in the context."
        ),
        "format": (
            "Use 2-3 short paragraphs (3-4 sentences each, keep paragraphs under 80 words). "
            "Include at least one analogy or everyday comparison to make a key idea concrete. "
            "Stop after your final clarifying sentence — no wrap-up or meta-commentary."
        ),
        "tone": "Conversational and clear.",
    },
    Intent.ANALYZE: {
        "task": (
            "First, silently identify the key themes, arguments, and tensions in the context. "
            "Then write a synthesized analysis that explains core themes, patterns, or conflicts. "
            "Go beyond description — explain WHY things are the way they are. "
            "Highlight where ideas converge and where they diverge. "
            "Cover a range of DISTINCT points — do not revisit the same example or argument more than once."
        ),
        "format": (
            "Write a cohesive narrative — do NOT use bullet points. "
            "Structure: (1) a 1-2 sentence opening framing the topic, "
            "(2) a middle section developing themes with specific evidence from the context, "
            "(3) a brief closing (1-2 sentences) drawn directly from the context — no speculation or hedging beyond what the text states. "
            "Keep to 2-4 paragraphs total. Do NOT repeat points, examples, or paraphrases across paragraphs."
        ),
        "tone": "Analytical, objective, and scholarly.",
    },
    Intent.COMPARE: {
        "task": (
            "Identify the two or more ideas, positions, theories, or documents being compared. "
            "For each, state the core claim or position clearly. "
            "Then systematically map: (1) where they agree or overlap, "
            "(2) where they diverge or conflict, (3) what drives the differences "
            "(e.g. differing assumptions, methods, or goals)."
        ),
        "format": (
            "Write a structured narrative — do NOT use bullet points. "
            "Structure: (1) a 1-2 sentence opening that names the items being compared and the axis of comparison, "
            "(2) a section on points of convergence with evidence from the context, "
            "(3) a section on points of divergence with evidence from the context, "
            "(4) a brief closing (1-2 sentences) on what the comparison reveals — grounded in the context, not speculation. "
            "Keep to 2-4 paragraphs total. Do NOT repeat points across paragraphs."
        ),
        "tone": "Balanced, precise, and scholarly.",
    },
    Intent.CRITIQUE: {
        "task": (
            "Identify the central argument or claim the user is asking about. "
            "Report how the text itself evaluates, defends, or challenges that argument. "
            "Surface any critiques, objections, limitations, or counterpoints that the text raises. "
            "If the text is one-sided, say so — do NOT invent counterarguments or limitations not present in the context. "
            "You may note what the text leaves unaddressed, but label it clearly as an omission rather than a flaw. "
            "Present a range of DISTINCT arguments from the text — do not revisit the same point or example more than once."
        ),
        "format": (
            "Write a structured evaluative narrative — do NOT use bullet points. "
            "Structure: (1) a 1-2 sentence opening that states the argument being examined, "
            "(2) a section reporting how the text supports or defends the argument with specific evidence, "
            "(3) a section on critiques, limitations, or counterpoints the text itself raises (if any), "
            "(4) a brief closing (1-2 sentences) that summarizes the text's own position — do NOT add speculative hedges or possibilities not stated in the text. "
            "Keep to 2-4 paragraphs total. Do NOT repeat points, examples, or paraphrases across paragraphs."
        ),
        "tone": "Evaluative, text-grounded, and intellectually rigorous.",
    },
    Intent.FACTUAL: {
        "task": (
            "Answer the user's question directly and concisely using ONLY the provided context. "
            "Extract the specific fact, name, date, or detail that the question asks for. "
            "If the answer is explicitly stated in the context, quote or paraphrase the relevant passage. "
            "Do NOT provide analysis, background, or tangential information."
        ),
        "format": (
            "Give the direct answer in 1-3 sentences. "
            "If helpful, include a brief quote from the context that supports the answer. "
            "Do NOT use bullet points or provide additional context beyond what is asked."
        ),
        "tone": "Direct, precise, and factual.",
    },
    Intent.COLLECTION: {
        "task": (
            "Describe the documents available in this collection based on the provided summaries. "
            "Identify the topics, themes, and scope of the corpus as a whole. "
            "Highlight how the documents relate to each other, if applicable."
        ),
        "format": (
            "Start with one sentence describing the overall scope of the collection. "
            "Then list each document with a 1-2 sentence description of its content. "
            "End with one sentence on common themes or connections between documents."
        ),
        "tone": "Informative and concise.",
    },
}

# ---------------------------------------------------------------------------
# Intent prompts for DEEP RESEARCH mode (80B model)
# ---------------------------------------------------------------------------
# The 80B model may benefit from different prompt characteristics in future
# (e.g. longer outputs, more nuanced instructions). For now these mirror
# the regular prompts so behaviour is consistent while we tune separately.
# ---------------------------------------------------------------------------

INTENT_INSTRUCTIONS_DEEP_RESEARCH: dict[Intent, dict[str, str]] = {
    Intent.OVERVIEW: {
        "task": (
            "Provide a brief, high-level description of this document's type and purpose. "
            "Do NOT describe detailed findings, specific arguments, or examples."
        ),
        "format": (
            "Your first sentence MUST state what type of document this is and its primary purpose. "
            "Limit your response to a maximum of 3 sentences."
        ),
        "tone": "Neutral and concise.",
    },
    Intent.SUMMARIZE: {
        "task": (
            "Extract the main claims and findings from the context. "
            "Merge overlapping points. Report only what the document states."
        ),
        "format": (
            "Start with one sentence identifying the document. "
            "Then list 3-5 key points as bullet points. "
            "Each bullet should be one direct sentence capturing one core idea."
        ),
        "tone": "Academic but accessible.",
    },
    Intent.EXPLAIN: {
        "task": (
            "Explain the content as if teaching someone curious but with no background in this field. "
            "Use everyday language. Avoid all jargon and technical terms. "
            "Do NOT introduce facts, definitions, or topics not present in the context."
        ),
        "format": (
            "Use 2-3 short paragraphs (3-4 sentences each, keep paragraphs under 80 words). "
            "Include at least one analogy or everyday comparison to make a key idea concrete. "
            "Stop after your final clarifying sentence — no wrap-up or meta-commentary."
        ),
        "tone": "Conversational and clear.",
    },
    Intent.ANALYZE: {
        "task": (
            "First, silently identify the key themes, arguments, and tensions in the context. "
            "Then write a synthesized analysis that explains core themes, patterns, or conflicts. "
            "Go beyond description — explain WHY things are the way they are. "
            "Highlight where ideas converge and where they diverge. "
            "Cover a range of DISTINCT points — do not revisit the same example or argument more than once."
        ),
        "format": (
            "Write a cohesive narrative — do NOT use bullet points. "
            "Structure: (1) a 1-2 sentence opening framing the topic, "
            "(2) a middle section developing themes with specific evidence from the context, "
            "(3) a brief closing (1-2 sentences) drawn directly from the context — no speculation or hedging beyond what the text states. "
            "Keep to 2-4 paragraphs total. Do NOT repeat points, examples, or paraphrases across paragraphs."
        ),
        "tone": "Analytical, objective, and scholarly.",
    },
    Intent.COMPARE: {
        "task": (
            "Identify the two or more ideas, positions, theories, or documents being compared. "
            "For each, state the core claim or position clearly. "
            "Then systematically map: (1) where they agree or overlap, "
            "(2) where they diverge or conflict, (3) what drives the differences "
            "(e.g. differing assumptions, methods, or goals)."
        ),
        "format": (
            "Write a structured narrative — do NOT use bullet points. "
            "Structure: (1) a 1-2 sentence opening that names the items being compared and the axis of comparison, "
            "(2) a section on points of convergence with evidence from the context, "
            "(3) a section on points of divergence with evidence from the context, "
            "(4) a brief closing (1-2 sentences) on what the comparison reveals — grounded in the context, not speculation. "
            "Keep to 2-4 paragraphs total. Do NOT repeat points across paragraphs."
        ),
        "tone": "Balanced, precise, and scholarly.",
    },
    Intent.CRITIQUE: {
        "task": (
            "Identify the central argument or claim the user is asking about. "
            "Report how the text itself evaluates, defends, or challenges that argument. "
            "Surface any critiques, objections, limitations, or counterpoints that the text raises. "
            "If the text is one-sided, say so — do NOT invent counterarguments or limitations not present in the context. "
            "You may note what the text leaves unaddressed, but label it clearly as an omission rather than a flaw. "
            "Present a range of DISTINCT arguments from the text — do not revisit the same point or example more than once."
        ),
        "format": (
            "Write a structured evaluative narrative — do NOT use bullet points. "
            "Structure: (1) a 1-2 sentence opening that states the argument being examined, "
            "(2) a section reporting how the text supports or defends the argument with specific evidence, "
            "(3) a section on critiques, limitations, or counterpoints the text itself raises (if any), "
            "(4) a brief closing (1-2 sentences) that summarizes the text's own position — do NOT add speculative hedges or possibilities not stated in the text. "
            "Keep to 2-4 paragraphs total. Do NOT repeat points, examples, or paraphrases across paragraphs."
        ),
        "tone": "Evaluative, text-grounded, and intellectually rigorous.",
    },
    Intent.FACTUAL: {
        "task": (
            "Answer the user's question directly and concisely using ONLY the provided context. "
            "Extract the specific fact, name, date, or detail that the question asks for. "
            "If the answer is explicitly stated in the context, quote or paraphrase the relevant passage. "
            "Do NOT provide analysis, background, or tangential information."
        ),
        "format": (
            "Give the direct answer in 1-3 sentences. "
            "If helpful, include a brief quote from the context that supports the answer. "
            "Do NOT use bullet points or provide additional context beyond what is asked."
        ),
        "tone": "Direct, precise, and factual.",
    },
    Intent.COLLECTION: {
        "task": (
            "Describe the documents available in this collection based on the provided summaries. "
            "Identify the topics, themes, and scope of the corpus as a whole. "
            "Highlight how the documents relate to each other, if applicable."
        ),
        "format": (
            "Start with one sentence describing the overall scope of the collection. "
            "Then list each document with a 1-2 sentence description of its content. "
            "End with one sentence on common themes or connections between documents."
        ),
        "tone": "Informative and concise.",
    },
}

# Backward-compatible alias (defaults to regular)
INTENT_INSTRUCTIONS = INTENT_INSTRUCTIONS_REGULAR


def _get_intent_instructions(mode: Optional[str] = None) -> dict[Intent, dict[str, str]]:
    """Return the intent instruction set for the given operating mode."""
    if mode == "power-deep-research":
        return INTENT_INSTRUCTIONS_DEEP_RESEARCH
    return INTENT_INSTRUCTIONS_REGULAR

_SYSTEM_MESSAGE = """You are a research assistant. Follow these rules strictly:

1. Use ONLY the provided context. Do not rely on outside knowledge.
2. Answer the user's SPECIFIC question — do not provide unrelated information.
3. You may synthesize ideas across multiple context chunks, even if the sources do not reference each other — this is expected. Only state "The provided context does not contain sufficient information to address this" if you genuinely cannot construct ANY relevant answer from the context. NEVER append this disclaimer after a substantive answer.
4. Stop generating immediately after completing your answer.
5. Do NOT include meta-commentary, self-evaluations, filler phrases, or sign-offs.
6. Do NOT end with speculative hedges, "future possibilities," or softening statements not grounded in the context."""

_CITATION_RULES = """
CITATION REQUIREMENTS:
- Context chunks are marked with [CHUNK START | SOURCE: SourceID | PAGE: X] or [CHUNK START | SOURCE: SourceID] (when page is unavailable)
- Extract the SourceID and PAGE (if present) from these markers
- When PAGE is present, cite as [SourceID, p. X]
- When PAGE is absent, cite as [SourceID]
- Example with page: [Chomsky_Skinner_Review, p. 1]
- Example without page: [Chomsky_Skinner_Review]
- REQUIRED: Every claim must have a citation"""


def _build_system_block(cfg: dict[str, str], citation_block: str, extra_block: str) -> str:
    return f"{_SYSTEM_MESSAGE}{citation_block}\n\nTask: {cfg['task']}\nFormat: {cfg['format']}\nTone: {cfg['tone']}{extra_block}"


def _prepare_config(intent: Optional[Intent], citations_enabled: bool, extra_instructions: Optional[str], mode: Optional[str] = None) -> tuple[dict[str, str], str, str]:
    intent = intent or Intent.OVERVIEW
    instructions = _get_intent_instructions(mode)
    cfg = instructions.get(intent, instructions[Intent.OVERVIEW]).copy()
    extra_block = f"\nAdditional constraints: {extra_instructions.strip()}" if extra_instructions and extra_instructions.strip() else ""
    citation_block = ""
    if citations_enabled:
        citation_block = f"\n{_CITATION_RULES}"
        cfg["format"] += " Include inline citations [SourceID, p. X] or [SourceID] for factual claims."
    return cfg, citation_block, extra_block


def build_messages(
    context: str,
    question: str,
    intent: Optional[Intent] = None,
    extra_instructions: Optional[str] = None,
    citations_enabled: bool = False,
    source_legend: Optional[str] = None,
    mode: Optional[str] = None,
) -> list[dict[str, str]]:
    """Build intent-aware chat messages for the LLM."""
    cfg, citation_block, extra_block = _prepare_config(intent, citations_enabled, extra_instructions, mode=mode)
    system_block = _build_system_block(cfg, citation_block, extra_block)
    legend_block = f"\n\n{source_legend}" if citations_enabled and source_legend else ""
    user_block = f"Context:\n{context}{legend_block}\n\nQuestion: {question}\n\nAnswer:"
    return [{"role": "system", "content": system_block}, {"role": "user", "content": user_block}]
