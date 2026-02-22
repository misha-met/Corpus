from __future__ import annotations

from typing import Optional

from .intent import Intent


# ---------------------------------------------------------------------------
# Ingestion-only document summary system prompt
# This is intentionally separate from the RAG query prompts — no citations,
# different style goal: orient the reader to what the document is and contains.
# ---------------------------------------------------------------------------

_INGEST_SUMMARY_SYSTEM = """You are a research assistant that writes concise document overviews.
Your job is to help a researcher quickly understand what a newly-ingested document is.

Rules:
1. Use ONLY the provided text. Do not rely on outside knowledge.
2. Do NOT include citation markers such as [1], [2], etc. — there are no numbered passages.
3. Do NOT use bullet points. Write in clear prose paragraphs.
4. Do NOT include meta-commentary, self-evaluations, or sign-offs.
5. Stop immediately after your final sentence."""

_INGEST_SUMMARY_QUESTION = (
    "Write a 3-4 paragraph summary of this document. "
    "Start the first paragraph by explaining what the document is — "
    "its type (e.g. book, report, article), its subject matter, and its scope or purpose. "
    "Open with one clear sentence (20-30 words) that names the document and states its main subject. "
    "Use the middle paragraphs to describe the main topics, arguments, or findings covered. "
    "Close with a short paragraph noting the document's relevance or usefulness for research."
)


def build_ingest_summary_messages(context: str) -> list[dict[str, str]]:
    """Build messages for document ingestion summary generation.

    Uses a dedicated prompt that is completely separate from the RAG query
    prompts — no citation rules, no intent instructions, correct style for
    describing what a document *is*.
    """
    return [
        {"role": "system", "content": _INGEST_SUMMARY_SYSTEM},
        {"role": "user", "content": f"Document text:\n{context}\n\nTask: {_INGEST_SUMMARY_QUESTION}\n\nSummary:"},
    ]


# ---------------------------------------------------------------------------
# System message with paragraph formatting rules
# ---------------------------------------------------------------------------

_SYSTEM_MESSAGE = """You are a research assistant. Follow these rules strictly:

1. Use ONLY the provided context. Do not rely on outside knowledge.
2. Answer the user's SPECIFIC question — do not provide unrelated information.
3. Write in SHORT, focused paragraphs (60-100 words each). Each paragraph should develop ONE idea.
4. Start each body paragraph with a clear topic sentence.
5. If a paragraph exceeds 100 words, split it into two paragraphs.
6. You may synthesize ideas across multiple context chunks, even if sources don't reference each other.
7. Only state "The provided context does not contain sufficient information" if you genuinely cannot construct ANY relevant answer. NEVER append this disclaimer after a substantive answer.
8. Stop generating immediately after completing your answer.
9. Do NOT include meta-commentary, self-evaluations, filler phrases, or sign-offs.
10. Do NOT end with speculative hedges or "future possibilities" not grounded in the context.
11. Never reference retrieval internals in your answer (for example: "chunk", "chunk numbers", "retrieved chunks", "vector search", "reranker", or "index").
12. Refer to evidence naturally as "the text", "the passage", "the author", or "the document"."""

_CITATION_RULES = """
CITATION REQUIREMENTS (MANDATORY):
- Context passages are numbered [PASSAGE 1 | SOURCE: ...], [PASSAGE 2 | SOURCE: ...], etc.
- You MUST cite every factual claim using the passage number: [1], [2], [3], etc.
- Place the citation immediately after the sentence it supports, before the period when possible.
- Example: "The unemployment rate rose to 5.2% [1]. Housing starts declined by 12% [2]."
- If multiple passages support one claim, cite all of them: [1][3].
- Do NOT omit citations. Every statement derived from the context MUST have at least one [N] marker."""


# ---------------------------------------------------------------------------
# Intent-specific instructions
# ---------------------------------------------------------------------------

INTENT_INSTRUCTIONS_REGULAR: dict[Intent, dict[str, str]] = {
    Intent.SUMMARIZE: {
        "task": (
            "Extract the main claims and findings from the context. "
            "Merge overlapping points. Report only what the document states."
        ),
        "format": (
            "List 3-5 key points as bullet points. "
            "Each bullet should be one direct sentence (15-35 words) capturing one core idea. "
            "When citations are enabled, follow each bullet immediately with its inline citation [N]. "
            "Do NOT add interpretation or evaluation beyond what the source states."
        ),
        "tone": "Academic but accessible.",
    },
    
    Intent.EXPLAIN: {
        "task": (
            "Explain the content as if teaching someone curious but with no background in this field. "
            "Use everyday language. Avoid all jargon and technical terms. "
            "Do NOT introduce facts, definitions, or topics not present in the context. "
            "If the source material is already non-technical, focus on clarifying the structure "
            "of the argument and its practical implications rather than simplifying vocabulary."
        ),
        "format": (
            "Write 3-5 short paragraphs (3-5 sentences each, maximum 80 words per paragraph). "
            "Each paragraph should explain ONE concept or idea. "
            "\n\n"
            "Structure:\n"
            "• Opening paragraph: Introduce the core concept in simple terms.\n"
            "• Middle paragraphs (2-3): Develop the explanation. Each paragraph = one sub-idea.\n"
            "• Final paragraph: Provide a concrete takeaway or practical implication.\n"
            "\n"
            "CRITICAL: Include at least one analogy or everyday comparison to make the key idea concrete. "
            "If a sentence feels too complex, break it into two shorter sentences. "
            "Stop after your final clarifying sentence — no wrap-up or meta-commentary."
        ),
        "tone": "Conversational and clear.",
    },
    
    Intent.ANALYZE: {
        "task": (
            "Explain the core themes, patterns, or conflicts in the context. "
            "Go beyond description — explain WHY things are the way they are. "
            "Highlight where ideas converge and where they diverge. "
            "Cover a range of DISTINCT points — do not revisit the same example or argument more than once."
        ),
        "format": (
            "Write 6-8 short, focused paragraphs (60-100 words each). Use blank lines between paragraphs. "
            "\n\n"
            "Structure:\n"
            "• Opening paragraph: Frame the topic and preview 2-3 key themes you'll analyze.\n"
            "• Body paragraphs (4-6): Each paragraph develops ONE distinct theme or pattern. "
            "Start each with a topic sentence that signals what this paragraph explores. "
            "Use specific evidence from the context. "
            "Use transitions between paragraphs (\"Building on this,\" \"A second theme,\" \"This tension reveals\").\n"
            "• Closing paragraph: Synthesize what the analysis reveals, grounded in the context.\n"
            "\n"
            "CRITICAL RULES:\n"
            "• Each paragraph = 60-100 words maximum. If a point needs more, split into two paragraphs.\n"
            "• Start body paragraphs with topic sentences.\n"
            "• Do NOT repeat points, examples, or evidence across paragraphs.\n"
            "• Do NOT use bullet points — write in prose."
        ),
        "tone": "Analytical, objective, and scholarly.",
    },
    
    Intent.COMPARE: {
        "task": (
            "Identify the two or more ideas, positions, theories, or documents being compared. "
            "For each, state the core claim or position clearly. "
            "Then systematically map: (1) where they agree or overlap, "
            "(2) where they diverge or conflict, (3) what drives the differences "
            "(e.g., differing assumptions, methods, or goals)."
        ),
        "format": (
            "Write 6-8 short, focused paragraphs (60-100 words each). Use blank lines between paragraphs. "
            "\n\n"
            "Structure:\n"
            "• Opening paragraph: State what is being compared and the axis of comparison. "
            "Name the items clearly in your first sentence.\n"
            "• Shared ground (2-3 paragraphs): Explain where they converge. "
            "Each paragraph = ONE point of convergence. "
            "Start each with a topic sentence. Use evidence from the context.\n"
            "• Key differences (2-3 paragraphs): Explain where they diverge. "
            "Each paragraph = ONE point of divergence. "
            "Use transitions (\"In contrast,\" \"However,\" \"The divergence stems from\"). "
            "Explain what drives each difference.\n"
            "• Closing paragraph: What the comparison reveals, grounded in the context.\n"
            "\n"
            "CRITICAL RULES:\n"
            "• Each paragraph = 60-100 words maximum. If explaining a complex point, split it.\n"
            "• Start body paragraphs with topic sentences.\n"
            "• Do NOT repeat points or examples across paragraphs.\n"
            "• Do NOT use bullet points — write in prose."
        ),
        "tone": "Balanced, precise, and scholarly.",
    },
    
    Intent.CRITIQUE: {
        "task": (
            "Identify the central argument or claim the user is asking about. "
            "Report how the text itself evaluates, defends, or challenges that argument. "
            "Surface any critiques, objections, limitations, or counterpoints the text raises. "
            "If the text is one-sided, say so — do NOT invent counterarguments not present in the context. "
            "You may note what the text leaves unaddressed, but label it as an omission, not a flaw. "
            "Present a range of DISTINCT arguments — do not revisit the same point more than once."
        ),
        "format": (
            "Write 5-7 short, focused paragraphs (60-100 words each). Use blank lines between paragraphs. "
            "\n\n"
            "Structure:\n"
            "• Opening paragraph: State the argument being examined.\n"
            "• Strengths (2-3 paragraphs): How the text supports or defends the argument. "
            "Each paragraph = one distinct line of support. "
            "Start each with a topic sentence (\"The text's first defense is...\").\n"
            "• Limitations (1-2 paragraphs): Critiques or counterpoints the text itself raises (if any). "
            "Use transitions (\"However,\" \"A noted limitation,\" \"The text acknowledges\").\n"
            "• Closing paragraph: Summarize the text's own position on the argument.\n"
            "\n"
            "CRITICAL RULES:\n"
            "• Each paragraph = 60-100 words maximum. One point per paragraph.\n"
            "• Start body paragraphs with topic sentences.\n"
            "• Do NOT invent critiques not in the context.\n"
            "• Do NOT repeat points or examples.\n"
            "• Do NOT use bullet points — write in prose."
        ),
        "tone": "Evaluative, text-grounded, and intellectually rigorous.",
    },
    
    Intent.FACTUAL: {
        "task": (
            "Answer the user's question directly and concisely using ONLY the provided context. "
            "Extract the specific fact, name, date, or detail the question asks for. "
            "If the answer is explicitly stated, quote or paraphrase the relevant passage. "
            "If the answer requires combining facts from multiple passages, do so explicitly "
            "and cite each passage used. "
            "Do NOT provide analysis, background, or tangential information."
        ),
        "format": (
            "Give the direct answer in 1-3 sentences. "
            "Quote the specific passage from the context that supports your answer. "
            "If the provided context does not contain the specific detail asked for, "
            "state exactly what information is missing rather than inferring or guessing. "
            "Do NOT use bullet points. Do NOT provide additional context beyond what is asked."
        ),
        "tone": "Direct, precise, and factual.",
    },
    
    Intent.COLLECTION: {
        "task": (
            "Describe the documents available in this collection based on the provided summaries. "
            "Identify the topics, themes, and scope of the corpus as a whole. "
            "Mention document types (e.g., article, report, book chapter, memo) where apparent. "
            "Note any date ranges visible in the material. "
            "Flag any obvious gaps in corpus coverage (e.g., missing perspectives, periods, or formats). "
            "Highlight how the documents relate to each other, if applicable."
        ),
        "format": (
            "Write 3-5 short paragraphs (60-80 words each). "
            "\n\n"
            "Structure:\n"
            "• Opening paragraph: Describe the overall scope and focus of the collection, including document types and date ranges where apparent.\n"
            "• Middle paragraphs (1-3): Briefly describe each major document or topic cluster. "
            "One paragraph per document or theme.\n"
            "• Closing paragraph: Note common themes or connections between documents, and flag any notable gaps in coverage.\n"
            "\n"
            "Do NOT use bullet points. Write in prose."
        ),
        "tone": "Informative and concise.",
    },

    Intent.EXTRACT: {
        "task": (
            "Identify what type of entities the user is requesting (names, dates, figures, "
            "definitions, locations, events, etc.). "
            "Exhaustively extract every matching instance from the provided context. "
            "Do NOT paraphrase or interpret — report what is in the text. "
            "Do NOT omit instances even if they seem minor."
        ),
        "format": (
            "Output a clean, structured list or table. "
            "\n\n"
            "If extracting a single entity type: use a simple numbered or bulleted list. "
            "Each entry = one item on its own line. No extra prose around the list. "
            "\n\n"
            "If extracting multiple entity types: use a small table or grouped sub-lists, "
            "one section per entity type. Label each section clearly. "
            "\n\n"
            "Begin directly with the list or table — no preamble. "
            "After the list, add a single line: \"Total: N items found.\" "
            "If no matching entities are found, state: \"No [entity type] found in the provided context.\""
        ),
        "tone": "Precise and exhaustive.",
    },

    Intent.TIMELINE: {
        "task": (
            "Identify all events and dates mentioned in the context. "
            "Arrange them in chronological order. "
            "Note any causal links between events where the text makes them explicit. "
            "Flag any gaps in the chronology where the sequence is unclear or discontinuous. "
            "Do NOT infer dates not present in the context."
        ),
        "format": (
            "Present events as a chronological list. "
            "\n\n"
            "Each entry should follow this format: "
            "  [DATE / PERIOD] — [Event description (1-2 sentences max)]. "
            "Use '→' to indicate a direct causal link to the next event when the text supports it. "
            "\n\n"
            "If a date is approximate or a range, indicate this clearly (e.g., 'c. 1850', '1840s–1860s'). "
            "After the timeline, add a brief note (1-2 sentences) on any significant gaps or missing dates. "
            "Begin directly with the first item — no preamble."
        ),
        "tone": "Factual and sequential.",
    },

    Intent.HOW_TO: {
        "task": (
            "Identify the process, procedure, or set of steps described in the source material "
            "that the user is asking about. "
            "Present the steps in the order they appear in or are implied by the text. "
            "Focus on the process itself — not on conceptual background or theory. "
            "This intent is distinct from EXPLAIN: do not simply simplify vocabulary; "
            "focus on sequence and actionable steps."
        ),
        "format": (
            "Present the procedure as a numbered step-by-step list. "
            "\n\n"
            "Each step: [Step N]: [Clear, imperative action statement]. "
            "After the action statement, add 1-2 sentences explaining what happens in that step "
            "and why it matters (based strictly on the source material). "
            "\n\n"
            "If the source describes sub-steps or stages within a main step, use indented bullet points. "
            "Begin directly with Step 1 — no preamble. "
            "Close with a single sentence summarising the outcome of the complete procedure."
        ),
        "tone": "Clear, practical, and procedural.",
    },

    Intent.QUOTE_EVIDENCE: {
        "task": (
            "Identify the claim, argument, or question the user wants evidence for. "
            "Find direct quotes from the context that support or illuminate it. "
            "Minimise paraphrase — let the text speak for itself. "
            "Organise quotes by relevance to the user's query, most relevant first. "
            "Do NOT fabricate or modify quotes."
        ),
        "format": (
            "Present each piece of evidence as a block quote, followed by a one-sentence "
            "explanation of how it supports the user's query. "
            "\n\n"
            "Format each entry as:\n"
            "  > \"[Exact quote from the source.]\"\n"
            "  [One sentence: why this quote is relevant to the query.]\n"
            "\n"
            "Separate entries with a blank line. "
            "Begin directly with the first quote — no preamble. "
            "If more than 5 quotes are found, include only the 5 most directly relevant ones. "
            "If no relevant quotes can be found in the context, state: "
            "\"No direct quotes supporting this claim were found in the provided context.\""
        ),
        "tone": "Precise and textually grounded.",
    },
}

# ---------------------------------------------------------------------------
# Deep Research mode (currently mirrors regular)
# ---------------------------------------------------------------------------

INTENT_INSTRUCTIONS_DEEP_RESEARCH: dict[Intent, dict[str, str]] = {
    intent: {k: v for k, v in cfg.items()}
    for intent, cfg in INTENT_INSTRUCTIONS_REGULAR.items()
}

# Backward-compatible alias
INTENT_INSTRUCTIONS = INTENT_INSTRUCTIONS_REGULAR


def _get_intent_instructions(mode: Optional[str] = None) -> dict[Intent, dict[str, str]]:
    """Return the intent instruction set for the given operating mode."""
    if mode == "power-deep-research":
        return INTENT_INSTRUCTIONS_DEEP_RESEARCH
    return INTENT_INSTRUCTIONS_REGULAR


def _build_system_block(cfg: dict[str, str], citation_block: str, extra_block: str) -> str:
    return f"{_SYSTEM_MESSAGE}{citation_block}\n\nTask: {cfg['task']}\nFormat: {cfg['format']}\nTone: {cfg['tone']}{extra_block}"


def _prepare_config(
    intent: Optional[Intent],
    citations_enabled: bool,
    extra_instructions: Optional[str],
    mode: Optional[str] = None
) -> tuple[dict[str, str], str, str]:
    intent = intent or Intent.SUMMARIZE
    instructions = _get_intent_instructions(mode)
    cfg = instructions.get(intent, instructions[Intent.SUMMARIZE]).copy()
    extra_block = (
        f"\nAdditional constraints: {extra_instructions.strip()}"
        if extra_instructions and extra_instructions.strip()
        else ""
    )
    citation_block = ""
    if citations_enabled:
        citation_block = f"\n{_CITATION_RULES}"
        cfg["format"] += (
            " Include numbered inline citations [1], [2], etc. after every factual claim. "
            "Treat citation numbers as references to provided passages; do not use words like 'chunk' or mention retrieval internals."
        )
    return cfg, citation_block, extra_block


def build_messages(
    context: str,
    question: str,
    intent: Optional[Intent] = None,
    extra_instructions: Optional[str] = None,
    citations_enabled: bool = False,
    source_legend: Optional[str] = None,
    mode: Optional[str] = None,
    retrieval_budget: Optional[int] = None,
) -> list[dict[str, str]]:
    """Build intent-aware chat messages for the LLM."""
    cfg, citation_block, extra_block = _prepare_config(
        intent, citations_enabled, extra_instructions, mode=mode
    )
    system_block = _build_system_block(cfg, citation_block, extra_block)

    # Context-sparsity warning: if context fills < 10% of retrieval budget
    if retrieval_budget and retrieval_budget > 0 and context:
        context_tokens = len(context.split())
        fill_ratio = context_tokens / retrieval_budget
        if fill_ratio < 0.10:
            sparsity_warning = (
                "WARNING: Limited source material was retrieved for this query. "
                "Base your answer strictly and exclusively on the passages provided. "
                "Do not infer, extrapolate, or add details not explicitly present in the context. "
                "If the context is insufficient to fully answer the question, clearly state "
                "what cannot be determined."
            )
            system_block = sparsity_warning + "\n\n" + system_block

    legend_block = f"\n\n{source_legend}" if citations_enabled and source_legend else ""
    # Reminder injected right before "Answer:" — at the generation trigger point where
    # model attention peaks — to combat first-run variability where citation rules
    # buried in the system prompt are sometimes insufficiently weighted.
    citation_reminder = (
        "\n\nIMPORTANT: You MUST include inline citation numbers [1], [2], etc. "
        "after every factual claim. Use the PASSAGE numbers from the context above."
        if citations_enabled else ""
    )
    user_block = f"Context:\n{context}{legend_block}\n\nQuestion: {question}{citation_reminder}\n\nAnswer:"
    return [
        {"role": "system", "content": system_block},
        {"role": "user", "content": user_block}
    ]
