"""Prompt construction for the RAG query pipeline.

Builds the ``messages`` list passed to ``MlxGenerator`` for every query type.

Architecture
~~~~~~~~~~~~
- One ``build_messages()`` function is the single entry point for the query
  pipeline; it selects system message, citation rules, and intent-specific
  instructions then formats them into a chat-template-ready list.
- Intent instructions live in ``INTENT_INSTRUCTIONS_REGULAR`` and
  ``INTENT_INSTRUCTIONS_DEEP_RESEARCH`` — keyed dicts of ``task``,
  ``format``, and ``tone`` strings merged at call time.
- The ingest summary prompt (``build_ingest_summary_messages()``) is
  intentionally separate: no citation rules, different style goal, called
  only from ``RagEngine.ingest()``.
"""
from __future__ import annotations

import logging
from typing import Optional

from .intent import Intent

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Ingestion-only document summary system prompt
# This is intentionally separate from the RAG query prompts — no citations,
# different style goal: orient the reader to what the document is and contains.
# ---------------------------------------------------------------------------

_INGEST_SUMMARY_SYSTEM = """You are a research assistant that writes concise document summaries.

Rules:
1. Use ONLY the provided text. Do not rely on outside knowledge.
2. Write in plain prose. No bullet points, numbered lists, or inline references like [1] or (Author, Year).
3. No meta-commentary, sign-offs, or opinions on the document.
4. Stop after your final sentence."""

_INGEST_SUMMARY_QUESTION = (
    "Write a 3-4 paragraph summary of this document. "
    "Start the first paragraph by explaining what the document is — "
    "its type (e.g. book, report, article), its subject matter, and its scope or purpose. "
    "Open with 1-2 sentences that name the document and state its main subject. "
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
        {"role": "user", "content": f"Document text:\n{context}\n\nTask: {_INGEST_SUMMARY_QUESTION}"},
    ]


# ---------------------------------------------------------------------------
# System message with paragraph formatting rules
# ---------------------------------------------------------------------------

_SYSTEM_MESSAGE = """You are a research assistant. Follow these rules strictly:

1. Use ONLY the provided context. Do not rely on outside knowledge.
2. Answer the user's SPECIFIC question — do not provide unrelated information.
3. Write in short, focused paragraphs (3-5 sentences each). One idea per paragraph.
4. You may synthesize ideas across multiple passages even if they don't reference each other.
5. Only say "The provided context does not contain sufficient information" if you genuinely cannot answer at all. Never append this after a substantive answer.
6. Do not reference retrieval internals (chunks, vectors, reranker). Refer to sources as "the text", "the passage", or "the author".
7. Stop after completing your answer. No meta-commentary, sign-offs, or filler."""

_CITATION_RULES = """
CITATION REQUIREMENTS (MANDATORY):
- Context is provided as numbered [PASSAGE N] blocks with headers like [Source: ... | Page ... | Chunk ...].
- You MUST cite the primary supporting passage for each distinct claim using passage numbers: [1], [2], [3], etc.
- Cite the specific page whose chunk text supports the claim. Do not guess adjacent pages.
- If a claim is supported by the chunk labeled Page 34, cite that passage for Page 34, not Page 33 or 35.
- If multiple passages support one claim, cite all relevant passages: [1][3].
- Place citations inline immediately after the supported sentence (not as one citation block at the end).
- Do NOT omit source grounding. Every evidence-backed claim should include at least one [N] marker."""

_BENCHMARK_CITATION_RULES = """
CITATION REQUIREMENTS (MANDATORY):
- Context is provided as numbered [PASSAGE N] blocks with headers like [Source: ... | Page ... | Chunk ...].
- Every factual claim MUST use inline citations in this exact format: [N, p.XX].
- XX MUST be the nearest preceding [Page N] marker in that passage's text.
- If a claim is supported by a passage section marked [Page 34], cite [N, p.34], not [N, p.33] or [N, p.35].
- If multiple passages support one claim, cite all of them, e.g. [1, p.12][3, p.14].
- Place citations inline immediately after each supported sentence, not as an end-of-answer citation list.
- Do NOT omit source grounding. Every evidence-backed claim should include at least one [N, p.XX] marker."""

_CITATION_OUTPUT_DEFAULT = "default"
_CITATION_OUTPUT_BENCHMARK_PAGE = "benchmark_page"


# ---------------------------------------------------------------------------
# Intent-specific instructions
# ---------------------------------------------------------------------------

INTENT_INSTRUCTIONS_REGULAR: dict[Intent, dict[str, str]] = {
    Intent.OVERVIEW: {
        "task": (
            "Provide a high-level overview of the topic covered in the context. "
            "Identify the subject, scope, and main points without deep analysis. "
            "Report what the source covers, not what it argues."
        ),
        "format": (
            "Write 3-4 short paragraphs (3-5 sentences each). "
            "Open with what the source is about and its scope. "
            "Use middle paragraphs to describe the main topics or sections covered. "
            "Close with the source's relevance or key takeaway.\n\n"
            "Write in prose \u2014 no bullet points."
        ),
        "tone": "Informative and concise.",
    },

    Intent.SUMMARIZE: {
        "task": (
            "Extract and consolidate the main claims and findings from the context. "
            "Merge overlapping points. Report only what the source states."
        ),
        "format": (
            "List 3-5 key points as bullet points. "
            "Each bullet should be 1-2 clear sentences capturing one core idea. "
            "When citations are enabled, follow each bullet immediately with its inline citation [N]. "
            "Do NOT add interpretation or evaluation beyond what the source states."
        ),
        "tone": "Academic but accessible.",
    },
    
    Intent.EXPLAIN: {
        "task": (
            "Answer the user's question in plain, accessible language. Avoid jargon. "
            "Do NOT introduce facts, definitions, or topics not present in the context. "
            "If the source is already non-technical, focus on clarifying the argument's "
            "structure and practical implications rather than just simplifying vocabulary."
        ),
        "format": (
            "Write 3-5 short paragraphs (3-4 sentences maximum per paragraph). "
            "Each paragraph should explain ONE concept or idea.\n\n"
            "Structure:\n"
            "• Opening paragraph: Introduce the core concept in simple terms.\n"
            "• Middle paragraphs (2-3): Develop the explanation. Each paragraph = one sub-idea.\n"
            "• Final paragraph: Provide a concrete takeaway or practical implication.\n\n"
            "Where possible, use an analogy or everyday comparison to make the key idea concrete. "
            "If the context does not suggest one naturally, focus on clear, simple language instead. "
            "Stop after your final clarifying sentence — no wrap-up or meta-commentary."
        ),
        "tone": "Conversational and clear.",
    },
    
    Intent.ANALYZE: {
        "task": (
            "Answer the user's analytical question using evidence from the context. "
            "Explain causes, relationships, and significance — go beyond description to explain why."
        ),
        "format": (
            "Write 4-6 short paragraphs (3-5 sentences each). "
            "Use blank lines between paragraphs.\n\n"
            "Open by framing the question and previewing your key points. "
            "Develop one distinct point per body paragraph, each starting with a topic sentence. "
            "Use transitions between paragraphs (\"Building on this,\" \"A second theme,\" "
            "\"This tension reveals\"). "
            "Close by synthesizing what the analysis reveals.\n\n"
            "Use topic sentences. Write in prose — no bullet points."
        ),
        "tone": "Analytical, objective, and scholarly.",
    },
    
    Intent.COMPARE: {
        "task": (
            "Answer the user's comparison question. "
            "For each item compared, state the core claim or position clearly. "
            "Then systematically identify where they converge, diverge, and what drives the differences."
        ),
        "format": (
            "Write 4-7 short paragraphs (3-5 sentences each). "
            "Use blank lines between paragraphs.\n\n"
            "Open by stating what is being compared and on what basis. "
            "Then discuss convergences and divergences — one point per paragraph, "
            "each starting with a topic sentence. Allocate paragraphs based on "
            "where the substance is, not a fixed structure. "
            "Use transitions (\"In contrast,\" \"However,\" \"The divergence stems from\"). "
            "Close with what the comparison reveals.\n\n"
            "Write in prose — no bullet points."
        ),
        "tone": "Balanced, precise, and scholarly.",
    },
    
    Intent.CRITIQUE: {
        "task": (
            "Answer the user's question by reporting how the text itself evaluates, defends, "
            "or challenges the argument in question. "
            "Surface critiques, objections, or limitations the text raises. "
            "If the text is one-sided, say so — do NOT invent counterarguments not present in the context. "
            "You may note what the text leaves unaddressed, but label it as an omission, not a flaw."
        ),
        "format": (
            "Write 4-6 short paragraphs (3-5 sentences each). "
            "Use blank lines between paragraphs.\n\n"
            "Open by stating the argument being examined. "
            "Present how the text supports or defends it, then any limitations "
            "or counterpoints the text itself raises. If the text is one-sided, "
            "state that directly rather than fabricating balance. "
            "Close by summarizing the text's own position.\n\n"
            "Write in prose — no bullet points."
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
            "Give the direct answer concisely. "
            "For single-fact questions, 1-3 sentences. "
            "For multi-part questions, as many sentences as needed to cover each part. "
            "Quote or paraphrase the specific passage that supports your answer. "
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
            "Write 3-5 short paragraphs (3-4 sentences each).\n\n"
            "Structure:\n"
            "\u2022 Opening paragraph: Describe the overall scope and focus of the collection, "
            "including document types and date ranges where apparent.\n"
            "\u2022 Middle paragraphs (1-3): Briefly describe each major document or topic cluster. "
            "One paragraph per document or theme.\n"
            "\u2022 Closing paragraph: Note common themes or connections between documents, "
            "and flag any notable gaps in coverage.\n\n"
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
            "focus on sequence and actionable steps. "
            "If the context does not describe a clear process or procedure, state this directly "
            "rather than forcing a step-by-step structure."
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
# Deep Research mode
# Intent instructions diverge from regular mode in two ways:
# 1. Thinking-enabled intents (ANALYZE, COMPARE, CRITIQUE, EXPLAIN) get
#    rewritten task/format: reasoning scaffolding is removed because the
#    model's internal thinking phase handles it. Visible output instructions
#    describe the finished product, not how to reason about it.
# 2. Non-thinking intents get exhaustiveness additions to their task field
#    to leverage the deeper retrieval budget in deep-research mode.
# ---------------------------------------------------------------------------

# Start from a per-intent shallow copy so mutations don't bleed back.
INTENT_INSTRUCTIONS_DEEP_RESEARCH: dict[Intent, dict[str, str]] = {
    intent: dict(cfg) for intent, cfg in INTENT_INSTRUCTIONS_REGULAR.items()
}

# ── 8.2  Rewrite task/format for thinking-enabled intents ─────────────────

_PRESENT_DIRECTLY = (
    "Present your analysis directly. State your findings and support them "
    "with evidence from the passages. Do not walk through your reasoning "
    "step by step or narrate how you arrived at your conclusions."
)

INTENT_INSTRUCTIONS_DEEP_RESEARCH[Intent.ANALYZE] = {
    **INTENT_INSTRUCTIONS_DEEP_RESEARCH[Intent.ANALYZE],
    "task": (
        "Synthesize patterns and relationships across all provided passages. "
        "Identify where evidence converges or conflicts, assess the significance "
        "of what you find, and flag what the sources collectively leave unaddressed. "
        "Go beyond description to explain causes, relationships, and significance."
    ),
    "format": (
        "Write 4\u20136 short paragraphs. Use blank lines between paragraphs.\n\n"
        + _PRESENT_DIRECTLY + "\n\n"
        "Write in prose \u2014 no bullet points."
    ),
}

INTENT_INSTRUCTIONS_DEEP_RESEARCH[Intent.COMPARE] = {
    **INTENT_INSTRUCTIONS_DEEP_RESEARCH[Intent.COMPARE],
    "task": (
        "Go beyond listing similarities and differences. Analyze what drives "
        "the disagreements, evaluate which differences are substantively "
        "significant versus superficial, and identify unstated assumptions "
        "that explain why the positions diverge. If the sources share common "
        "ground, assess how robust that agreement is."
    ),
    "format": (
        "Write 4\u20137 short paragraphs. Use blank lines between paragraphs.\n\n"
        + _PRESENT_DIRECTLY + "\n\n"
        "Write in prose \u2014 no bullet points."
    ),
}

INTENT_INSTRUCTIONS_DEEP_RESEARCH[Intent.CRITIQUE] = {
    **INTENT_INSTRUCTIONS_DEEP_RESEARCH[Intent.CRITIQUE],
    "task": (
        "Evaluate argument strength, evidence quality, logical consistency, "
        "and unstated assumptions. Deliver a clear evaluative judgment, not "
        "just a list of strengths and weaknesses. If the text is one-sided, "
        "say so and assess how well the one-sided case is made. You may note "
        "what the text leaves unaddressed but should label omissions clearly. "
        "Do NOT fabricate counterarguments not present in the context."
    ),
    "format": (
        "Write 4\u20136 short paragraphs. Use blank lines between paragraphs.\n\n"
        + _PRESENT_DIRECTLY + "\n\n"
        "Write in prose \u2014 no bullet points."
    ),
}

INTENT_INSTRUCTIONS_DEEP_RESEARCH[Intent.EXPLAIN] = {
    **INTENT_INSTRUCTIONS_DEEP_RESEARCH[Intent.EXPLAIN],
    "task": (
        "Build a layered explanation that connects the core concept to its "
        "broader implications and practical significance. Anticipate where "
        "a reader would get confused and address those points proactively. "
        "If the concept relates to other ideas in the provided material, "
        "draw those connections. Do NOT introduce facts, definitions, or "
        "topics not present in the context."
    ),
    "format": (
        "Write 3\u20135 paragraphs, one concept per paragraph. "
        "Where possible, use an analogy or everyday comparison to make the "
        "key idea concrete.\n\n"
        + _PRESENT_DIRECTLY
    ),
}

# ── 8.3  Append exhaustiveness sentences to non-thinking intents ──────────

INTENT_INSTRUCTIONS_DEEP_RESEARCH[Intent.FACTUAL]["task"] += (
    " Examine every provided passage before answering. Where multiple passages "
    "address the question, corroborate across them and note any discrepancies."
)

INTENT_INSTRUCTIONS_DEEP_RESEARCH[Intent.OVERVIEW]["task"] += (
    " Draw on all provided passages to give a comprehensive overview. "
    "Do not anchor on a single source when others provide additional scope."
)

INTENT_INSTRUCTIONS_DEEP_RESEARCH[Intent.SUMMARIZE]["task"] += (
    " Synthesize across all provided passages. Ensure claims from lower-ranked "
    "passages are not omitted if they introduce distinct points not covered "
    "by higher-ranked ones."
)

INTENT_INSTRUCTIONS_DEEP_RESEARCH[Intent.COLLECTION]["task"] += (
    " Be thorough \u2014 describe every document represented in the provided "
    "summaries, not just the most prominent ones."
)

INTENT_INSTRUCTIONS_DEEP_RESEARCH[Intent.EXTRACT]["task"] += (
    " Examine every provided passage exhaustively. Do not stop after finding "
    "the first match \u2014 continue through all passages to ensure completeness."
)

INTENT_INSTRUCTIONS_DEEP_RESEARCH[Intent.TIMELINE]["task"] += (
    " Check every provided passage for dates and events. Ensure the timeline "
    "reflects the full range of the provided material, not just the most "
    "prominent passages."
)

INTENT_INSTRUCTIONS_DEEP_RESEARCH[Intent.HOW_TO]["task"] += (
    " Draw on all provided passages to capture the complete procedure. "
    "If different passages describe overlapping or complementary steps, "
    "reconcile them into a coherent sequence."
)

INTENT_INSTRUCTIONS_DEEP_RESEARCH[Intent.QUOTE_EVIDENCE]["task"] += (
    " Search every provided passage for relevant quotes. Do not stop at "
    "the first few matches \u2014 the most relevant evidence may appear in "
    "less prominent passages."
)

# Backward-compatible alias
INTENT_INSTRUCTIONS = INTENT_INSTRUCTIONS_REGULAR


def _get_intent_instructions(mode: Optional[str] = None) -> dict[Intent, dict[str, str]]:
    """Return the intent instruction set for the given operating mode."""
    if mode == "deep-research":
        return INTENT_INSTRUCTIONS_DEEP_RESEARCH
    return INTENT_INSTRUCTIONS_REGULAR


def _build_system_block(cfg: dict[str, str], citation_block: str, extra_block: str) -> str:
    return f"{_SYSTEM_MESSAGE}{citation_block}\n\nTask: {cfg['task']}\nFormat: {cfg['format']}\nTone: {cfg['tone']}{extra_block}"


def _prepare_config(
    intent: Optional[Intent],
    citations_enabled: bool,
    extra_instructions: Optional[str],
    mode: Optional[str] = None,
    citation_output_mode: str = _CITATION_OUTPUT_DEFAULT,
) -> tuple[dict[str, str], str, str]:
    intent = intent or Intent.SUMMARIZE
    instructions = _get_intent_instructions(mode)
    if intent not in instructions:
        logger.warning(
            "No prompt config for intent %s — falling back to SUMMARIZE", intent
        )
    cfg = instructions.get(intent, instructions[Intent.SUMMARIZE]).copy()
    extra_block = (
        f"\nAdditional constraints: {extra_instructions.strip()}"
        if extra_instructions and extra_instructions.strip()
        else ""
    )
    citation_block = ""
    if citations_enabled:
        benchmark_mode = citation_output_mode == _CITATION_OUTPUT_BENCHMARK_PAGE
        if citation_output_mode not in {
            _CITATION_OUTPUT_DEFAULT,
            _CITATION_OUTPUT_BENCHMARK_PAGE,
        }:
            logger.warning(
                "Unknown citation_output_mode '%s'; falling back to default",
                citation_output_mode,
            )
            benchmark_mode = False

        citation_rules = _BENCHMARK_CITATION_RULES if benchmark_mode else _CITATION_RULES
        citation_block = f"\n{citation_rules}"
        if benchmark_mode:
            cfg["format"] += (
                " Include benchmark citations as [N, p.XX] after every factual claim. "
                "Use XX from the nearest preceding [Page N] marker in that passage text."
            )
        else:
            cfg["format"] += (
                " Include numbered inline citations [1], [2], etc. after every factual claim. "
                "Treat citation numbers as references to provided passages; do not use words like 'chunk' or mention retrieval internals."
            )

        if intent == Intent.COMPARE:
            cfg["format"] += (
                " When comparing positions, cite each side separately within the same sentence. "
                "Do not batch multiple page citations at the end of a paragraph. "
                "Each distinct claim gets its own inline citation immediately after the supporting clause."
            )
        elif intent == Intent.OVERVIEW:
            cfg["format"] += (
                " Even though you are summarizing broadly, cite the specific page for each main point individually. "
                "Do not group citations at the end of a paragraph. "
                "If one sentence covers multiple topics from different pages, split it into separate sentences with individual citations."
            )
        elif intent == Intent.ANALYZE:
            cfg["format"] += (
                " Cite at the clause level, not the paragraph level. "
                "Each analytical claim should carry its own inline citation immediately after the supporting clause."
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
    citation_output_mode: str = _CITATION_OUTPUT_DEFAULT,
) -> list[dict[str, str]]:
    """Build intent-aware chat messages for the LLM."""
    cfg, citation_block, extra_block = _prepare_config(
        intent,
        citations_enabled,
        extra_instructions,
        mode=mode,
        citation_output_mode=citation_output_mode,
    )
    system_block = _build_system_block(cfg, citation_block, extra_block)

    # Context-sparsity warning: if context fills < 10% of retrieval budget.
    # Token estimate uses ×1.35 correction for subword tokenization — word-split
    # underestimates actual token count by ~25-30% for English text.
    # (Heuristic consistent with count_tokens() elsewhere in the codebase.)
    if retrieval_budget and retrieval_budget > 0 and context:
        context_tokens = int(len(context.split()) * 1.35)
        fill_ratio = context_tokens / retrieval_budget
        if fill_ratio < 0.10:
            sparsity_warning = (
                "WARNING: Limited source material was retrieved for this query. "
                "Base your answer strictly and exclusively on the passages provided. "
                "Do not infer, extrapolate, or add details not explicitly present in the context. "
                "If the context is insufficient to fully answer the question, clearly state "
                "what cannot be determined."
            )
            # Append after the identity/rules block so the model's identity
            # framing is always the first thing in the system message.
            system_block = system_block + "\n\n" + sparsity_warning

    legend_block = f"\n\n{source_legend}" if citations_enabled and source_legend else ""
    # Reminder injected at the end of the user message — at the generation trigger
    # point where model attention peaks — to reinforce citation rules already stated
    # in the system prompt.
    citation_reminder = ""
    if citation_output_mode == _CITATION_OUTPUT_BENCHMARK_PAGE and citations_enabled:
        citation_reminder = (
            "\n\nIMPORTANT: You MUST include inline citations as [N, p.XX] after every factual claim. "
            "N is the PASSAGE number, XX is the nearest preceding [Page N] marker in that passage's text."
        )
    elif citations_enabled:
        citation_reminder = (
            "\n\nIMPORTANT: You MUST include inline citation numbers [1], [2], etc. "
            "after every factual claim. Use the PASSAGE numbers from the context above."
        )
    else:
        citation_reminder = ""
    user_block = f"Context:\n{context}{legend_block}\n\nQuestion: {question}{citation_reminder}"
    return [
        {"role": "system", "content": system_block},
        {"role": "user", "content": user_block}
    ]
