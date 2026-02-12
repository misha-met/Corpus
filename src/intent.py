"""Intent classification for RAG query processing.

Architecture
~~~~~~~~~~~~
1. **Heuristic classifier** (primary) — fast regex + structural signal scoring.
   Returns an ``IntentResult`` with confidence in (0, 1].
2. **LLM fallback** (optional) — when heuristic confidence is below
    ``llm_fallback_threshold``, uses an MLX language model to classify via
    JSON generation.
3. **Overview gate** — any result below ``confidence_threshold`` is demoted to
   ``OVERVIEW`` so vague / garbage queries don't trigger specialised pipelines.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


# ── Data types ────────────────────────────────────────────────────────────────

class Intent(Enum):
    OVERVIEW = "overview"
    SUMMARIZE = "summarize"
    EXPLAIN = "explain"
    ANALYZE = "analyze"
    COMPARE = "compare"
    CRITIQUE = "critique"
    FACTUAL = "factual"
    COLLECTION = "collection"


@dataclass(frozen=True)
class IntentResult:
    intent: Intent
    confidence: float
    method: str

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Confidence must be between 0.0 and 1.0")


# ── Intent → JSON mapping (shared) ───────────────────────────────────────────

_INTENT_MAP: dict[str, Intent] = {
    "overview": Intent.OVERVIEW,
    "summarize": Intent.SUMMARIZE,
    "explain": Intent.EXPLAIN,
    "analyze": Intent.ANALYZE,
    "compare": Intent.COMPARE,
    "critique": Intent.CRITIQUE,
    "factual": Intent.FACTUAL,
    "collection": Intent.COLLECTION,
}


# ── Heuristic patterns ───────────────────────────────────────────────────────

_INTENT_PATTERNS: dict[Intent, list[re.Pattern]] = {
    # ---- COLLECTION: corpus-level queries about available documents ----
    Intent.COLLECTION: [
        re.compile(r"\bwhat\s+(documents?|docs?|files?|sources?)\s+(do\s+)?(we|i|you)\s+have\b", re.IGNORECASE),
        re.compile(r"\bwhat\s+(documents?|docs?|files?|sources?)\s+(are|is)\s+(in\s+)?(here|there|available)\b", re.IGNORECASE),
        re.compile(r"\bwhat('s| is)\s+(in\s+)?(here|this\s+(collection|corpus|workspace|library))\b", re.IGNORECASE),
        re.compile(r"\blist\s+(all\s+)?(the\s+)?(documents?|docs?|files?|sources?)\b", re.IGNORECASE),
        re.compile(r"\bwhat\s+are\s+(we|these)\s+(looking\s+at|working\s+with)\b", re.IGNORECASE),
        re.compile(r"\bwhat\s+are\s+(the|all\s+the)\s+(documents?|docs?|files?|sources?)\b", re.IGNORECASE),
        re.compile(r"\bsummari[sz]e\s+(all|every(thing)?|the\s+(whole|entire))\b", re.IGNORECASE),
        re.compile(r"\boverview\s+of\s+(all|every|the\s+entire)\b", re.IGNORECASE),
        re.compile(r"\bwhat('s| is)\s+the\s+(corpus|collection)\s+(about|contain)", re.IGNORECASE),
        re.compile(r"\bshow\s+(me\s+)?(all|the)\s+(documents?|docs?|files?|sources?)\b", re.IGNORECASE),
        re.compile(r"\bgive\s+(me\s+)?(an?\s+)?overview\s+of\s+(all|everything)\b", re.IGNORECASE),
        re.compile(r"\bwhat\s+topics?\s+(are|do)\s+(these|the)\s+(documents?|docs?)\s+cover\b", re.IGNORECASE),
        re.compile(r"\bdescribe\s+(all|the)\s+(documents?|docs?|sources?)\b", re.IGNORECASE),
        re.compile(r"\bwhat\s+are\s+the\s+docs\s+in\s+here\b", re.IGNORECASE),
        re.compile(r"\bshow\s+(me\s+)?all\s+(the\s+)?sources\b", re.IGNORECASE),
        re.compile(r"\bdescribe\s+all\s+(the\s+)?sources\b", re.IGNORECASE),
        # Informal corpus queries
        re.compile(r"\bwhat\s+(do|can)\s+(we|i|you)\s+(have|see|access|read)\b", re.IGNORECASE),
        re.compile(r"\bwhat('s| is)\s+available\b", re.IGNORECASE),
        re.compile(r"\bwhat('s| is)\s+in\s+(the\s+)?(database|index|system)\b", re.IGNORECASE),
    ],
    # ---- FACTUAL: direct fact extraction (who/what/when/where/which) ----
    Intent.FACTUAL: [
        re.compile(r"\bwhat\s+particular\b", re.IGNORECASE),
        re.compile(r"\bwhat\s+(specific|exact)\b", re.IGNORECASE),
        re.compile(r"\bwho\s+(is|was|are|were|did|does|wrote|said)\b", re.IGNORECASE),
        re.compile(r"\bwhen\s+(did|was|were|is)\b", re.IGNORECASE),
        re.compile(r"\bwhere\s+(did|was|were|is|does)\b", re.IGNORECASE),
        re.compile(r"\bwhich\s+(specific|particular)?\s*\w+\s+(is|was|are|were|did|does)\b", re.IGNORECASE),
        re.compile(r"\bname\s+the\b", re.IGNORECASE),
        re.compile(r"\bidentify\s+the\b", re.IGNORECASE),
        re.compile(r"\bwhat\s+(year|date|time|number|name|title|author)\b", re.IGNORECASE),
        re.compile(r"\bwhat\s+(is|are|was|were)\s+the\s+(name|title|author|year|date)\b", re.IGNORECASE),
        re.compile(r"\bhow\s+many\b", re.IGNORECASE),
        re.compile(r"\bwhat\s+(does|did)\s+\w+\s+(say|write|argue|claim|state|mention)\s+about\b", re.IGNORECASE),
        re.compile(r"\baccording\s+to\b", re.IGNORECASE),
        re.compile(r"\bwhat\s+(llm|model|algorithm|method|technique|tool|framework)\b", re.IGNORECASE),
        re.compile(r"\bwhat\s+is\s+the\s+(author|writer)\s+(talking|writing|referring)\b", re.IGNORECASE),
        re.compile(r"\bwhat\s+does\s+the\s+(author|writer|text|document|paper)\s+(say|state|claim|argue|mention)\b", re.IGNORECASE),
        re.compile(r"\bfind\s+(me\s+)?(the|a|all)\b", re.IGNORECASE),
    ],
    # ---- OVERVIEW: high-level single-document description ----
    Intent.OVERVIEW: [
        re.compile(r"^what\s+is\s+this\s*\??$", re.IGNORECASE),
        re.compile(r"\bwhat\s+is\s+(this|the)\s+(paper|text|document|article)\s+(about\s*)?\??", re.IGNORECASE),
        re.compile(r"\bwhat\s+is\s+this\s+about\b", re.IGNORECASE),
        re.compile(r"\btell\s+me\s+about\s+(this|the)\s*(document|paper|text|article)?\b", re.IGNORECASE),
        re.compile(r"\bwhat('s| is)\s+the\s+point\s+of\s+this\b", re.IGNORECASE),
        re.compile(r"\bwhat\s+am\s+i\s+(reading|looking at)\b", re.IGNORECASE),
        re.compile(r"\bgive\s+me\s+(a|the)\s+(gist|overview)\b", re.IGNORECASE),
        re.compile(r"\bquick\s+(overview|summary)\b", re.IGNORECASE),
        re.compile(r"\bin\s+a\s+nutshell\b", re.IGNORECASE),
        re.compile(r"\bwhat\s+is\s+this\s+paper\b", re.IGNORECASE),
    ],
    # ---- EXPLAIN: simplification for non-experts ----
    Intent.EXPLAIN: [
        re.compile(r"\bexplain\b", re.IGNORECASE),
        re.compile(r"\bsimple\s+terms\b", re.IGNORECASE),
        re.compile(r"\bsimplif", re.IGNORECASE),
        re.compile(r"\blayman", re.IGNORECASE),
        re.compile(r"\bnon-expert", re.IGNORECASE),
        re.compile(r"\beasy\s+to\s+understand\b", re.IGNORECASE),
        re.compile(r"\bbreak\s*(it\s*)?down\b", re.IGNORECASE),
        re.compile(r"\bELI5\b", re.IGNORECASE),
        re.compile(r"\bwhat\s+does\s+(this|that|it)\s+mean\b", re.IGNORECASE),
        re.compile(r"\bhelp\s+(me\s+)?understand\b", re.IGNORECASE),
        re.compile(r"\bin\s+(plain|everyday)\s+(english|language|words)\b", re.IGNORECASE),
        re.compile(r"\bdumb\s+(it|this|that)?\s*down\b", re.IGNORECASE),
    ],
    # ---- COMPARE: side-by-side comparative analysis ----
    Intent.COMPARE: [
        re.compile(r"\bcompare\b", re.IGNORECASE),
        re.compile(r"\bcontrast\b", re.IGNORECASE),
        re.compile(r"\bdiffer(ence|s|ent)?\b", re.IGNORECASE),
        re.compile(r"\bsimilarit(y|ies)\b", re.IGNORECASE),
        re.compile(r"\b(how|in\s+what\s+way)\s+does\b.*\b(relate|compare|differ)\b", re.IGNORECASE),
        re.compile(r"\bversus\b|\bvs\.?\b", re.IGNORECASE),
        re.compile(r"\b(both|each|two|these)\b.*\b(approach|view|theory|position|argument)s?\b", re.IGNORECASE),
        re.compile(r"\bside.by.side\b", re.IGNORECASE),
        re.compile(r"\bhow\s+(is|are|does|do)\b.*\b(like|unlike)\b", re.IGNORECASE),
    ],
    # ---- CRITIQUE: evaluative analysis of arguments ----
    Intent.CRITIQUE: [
        re.compile(r"\bcritique\b", re.IGNORECASE),
        re.compile(r"\bcritici[sz]", re.IGNORECASE),
        re.compile(r"\bevaluate\s+(the|this|that|whether|if|an?)\b", re.IGNORECASE),
        re.compile(r"\bassess\s+(the|this|that|whether|if|an?)\b", re.IGNORECASE),
        re.compile(r"\bwhy\b.*\b(controversial|debate|disagree|critic)", re.IGNORECASE),
        re.compile(r"\bcontrovers", re.IGNORECASE),
        re.compile(r"\bwhat\s+(are|were)\s+the\s+(criticism|objection)s?\b", re.IGNORECASE),
        re.compile(r"\bhow\s+did\s+(people|scholars|critics)\s+react\b", re.IGNORECASE),
        re.compile(r"\bwhat\s+(is|was)\s+the\s+debate\b", re.IGNORECASE),
        re.compile(r"\bstrengths?\s+and\s+weaknesses?\b", re.IGNORECASE),
        re.compile(r"\bpros?\s+and\s+cons?\b", re.IGNORECASE),
        re.compile(r"\bto\s+what\s+extent\b", re.IGNORECASE),
        re.compile(r"\bhow\s+(valid|sound|strong|weak|convincing)\b", re.IGNORECASE),
        re.compile(r"\bis\s+(it|this|that|the\s+\w+)\b.*\b(valid|sound|convincing|justified|well.supported)\b", re.IGNORECASE),
        re.compile(r"\bhow\s+well\s+does\b", re.IGNORECASE),
    ],
    # ---- ANALYZE: mechanism / cause / theme analysis ----
    Intent.ANALYZE: [
        re.compile(r"\bhow\s+does\b", re.IGNORECASE),
        re.compile(r"\bin\s+what\s+way\b", re.IGNORECASE),
        re.compile(r"\banalyze\b", re.IGNORECASE),
        re.compile(r"\bwhat\s+role\s+does\b", re.IGNORECASE),
        re.compile(r"\bwhy\s+(is|are|does|do|did|was|were)\b", re.IGNORECASE),
    ],
    # ---- SUMMARIZE: detailed summary with key points ----
    Intent.SUMMARIZE: [
        re.compile(r"\bsummari[sz]e\b", re.IGNORECASE),
        re.compile(r"\bdetailed\s+(summary|overview)\b", re.IGNORECASE),
        re.compile(r"\bmain\s+(point|idea|argument|theme)s?\b", re.IGNORECASE),
        re.compile(r"\bkey\s+(point|takeaway|finding)s?\b", re.IGNORECASE),
        re.compile(r"\btl;?dr\b", re.IGNORECASE),
        re.compile(r"\bbullet\s*points?\b", re.IGNORECASE),
        re.compile(r"\blist\s+(the\s+)?(main|key)\b", re.IGNORECASE),
    ],
}

_HEURISTIC_CONFIDENCE = {"strong_match": 0.85, "single_match": 0.70, "weak_match": 0.50}
_TECHNICAL_TERM_HINTS = {"stimulus", "reinforcement", "skinner"}
_LOW_INFO_COMMON_WORDS = {
    "what", "why", "how", "who", "when", "where", "which", "explain", "summarize",
    "compare", "analyze", "critique", "document", "docs", "paper", "text", "about",
    "mean", "help", "understand", "list", "show", "tell", "overview", "details",
}


# ── Structural signal patterns ────────────────────────────────────────────────

_COMMAND_VERB_INTENTS: dict[re.Pattern, Intent] = {
    re.compile(r"^\s*compare\b", re.IGNORECASE): Intent.COMPARE,
    re.compile(r"^\s*contrast\b", re.IGNORECASE): Intent.COMPARE,
    re.compile(r"^\s*summari[sz]e\b", re.IGNORECASE): Intent.SUMMARIZE,
    re.compile(r"^\s*explain\b", re.IGNORECASE): Intent.EXPLAIN,
    re.compile(r"^\s*analy[sz]e\b", re.IGNORECASE): Intent.ANALYZE,
    re.compile(r"^\s*trace\b", re.IGNORECASE): Intent.ANALYZE,
    re.compile(r"^\s*extract\b", re.IGNORECASE): Intent.FACTUAL,
    re.compile(r"^\s*identify\b", re.IGNORECASE): Intent.FACTUAL,
}

_COMPARATIVE_STRUCTURES: list[re.Pattern] = [
    re.compile(r"\bvs\.?\b", re.IGNORECASE),
    re.compile(r"\bversus\b", re.IGNORECASE),
    re.compile(r"\bdifferences?\s+between\b", re.IGNORECASE),
    re.compile(r"\bsimilarit(y|ies)\s+between\b", re.IGNORECASE),
    re.compile(r"\bin\s+light\s+of\b", re.IGNORECASE),
    re.compile(r"\bbetween\b.+\band\b", re.IGNORECASE),
    re.compile(r"\brelate\s+to\b", re.IGNORECASE),
]

_EXTRACTION_STRUCTURES: list[re.Pattern] = [
    re.compile(r"\bextract\b", re.IGNORECASE),
    re.compile(r"\blist\s+all\s+(names?|dates?|years?|titles?|authors?|citations?)\b", re.IGNORECASE),
    re.compile(r"\bgive\s+me\s+the\s+(dates?|years?|names?|citations?)\b", re.IGNORECASE),
    re.compile(r"\bformat\b.*\bas\s+(a\s+)?table\b", re.IGNORECASE),
    re.compile(r"\btabular\b", re.IGNORECASE),
]

_SUMMARIZATION_STRUCTURES: list[re.Pattern] = [
    re.compile(r"\btl;?dr\b", re.IGNORECASE),
    re.compile(r"\bmain\s+points?\b", re.IGNORECASE),
    re.compile(r"\bkey\s+points?\b", re.IGNORECASE),
    re.compile(r"\boverview\s+of\b", re.IGNORECASE),
]


# ── Heuristic helpers ─────────────────────────────────────────────────────────

def _apply_structural_intent_signals(query: str, scores: dict[Intent, int]) -> None:
    """Apply structure-aware intent boosts beyond plain keyword presence."""
    normalized = query.strip()

    corpus_scope = bool(
        re.search(r"\b(all|every|entire|whole)\b", normalized, re.IGNORECASE)
        and re.search(r"\b(documents?|docs?|sources?|collection|corpus)\b", normalized, re.IGNORECASE)
    )
    overview_everything_scope = bool(
        re.search(r"\boverview\s+of\s+everything\b", normalized, re.IGNORECASE)
        or re.search(r"\bsummari[sz]e\s+everything\b", normalized, re.IGNORECASE)
    )

    for pattern, intent in _COMMAND_VERB_INTENTS.items():
        if pattern.search(normalized):
            if intent in (Intent.SUMMARIZE, Intent.FACTUAL) and corpus_scope:
                continue
            scores[intent] += 3

    if any(pattern.search(normalized) for pattern in _COMPARATIVE_STRUCTURES):
        scores[Intent.COMPARE] += 2

    if any(pattern.search(normalized) for pattern in _EXTRACTION_STRUCTURES):
        scores[Intent.FACTUAL] += 3

    if any(pattern.search(normalized) for pattern in _SUMMARIZATION_STRUCTURES):
        scores[Intent.SUMMARIZE] += 2

    if corpus_scope or overview_everything_scope:
        scores[Intent.COLLECTION] += 4

    # Noun-phrase analytical framing: "X's critique of Y" usually asks for
    # analysis of arguments, not an imperative to "critique".
    if re.search(r"\b\w+(?:'s|s)\s+critique\s+of\b", normalized, re.IGNORECASE):
        scores[Intent.ANALYZE] += 2

    # Causal chain phrasing is analytical / multi-hop.
    if re.search(r"\btrace\b.*\b(chain|path|link)\b", normalized, re.IGNORECASE):
        scores[Intent.ANALYZE] += 3
    if "->" in normalized:
        scores[Intent.ANALYZE] += 2


def _has_technical_terms(query: str) -> bool:
    if re.search(r"['\"`].+?['\"`]", query) or re.search(r"\b[A-Z][a-z]{2,}\b", query):
        return True
    return any(term in query.lower() for term in _TECHNICAL_TERM_HINTS)


def _is_technical_how_why(query: str) -> bool:
    # "how many" is a factual question, not analytical.
    if re.match(r"^\s*how\s+many\b", query, re.IGNORECASE):
        return False
    return bool(re.match(r"^\s*(how|why)\b", query, re.IGNORECASE)) and _has_technical_terms(query)


def _classify_heuristic(query: str) -> IntentResult:
    """Classify intent using regex pattern matching + structural signals."""
    scores: dict[Intent, int] = {intent: 0 for intent in Intent}
    for intent, patterns in _INTENT_PATTERNS.items():
        for pattern in patterns:
            if pattern.search(query):
                scores[intent] += 1

    _apply_structural_intent_signals(query, scores)

    # ---- noun-phrase de-boost ----
    # "Chomsky's critique", "the critique of X" → the word "critique" is a
    # noun, NOT an instruction to critique.
    if scores[Intent.CRITIQUE] > 0:
        noun_critique = re.search(
            r"(?:\b\w+(?:'s|s)\s+|(?:the|a|an|this|that|his|her|their|its)\s+)critique\b",
            query, re.IGNORECASE,
        )
        if noun_critique:
            scores[Intent.CRITIQUE] = max(0, scores[Intent.CRITIQUE] - 1)
            logger.debug(
                "De-boosted CRITIQUE: 'critique' appears as noun phrase (%s)",
                noun_critique.group(),
            )

    analyze_bias = _is_technical_how_why(query)
    if analyze_bias:
        analytical = [Intent.COMPARE, Intent.CRITIQUE, Intent.ANALYZE]
        best_analytical = max(analytical, key=lambda k: scores[k])
        if any(scores[i] > 0 for i in analytical):
            scores[best_analytical] += 2
        else:
            scores[Intent.ANALYZE] += 2

    best_intent = max(scores, key=lambda k: scores[k])
    best_score = scores[best_intent]

    if best_score == 0:
        return IntentResult(intent=Intent.OVERVIEW, confidence=0.40, method="fallback")

    matching_intents = [i for i, s in scores.items() if s > 0]

    # ---- Tie-break rules ----

    # COMPARE/CRITIQUE wins over weak ANALYZE evidence.
    if best_intent == Intent.ANALYZE and (
        scores[Intent.ANALYZE] <= 1
        and (scores[Intent.COMPARE] > 0 or scores[Intent.CRITIQUE] > 0)
    ):
        if scores[Intent.COMPARE] >= scores[Intent.CRITIQUE]:
            best_intent = Intent.COMPARE
            best_score = scores[Intent.COMPARE]
        else:
            best_intent = Intent.CRITIQUE
            best_score = scores[Intent.CRITIQUE]

    # COLLECTION wins ties with SUMMARIZE ("Summarize all documents").
    if Intent.COLLECTION in matching_intents and Intent.SUMMARIZE in matching_intents:
        if scores[Intent.COLLECTION] >= scores[Intent.SUMMARIZE]:
            best_intent = Intent.COLLECTION
            best_score = scores[Intent.COLLECTION]

    # COMPARE wins ties with CRITIQUE ("similarity in Chomsky's critique").
    if (
        scores[Intent.COMPARE] > 0
        and scores[Intent.CRITIQUE] > 0
        and scores[Intent.COMPARE] >= scores[Intent.CRITIQUE]
    ):
        best_intent = Intent.COMPARE
        best_score = scores[Intent.COMPARE]
        scores[Intent.CRITIQUE] = 0

    # Recalculate after tie-breaks.
    matching_intents = [i for i, s in scores.items() if s > 0]

    if len(matching_intents) > 1 and best_score == 1:
        confidence = _HEURISTIC_CONFIDENCE["weak_match"]
    elif best_score >= 2:
        confidence = _HEURISTIC_CONFIDENCE["strong_match"]
    else:
        confidence = _HEURISTIC_CONFIDENCE["single_match"]

    if best_intent in (Intent.ANALYZE, Intent.COMPARE, Intent.CRITIQUE) and analyze_bias:
        confidence = min(0.95, confidence + 0.15)

    return IntentResult(intent=best_intent, confidence=confidence, method="heuristic")


def is_low_information_query(query: str) -> bool:
    """Return True when query is likely underspecified or gibberish-like."""
    normalized = query.strip().lower()
    if not normalized:
        return True

    tokens = re.findall(r"[a-zA-Z']+", normalized)
    if not tokens:
        return True

    # Repeated-token noise, e.g. "wa wa wa"
    if len(tokens) >= 2 and len(set(tokens)) == 1:
        return True

    has_pattern_signal = any(
        pattern.search(normalized)
        for patterns in _INTENT_PATTERNS.values()
        for pattern in patterns
    )
    has_common_intent_word = any(token in _LOW_INFO_COMMON_WORDS for token in tokens)
    has_question_mark = "?" in normalized

    if len(tokens) <= 2 and not has_pattern_signal and not has_common_intent_word:
        return True
    if len(tokens) <= 3 and not has_pattern_signal and not has_common_intent_word and not has_question_mark:
        return True

    return False


# ── LLM classification helpers ────────────────────────────────────────────────

def _build_classification_prompt(query: str) -> str:
    """Build a minimal prompt for LLM-based intent classification."""
    return f"""You are a strict JSON generator.
Return ONLY a single JSON object and nothing else.
No markdown, no code fences, no explanations.

Classify the user's intent into exactly one category.

Categories:
- overview: User wants a brief, high-level description of what the document is and its purpose
- summarize: User wants a detailed summary with key points and bullet points
- explain: User wants the content explained simply, for non-experts
- compare: User wants a side-by-side comparison or contrast between two or more named ideas, theories, positions, or documents
- critique: User explicitly asks to evaluate, critique, or assess the merits of an argument — uses words like "evaluate", "critique", "strengths", "weaknesses", "how convincing"
- analyze: User wants to understand how or why something works, or wants analysis of themes, patterns, causes, or mechanisms (default for "how does X" and "why does X" questions)
- factual: User wants a direct factual answer extracted from the text (who, what, when, where, which, how many)
- collection: User wants to know what documents are available, or wants an overview of all documents in the corpus

User query: "{query}"

Respond with ONLY a JSON object in this exact format:
{{"intent": "<overview|summarize|explain|compare|critique|analyze|factual|collection>", "confidence": <0.0-1.0>}}"""


def _parse_llm_response(response: str) -> Optional[Tuple[Intent, float]]:
    """Parse JSON intent classification from LLM response."""
    response = response.strip()
    if "```" in response:
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", response, re.DOTALL)
        if match:
            response = match.group(1)

    start, end = response.find("{"), response.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None

    try:
        data = json.loads(response[start:end + 1])
        intent = _INTENT_MAP.get(data.get("intent", "").lower().strip())
        if intent is None:
            return None
        return (intent, max(0.0, min(1.0, float(data.get("confidence", 0.5)))))
    except (json.JSONDecodeError, ValueError, TypeError):
        return None


# ── Main classifier ───────────────────────────────────────────────────────────

class IntentClassifier:
    """Classifies user queries into intents.

    Flow: heuristic → (optional) LLM fallback → overview gate.

    Parameters
    ----------
    confidence_threshold:
        Final confidence gate.  Results below this are demoted to OVERVIEW.
    llm_model_id:
        HuggingFace / MLX model ID for the LLM fallback (e.g.
        ``mlx-community/LFM2-8B-A1B-4bit``).  ``None`` disables the fallback.
    llm_fallback_threshold:
        If the heuristic confidence is below this value *and* an LLM model is
        configured, the LLM fallback is invoked.
    """

    def __init__(
        self,
        confidence_threshold: float = 0.6,
        llm_model_id: Optional[str] = None,
        llm_fallback_threshold: float = 0.70,
        eager_load_llm: bool = True,
    ) -> None:
        self._confidence_threshold = confidence_threshold
        self._llm_model_id = llm_model_id
        self._llm_fallback_threshold = llm_fallback_threshold
        self._eager_load_llm = eager_load_llm
        # LLM state
        self._llm_model = None
        self._llm_tokenizer = None
        if self._llm_model_id is not None and self._eager_load_llm:
            self._load_llm_model()

    # ── public API ────────────────────────────────────────────────────────

    def classify(self, query: str) -> IntentResult:
        """Classify query intent.  Falls back to OVERVIEW when uncertain."""
        if not query.strip():
            return IntentResult(intent=Intent.OVERVIEW, confidence=1.0, method="fallback")

        result = _classify_heuristic(query)

        # If heuristic is confident enough, skip the LLM.
        if result.confidence >= self._llm_fallback_threshold or self._llm_model_id is None:
            return self._apply_overview_gate(result)

        # Low-confidence heuristic → try LLM fallback.
        logger.info(
            "Heuristic confidence %.2f < %.2f — invoking LLM fallback (%s)",
            result.confidence, self._llm_fallback_threshold, self._llm_model_id,
        )
        try:
            llm_result = self._classify_with_llm(query)
            if llm_result is not None:
                # Use LLM result if it's confident; otherwise keep heuristic.
                if llm_result.confidence >= self._confidence_threshold:
                    return llm_result
                logger.info(
                    "LLM fallback confidence %.2f still below threshold; keeping heuristic",
                    llm_result.confidence,
                )
        except Exception as exc:
            logger.warning("LLM intent classification failed: %s", exc)

        return self._apply_overview_gate(result)

    # ── private helpers ───────────────────────────────────────────────────

    def _apply_overview_gate(self, result: IntentResult) -> IntentResult:
        """Demote to OVERVIEW if confidence is below the final threshold."""
        if result.confidence < self._confidence_threshold:
            logger.info(
                "Intent '%s' confidence %.2f below threshold — falling back to overview",
                result.intent.value, result.confidence,
            )
            return IntentResult(
                intent=Intent.OVERVIEW,
                confidence=result.confidence,
                method=f"{result.method}+overview_fallback",
            )
        return result

    def _load_llm_model(self) -> None:
        """Hard-load the LLM fallback model when classifier is created."""
        if self._llm_model is not None:
            return
        if self._llm_model_id is None:
            return
        logger.info("Loading intent fallback model: %s", self._llm_model_id)
        import mlx_lm  # noqa: local lazy import
        self._llm_model, self._llm_tokenizer = mlx_lm.load(self._llm_model_id)

    def _classify_with_llm(self, query: str) -> Optional[IntentResult]:
        """Classify intent using the LLM fallback model."""
        self._load_llm_model()
        if self._llm_model is None:
            return None

        import mlx_lm  # noqa: local lazy import

        prompt = _build_classification_prompt(query)

        # Apply chat template if the tokenizer supports it.
        if hasattr(self._llm_tokenizer, "apply_chat_template"):
            messages = [{"role": "user", "content": prompt}]
            formatted = self._llm_tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
        else:
            formatted = prompt

        response = mlx_lm.generate(
            self._llm_model,
            self._llm_tokenizer,
            prompt=formatted,
            max_tokens=60,
            temp=0.0,
        )

        parsed = _parse_llm_response(response)
        if parsed is None:
            logger.warning("Failed to parse LLM intent response: %s", response[:200])
            return None

        return IntentResult(intent=parsed[0], confidence=parsed[1], method="llm-fallback")


# ── Convenience function ──────────────────────────────────────────────────────

def classify_intent(
    query: str,
    confidence_threshold: float = 0.6,
    llm_model_id: Optional[str] = None,
    llm_fallback_threshold: float = 0.70,
) -> IntentResult:
    """One-shot convenience wrapper around :class:`IntentClassifier`."""
    return IntentClassifier(
        confidence_threshold=confidence_threshold,
        llm_model_id=llm_model_id,
        llm_fallback_threshold=llm_fallback_threshold,
    ).classify(query)
