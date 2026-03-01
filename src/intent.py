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
from difflib import get_close_matches
from dataclasses import dataclass
from enum import Enum
from typing import Optional

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
    EXTRACT = "extract"
    TIMELINE = "timeline"
    HOW_TO = "how_to"
    QUOTE_EVIDENCE = "quote_evidence"


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
    "extract": Intent.EXTRACT,
    "timeline": Intent.TIMELINE,
    "how_to": Intent.HOW_TO,
    "quote_evidence": Intent.QUOTE_EVIDENCE,
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
        # Source/document selection across the corpus
        re.compile(r"\bwhich\s+(of\s+)?(these|the)\s+(documents?|docs?|files?|sources?)\s+(is|are)\b", re.IGNORECASE),
        re.compile(r"\bwhich\s+(documents?|docs?|files?|sources?)\s+(is|are)\b", re.IGNORECASE),
        re.compile(r"\bwhich\s+(documents?|docs?|files?|sources?)\b.*\b(about|on|regarding|discuss(?:es|ing)|criti(?:que|ques|quing)|critic(?:ism|ize|izes|ized|izing))\b", re.IGNORECASE),
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
        # Exhaustive enumeration queries (Fix 7)
        re.compile(r"\blist\s+(every|all)\b", re.IGNORECASE),
        re.compile(r"\bname\s+all\b", re.IGNORECASE),
        re.compile(r"\bwho\s+are\s+all\s+(the\s+)?\w", re.IGNORECASE),
        # Entity-mention / reference-finding queries (Fix 10: moved from COLLECTION)
        re.compile(r"\bwhat\s+mentions?\s+of\b.+\bare\s+there\b", re.IGNORECASE),
        re.compile(r"\bwhere\s+is\b.+\bmentioned\b", re.IGNORECASE),
        re.compile(r"\bwhere\s+does\b.+\bappear\b", re.IGNORECASE),
        re.compile(r"\bfind\s+(all|every)\s+(references?|mentions?)\s+(to|of)\b", re.IGNORECASE),
        re.compile(r"\bhow\s+many\s+times\s+is\b.+\bmentioned\b", re.IGNORECASE),
        re.compile(r"\bwhat\s+does\s+the\s+text\s+say\s+about\b", re.IGNORECASE),
        re.compile(r"\bwhat\s+is\s+said\s+about\b", re.IGNORECASE),
        re.compile(r"\bwhere\s+is\b.+\bdiscuss(?:ed|es|ing)?\b", re.IGNORECASE),
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
    # ---- EXTRACT: structured data extraction (entities, dates, figures) ----
    Intent.EXTRACT: [
        re.compile(r"\bextract\s+(all|every|the|any)?\s*(names?|dates?|years?|figures?|numbers?|entities|data|definitions?|terms?)\b", re.IGNORECASE),
        re.compile(r"\blist\s+(all\s+)?(the\s+)?(names?|dates?|years?|figures?|numbers?|entities|terms?|authors?|titles?|places?|locations?)\b", re.IGNORECASE),
        re.compile(r"\bgive\s+me\s+(all\s+)?(the\s+)?(names?|dates?|years?|figures?|numbers?|entities|terms?)\b", re.IGNORECASE),
        re.compile(r"\bwhat\s+(are|were|is)\s+(all\s+)?(the\s+)?(names?|dates?|figures?|numbers?|entities|terms?|places?)\b", re.IGNORECASE),
        re.compile(r"\bformat\b.{0,30}\bas\s+(a\s+)?(table|list|structured)\b", re.IGNORECASE),
        re.compile(r"\btabular\s+format\b", re.IGNORECASE),
        re.compile(r"\bpull\s+(out|together)\s+(all\s+)?(the\s+)?(data|entities|names?|dates?|figures?)\b", re.IGNORECASE),
        re.compile(r"\bcatalog\b.{0,30}\b(names?|dates?|entities|terms?)\b", re.IGNORECASE),
        re.compile(r"\bcompile\s+(a\s+)?(list|table)\b", re.IGNORECASE),
    ],
    # ---- TIMELINE: chronological ordering of events ----
    Intent.TIMELINE: [
        re.compile(r"\btimeline\b", re.IGNORECASE),
        re.compile(r"\bchronolog(y|ical|ically)\b", re.IGNORECASE),
        re.compile(r"\bin\s+(chronological|date|time)\s+order\b", re.IGNORECASE),
        re.compile(r"\bordered?\s+by\s+(date|time|year|when)\b", re.IGNORECASE),
        re.compile(r"\bsequence\s+of\s+(events?|steps?|actions?)\b", re.IGNORECASE),
        re.compile(r"\bwhat\s+(events?|things?\s+)?(happened|occurred)\s+(and\s+)?when\b", re.IGNORECASE),
        re.compile(r"\bhistory\s+of\s+events?\b", re.IGNORECASE),
        re.compile(r"\bwhen\s+did\b.{0,40}\b(and|also|then|next|follow)\b", re.IGNORECASE),
        re.compile(r"\blist\s+(all\s+)?events?\s+(in|with)\s+(order|dates?|years?)\b", re.IGNORECASE),
        re.compile(r"\btrace\s+(the\s+)?(history|development|progression|sequence)\b", re.IGNORECASE),
    ],
    # ---- HOW_TO: procedural / step-by-step instructions ----
    Intent.HOW_TO: [
        re.compile(r"\bhow\s+(do|does|did|can|to)\b.{0,30}\b(step|procedure|process|method|approach|technique)\b", re.IGNORECASE),
        re.compile(r"\bsteps?\s+(to|for|in)\b", re.IGNORECASE),
        re.compile(r"\bstep.by.step\b", re.IGNORECASE),
        re.compile(r"\bprocedure\s+for\b", re.IGNORECASE),
        re.compile(r"\bhow\s+to\b", re.IGNORECASE),
        re.compile(r"\binstructions?\s+(for|to|on)\b", re.IGNORECASE),
        re.compile(r"\bmethod\s+(for|to)\b", re.IGNORECASE),
        re.compile(r"\bprocess\s+(of|for)\b.{0,20}\b(making|doing|creating|applying|using|performing)\b", re.IGNORECASE),
        re.compile(r"\bwhat\s+(are|were)\s+the\s+steps?\b", re.IGNORECASE),
        re.compile(r"\bwalk\s+(me\s+)?through\b", re.IGNORECASE),
    ],
    # ---- QUOTE_EVIDENCE: direct textual evidence / quotes ----
    Intent.QUOTE_EVIDENCE: [
        re.compile(r"\b(exact\s+)?(quote|quotes|quotation|quoted)\b", re.IGNORECASE),
        re.compile(r"\bverbatim\b", re.IGNORECASE),
        re.compile(r"\bword\s*-?\s*for\s*-?\s*word\b", re.IGNORECASE),
        re.compile(r"\bdirect\s+(passage|evidence|support|quote|citation)\b", re.IGNORECASE),
        re.compile(r"\btextual\s+evidence\b", re.IGNORECASE),
        re.compile(r"\bwhat\s+does\s+the\s+(text|document|author|passage|source)\s+say\s+exactly\b", re.IGNORECASE),
        re.compile(r"\bshow\s+me\s+(the\s+)?exact\b.{0,20}\b(words?|text|passage|lines?)\b", re.IGNORECASE),
        re.compile(r"\bfind\s+(me\s+)?(a\s+)?(quote|passage|excerpt)\b", re.IGNORECASE),
        re.compile(r"\bevidence\s+(of|for|that|supporting)\b", re.IGNORECASE),
        re.compile(r"\bsupport(ing|s|ed)?\s+(this\s+)?(claim|argument|point|statement|view|position)\b", re.IGNORECASE),
    ],
}

_HEURISTIC_CONFIDENCE = {"strong_match": 0.85, "single_match": 0.70, "weak_match": 0.50}
_TECHNICAL_TERM_HINTS = {"stimulus", "reinforcement", "skinner"}
_OVERVIEW_SUBJECT_WORDS = {
    "this", "that", "it", "paper", "document", "article", "text",
    "docs", "doc", "source", "sources", "collection", "corpus",
}
_NON_DEFINITION_QUERY_WORDS = {
    "main", "key", "point", "points", "overview", "summary", "gist",
    "argument", "arguments", "difference", "differences", "similarity",
    "similarities", "role", "effect", "effects",
    "we", "i", "you", "here", "there", "in", "at", "looking",
    "available",
}
_LOW_INFO_COMMON_WORDS = {
    "what", "why", "how", "who", "when", "where", "which", "explain", "summarize",
    "compare", "analyze", "critique", "document", "docs", "paper", "text", "about",
    "mean", "help", "understand", "list", "show", "tell", "overview", "details",
    "extract", "timeline", "quote", "steps", "instructions", "procedure",
}

_INTENT_NORMALIZATION_LEXICON = {
    "which", "what", "document", "documents", "doc", "docs", "source", "sources", "file", "files",
    "about", "covers", "cover", "discuss", "discusses", "mention", "mentions", "regarding",
    "critique", "criticism", "review", "overview", "summarize", "summarise",
}


def _normalize_for_intent(query: str) -> str:
    """Lightweight typo-tolerant normalization for intent/routing decisions."""
    if not query:
        return ""

    parts = re.split(r"(\W+)", query.lower())
    normalized_parts: list[str] = []
    for part in parts:
        if not part or not part.isalpha() or len(part) < 3:
            normalized_parts.append(part)
            continue
        if part in _INTENT_NORMALIZATION_LEXICON:
            normalized_parts.append(part)
            continue
        match = get_close_matches(part, _INTENT_NORMALIZATION_LEXICON, n=1, cutoff=0.82)
        normalized_parts.append(match[0] if match else part)
    return "".join(normalized_parts)


def is_source_selection_query(query: str) -> bool:
    """Return True when the user asks which source/document matches a topic.

    Uses typo-tolerant normalization first, then robust structural matching.
    """
    normalized = _normalize_for_intent(query).strip().lower()
    if not normalized:
        return False

    has_doc_term = bool(
        re.search(r"\b(doc|docs|document|documents|source|sources|file|files)\b", normalized)
    )
    if not has_doc_term:
        return False

    has_selector = bool(
        re.search(r"\bwhich\b", normalized)
        or re.search(r"\bwhich\s+(one|ones|of\s+these|of\s+the)\b", normalized)
        or re.search(r"\bwhat\s+(doc|docs|document|documents|source|sources|file|files)\b", normalized)
    )
    has_topic_relation = bool(
        re.search(
            r"\b(cover|covers|covered|about|regarding|mention|mentions|discuss|discusses|review|reviews|critique|criticism|criticize|criticises|criticizes)\b",
            normalized,
        )
    )
    return has_selector or has_topic_relation


# ── Structural signal patterns ────────────────────────────────────────────────

_COMMAND_VERB_INTENTS: dict[re.Pattern, Intent] = {
    re.compile(r"^\s*compare\b", re.IGNORECASE): Intent.COMPARE,
    re.compile(r"^\s*contrast\b", re.IGNORECASE): Intent.COMPARE,
    re.compile(r"^\s*summari[sz]e\b", re.IGNORECASE): Intent.SUMMARIZE,
    re.compile(r"^\s*explain\b", re.IGNORECASE): Intent.EXPLAIN,
    re.compile(r"^\s*analy[sz]e\b", re.IGNORECASE): Intent.ANALYZE,
    re.compile(r"^\s*trace\s+(the\s+)?(history|development|progression|sequence|evolution)\b", re.IGNORECASE): Intent.TIMELINE,
    re.compile(r"^\s*trace\b", re.IGNORECASE): Intent.ANALYZE,
    re.compile(r"^\s*extract\b", re.IGNORECASE): Intent.EXTRACT,
    re.compile(r"^\s*identify\b", re.IGNORECASE): Intent.FACTUAL,
    re.compile(r"^\s*list\s+(all\s+)?(the\s+)?(names?|dates?|years?|figures?|entities|terms?)\b", re.IGNORECASE): Intent.EXTRACT,
    re.compile(r"^\s*create\s+(a\s+)?timeline\b", re.IGNORECASE): Intent.TIMELINE,
    re.compile(r"^\s*show\s+(me\s+)?the\s+timeline\b", re.IGNORECASE): Intent.TIMELINE,
    re.compile(r"^\s*how\s+to\b", re.IGNORECASE): Intent.HOW_TO,
    re.compile(r"^\s*quote\b", re.IGNORECASE): Intent.QUOTE_EVIDENCE,
    re.compile(r"^\s*find\s+(me\s+)?(a\s+)?quote\b", re.IGNORECASE): Intent.QUOTE_EVIDENCE,
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

# Patterns that indicate a factual extraction query (boosts FACTUAL intent).
# Compare with _EXTRACT_STRUCTURES which boosts the dedicated EXTRACT intent.
_FACTUAL_EXTRACTION_PATTERNS: list[re.Pattern] = [
    re.compile(r"\bextract\b", re.IGNORECASE),
    re.compile(r"\blist\s+all\s+(names?|dates?|years?|titles?|authors?|citations?)\b", re.IGNORECASE),
    re.compile(r"\bgive\s+me\s+the\s+(dates?|years?|names?|citations?)\b", re.IGNORECASE),
    re.compile(r"\bformat\b.*\bas\s+(a\s+)?table\b", re.IGNORECASE),
    re.compile(r"\btabular\b", re.IGNORECASE),
    # Exhaustive enumeration — recall-oriented factual queries (Fix 7)
    re.compile(r"\blist\s+(every|all)\b", re.IGNORECASE),
    re.compile(r"\bname\s+all\b", re.IGNORECASE),
    re.compile(r"\bwho\s+are\s+all\b", re.IGNORECASE),
    # Entity-mention / reference-finding — chunk-level recall queries
    re.compile(r"\bmentions?\s+of\b", re.IGNORECASE),
    re.compile(r"\bmentioned\b", re.IGNORECASE),
    re.compile(r"\breferences?\s+to\b", re.IGNORECASE),
    re.compile(r"\bwhere\s+is\b.+\bdiscuss", re.IGNORECASE),
    re.compile(r"\bwhere\s+does\b.+\bappear\b", re.IGNORECASE),
    re.compile(r"\bwhat\s+(does|did)\s+the\s+text\s+say\s+about\b", re.IGNORECASE),
    re.compile(r"\bwhat\s+is\s+said\s+about\b", re.IGNORECASE),
]

_SUMMARIZATION_STRUCTURES: list[re.Pattern] = [
    re.compile(r"\btl;?dr\b", re.IGNORECASE),
    re.compile(r"\bmain\s+points?\b", re.IGNORECASE),
    re.compile(r"\bkey\s+points?\b", re.IGNORECASE),
    re.compile(r"\boverview\s+of\b", re.IGNORECASE),
]

_EXTRACT_STRUCTURES: list[re.Pattern] = [
    re.compile(r"\bextract\b.{0,30}\b(entities|names?|dates?|years?|figures?|numbers?|definitions?|terms?|data)\b", re.IGNORECASE),
    re.compile(r"\blist\b.{0,20}\b(all|every)\b.{0,20}\b(names?|dates?|years?|figures?|entities|terms?)\b", re.IGNORECASE),
    re.compile(r"\bformat\b.{0,30}\bas\s+(a\s+)?(table|structured\s+list)\b", re.IGNORECASE),
    re.compile(r"\btabular\b", re.IGNORECASE),
    re.compile(r"\bpull\s+out\b", re.IGNORECASE),
    re.compile(r"\bcompile\s+(a\s+)?(list|table|inventory)\b", re.IGNORECASE),
]

_TIMELINE_STRUCTURES: list[re.Pattern] = [
    re.compile(r"\btimeline\b", re.IGNORECASE),
    re.compile(r"\bchronolog", re.IGNORECASE),
    re.compile(r"\bin\s+(chronological|date|time)\s+order\b", re.IGNORECASE),
    re.compile(r"\bsequence\s+of\s+events?\b", re.IGNORECASE),
]

_HOW_TO_STRUCTURES: list[re.Pattern] = [
    re.compile(r"\bhow\s+to\b", re.IGNORECASE),
    re.compile(r"\bstep.by.step\b", re.IGNORECASE),
    re.compile(r"\bsteps?\s+(to|for|in)\b", re.IGNORECASE),
    re.compile(r"\bwalk\s+(me\s+)?through\b", re.IGNORECASE),
    re.compile(r"\binstructions?\s+(for|to|on)\b", re.IGNORECASE),
]

_QUOTE_EVIDENCE_STRUCTURES: list[re.Pattern] = [
    re.compile(r"\b(exact\s+)?quote\b", re.IGNORECASE),
    re.compile(r"\bverbatim\b", re.IGNORECASE),
    re.compile(r"\btextual\s+evidence\b", re.IGNORECASE),
    re.compile(r"\bdirect\s+(passage|evidence|support)\b", re.IGNORECASE),
    re.compile(r"\bsupporting\s+(this\s+)?(claim|argument|point)\b", re.IGNORECASE),
    re.compile(r"\bsay\s+exactly\b", re.IGNORECASE),
    re.compile(r"\bword\s*-?\s*for\s*-?\s*word\b", re.IGNORECASE),
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

    if any(pattern.search(normalized) for pattern in _FACTUAL_EXTRACTION_PATTERNS):
        scores[Intent.FACTUAL] += 3

    if any(pattern.search(normalized) for pattern in _SUMMARIZATION_STRUCTURES):
        scores[Intent.SUMMARIZE] += 2

    if any(pattern.search(normalized) for pattern in _EXTRACT_STRUCTURES):
        scores[Intent.EXTRACT] += 3

    if any(pattern.search(normalized) for pattern in _TIMELINE_STRUCTURES):
        scores[Intent.TIMELINE] += 3

    if any(pattern.search(normalized) for pattern in _HOW_TO_STRUCTURES):
        scores[Intent.HOW_TO] += 2

    if any(pattern.search(normalized) for pattern in _QUOTE_EVIDENCE_STRUCTURES):
        scores[Intent.QUOTE_EVIDENCE] += 3

    if _is_definition_style_query(normalized):
        scores[Intent.FACTUAL] += 2

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


def _is_definition_style_query(query: str) -> bool:
    """Detect short definitional prompts like 'What is community?'"""
    match = re.match(
        r"^\s*what\s+(?:is|are)\s+(?:an?\s+)?([a-z][a-z0-9'\-]*(?:\s+[a-z][a-z0-9'\-]*){0,2})\s*\??\s*$",
        query,
        re.IGNORECASE,
    )
    if not match:
        return False

    subject_tokens = set(match.group(1).lower().split())
    if subject_tokens & _OVERVIEW_SUBJECT_WORDS:
        return False
    if subject_tokens & _NON_DEFINITION_QUERY_WORDS:
        return False

    return True


# ── Why-question specificity override (Fix 6) ────────────────────────────────

_WHY_ENTITY_STOPWORDS: set[str] = {
    # Determiners / pronouns / auxiliaries
    "the", "a", "an", "is", "are", "was", "were", "do", "does", "did",
    "in", "on", "at", "to", "for", "of", "and", "or", "but", "not",
    "this", "that", "if", "how", "why", "what", "who", "where", "which",
    "has", "have", "had", "will", "would", "could", "should", "can",
    "with", "from", "about", "into", "through", "also", "just",
    "i", "me", "my", "we", "us", "you", "your", "he", "him", "his",
    "she", "her", "they", "them", "their", "it", "its",
    "some", "any", "no", "all", "every", "each", "many", "much",
    # Abstract / thematic nouns often capitalised in titles / topic phrases
    "love", "death", "fate", "time", "life", "nature", "power", "truth",
    "beauty", "peace", "war", "justice", "honor", "honour", "hope", "fear",
    "theme", "role", "concept", "idea", "theory", "importance",
}

_WHY_ACTION_VERB_RE = re.compile(
    r'\b('
    r'ask(?:s|ed|ing)?|read(?:s|ing)?|kill(?:s|ed|ing)?|'
    r'say(?:s|ing)?|said|go(?:es|ing)?|went|'
    r'mak(?:e|es|ing)|made|tak(?:e|es|ing)|took|'
    r'giv(?:e|es|ing)|gave|tell(?:s|ing)?|told|'
    r'call(?:s|ed|ing)?|send(?:s|ing)?|sent|'
    r'writ(?:e|es|ing)?|wrote|fight(?:s|ing)?|fought|'
    r'die(?:s|d)?|dying|drink(?:s|ing)?|drank|'
    r'speak(?:s|ing)?|spoke|hid(?:e|es|ing)?|'
    r'marry|marries|married|marrying|'
    r'refus(?:e|es|ed|ing)|want(?:s|ed|ing)?|'
    r'poison(?:s|ed|ing)?|stab(?:s|bed|bing)?|banish(?:es|ed|ing)?|'
    r'meet(?:s|ing)?|met|leav(?:e|es|ing)?|left|'
    r'buy(?:s|ing)?|bought|sell(?:s|ing)?|sold|'
    r'steal(?:s|ing)?|stole|challeng(?:e|es|ed|ing)|'
    r'invit(?:e|es|ed|ing)|warn(?:s|ed|ing)?|'
    r'run(?:s|ning)?|ran|carry|carries|carried|carrying|'
    r'bring(?:s|ing)?|brought|hold(?:s|ing)?|held|'
    r'open(?:s|ed|ing)?|clos(?:e|es|ed|ing)|'
    r'danc(?:e|es|ed|ing)|sing(?:s|ing)?|sang|sung'
    r')\b',
    re.IGNORECASE,
)


def _detect_why_specificity(query: str) -> tuple[Optional[str], Optional[str]]:
    """Detect if a why-question references a named entity performing a specific action.

    Returns ``(entity, verb)`` when both are found, ``(None, None)`` otherwise.
    """
    words = query.split()
    if len(words) < 4:
        return None, None

    # Detect proper noun: capitalised word not at position 0 and not a stopword
    entity: Optional[str] = None
    for word in words[1:]:
        clean = re.sub(r"[^A-Za-z]", "", word)
        if (
            clean
            and len(clean) >= 2
            and clean[0].isupper()
            and clean.lower() not in _WHY_ENTITY_STOPWORDS
        ):
            entity = clean
            break

    if entity is None:
        return None, None

    # Detect concrete action verb
    match = _WHY_ACTION_VERB_RE.search(query)
    if match is None:
        return None, None

    return entity, match.group(1)


def _compute_intent_scores(normalized_query: str) -> tuple[dict["Intent", int], bool]:
    """Score each intent from regex patterns and structural signals.

    Returns (scores, analyze_bias).
    """
    scores: dict[Intent, int] = {intent: 0 for intent in Intent}
    for intent, patterns in _INTENT_PATTERNS.items():
        for pattern in patterns:
            if pattern.search(normalized_query):
                scores[intent] += 1

    _apply_structural_intent_signals(normalized_query, scores)

    # Noun-phrase de-boost: "Chomsky's critique" is a noun, not an instruction
    if scores[Intent.CRITIQUE] > 0:
        noun_critique = re.search(
            r"(?:\b\w+(?:'s|s)\s+|(?:the|a|an|this|that|his|her|their|its)\s+)critique\b",
            normalized_query, re.IGNORECASE,
        )
        if noun_critique:
            scores[Intent.CRITIQUE] = max(0, scores[Intent.CRITIQUE] - 1)
            logger.debug("De-boosted CRITIQUE: noun phrase (%s)", noun_critique.group())

    analyze_bias = _is_technical_how_why(normalized_query)
    if analyze_bias:
        analytical = [Intent.COMPARE, Intent.CRITIQUE, Intent.ANALYZE]
        best_analytical = max(analytical, key=lambda k: scores[k])
        if any(scores[i] > 0 for i in analytical):
            scores[best_analytical] += 2
        else:
            scores[Intent.ANALYZE] += 2

    return scores, analyze_bias


def _apply_tiebreaks(scores: dict["Intent", int]) -> tuple["Intent", int]:
    """Apply tie-break rules and return (best_intent, best_score)."""
    best_intent = max(scores, key=lambda k: scores[k])
    best_score = scores[best_intent]
    matching_intents = [i for i, s in scores.items() if s > 0]

    # Dedicated intents take priority when they have the highest score
    _dedicated_intents = (Intent.EXTRACT, Intent.TIMELINE, Intent.HOW_TO, Intent.QUOTE_EVIDENCE)
    _dedicated_best_score = max(scores[i] for i in _dedicated_intents)
    _dedicated_wins = _dedicated_best_score > 0 and _dedicated_best_score >= max(
        scores[Intent.COLLECTION], scores[Intent.FACTUAL],
        scores[Intent.SUMMARIZE], scores[Intent.ANALYZE],
    )

    # COMPARE/CRITIQUE wins over weak ANALYZE evidence
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

    # COLLECTION wins ties with SUMMARIZE
    if (
        not _dedicated_wins
        and Intent.COLLECTION in matching_intents
        and Intent.SUMMARIZE in matching_intents
        and scores[Intent.COLLECTION] >= scores[Intent.SUMMARIZE]
    ):
        best_intent = Intent.COLLECTION
        best_score = scores[Intent.COLLECTION]

    # COLLECTION wins ties with FACTUAL for document-selection queries
    if (
        not _dedicated_wins
        and Intent.COLLECTION in matching_intents
        and Intent.FACTUAL in matching_intents
        and scores[Intent.COLLECTION] >= scores[Intent.FACTUAL]
    ):
        best_intent = Intent.COLLECTION
        best_score = scores[Intent.COLLECTION]

    # COMPARE wins ties with CRITIQUE
    if (
        scores[Intent.COMPARE] > 0
        and scores[Intent.CRITIQUE] > 0
        and scores[Intent.COMPARE] >= scores[Intent.CRITIQUE]
    ):
        best_intent = Intent.COMPARE
        best_score = scores[Intent.COMPARE]
        scores[Intent.CRITIQUE] = 0

    return best_intent, best_score


def _compute_confidence(
    scores: dict["Intent", int],
    best_intent: "Intent",
    best_score: int,
    analyze_bias: bool,
) -> float:
    """Compute confidence from score distribution and analysis bias."""
    matching_intents = [i for i, s in scores.items() if s > 0]

    if len(matching_intents) > 1 and best_score == 1:
        confidence = _HEURISTIC_CONFIDENCE["weak_match"]
    elif best_score >= 2:
        confidence = _HEURISTIC_CONFIDENCE["strong_match"]
    else:
        confidence = _HEURISTIC_CONFIDENCE["single_match"]

    if best_intent in (Intent.ANALYZE, Intent.COMPARE, Intent.CRITIQUE) and analyze_bias:
        confidence = min(0.95, confidence + 0.15)

    return confidence


def _classify_heuristic(query: str) -> IntentResult:
    """Classify intent using regex pattern matching + structural signals."""
    normalized_query = _normalize_for_intent(query)

    scores, analyze_bias = _compute_intent_scores(normalized_query)

    best_intent = max(scores, key=lambda k: scores[k])
    if scores[best_intent] == 0:
        return IntentResult(intent=Intent.OVERVIEW, confidence=0.40, method="fallback")

    best_intent, best_score = _apply_tiebreaks(scores)
    confidence = _compute_confidence(scores, best_intent, best_score, analyze_bias)

    # Why-question specificity override (Fix 6)
    if best_intent in (Intent.ANALYZE, Intent.EXPLAIN) and re.match(
        r"^\s*why\b", query, re.IGNORECASE
    ):
        entity, verb = _detect_why_specificity(query)
        if entity is not None and verb is not None:
            original = best_intent.value
            best_intent = Intent.FACTUAL
            confidence = max(confidence, _HEURISTIC_CONFIDENCE["strong_match"])
            logger.info(
                "Why-question specificity override: %s -> factual (entity='%s', action='%s')",
                original, entity, verb,
            )

    return IntentResult(intent=best_intent, confidence=confidence, method="heuristic")


def is_low_information_query(query: str) -> bool:
    """Return True when query is likely underspecified or gibberish-like."""
    normalized = _normalize_for_intent(query).strip().lower()
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
- collection: User wants to know what documents are available or wants an overview of all documents in the corpus
- extract: User wants specific structured data pulled out — names, dates, figures, definitions, entities — formatted as a list or table
- timeline: User wants events ordered chronologically; query uses words like "timeline", "chronological", "sequence of events", "when did X happen then Y"
- how_to: User wants step-by-step procedural instructions describing a process from the source material; query uses "how to", "steps to", "instructions for", "walk me through"
- quote_evidence: User wants direct verbatim quotes or textual evidence supporting a claim; query uses "quote", "verbatim", "word for word", "direct evidence", "supporting this claim"

User query: "{query}"

Respond with ONLY a JSON object in this exact format:
{{"intent": "<overview|summarize|explain|compare|critique|analyze|factual|collection|extract|timeline|how_to|quote_evidence>", "confidence": <0.0-1.0>}}"""


def _parse_llm_response(response: str) -> Optional[tuple[Intent, float]]:
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
        from mlx_lm.generate import make_sampler

        prompt = _build_classification_prompt(query)

        # Apply chat template if the tokenizer supports it.
        if hasattr(self._llm_tokenizer, "apply_chat_template"):
            messages = [{"role": "user", "content": prompt}]
            formatted = self._llm_tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
        else:
            formatted = prompt

        # Use sampler instead of temperature kwarg — mlx_lm.generate_step()
        # does not accept temperature directly; sampling is controlled via
        # the sampler callable.
        sampler = make_sampler(temp=0.0, top_p=0.1)
        response = mlx_lm.generate(
            self._llm_model,
            self._llm_tokenizer,
            prompt=formatted,
            max_tokens=60,
            sampler=sampler,
        )

        parsed = _parse_llm_response(response)
        if parsed is None:
            logger.warning("Failed to parse LLM intent response: %s", response[:200])
            return None

        return IntentResult(intent=parsed[0], confidence=parsed[1], method="llm-fallback")

