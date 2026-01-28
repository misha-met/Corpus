"""
Intent classification module for RAG query processing.

Classifies user queries into intent types to enable intent-aware
prompt construction and retrieval augmentation.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


class Intent(Enum):
    """Supported intent types (v1.1 - with overview for high-level queries)."""
    OVERVIEW = "overview"    # New: bird's-eye view, document type + purpose
    SUMMARIZE = "summarize"  # Detailed summary with key points
    EXPLAIN = "explain"      # Simple language for non-experts
    ANALYZE = "analyze"      # Critique, controversy, evaluation


@dataclass(frozen=True)
class IntentResult:
    """Result of intent classification."""
    intent: Intent
    confidence: float
    method: str  # "llm", "heuristic", or "fallback"

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Confidence must be between 0.0 and 1.0")


# Heuristic patterns for fallback classification
# NOTE: Order does not control precedence; all intents are scored and the highest score wins.
_INTENT_PATTERNS: dict[Intent, list[re.Pattern]] = {
    # OVERVIEW: High-level "what is this?" queries - bird's-eye view
    Intent.OVERVIEW: [
        re.compile(r"^what\s+is\s+this\s*\??$", re.IGNORECASE),  # Exact "what is this?"
        re.compile(r"\bwhat\s+is\s+(this|the)\s+(paper|text|document|article)\s+(about\s*)?\??", re.IGNORECASE),
        re.compile(r"\bwhat\s+is\s+this\s+about\b", re.IGNORECASE),
        re.compile(r"\btell\s+me\s+about\s+(this|the)\s*(document|paper|text|article)?\b", re.IGNORECASE),
        re.compile(r"\bwhat('s| is)\s+the\s+point\s+of\s+this\b", re.IGNORECASE),
        re.compile(r"\bwhat\s+am\s+i\s+(reading|looking at)\b", re.IGNORECASE),
        re.compile(r"\bgive\s+me\s+(a|the)\s+(gist|overview)\b", re.IGNORECASE),
        re.compile(r"\bquick\s+(overview|summary)\b", re.IGNORECASE),
        re.compile(r"\bin\s+a\s+nutshell\b", re.IGNORECASE),
        re.compile(r"\bwhat\s+is\s+this\s+paper\b", re.IGNORECASE),  # "what is this paper"
    ],
    # EXPLAIN: Simple language for non-experts
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
    ],
    # ANALYZE: Critique, controversy, evaluation
    Intent.ANALYZE: [
        re.compile(r"\bhow\s+does\b", re.IGNORECASE),
        re.compile(r"\bin\s+what\s+way\b", re.IGNORECASE),
        re.compile(r"\bcompare\b", re.IGNORECASE),
        re.compile(r"\bcritique\b", re.IGNORECASE),
        re.compile(r"\bwhy\b.*\b(controversial|debate|disagree|critic)", re.IGNORECASE),
        re.compile(r"\bcontrovers", re.IGNORECASE),
        re.compile(r"\bcritici[sz]", re.IGNORECASE),
        re.compile(r"\bwhat\s+(are|were)\s+the\s+(criticism|objection|argument)", re.IGNORECASE),
        re.compile(r"\bhow\s+did\s+(people|scholars|critics)\s+react\b", re.IGNORECASE),
        re.compile(r"\bwhat\s+(is|was)\s+the\s+debate\b", re.IGNORECASE),
        re.compile(r"\banalyze\b", re.IGNORECASE),
        re.compile(r"\bevaluate\b", re.IGNORECASE),
        re.compile(r"\bstrengths?\s+and\s+weaknesses?\b", re.IGNORECASE),
        re.compile(r"\bpros?\s+and\s+cons?\b", re.IGNORECASE),
    ],
    # SUMMARIZE: Detailed summary - explicit requests for depth
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

# Confidence scores for heuristic matches
_HEURISTIC_CONFIDENCE = {
    "strong_match": 0.85,  # Multiple patterns or very specific pattern
    "single_match": 0.70,  # Single pattern match
    "weak_match": 0.50,    # Ambiguous match
}

_TECHNICAL_TERM_HINTS = {
    "stimulus",
    "reinforcement",
    "skinner",
}


def _has_technical_terms(query: str) -> bool:
    """Heuristic for technical nouns or quoted terms in a query."""
    if re.search(r"['\"`].+?['\"`]", query):
        return True
    if re.search(r"\b[A-Z][a-z]{2,}\b", query):
        return True
    lowered = query.lower()
    return any(term in lowered for term in _TECHNICAL_TERM_HINTS)


def _is_technical_how_why(query: str) -> bool:
    """Detect technical 'How/Why' questions to bias ANALYZE intent."""
    if not re.match(r"^\s*(how|why)\b", query, re.IGNORECASE):
        return False
    return _has_technical_terms(query)


def _classify_heuristic(query: str) -> IntentResult:
    """
    Classify intent using regex pattern matching.
    
    Returns IntentResult with confidence based on match strength.
    Falls back to OVERVIEW with low confidence if no patterns match.
    """
    scores: dict[Intent, int] = {intent: 0 for intent in Intent}
    
    for intent, patterns in _INTENT_PATTERNS.items():
        for pattern in patterns:
            if pattern.search(query):
                scores[intent] += 1

    analyze_bias = _is_technical_how_why(query)
    if analyze_bias:
        scores[Intent.ANALYZE] += 2
    
    # Find best match
    best_intent = max(scores, key=lambda k: scores[k])
    best_score = scores[best_intent]
    
    if best_score == 0:
        # No matches - fall back to OVERVIEW (not summarize) with low confidence
        return IntentResult(
            intent=Intent.OVERVIEW,
            confidence=0.40,
            method="fallback",
        )
    
    # Check for ambiguity (multiple intents have matches)
    matching_intents = [i for i, s in scores.items() if s > 0]
    
    if len(matching_intents) > 1 and scores[best_intent] == 1:
        # Ambiguous - single match but multiple intents detected
        confidence = _HEURISTIC_CONFIDENCE["weak_match"]
        if best_intent == Intent.ANALYZE and analyze_bias:
            confidence = min(0.95, confidence + 0.15)
        return IntentResult(
            intent=best_intent,
            confidence=confidence,
            method="heuristic",
        )
    
    if best_score >= 2:
        # Strong match - multiple patterns matched
        confidence = _HEURISTIC_CONFIDENCE["strong_match"]
        if best_intent == Intent.ANALYZE and analyze_bias:
            confidence = min(0.95, confidence + 0.10)
        return IntentResult(
            intent=best_intent,
            confidence=confidence,
            method="heuristic",
        )
    
    # Single clear match
    confidence = _HEURISTIC_CONFIDENCE["single_match"]
    if best_intent == Intent.ANALYZE and analyze_bias:
        confidence = min(0.95, confidence + 0.15)
    return IntentResult(
        intent=best_intent,
        confidence=confidence,
        method="heuristic",
    )


def _build_classification_prompt(query: str) -> str:
    """Build a minimal prompt for LLM-based intent classification."""
    return f"""Classify the user's intent into exactly one category.

Categories:
- overview: User wants a brief, high-level description of what the document is and its purpose
- summarize: User wants a detailed summary with key points and bullet points
- explain: User wants the content explained simply, for non-experts
- analyze: User wants analysis, critique, controversy, or evaluation

User query: "{query}"

Respond with ONLY a JSON object in this exact format:
{{"intent": "<overview|summarize|explain|analyze>", "confidence": <0.0-1.0>}}"""


def _parse_llm_response(response: str) -> Optional[Tuple[Intent, float]]:
    """Parse the LLM classification response."""
    import json
    
    # Try to extract JSON from response
    response = response.strip()
    
    # Handle potential markdown code blocks
    if "```" in response:
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", response, re.DOTALL)
        if match:
            response = match.group(1)
    
    # Find JSON object in response
    json_match = re.search(r"\{[^}]+\}", response)
    if not json_match:
        return None
    
    try:
        data = json.loads(json_match.group())
        intent_str = data.get("intent", "").lower().strip()
        confidence = float(data.get("confidence", 0.5))
        
        # Map string to Intent enum
        intent_map = {
            "overview": Intent.OVERVIEW,
            "summarize": Intent.SUMMARIZE,
            "explain": Intent.EXPLAIN,
            "analyze": Intent.ANALYZE,
        }
        
        intent = intent_map.get(intent_str)
        if intent is None:
            return None
        
        # Clamp confidence to valid range
        confidence = max(0.0, min(1.0, confidence))
        
        return (intent, confidence)
    except (json.JSONDecodeError, ValueError, TypeError):
        return None


class IntentClassifier:
    """
    Classifies user queries into intent types.
    
    Supports LLM-based classification with heuristic fallback.
    """
    
    def __init__(
        self,
        generator: Optional[object] = None,
        confidence_threshold: float = 0.6,
        use_llm: bool = True,
    ) -> None:
        """
        Initialize the intent classifier.
        
        Args:
            generator: Optional MlxGenerator instance for LLM classification
            confidence_threshold: Minimum confidence to accept classification
            use_llm: Whether to attempt LLM classification (falls back to heuristic)
        """
        self._generator = generator
        self._confidence_threshold = confidence_threshold
        self._use_llm = use_llm and generator is not None
    
    def classify(self, query: str) -> IntentResult:
        """
        Classify the intent of a user query.
        
        Returns IntentResult with intent, confidence, and method used.
        If confidence < threshold, returns OVERVIEW as fallback (not summarize).
        """
        if not query.strip():
            return IntentResult(
                intent=Intent.OVERVIEW,
                confidence=1.0,
                method="fallback",
            )
        
        result: Optional[IntentResult] = None
        
        # Try LLM classification first
        if self._use_llm and self._generator is not None:
            try:
                result = self._classify_with_llm(query)
            except Exception as e:
                logger.warning(f"LLM classification failed: {e}, falling back to heuristic")
                result = None
        
        # Fall back to heuristic if LLM failed or not available
        if result is None:
            result = _classify_heuristic(query)
        
        # Apply confidence threshold fallback -> OVERVIEW (not summarize)
        if result.confidence < self._confidence_threshold:
            logger.info(
                f"Intent '{result.intent.value}' confidence {result.confidence:.2f} "
                f"below threshold {self._confidence_threshold}, falling back to overview"
            )
            return IntentResult(
                intent=Intent.OVERVIEW,
                confidence=result.confidence,
                method=f"{result.method}+overview_fallback",
            )
        
        return result
    
    def _classify_with_llm(self, query: str) -> Optional[IntentResult]:
        """Classify using the LLM generator."""
        if self._generator is None:
            return None
        
        prompt = _build_classification_prompt(query)
        
        # Import here to avoid circular dependency
        from .generator import GenerationConfig
        
        # Use low temperature for classification
        config = GenerationConfig(
            max_tokens=50,
            temperature=0.1,
            top_p=0.9,
        )
        
        response = self._generator.generate(prompt, config=config)
        parsed = _parse_llm_response(response)
        
        if parsed is None:
            logger.warning(f"Failed to parse LLM response: {response}")
            return None
        
        intent, confidence = parsed
        return IntentResult(
            intent=intent,
            confidence=confidence,
            method="llm",
        )


def classify_intent(
    query: str,
    generator: Optional[object] = None,
    confidence_threshold: float = 0.6,
    use_llm: bool = True,
) -> IntentResult:
    """
    Convenience function to classify intent.
    
    Args:
        query: The user's query string
        generator: Optional MlxGenerator for LLM classification
        confidence_threshold: Minimum confidence to accept (default 0.6)
        use_llm: Whether to use LLM classification (default True)
    
    Returns:
        IntentResult with intent, confidence, and classification method
    """
    classifier = IntentClassifier(
        generator=generator,
        confidence_threshold=confidence_threshold,
        use_llm=use_llm,
    )
    return classifier.classify(query)
