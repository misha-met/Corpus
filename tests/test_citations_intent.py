"""Tests for citation formatting, source legends, intent classification, and generation helpers."""
from __future__ import annotations

import json

import pytest

from src.intent import (
    Intent,
    IntentClassifier,
    IntentResult,
    _classify_heuristic,
    _parse_llm_response,
)
from src.retrieval import (
    build_source_legend,
    format_chunk_for_citation,
    format_context_with_citations,
)
from src.generation import (
    INTENT_INSTRUCTIONS,
    _CITATION_RULES,
    _SYSTEM_MESSAGE,
    _prepare_config,
    build_messages,
)
from tests.conftest import Timer, get_test_logger

logger = get_test_logger("citations_intent")


# ===========================================================================
# Citation formatting
# ===========================================================================

class TestCitationFormatting:
    def test_format_chunk_with_page(self):
        result = format_chunk_for_citation(
            "Some text here.", source_id="doc1", display_page="5"
        )
        assert "[CHUNK START | SOURCE: doc1 | PAGE: 5]" in result
        assert "Some text here." in result
        assert "[CHUNK END]" in result

    def test_format_chunk_without_page(self):
        result = format_chunk_for_citation(
            "Some text.", source_id="doc1", display_page=None
        )
        assert "[CHUNK START | SOURCE: doc1]" in result
        assert "PAGE" not in result

    def test_format_chunk_empty_page(self):
        result = format_chunk_for_citation(
            "Content.", source_id="src", display_page=""
        )
        # Empty string is falsy -> no page field
        assert "PAGE" not in result

    def test_format_context_with_citations_basic(self):
        texts = ["Content A.", "Content B."]
        metadatas = [
            {"source_id": "doc1", "display_page": "1"},
            {"source_id": "doc2", "display_page": "3"},
        ]
        context, mapping = format_context_with_citations(texts, metadatas)
        assert "SOURCE: doc1" in context
        assert "SOURCE: doc2" in context
        assert "Content A." in context
        assert "Content B." in context

    def test_format_context_mismatched_lengths(self):
        with pytest.raises(ValueError, match="same length"):
            format_context_with_citations(["a", "b"], [{"source_id": "x"}])

    def test_format_context_source_mapping(self):
        texts = ["Content."]
        metadatas = [
            {"source_id": "abc123", "display_page": "1", "doc_name": "My Document.pdf"},
        ]
        _, mapping = format_context_with_citations(texts, metadatas)
        assert mapping.get("abc123") == "My Document.pdf"

    def test_format_context_no_mapping_when_same(self):
        """No mapping entry when source_id matches doc_name."""
        texts = ["Content."]
        metadatas = [{"source_id": "doc1", "display_page": "1", "doc_name": "doc1"}]
        _, mapping = format_context_with_citations(texts, metadatas)
        assert "doc1" not in mapping


# ===========================================================================
# Source legend
# ===========================================================================

class TestSourceLegend:
    def test_build_legend_basic(self):
        mapping = {"abc123": "My Document.pdf", "def456": "Paper.pdf"}
        legend = build_source_legend(mapping)
        assert "SOURCE LEGEND:" in legend
        assert "abc123" in legend
        assert "My Document.pdf" in legend

    def test_build_legend_filters_trivial(self):
        """Legend should filter redundant mappings where id==name."""
        mapping = {"doc1": "doc1", "abc": "Paper.pdf"}
        legend = build_source_legend(mapping)
        assert "doc1 →" not in legend  # trivial mapping filtered
        assert "abc" in legend

    def test_build_legend_empty_mapping(self):
        assert build_source_legend({}) == ""

    def test_build_legend_all_trivial(self):
        mapping = {"doc1": "doc1", "doc2": "doc2"}
        assert build_source_legend(mapping) == ""


# ===========================================================================
# Citation output: enabled vs disabled
# ===========================================================================

class TestCitationToggle:
    def test_citations_enabled_in_messages(self):
        messages = build_messages(
            context="Context text",
            question="What is this?",
            intent=Intent.OVERVIEW,
            citations_enabled=True,
        )
        system_msg = messages[0]["content"]
        assert "CITATION" in system_msg or "citation" in system_msg

    def test_citations_disabled_in_messages(self):
        messages = build_messages(
            context="Context text",
            question="What is this?",
            intent=Intent.OVERVIEW,
            citations_enabled=False,
        )
        system_msg = messages[0]["content"]
        assert "CITATION REQUIREMENTS" not in system_msg

    def test_source_legend_included_when_enabled(self):
        messages = build_messages(
            context="Context text",
            question="What is this?",
            intent=Intent.OVERVIEW,
            citations_enabled=True,
            source_legend="SOURCE LEGEND:\n- abc → Doc.pdf",
        )
        user_msg = messages[1]["content"]
        assert "SOURCE LEGEND" in user_msg

    def test_source_legend_excluded_when_disabled(self):
        messages = build_messages(
            context="Context text",
            question="What is this?",
            intent=Intent.OVERVIEW,
            citations_enabled=False,
            source_legend="SOURCE LEGEND:\n- abc → Doc.pdf",
        )
        user_msg = messages[1]["content"]
        assert "SOURCE LEGEND" not in user_msg


# ===========================================================================
# Intent classification (heuristic)
# ===========================================================================

class TestIntentClassification:
    @pytest.mark.parametrize(
        "query,expected",
        [
            ("What is this paper about?", Intent.OVERVIEW),
            ("Summarize the key points", Intent.SUMMARIZE),
            ("Explain this in simple terms", Intent.EXPLAIN),
            ("Compare the two approaches", Intent.ANALYZE),
            ("What are the main arguments?", Intent.SUMMARIZE),
            ("Give me the gist", Intent.OVERVIEW),
            ("ELI5", Intent.EXPLAIN),
            ("How does reinforcement affect language?", Intent.ANALYZE),
            ("tl;dr", Intent.SUMMARIZE),
            ("What does this mean?", Intent.EXPLAIN),
        ],
    )
    def test_heuristic_classification(self, query: str, expected: Intent):
        result = _classify_heuristic(query)
        assert result.intent == expected, (
            f"Query '{query}': expected {expected.value}, got {result.intent.value} "
            f"(confidence={result.confidence:.2f})"
        )

    def test_empty_query_fallback(self):
        classifier = IntentClassifier(generator=None, use_llm=False)
        result = classifier.classify("")
        assert result.intent == Intent.OVERVIEW
        assert result.confidence == 1.0

    def test_ambiguous_query_fallback(self):
        """Ambiguous queries with low confidence should fallback to OVERVIEW."""
        # "random nonsense" should have no pattern matches
        result = _classify_heuristic("xyzzy flurble blargh")
        assert result.intent == Intent.OVERVIEW
        assert result.confidence < 0.5

    def test_intent_result_validation(self):
        with pytest.raises(ValueError):
            IntentResult(intent=Intent.OVERVIEW, confidence=1.5, method="test")
        with pytest.raises(ValueError):
            IntentResult(intent=Intent.OVERVIEW, confidence=-0.1, method="test")


# ===========================================================================
# LLM response parsing
# ===========================================================================

class TestLLMResponseParsing:
    def test_parse_valid_json(self):
        response = '{"intent": "summarize", "confidence": 0.85}'
        parsed = _parse_llm_response(response)
        assert parsed is not None
        assert parsed[0] == Intent.SUMMARIZE
        assert abs(parsed[1] - 0.85) < 0.01

    def test_parse_json_in_code_block(self):
        response = '```json\n{"intent": "explain", "confidence": 0.7}\n```'
        parsed = _parse_llm_response(response)
        assert parsed is not None
        assert parsed[0] == Intent.EXPLAIN

    def test_parse_invalid_json(self):
        assert _parse_llm_response("not json at all") is None

    def test_parse_unknown_intent(self):
        response = '{"intent": "unknown_type", "confidence": 0.5}'
        assert _parse_llm_response(response) is None

    def test_parse_confidence_clamped(self):
        response = '{"intent": "overview", "confidence": 2.5}'
        parsed = _parse_llm_response(response)
        assert parsed is not None
        assert parsed[1] <= 1.0


# ===========================================================================
# Generation message building
# ===========================================================================

class TestMessageBuilding:
    def test_messages_structure(self):
        messages = build_messages("context", "question")
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"

    def test_all_intents_have_instructions(self):
        for intent in Intent:
            assert intent in INTENT_INSTRUCTIONS
            cfg = INTENT_INSTRUCTIONS[intent]
            assert "task" in cfg
            assert "format" in cfg
            assert "tone" in cfg

    def test_system_message_contains_rules(self):
        assert "ONLY the provided context" in _SYSTEM_MESSAGE

    def test_extra_instructions_included(self):
        messages = build_messages(
            "ctx", "q", extra_instructions="Be very concise."
        )
        system = messages[0]["content"]
        assert "Be very concise" in system

    def test_citation_rules_format(self):
        assert "[SourceID, p. X]" in _CITATION_RULES


# ===========================================================================
# Latency: classification
# ===========================================================================

class TestIntentLatency:
    def test_heuristic_classification_latency(self):
        queries = [
            "What is this paper about?",
            "Summarize the main arguments",
            "How does Chomsky's theory compare to Skinner's?",
            "Explain the poverty of the stimulus",
        ]
        for q in queries:
            with Timer("intent_heuristic", query=q) as t:
                result = _classify_heuristic(q)
            logger.info(
                f"intent_heuristic '{q[:40]}': {t.result.elapsed_ms:.3f}ms "
                f"-> {result.intent.value} ({result.confidence:.2f})"
            )
            assert t.result.elapsed_ms < 50, "Heuristic should be very fast"
