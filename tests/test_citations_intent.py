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
    is_low_information_query,
    is_source_selection_query,
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
        assert "[PASSAGE 1 | SOURCE: doc1 | PAGE: 5]" in result
        assert "Some text here." in result
        assert "[PASSAGE END]" in result

    def test_format_chunk_without_page(self):
        result = format_chunk_for_citation(
            "Some text.", source_id="doc1", display_page=None
        )
        assert "[PASSAGE 1 | SOURCE: doc1]" in result
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
            ("Compare the two approaches", Intent.COMPARE),
            ("What are the main arguments?", Intent.SUMMARIZE),
            ("Give me the gist", Intent.OVERVIEW),
            ("ELI5", Intent.EXPLAIN),
            ("How does reinforcement affect language?", Intent.ANALYZE),
            ("tl;dr", Intent.SUMMARIZE),
            ("What does this mean?", Intent.EXPLAIN),
            # FACTUAL intent tests
            ("What particular LLM is the author talking about?", Intent.FACTUAL),
            ("Who wrote this paper?", Intent.FACTUAL),
            ("When did Chomsky publish his review?", Intent.FACTUAL),
            ("How many core counts does the chip have?", Intent.FACTUAL),
            ("What does the author say about behaviorism?", Intent.FACTUAL),
            ("What specific method did they use?", Intent.FACTUAL),
            ("What is community?", Intent.FACTUAL),
            ("What is epistemology?", Intent.FACTUAL),
            ("Name the key theorists", Intent.FACTUAL),
            ("Extract all publication years and cited authors", Intent.EXTRACT),
            ("Format the results as a table", Intent.EXTRACT),
            ("what mentions of ChatGPT are there", Intent.FACTUAL),
            ("where is mortality discussed in the text", Intent.FACTUAL),
            ("find all references to the Nurse", Intent.FACTUAL),
            # COLLECTION intent tests
            ("What documents do we have?", Intent.COLLECTION),
            ("What are we looking at?", Intent.COLLECTION),
            ("List all the documents", Intent.COLLECTION),
            ("What are the docs in here?", Intent.COLLECTION),
            ("What docs are in here?", Intent.COLLECTION),
            ("Summarize all documents", Intent.COLLECTION),
            ("Show me all the sources", Intent.COLLECTION),
            ("What topics do the documents cover?", Intent.COLLECTION),
            ("Give me an overview of everything", Intent.COLLECTION),
            ("Which of these docs is a critique of ChatGPT?", Intent.COLLECTION),
            # Structural comparative / analysis
            ("Differences between Skinner and Chomsky", Intent.COMPARE),
            ("Critique of Skinner in light of modern LLMs", Intent.COMPARE),
            ("Trace the chain Skinner -> Chomsky -> modern LLM criticism", Intent.ANALYZE),
            ("What is chomskys critique of skinner", Intent.ANALYZE),
            ("whic document is abiut connectioism", Intent.COLLECTION),
        ],
    )
    def test_heuristic_classification(self, query: str, expected: Intent):
        result = _classify_heuristic(query)
        assert result.intent == expected, (
            f"Query '{query}': expected {expected.value}, got {result.intent.value} "
            f"(confidence={result.confidence:.2f})"
        )

    def test_empty_query_fallback(self):
        classifier = IntentClassifier()
        result = classifier.classify("")
        assert result.intent == Intent.OVERVIEW
        assert result.confidence == 1.0

    def test_llm_fallback_used_when_low_confidence(self):
        """When heuristic confidence is low and LLM model is configured,
        the LLM fallback should be attempted."""
        classifier = IntentClassifier(
            llm_model_id="fake-model",
            llm_fallback_threshold=0.90,  # Force fallback on everything
            confidence_threshold=0.6,
            eager_load_llm=False,
        )
        # Monkey-patch _classify_with_llm to return a known result
        classifier._classify_with_llm = lambda q: IntentResult(
            intent=Intent.COMPARE, confidence=0.85, method="llm-fallback",
        )
        result = classifier.classify("Compare Skinner and Chomsky")
        assert result.intent == Intent.COMPARE
        assert result.method == "llm-fallback"

    def test_llm_fallback_skipped_when_confident(self):
        """When heuristic confidence is high, LLM fallback should be skipped."""
        classifier = IntentClassifier(
            llm_model_id="fake-model",
            llm_fallback_threshold=0.70,
            confidence_threshold=0.6,
            eager_load_llm=False,
        )
        # If the LLM were called, it would crash — but it shouldn't be called
        classifier._classify_with_llm = lambda q: (_ for _ in ()).throw(
            RuntimeError("LLM should not be called")
        )
        result = classifier.classify("Compare Skinner and Chomsky")
        assert result.intent == Intent.COMPARE
        assert result.method == "heuristic"

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

    def test_parse_factual_intent(self):
        response = '{"intent": "factual", "confidence": 0.9}'
        parsed = _parse_llm_response(response)
        assert parsed is not None
        assert parsed[0] == Intent.FACTUAL

    def test_parse_collection_intent(self):
        response = '{"intent": "collection", "confidence": 0.85}'
        parsed = _parse_llm_response(response)
        assert parsed is not None
        assert parsed[0] == Intent.COLLECTION

    def test_parse_compare_intent(self):
        response = '{"intent": "compare", "confidence": 0.90}'
        parsed = _parse_llm_response(response)
        assert parsed is not None
        assert parsed[0] == Intent.COMPARE

    def test_parse_critique_intent(self):
        response = '{"intent": "critique", "confidence": 0.88}'
        parsed = _parse_llm_response(response)
        assert parsed is not None
        assert parsed[0] == Intent.CRITIQUE


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
            if intent == Intent.OVERVIEW:
                continue
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
        assert "[1], [2], [3]" in _CITATION_RULES
        assert "MUST cite" in _CITATION_RULES.upper() or "MUST CITE" in _CITATION_RULES.upper()


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
            "What particular LLM is the author talking about?",
            "What documents do we have?",
        ]
        for q in queries:
            with Timer("intent_heuristic", query=q) as t:
                result = _classify_heuristic(q)
            logger.info(
                f"intent_heuristic '{q[:40]}': {t.result.elapsed_ms:.3f}ms "
                f"-> {result.intent.value} ({result.confidence:.2f})"
            )
            assert t.result.elapsed_ms < 50, "Heuristic should be very fast"


# ===========================================================================
# FACTUAL intent: classification + message building + cite modes
# ===========================================================================

class TestFactualIntent:
    """Tests for the FACTUAL intent — direct fact extraction from context."""

    @pytest.mark.parametrize(
        "query",
        [
            "What particular LLM is the author talking about?",
            "Who is Chomsky?",
            "When was the review published?",
            "What year did this happen?",
            "How many parameters does the model have?",
            "What does the author say about behaviorism?",
            "Name the key theorists mentioned",
            "According to the text, what is Universal Grammar?",
            "What specific technique was used?",
            "Identify the main claim",
            "What model is discussed in this paper?",
            "Which framework does the author propose?",
        ],
    )
    def test_factual_classification(self, query: str):
        result = _classify_heuristic(query)
        assert result.intent == Intent.FACTUAL, (
            f"Query '{query}': expected factual, got {result.intent.value} "
            f"(confidence={result.confidence:.2f})"
        )

    def test_factual_confidence_above_threshold(self):
        """FACTUAL queries should have confidence above the default 0.6 threshold."""
        result = _classify_heuristic("What particular LLM is the author talking about?")
        assert result.confidence >= 0.60

    def test_factual_generation_messages_without_cite(self):
        """FACTUAL intent messages should contain direct-answer instructions."""
        messages = build_messages(
            context="Chomsky argues that language is innate.",
            question="What does Chomsky argue?",
            intent=Intent.FACTUAL,
            citations_enabled=False,
        )
        system = messages[0]["content"]
        assert "directly" in system.lower() or "direct" in system.lower()
        assert "CITATION REQUIREMENTS" not in system

    def test_factual_generation_messages_with_cite(self):
        """FACTUAL + citations should include both extraction and citation instructions."""
        messages = build_messages(
            context="[CHUNK START | SOURCE: doc1 | PAGE: 3]\nChomsky argues that language is innate.\n[CHUNK END]",
            question="What does Chomsky argue?",
            intent=Intent.FACTUAL,
            citations_enabled=True,
        )
        system = messages[0]["content"]
        assert "directly" in system.lower() or "direct" in system.lower()
        assert "CITATION" in system

    def test_factual_not_confused_with_analyze(self):
        """Queries like 'what does X say about Y' should be FACTUAL, not ANALYZE."""
        queries_should_be_factual = [
            "What does the author say about Skinner?",
            "What particular method is used?",
            "Who wrote the original paper?",
        ]
        for q in queries_should_be_factual:
            result = _classify_heuristic(q)
            assert result.intent == Intent.FACTUAL, (
                f"Query '{q}': expected factual, got {result.intent.value}"
            )

    def test_factual_instructions_in_all_modes(self):
        """FACTUAL instructions should be present in both regular and deep-research."""
        from src.generation import INTENT_INSTRUCTIONS_REGULAR, INTENT_INSTRUCTIONS_DEEP_RESEARCH
        assert Intent.FACTUAL in INTENT_INSTRUCTIONS_REGULAR
        assert Intent.FACTUAL in INTENT_INSTRUCTIONS_DEEP_RESEARCH
        assert "task" in INTENT_INSTRUCTIONS_REGULAR[Intent.FACTUAL]
        assert "task" in INTENT_INSTRUCTIONS_DEEP_RESEARCH[Intent.FACTUAL]


# ===========================================================================
# COMPARE intent: classification + message building
# ===========================================================================

class TestCompareIntent:
    """Tests for the COMPARE intent — side-by-side analysis of ideas."""

    @pytest.mark.parametrize(
        "query",
        [
            "Compare Chomsky's and Skinner's views on language",
            "Contrast the two approaches to learning",
            "What is the difference between behaviorism and nativism?",
            "How does connectionism differ from symbolic AI?",
            "Compare and contrast these positions",
        ],
    )
    def test_compare_classification(self, query: str):
        result = _classify_heuristic(query)
        assert result.intent == Intent.COMPARE, (
            f"Query '{query}': expected compare, got {result.intent.value} "
            f"(confidence={result.confidence:.2f})"
        )

    def test_compare_generation_messages(self):
        """COMPARE messages should contain comparison-specific instructions."""
        messages = build_messages(
            context="Theory A says X. Theory B says Y.",
            question="Compare these two theories",
            intent=Intent.COMPARE,
            citations_enabled=False,
        )
        system = messages[0]["content"]
        assert "converge" in system.lower() or "diverge" in system.lower() or "compared" in system.lower()

    def test_compare_instructions_in_all_modes(self):
        from src.generation import INTENT_INSTRUCTIONS_REGULAR, INTENT_INSTRUCTIONS_DEEP_RESEARCH
        assert Intent.COMPARE in INTENT_INSTRUCTIONS_REGULAR
        assert Intent.COMPARE in INTENT_INSTRUCTIONS_DEEP_RESEARCH
        assert "task" in INTENT_INSTRUCTIONS_REGULAR[Intent.COMPARE]
        assert "task" in INTENT_INSTRUCTIONS_DEEP_RESEARCH[Intent.COMPARE]


# ===========================================================================
# CRITIQUE intent: classification + message building
# ===========================================================================

class TestCritiqueIntent:
    """Tests for the CRITIQUE intent — evaluative analysis of arguments."""

    @pytest.mark.parametrize(
        "query",
        [
            "Evaluate the argument that language is innate",
            "Critique Skinner's behaviorist model",
            "What are the strengths and weaknesses of connectionism?",
            "Assess whether the author's conclusion follows from the evidence",
            "How convincing is the poverty of the stimulus argument?",
        ],
    )
    def test_critique_classification(self, query: str):
        result = _classify_heuristic(query)
        assert result.intent == Intent.CRITIQUE, (
            f"Query '{query}': expected critique, got {result.intent.value} "
            f"(confidence={result.confidence:.2f})"
        )

    def test_critique_generation_messages(self):
        """CRITIQUE messages should contain text-grounded evaluative instructions."""
        messages = build_messages(
            context="The argument claims X. Evidence shows Y.",
            question="Evaluate this argument",
            intent=Intent.CRITIQUE,
            citations_enabled=False,
        )
        system = messages[0]["content"]
        assert "text" in system.lower() or "context" in system.lower()
        assert "invent" in system.lower() or "one-sided" in system.lower()

    def test_critique_instructions_in_all_modes(self):
        from src.generation import INTENT_INSTRUCTIONS_REGULAR, INTENT_INSTRUCTIONS_DEEP_RESEARCH
        assert Intent.CRITIQUE in INTENT_INSTRUCTIONS_REGULAR
        assert Intent.CRITIQUE in INTENT_INSTRUCTIONS_DEEP_RESEARCH
        assert "task" in INTENT_INSTRUCTIONS_REGULAR[Intent.CRITIQUE]
        assert "task" in INTENT_INSTRUCTIONS_DEEP_RESEARCH[Intent.CRITIQUE]


# ===========================================================================
# COLLECTION intent: classification + message building + summary routing
# ===========================================================================

class TestCollectionIntent:
    """Tests for the COLLECTION intent — corpus-level document queries."""

    @pytest.mark.parametrize(
        "query",
        [
            "What documents do we have?",
            "What are we looking at?",
            "What are the docs in here?",
            "What docs are in here?",
            "List all the documents",
            "Show me all the sources",
            "Summarize all documents",
            "What's in this collection?",
            "Give me an overview of everything",
            "What topics do the documents cover?",
            "Describe all the sources",
            "What is in here?",
            "Overview of all documents",
            "Which of these docs is a critique of ChatGPT?",
        ],
    )
    def test_collection_classification(self, query: str):
        result = _classify_heuristic(query)
        assert result.intent == Intent.COLLECTION, (
            f"Query '{query}': expected collection, got {result.intent.value} "
            f"(confidence={result.confidence:.2f})"
        )

    def test_collection_confidence_above_threshold(self):
        result = _classify_heuristic("What documents do we have?")
        assert result.confidence >= 0.60

    def test_collection_generation_messages(self):
        """COLLECTION messages should include user-facing wording constraints."""
        messages = build_messages(
            context="Source: doc1\nSummary: About linguistics.\n\nSource: doc2\nSummary: About philosophy.",
            question="What documents do we have?",
            intent=Intent.COLLECTION,
            citations_enabled=False,
        )
        system = messages[0]["content"]
        assert "reference retrieval internals" in system.lower()

    def test_collection_citations_auto_disabled(self):
        """COLLECTION intent uses summaries, so citation rules should NOT be injected."""
        messages = build_messages(
            context="Source: doc1\nSummary: About linguistics.",
            question="What are the docs in here?",
            intent=Intent.COLLECTION,
            citations_enabled=False,
        )
        system = messages[0]["content"]
        assert "CITATION REQUIREMENTS" not in system

    def test_collection_not_confused_with_overview(self):
        """Corpus-level queries should not fall to OVERVIEW."""
        corpus_queries = [
            "What are we looking at?",
            "What documents do we have?",
            "List all documents",
        ]
        for q in corpus_queries:
            result = _classify_heuristic(q)
            assert result.intent == Intent.COLLECTION, (
                f"Query '{q}': expected collection, got {result.intent.value}"
            )

    @pytest.mark.parametrize(
        "query",
        [
            "what mentions of ChatGPT are there",
            "where is mortality discussed in the text",
            "find all references to the Nurse",
        ],
    )
    def test_entity_mentions_not_collection(self, query: str):
        """Entity-mention queries should route to FACTUAL chunk retrieval, not COLLECTION summaries."""
        result = _classify_heuristic(query)
        assert result.intent == Intent.FACTUAL, (
            f"Query '{query}': expected factual, got {result.intent.value}"
        )

    def test_single_doc_overview_stays_overview(self):
        """Single-document overview queries should still be OVERVIEW, not COLLECTION."""
        queries_should_be_overview = [
            "What is this paper about?",
            "What is this document about?",
            "Give me the gist",
        ]
        for q in queries_should_be_overview:
            result = _classify_heuristic(q)
            assert result.intent == Intent.OVERVIEW, (
                f"Query '{q}': expected overview, got {result.intent.value}"
            )

    def test_collection_instructions_in_all_modes(self):
        from src.generation import INTENT_INSTRUCTIONS_REGULAR, INTENT_INSTRUCTIONS_DEEP_RESEARCH
        assert Intent.COLLECTION in INTENT_INSTRUCTIONS_REGULAR
        assert Intent.COLLECTION in INTENT_INSTRUCTIONS_DEEP_RESEARCH

    def test_collection_summary_context_format(self):
        """Verify that summary-based context is properly formatted for COLLECTION."""
        summaries = {
            "doc_linguistics": "This document covers Chomsky's theory of generative grammar.",
            "doc_philosophy": "This document covers epistemology and normative ethics.",
        }
        summary_blocks = [
            f"Source: {source}\nSummary: {summary}"
            for source, summary in summaries.items()
        ]
        context = "\n\n".join(summary_blocks)

        messages = build_messages(
            context=context,
            question="What are the docs in here?",
            intent=Intent.COLLECTION,
            citations_enabled=False,
        )
        user_msg = messages[1]["content"]
        assert "doc_linguistics" in user_msg
        assert "doc_philosophy" in user_msg
        assert "Chomsky" in user_msg
        assert "epistemology" in user_msg


# ===========================================================================
# Cross-intent boundary tests
# ===========================================================================

class TestIntentBoundaries:
    """Edge cases where queries could be ambiguous between intents."""

    def test_what_is_overview_not_factual(self):
        """'What is this?' is OVERVIEW, not FACTUAL."""
        result = _classify_heuristic("What is this?")
        assert result.intent == Intent.OVERVIEW

    def test_what_is_paper_overview_not_factual(self):
        result = _classify_heuristic("What is this paper about?")
        assert result.intent == Intent.OVERVIEW

    def test_what_is_term_is_factual(self):
        result = _classify_heuristic("What is community?")
        assert result.intent == Intent.FACTUAL

    def test_summarize_all_is_collection(self):
        """'Summarize all documents' should be COLLECTION, not SUMMARIZE."""
        result = _classify_heuristic("Summarize all documents")
        assert result.intent == Intent.COLLECTION

    def test_summarize_single_is_summarize(self):
        """'Summarize the key points' should be SUMMARIZE, not COLLECTION."""
        result = _classify_heuristic("Summarize the key points")
        assert result.intent == Intent.SUMMARIZE

    def test_who_wrote_is_factual(self):
        result = _classify_heuristic("Who wrote the original paper?")
        assert result.intent == Intent.FACTUAL

    def test_compare_is_compare(self):
        result = _classify_heuristic("Compare these two approaches")
        assert result.intent == Intent.COMPARE

    def test_explain_is_explain(self):
        result = _classify_heuristic("Explain this in simple terms")
        assert result.intent == Intent.EXPLAIN

    def test_what_model_is_factual(self):
        result = _classify_heuristic("What LLM is being used?")
        assert result.intent == Intent.FACTUAL


class TestLowInformationQueryDetection:
    @pytest.mark.parametrize(
        "query,expected",
        [
            ("wa wa wa", True),
            ("googly moogly", True),
            ("", True),
            ("What is this paper about?", False),
            ("Summarize the key points", False),
            ("What docs are in here", False),
            ("ELI5", False),
        ],
    )
    def test_low_information_detector(self, query: str, expected: bool):
        assert is_low_information_query(query) is expected


class TestSourceSelectionRoutingGuard:
    @pytest.mark.parametrize(
        "query,expected",
        [
            ("Which of these docs covers a critique of ChatGPT?", True),
            ("which of thse docs covers aa crituqe of chatgpt", True),
            ("What source discusses reinforcement learning risks?", True),
            ("Summarize the key points of this paper", False),
            ("Explain this in simple terms", False),
        ],
    )
    def test_source_selection_query_detector(self, query: str, expected: bool):
        assert is_source_selection_query(query) is expected


class TestQueryExpansionRemoved:
    """Verify the dead query expansion infrastructure has been fully removed."""

    def test_no_expand_query_function(self) -> None:
        import src.rag_engine as _mod
        assert not hasattr(_mod, "_expand_query"), (
            "_expand_query still exists — it should have been removed"
        )

    def test_no_expansion_terms_dict(self) -> None:
        import src.rag_engine as _mod
        assert not hasattr(_mod, "_EXPANSION_TERMS"), (
            "_EXPANSION_TERMS still exists — it should have been removed"
        )

    def test_query_method_no_expansion_param(self) -> None:
        import inspect, src.rag_engine as _mod
        sig = inspect.signature(_mod.RagEngine.query)
        assert "enable_query_expansion" not in sig.parameters, (
            "query() still accepts enable_query_expansion"
        )


# ===========================================================================
# New intents: EXTRACT, TIMELINE, HOW_TO, QUOTE_EVIDENCE
# ===========================================================================

class TestExtractIntent:
    """Tests for the EXTRACT intent — structured data extraction."""

    @pytest.mark.parametrize(
        "query",
        [
            "Extract all dates mentioned in the document",
            "Extract all author names",
            "Extract all figures and numbers from the text",
            "List all the names in chronological order",
            "Give me all the entities in a table",
            "Pull out all the dates from the text",
            "Compile a list of all the terms defined",
            "Format the results as a table",
        ],
    )
    def test_extract_classification(self, query: str):
        result = _classify_heuristic(query)
        assert result.intent == Intent.EXTRACT, (
            f"Query '{query}': expected extract, got {result.intent.value} "
            f"(confidence={result.confidence:.2f})"
        )

    def test_extract_in_intent_map(self):
        """EXTRACT should be in the _INTENT_MAP for LLM response parsing."""
        from src.intent import _INTENT_MAP
        assert "extract" in _INTENT_MAP
        assert _INTENT_MAP["extract"] == Intent.EXTRACT

    def test_extract_instructions_present(self):
        from src.generation import INTENT_INSTRUCTIONS_REGULAR, INTENT_INSTRUCTIONS_DEEP_RESEARCH
        assert Intent.EXTRACT in INTENT_INSTRUCTIONS_REGULAR
        assert Intent.EXTRACT in INTENT_INSTRUCTIONS_DEEP_RESEARCH
        cfg = INTENT_INSTRUCTIONS_REGULAR[Intent.EXTRACT]
        assert "task" in cfg and "format" in cfg and "tone" in cfg
        assert "exhaustiv" in cfg["task"].lower()

    def test_extract_build_messages(self):
        messages = build_messages(
            context="The battle of Waterloo occurred in 1815. Napoleon was defeated.",
            question="Extract all dates mentioned.",
            intent=Intent.EXTRACT,
            citations_enabled=False,
        )
        system = messages[0]["content"]
        assert "list" in system.lower() or "table" in system.lower()


class TestTimelineIntent:
    """Tests for the TIMELINE intent — chronological event ordering."""

    @pytest.mark.parametrize(
        "query",
        [
            "Create a timeline of events",
            "Show me the timeline",
            "List events in chronological order",
            "What is the chronology of events in this document?",
            "Give me a chronological sequence of events",
            "Trace the history of development in the text",
        ],
    )
    def test_timeline_classification(self, query: str):
        result = _classify_heuristic(query)
        assert result.intent == Intent.TIMELINE, (
            f"Query '{query}': expected timeline, got {result.intent.value} "
            f"(confidence={result.confidence:.2f})"
        )

    def test_timeline_in_intent_map(self):
        from src.intent import _INTENT_MAP
        assert "timeline" in _INTENT_MAP
        assert _INTENT_MAP["timeline"] == Intent.TIMELINE

    def test_timeline_instructions_present(self):
        from src.generation import INTENT_INSTRUCTIONS_REGULAR, INTENT_INSTRUCTIONS_DEEP_RESEARCH
        assert Intent.TIMELINE in INTENT_INSTRUCTIONS_REGULAR
        assert Intent.TIMELINE in INTENT_INSTRUCTIONS_DEEP_RESEARCH
        cfg = INTENT_INSTRUCTIONS_REGULAR[Intent.TIMELINE]
        assert "task" in cfg and "format" in cfg and "tone" in cfg
        assert "chronolog" in cfg["task"].lower() or "date" in cfg["task"].lower()

    def test_timeline_build_messages(self):
        messages = build_messages(
            context="In 1789 the Revolution began. In 1815 Napoleon was defeated.",
            question="Create a timeline of events.",
            intent=Intent.TIMELINE,
            citations_enabled=False,
        )
        system = messages[0]["content"]
        assert "chronolog" in system.lower() or "order" in system.lower()


class TestHowToIntent:
    """Tests for the HOW_TO intent — step-by-step procedural queries."""

    @pytest.mark.parametrize(
        "query",
        [
            "How to apply this technique?",
            "What are the steps to reproduce the experiment?",
            "Walk me through the process",
            "Give step-by-step instructions for this procedure",
            "What are the steps in this method?",
            "Instructions for applying this approach",
        ],
    )
    def test_how_to_classification(self, query: str):
        result = _classify_heuristic(query)
        assert result.intent == Intent.HOW_TO, (
            f"Query '{query}': expected how_to, got {result.intent.value} "
            f"(confidence={result.confidence:.2f})"
        )

    def test_how_to_in_intent_map(self):
        from src.intent import _INTENT_MAP
        assert "how_to" in _INTENT_MAP
        assert _INTENT_MAP["how_to"] == Intent.HOW_TO

    def test_how_to_instructions_present(self):
        from src.generation import INTENT_INSTRUCTIONS_REGULAR, INTENT_INSTRUCTIONS_DEEP_RESEARCH
        assert Intent.HOW_TO in INTENT_INSTRUCTIONS_REGULAR
        assert Intent.HOW_TO in INTENT_INSTRUCTIONS_DEEP_RESEARCH
        cfg = INTENT_INSTRUCTIONS_REGULAR[Intent.HOW_TO]
        assert "task" in cfg and "format" in cfg and "tone" in cfg
        assert "step" in cfg["format"].lower() or "procedur" in cfg["format"].lower()

    def test_how_to_build_messages(self):
        messages = build_messages(
            context="First, prepare the reagents. Second, mix them carefully.",
            question="How to apply this technique?",
            intent=Intent.HOW_TO,
            citations_enabled=False,
        )
        system = messages[0]["content"]
        assert "step" in system.lower() or "procedur" in system.lower()


class TestQuoteEvidenceIntent:
    """Tests for the QUOTE_EVIDENCE intent — direct textual evidence."""

    @pytest.mark.parametrize(
        "query",
        [
            "Find me a quote supporting this claim",
            "What does the text say exactly about this?",
            "Give me a verbatim passage about the argument",
            "Find direct evidence supporting this position",
            "Quote the relevant passage word for word",
            "What textual evidence supports this argument?",
        ],
    )
    def test_quote_evidence_classification(self, query: str):
        result = _classify_heuristic(query)
        assert result.intent == Intent.QUOTE_EVIDENCE, (
            f"Query '{query}': expected quote_evidence, got {result.intent.value} "
            f"(confidence={result.confidence:.2f})"
        )

    def test_quote_evidence_in_intent_map(self):
        from src.intent import _INTENT_MAP
        assert "quote_evidence" in _INTENT_MAP
        assert _INTENT_MAP["quote_evidence"] == Intent.QUOTE_EVIDENCE

    def test_quote_evidence_instructions_present(self):
        from src.generation import INTENT_INSTRUCTIONS_REGULAR, INTENT_INSTRUCTIONS_DEEP_RESEARCH
        assert Intent.QUOTE_EVIDENCE in INTENT_INSTRUCTIONS_REGULAR
        assert Intent.QUOTE_EVIDENCE in INTENT_INSTRUCTIONS_DEEP_RESEARCH
        cfg = INTENT_INSTRUCTIONS_REGULAR[Intent.QUOTE_EVIDENCE]
        assert "task" in cfg and "format" in cfg and "tone" in cfg
        assert "quot" in cfg["task"].lower() or "verbatim" in cfg["format"].lower()

    def test_quote_evidence_build_messages(self):
        messages = build_messages(
            context="The author states: 'Language is innate to all humans.'",
            question="Find a direct quote supporting the nativist view.",
            intent=Intent.QUOTE_EVIDENCE,
            citations_enabled=False,
        )
        system = messages[0]["content"]
        assert "quote" in system.lower() or "verbatim" in system.lower()


class TestNewIntentEnum:
    """Verify all new intents are correctly wired into the enum and classifier."""

    def test_all_new_intents_in_enum(self):
        assert Intent.EXTRACT.value == "extract"
        assert Intent.TIMELINE.value == "timeline"
        assert Intent.HOW_TO.value == "how_to"
        assert Intent.QUOTE_EVIDENCE.value == "quote_evidence"

    def test_all_new_intents_in_intent_map(self):
        from src.intent import _INTENT_MAP
        for key in ("extract", "timeline", "how_to", "quote_evidence"):
            assert key in _INTENT_MAP, f"'{key}' missing from _INTENT_MAP"

    def test_all_new_intents_have_generation_instructions(self):
        from src.generation import INTENT_INSTRUCTIONS_REGULAR
        for intent in (Intent.EXTRACT, Intent.TIMELINE, Intent.HOW_TO, Intent.QUOTE_EVIDENCE):
            assert intent in INTENT_INSTRUCTIONS_REGULAR, (
                f"{intent.value} missing from INTENT_INSTRUCTIONS_REGULAR"
            )
            cfg = INTENT_INSTRUCTIONS_REGULAR[intent]
            for key in ("task", "format", "tone"):
                assert key in cfg, f"'{key}' missing from {intent.value} instructions"

    def test_llm_classification_prompt_includes_new_intents(self):
        from src.intent import _build_classification_prompt
        prompt = _build_classification_prompt("test query")
        for intent_name in ("extract", "timeline", "how_to", "quote_evidence"):
            assert intent_name in prompt, (
                f"'{intent_name}' not found in LLM classification prompt"
            )

    def test_parse_llm_response_handles_new_intents(self):
        from src.intent import _parse_llm_response
        for intent_name, expected_intent in [
            ("extract", Intent.EXTRACT),
            ("timeline", Intent.TIMELINE),
            ("how_to", Intent.HOW_TO),
            ("quote_evidence", Intent.QUOTE_EVIDENCE),
        ]:
            response = f'{{"intent": "{intent_name}", "confidence": 0.9}}'
            result = _parse_llm_response(response)
            assert result is not None, f"Failed to parse intent '{intent_name}'"
            assert result[0] == expected_intent

    def test_intent_override_with_new_intents(self):
        """rag_engine._classify_intent should handle new intent strings."""
        # We test the intent_map directly by checking that all new keys resolve
        from src.intent import _INTENT_MAP
        for key in ("extract", "timeline", "how_to", "quote_evidence"):
            assert key in _INTENT_MAP
            intent = _INTENT_MAP[key]
            assert isinstance(intent, Intent)


class TestGenerationImprovements:
    """Verify the spec-mandated improvements to existing intent instructions."""

    def test_summarize_bullet_range(self):
        from src.generation import INTENT_INSTRUCTIONS_REGULAR
        cfg = INTENT_INSTRUCTIONS_REGULAR[Intent.SUMMARIZE]
        # Sentence-count constraints replaced word-count constraints (plan phase 2A).
        # SUMMARIZE bullets should now specify "1-2" sentences, not a word range.
        assert "1-2" in cfg["format"], (
            "SUMMARIZE format should use sentence-count constraints (1-2 sentences per bullet)"
        )
        assert "15-35" not in cfg["format"], (
            "Word-count constraint '15-35 words' should be replaced with sentence count"
        )

    def test_summarize_has_no_opening_sentence_requirement(self):
        from src.generation import INTENT_INSTRUCTIONS_REGULAR
        cfg = INTENT_INSTRUCTIONS_REGULAR[Intent.SUMMARIZE]
        assert "Start with one sentence" not in cfg["format"], (
            "SUMMARIZE format should not require an opening document-identification sentence"
        )

    def test_factual_multi_passage_guidance(self):
        from src.generation import INTENT_INSTRUCTIONS_REGULAR
        cfg = INTENT_INSTRUCTIONS_REGULAR[Intent.FACTUAL]
        assert "multiple" in cfg["task"].lower() or "combining" in cfg["task"].lower(), (
            "FACTUAL task should mention multi-passage synthesis"
        )

    def test_collection_document_types_guidance(self):
        from src.generation import INTENT_INSTRUCTIONS_REGULAR
        cfg = INTENT_INSTRUCTIONS_REGULAR[Intent.COLLECTION]
        assert "document type" in cfg["task"].lower() or "types" in cfg["task"].lower(), (
            "COLLECTION task should mention document types"
        )
        assert "gap" in cfg["task"].lower(), (
            "COLLECTION task should mention corpus coverage gaps"
        )


class TestIntentOverrideAliases:
    def test_rag_engine_accepts_summarise_alias(self):
        from src.rag_engine import RagEngine

        engine = RagEngine.__new__(RagEngine)
        result = RagEngine._classify_intent(
            engine,
            query_text="anything",
            intent_override="summarise",
            no_generate=True,
        )

        assert result.intent == Intent.SUMMARIZE
        assert result.method == "manual"
        assert result.confidence == 1.0

    def test_explain_fallback_for_nontechnical(self):
        from src.generation import INTENT_INSTRUCTIONS_REGULAR
        cfg = INTENT_INSTRUCTIONS_REGULAR[Intent.EXPLAIN]
        assert "non-technical" in cfg["task"].lower() or "already" in cfg["task"].lower(), (
            "EXPLAIN task should have fallback for non-technical source material"
        )
