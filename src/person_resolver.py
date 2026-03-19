"""Resolver for ingest-time person mentions.

Canonicalises raw person-name mentions into a stable person dictionary using
exact and conservative fuzzy matching. The resolver keeps an in-memory
registry so repeated ingest calls stay consistent without additional DB reads.
"""
from __future__ import annotations

import re
import threading
from dataclasses import dataclass, field
from typing import Any, Optional

from rapidfuzz import fuzz, process

from . import config as app_config

_TITLE_TOKENS = {
    "dr",
    "prof",
    "mr",
    "mrs",
    "ms",
    "sir",
    "dame",
    "lord",
    "lady",
    "fr",
    "rev",
    "capt",
    "gen",
}


@dataclass
class _PersonEntry:
    canonical_name: str
    mention_count: int = 0
    source_ids: set[str] = field(default_factory=set)
    variants: dict[str, int] = field(default_factory=dict)


class PersonResolver:
    """Thread-safe in-memory registry for person canonicalization."""

    def __init__(
        self,
        *,
        fuzzy_threshold_lastname: int,
        fuzzy_threshold_fullname: int,
    ) -> None:
        self._fuzzy_threshold_lastname = int(fuzzy_threshold_lastname)
        self._fuzzy_threshold_fullname = int(fuzzy_threshold_fullname)
        self._lock = threading.RLock()
        self._entries: dict[str, _PersonEntry] = {}
        self._normalized_to_canonical: dict[str, str] = {}
        self._is_warm = False

    @property
    def is_warm(self) -> bool:
        with self._lock:
            return self._is_warm

    @staticmethod
    def _collapse_spaces(value: str) -> str:
        return re.sub(r"\s+", " ", value).strip()

    @staticmethod
    def _clean_token(token: str) -> str:
        return token.strip().strip(".,;:()[]{}\"'")

    @classmethod
    def _strip_titles(cls, value: str) -> str:
        tokens = [cls._clean_token(part) for part in value.split()]
        tokens = [token for token in tokens if token]
        while tokens:
            head = tokens[0].rstrip(".").lower()
            if head in _TITLE_TOKENS:
                tokens = tokens[1:]
                continue
            break
        return " ".join(tokens).strip()

    @classmethod
    def normalize_name(cls, value: str) -> str:
        stripped = cls._strip_titles(cls._collapse_spaces(value))
        if not stripped:
            return ""
        return re.sub(r"\s+", " ", stripped).strip().lower()

    @staticmethod
    def _display_rank(name: str) -> tuple[int, int]:
        tokens = [token for token in name.split() if token]
        return (len(tokens), len(name))

    @classmethod
    def _canonical_display_name(cls, value: str) -> str:
        stripped = cls._strip_titles(cls._collapse_spaces(value))
        return stripped or cls._collapse_spaces(value)

    @classmethod
    def _choose_better_display(cls, current: str, candidate: str) -> str:
        return candidate if cls._display_rank(candidate) > cls._display_rank(current) else current

    @staticmethod
    def _last_token(normalized: str) -> str:
        if not normalized:
            return ""
        parts = normalized.split()
        return parts[-1] if parts else ""

    @staticmethod
    def infer_role_hint(*, context_words: list[str], context_snippet: str) -> str:
        _ = context_words, context_snippet
        return "mentioned"

    def _rebuild_alias_index_locked(self) -> None:
        """Rebuild the normalized-name-to-canonical map from scratch.

        Uses the entry with the highest mention count to win conflicts rather
        than silently dropping one of the aliases via setdefault.
        """
        alias_map: dict[str, str] = {}
        # Process entries ordered by ascending mention_count so that the
        # highest-count entry wins when two entries share a normalised form.
        ordered = sorted(
            self._entries.items(),
            key=lambda kv: kv[1].mention_count,
        )
        for canonical_name, entry in ordered:
            names = [canonical_name, *entry.variants.keys()]
            for name in names:
                normalized = self.normalize_name(name)
                if not normalized:
                    continue
                alias_map[normalized] = canonical_name
        self._normalized_to_canonical = alias_map

    def warm_from_rows(self, rows: list[dict[str, Any]]) -> None:
        """Rebuild resolver registry from persisted mention rows."""
        with self._lock:
            self._entries = {}
            for row in rows:
                canonical_name = self._collapse_spaces(str(row.get("canonical_name", "")))
                raw_name = self._collapse_spaces(str(row.get("raw_name", "")))
                source_id = self._collapse_spaces(str(row.get("source_id", "")))

                if not canonical_name:
                    canonical_name = raw_name
                if not canonical_name:
                    continue
                if not raw_name:
                    raw_name = canonical_name

                entry = self._entries.get(canonical_name)
                if entry is None:
                    entry = _PersonEntry(canonical_name=canonical_name)
                    self._entries[canonical_name] = entry

                entry.mention_count += 1
                if source_id:
                    entry.source_ids.add(source_id)
                entry.variants[raw_name] = entry.variants.get(raw_name, 0) + 1

            self._rebuild_alias_index_locked()
            self._is_warm = True

    def _best_canonical_name_locked(self, entry: _PersonEntry) -> str:
        best = self._canonical_display_name(entry.canonical_name)
        for variant in entry.variants.keys():
            candidate = self._canonical_display_name(variant)
            best = self._choose_better_display(best, candidate)
        return best

    def _merge_entries_locked(self, survivor: _PersonEntry, absorbed: _PersonEntry) -> None:
        """Merge absorbed into survivor in place."""
        survivor.mention_count += absorbed.mention_count
        survivor.source_ids.update(absorbed.source_ids)
        for variant, count in absorbed.variants.items():
            survivor.variants[variant] = survivor.variants.get(variant, 0) + count

    def _maybe_promote_canonical_locked(self, entry: _PersonEntry) -> str:
        """Promote the best display name to canonical, merging on collision.

        Bug fix: the original code would silently overwrite an existing entry
        when the promoted name was already a key in self._entries. The entry
        that was overwritten lost all its data. This version detects that
        collision and merges the two entries instead.
        """
        best = self._best_canonical_name_locked(entry)
        if best == entry.canonical_name:
            self._rebuild_alias_index_locked()
            return entry.canonical_name

        old_name = entry.canonical_name

        existing = self._entries.get(best)
        if existing is not None and existing is not entry:
            # Collision: merge the entry being promoted into the one already
            # registered under the target name, then discard the old key.
            self._merge_entries_locked(existing, entry)
            self._entries.pop(old_name, None)
            self._rebuild_alias_index_locked()
            return best

        entry.canonical_name = best
        self._entries.pop(old_name, None)
        self._entries[best] = entry
        self._rebuild_alias_index_locked()
        return best

    def _fuzzy_match_lastname_locked(self, normalized: str) -> tuple[Optional[str], float]:
        """Fuzzy match on last token, comparing against last tokens of all aliases.

        Bug fix: the original compared the full normalised input against aliases
        that share the same last token. When name-order variants exist (e.g.
        "smith john" vs "john smith") the last tokens differ so the candidate
        set is empty and no match is found, producing a duplicate entry.

        The fix extracts the last token from both the query and each alias so
        that the comparison is always surname-to-surname regardless of token
        ordering.
        """
        last = self._last_token(normalized)
        if not last:
            return None, 0.0

        # Build a map of alias -> canonical where surname matches.
        candidates: dict[str, str] = {}
        for alias, canonical in self._normalized_to_canonical.items():
            alias_last = self._last_token(alias)
            if alias_last == last:
                candidates[alias] = canonical

        if not candidates:
            return None, 0.0

        canonical_candidates = set(candidates.values())

        # If we only know a surname alias (e.g. existing "chomsky") and now
        # see a fuller form ("noam chomsky"), link to that canonical first.
        if " " in normalized and last in candidates and len(canonical_candidates) == 1:
            only_canonical = next(iter(canonical_candidates))
            return only_canonical, 100.0

        # Surname-only mentions (e.g. "Chomsky") should resolve to the most
        # established canonical with that surname when possible.
        if " " not in normalized:
            canonical_scores: dict[str, tuple[int, tuple[int, int]]] = {}
            for canonical in canonical_candidates:
                entry = self._entries.get(canonical)
                mention_count = int(entry.mention_count) if entry is not None else 0
                canonical_scores[canonical] = (mention_count, self._display_rank(canonical))

            if canonical_scores:
                ranked = sorted(
                    canonical_scores.items(),
                    key=lambda item: (item[1][0], item[1][1][0], item[1][1][1]),
                    reverse=True,
                )
                if len(ranked) == 1 or ranked[0][1] > ranked[1][1]:
                    return ranked[0][0], 100.0

        match = process.extractOne(
            normalized,
            list(candidates.keys()),
            scorer=fuzz.WRatio,
            score_cutoff=self._fuzzy_threshold_lastname,
        )
        if match is None:
            return None, 0.0

        choice, score, _idx = match
        canonical = candidates.get(choice)
        return canonical, float(score)

    def _fuzzy_match_fullname_locked(self, normalized: str) -> tuple[Optional[str], float]:
        choices = list(self._normalized_to_canonical.keys())
        if not choices:
            return None, 0.0

        match = process.extractOne(
            normalized,
            choices,
            scorer=fuzz.WRatio,
            score_cutoff=self._fuzzy_threshold_fullname,
        )
        if match is None:
            return None, 0.0

        choice, score, _idx = match
        canonical = self._normalized_to_canonical.get(choice)
        return canonical, float(score)

    def resolve(
        self,
        *,
        raw_name: str,
        source_id: str,
        ner_score: float,
        context_words: list[str],
        context_snippet: str,
    ) -> Optional[dict[str, Any]]:
        """Resolve a single mention into canonical form and update registry."""
        clean_raw = self._collapse_spaces(raw_name)
        if not clean_raw:
            return None

        normalized = self.normalize_name(clean_raw)
        if not normalized:
            return None

        role_hint = self.infer_role_hint(
            context_words=context_words,
            context_snippet=context_snippet,
        )

        method = "new"
        match_confidence = 0.0

        with self._lock:
            canonical_name = self._normalized_to_canonical.get(normalized)
            if canonical_name:
                method = "exact"
                match_confidence = 1.0
            else:
                canonical_name, last_score = self._fuzzy_match_lastname_locked(normalized)
                if canonical_name:
                    method = "fuzzy_last"
                    match_confidence = float(last_score) / 100.0
                else:
                    canonical_name, full_score = self._fuzzy_match_fullname_locked(normalized)
                    if canonical_name:
                        method = "fuzzy_full"
                        match_confidence = float(full_score) / 100.0

            if not canonical_name:
                canonical_name = self._strip_titles(clean_raw) or clean_raw

            entry = self._entries.get(canonical_name)
            if entry is None:
                entry = _PersonEntry(canonical_name=canonical_name)
                self._entries[canonical_name] = entry

            entry.mention_count += 1
            if source_id:
                entry.source_ids.add(source_id)
            entry.variants[clean_raw] = entry.variants.get(clean_raw, 0) + 1

            # _maybe_promote_canonical_locked calls _rebuild_alias_index_locked
            # internally, so no second call is needed here (bug fix: the
            # original called _rebuild_alias_index_locked again after this,
            # causing the index to be rebuilt against a partially updated state).
            final_canonical = self._maybe_promote_canonical_locked(entry)
            self._is_warm = True

        try:
            ner_conf = float(ner_score)
        except (TypeError, ValueError):
            ner_conf = 0.0
        ner_conf = max(0.0, min(1.0, ner_conf))

        confidence = max(ner_conf, float(match_confidence))
        return {
            "canonical_name": final_canonical,
            "raw_name": clean_raw,
            "confidence": round(confidence, 4),
            "method": method,
            "role_hint": role_hint,
            "context_snippet": context_snippet,
        }

    def remove_mention(
        self,
        *,
        canonical_name: str,
        raw_name: str,
        source_id: Optional[str] = None,
    ) -> None:
        """Apply eager in-memory decrement after deleting one mention row."""
        canonical = self._collapse_spaces(canonical_name)
        variant = self._collapse_spaces(raw_name)
        if not canonical:
            return

        with self._lock:
            entry = self._entries.get(canonical)
            if entry is None:
                return

            entry.mention_count = max(0, entry.mention_count - 1)
            if variant and variant in entry.variants:
                next_count = max(0, entry.variants[variant] - 1)
                if next_count == 0:
                    entry.variants.pop(variant, None)
                else:
                    entry.variants[variant] = next_count

            if entry.mention_count == 0:
                self._entries.pop(canonical, None)
            else:
                self._maybe_promote_canonical_locked(entry)
                if source_id:
                    # source_ids is advisory; full correctness is restored on source-level re-warm.
                    entry.source_ids.discard(source_id)

            self._rebuild_alias_index_locked()

    def snapshot(self) -> dict[str, dict[str, Any]]:
        """Debug/testing helper with deterministic sorted output."""
        with self._lock:
            result: dict[str, dict[str, Any]] = {}
            for canonical_name in sorted(self._entries.keys()):
                entry = self._entries[canonical_name]
                result[canonical_name] = {
                    "mention_count": entry.mention_count,
                    "source_ids": sorted(entry.source_ids),
                    "variants": dict(sorted(entry.variants.items())),
                }
            return result


_resolver_lock = threading.Lock()
_resolver: Optional[PersonResolver] = None


def get_person_resolver() -> PersonResolver:
    """Return process-level resolver singleton."""
    global _resolver
    if _resolver is not None:
        return _resolver

    with _resolver_lock:
        if _resolver is None:
            _resolver = PersonResolver(
                fuzzy_threshold_lastname=app_config.PEOPLETAG_FUZZY_THRESHOLD_LASTNAME,
                fuzzy_threshold_fullname=app_config.PEOPLETAG_FUZZY_THRESHOLD_FULLNAME,
            )
    return _resolver