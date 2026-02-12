"""Pipeline latency profiler – activated by ``--latency``.

Collects wall-clock timings for every pipeline stage and prints a formatted
breakdown at the end of the query.  Zero cost when disabled (the ``Profiler``
object acts as a no-op when ``enabled=False``).
"""
from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Generator, Optional


@dataclass
class _Span:
    label: str
    start: float = 0.0
    end: float = 0.0
    detail: str = ""

    @property
    def elapsed_ms(self) -> float:
        return (self.end - self.start) * 1000


@dataclass
class LatencyProfiler:
    """Lightweight profiler that collects named spans."""

    enabled: bool = False
    _spans: list[_Span] = field(default_factory=list)
    _wall_start: float = 0.0
    _wall_end: float = 0.0

    # -- recording ----------------------------------------------------------

    def start_wall(self) -> None:
        if self.enabled:
            self._wall_start = time.perf_counter()

    def end_wall(self) -> None:
        if self.enabled:
            self._wall_end = time.perf_counter()

    @contextmanager
    def span(self, label: str, detail: str = "") -> Generator[None, None, None]:
        """Context-manager that records a named timing span."""
        if not self.enabled:
            yield
            return
        s = _Span(label=label, start=time.perf_counter(), detail=detail)
        try:
            yield
        finally:
            s.end = time.perf_counter()
            self._spans.append(s)

    def record(self, label: str, elapsed_ms: float, detail: str = "") -> None:
        """Manually record a span that was already measured elsewhere."""
        if not self.enabled:
            return
        s = _Span(label=label, detail=detail)
        s.start = 0.0
        s.end = elapsed_ms / 1000  # store in seconds for consistency
        # Override so elapsed_ms property works:
        s.start = 0.0
        s.end = elapsed_ms / 1000
        # Actually, let's just store raw ms
        self._spans.append(s)
        # Patch so elapsed_ms works:
        self._spans[-1].start = 0.0
        self._spans[-1].end = elapsed_ms / 1000

    # -- reporting ----------------------------------------------------------

    @property
    def wall_ms(self) -> float:
        return (self._wall_end - self._wall_start) * 1000

    @property
    def accounted_ms(self) -> float:
        return sum(s.elapsed_ms for s in self._spans)

    def format_report(self) -> str:
        """Return a human-readable latency breakdown."""
        if not self.enabled or not self._spans:
            return ""

        lines: list[str] = []
        lines.append("")
        lines.append("=" * 76)
        lines.append("  LATENCY PROFILE")
        lines.append("=" * 76)

        max_ms = max((s.elapsed_ms for s in self._spans), default=1)

        for s in self._spans:
            bar_len = int(30 * s.elapsed_ms / max(max_ms, 1))
            bar = "\u2588" * bar_len
            detail_str = f"  ({s.detail})" if s.detail else ""
            lines.append(
                f"  {s.label:<42s} {s.elapsed_ms:>9.1f} ms  {bar}{detail_str}"
            )

        lines.append("-" * 76)
        lines.append(f"  {'Accounted':<42s} {self.accounted_ms:>9.1f} ms")
        lines.append(f"  {'Wall-clock total':<42s} {self.wall_ms:>9.1f} ms")

        unaccounted = self.wall_ms - self.accounted_ms
        if abs(unaccounted) > 10:
            lines.append(f"  {'Unaccounted (overhead / overlap)':<42s} {unaccounted:>9.1f} ms")

        lines.append("=" * 76)
        lines.append("")
        return "\n".join(lines)
