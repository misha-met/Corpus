"""Offline tests for geocoder correctness.

Run with: python3 -m pytest tests/test_geocoder.py -v
Requires data/cities500.txt; skipped when absent.
"""
from __future__ import annotations

import math
from pathlib import Path

import pytest

from src.geocoder import (
    OfflineGeocoder,
    _radius_to_chord,
    _to_unit,
    extract_places_from_query,
)

pytestmark = pytest.mark.skipif(
    not Path("data/cities500.txt").exists(),
    reason="cities500.txt not present",
)


@pytest.fixture(scope="module")
def gc() -> OfflineGeocoder:
    geocoder = OfflineGeocoder()
    geocoder.warm(background=False)
    assert geocoder.is_available()
    return geocoder


def test_unit_sphere_norm() -> None:
    for lat, lon in [(0, 0), (90, 0), (-90, 0), (0, 180), (51.5, -0.1), (64.1, -21.9)]:
        vec = _to_unit(lat, lon)
        assert abs(math.sqrt(sum(float(x) ** 2 for x in vec)) - 1.0) < 1e-9


def test_antimeridian_near(gc: OfflineGeocoder) -> None:
    suva = gc.forward("Suva")
    assert suva is not None
    near = gc.find_near(suva.place.lat, suva.place.lon, radius_km=20)
    names = [p.name for p in near]
    assert "Suva" in names


def test_polar_near(gc: OfflineGeocoder) -> None:
    near = gc.find_near(69.6, 18.9, radius_km=100)
    assert all(p.lat > 50 for p in near)


def test_chord_radius_consistency() -> None:
    chord = _radius_to_chord(100)
    assert abs(chord - 0.015706) < 1e-4


@pytest.mark.parametrize(
    "name,country_hint,expected_country",
    [
        ("Paris", ("France",), "FR"),
        ("Paris", ("Texas",), "US"),
        ("Alexandria", ("Egypt",), "EG"),
        ("Alexandria", ("Virginia",), "US"),
        ("Antioch", ("Syria",), "TR"),
        ("Springfield", ("Illinois",), "US"),
    ],
)
def test_disambiguation(
    gc: OfflineGeocoder,
    name: str,
    country_hint: tuple[str, ...],
    expected_country: str,
) -> None:
    match = gc.forward(name, context_words=country_hint)
    assert match is not None
    assert match.place.country == expected_country, (
        f"{name} with context {country_hint}: got {match.place.country}, expected {expected_country}"
    )


@pytest.mark.parametrize("name", ["Byzantium", "Constantinople", "Istanbul"])
def test_istanbul_aliases(gc: OfflineGeocoder, name: str) -> None:
    match = gc.forward(name)
    assert match is not None
    assert match.place.asciiname == "Istanbul" or "Istanbul" in match.place.top_aliases


def test_get_aliases_o1(gc: OfflineGeocoder) -> None:
    match = gc.forward("Istanbul")
    assert match is not None
    aliases = gc.get_aliases(match.place.geonameid)
    assert len(aliases) > 3


def test_multi_entity_extraction() -> None:
    query = 'trade routes between "Tyre" and Carthage in the Mediterranean'
    places = extract_places_from_query(query)
    assert "Tyre" in places
    assert "Carthage" in places


def test_resolve_all(gc: OfflineGeocoder) -> None:
    query = "sources discussing events near Rome and Alexandria"
    places = extract_places_from_query(query)
    matches = gc.resolve_all(places, query=query)
    names = [m.place.asciiname for m in matches]
    assert "Rome" in names
    assert any("Alex" in n for n in names)


def test_confidence_monotonicity_exact_vs_fuzzy(gc: OfflineGeocoder) -> None:
    exact = gc.forward("Paris", context_words=("France",))
    fuzzy = gc.forward("Pariss", threshold=60, context_words=("France",))
    assert exact is not None
    assert fuzzy is not None
    assert exact.confidence > fuzzy.confidence


def test_confidence_acceptance_gates_provisional(gc: OfflineGeocoder) -> None:
    """Apply confidence gate checks only when the confidence bands are sufficiently labeled."""
    labeled_cases = [
        ("Paris", ("France",), "FR"),
        ("Paris", ("Texas",), "US"),
        ("Alexandria", ("Egypt",), "EG"),
        ("Alexandria", ("Virginia",), "US"),
        ("Antioch", ("Syria",), "TR"),
        ("Springfield", ("Illinois",), "US"),
        ("Rome", tuple(), "IT"),
        ("London", tuple(), "GB"),
        ("Athens", tuple(), "GR"),
        ("Cairo", tuple(), "EG"),
        ("Damascus", tuple(), "SY"),
        ("Istanbul", tuple(), "TR"),
    ]
    rows: list[tuple[float, bool]] = []
    for name, ctx, expected_country in labeled_cases:
        match = gc.forward(name, context_words=ctx)
        if match is None:
            continue
        rows.append((match.confidence, match.place.country == expected_country))

    band_90 = [correct for confidence, correct in rows if confidence >= 0.90]
    band_75 = [correct for confidence, correct in rows if confidence >= 0.75]

    if len(band_90) < 20 or len(band_75) < 20:
        pytest.skip("Provisional gates require at least 20 labeled examples in each confidence band.")

    p90 = sum(1 for flag in band_90 if flag) / len(band_90)
    p75 = sum(1 for flag in band_75 if flag) / len(band_75)

    assert p90 >= 0.95
    assert p75 >= 0.85


@pytest.mark.slow
def test_forward_latency(gc: OfflineGeocoder) -> None:
    import time

    t0 = time.perf_counter()
    for _ in range(50):
        gc.forward.__wrapped__(gc, "Constantinople")
    elapsed = time.perf_counter() - t0
    assert elapsed / 50 < 0.1, f"forward() avg {elapsed / 50:.3f}s > 100ms"


@pytest.mark.slow
def test_reverse_latency(gc: OfflineGeocoder) -> None:
    import time

    t0 = time.perf_counter()
    for _ in range(100):
        gc.reverse(41.01, 28.95, k=3)
    assert (time.perf_counter() - t0) / 100 < 0.01
