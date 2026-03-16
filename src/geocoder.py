"""Offline geocoder based on GeoNames cities500.txt.

Attribution: GeoNames (https://www.geonames.org)
License: Creative Commons Attribution 4.0 International
See: https://download.geonames.org/export/dump/readme.txt

Architecture:
- places_by_id: dict[int, GeoPlace]         one record per geonameid
- alias_to_ids: dict[str, list[int]]         normalized alias → [geonameid]
- _id_to_aliases: dict[int, list[str]]       geonameid → original aliases
- _ngram_to_ids: dict[str, set[int]]         trigram → geonameid set
- kdtree: cKDTree on 3D unit-sphere vectors
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re
import threading
import time
import unicodedata
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process
from scipy.spatial import cKDTree

from .config import (
    GEOTAG_ENTITY_TYPE_PENALTY,
    GEOTAG_FUZZY_MARGIN_THRESHOLD,
    GEOTAG_FUZZY_SCORE_FLOOR,
    GEOTAG_GENERIC_TOKEN_PENALTY,
    USE_HARDENED_GEOCODER,
)
from .geo_types import GeoMethod, GeocoderState

try:
    import pycountry as _pycountry
except ImportError:
    _pycountry = None

log = logging.getLogger(__name__)

# ── Tunables ──────────────────────────────────────────────────────────
_GEONAMES_PATH = "data/cities500.txt"
_VERSION_PATH = "data/geo_version.json"
_NGRAM_SIZE = 3
_NGRAM_CANDIDATES = 1_500
_DEFAULT_THRESHOLD = 72
_AMBIGUITY_WINDOW = 5.0
_STALE_DAYS = 180
_FORWARD_CACHE_SIZE = 2_048
_EARTH_RADIUS_KM = 6_371.0
_GEO_BOOST_MAX = 0.18
_GEO_DECAY_HALF_KM = 25.0
_MAX_EXPANSION_TERMS = 6
_MAX_DISAMBIG_ALIASES = 60

_GENERIC_PLACE_TERMS = frozenset({
    "city", "region", "area", "province", "state", "district", "county",
    "territory", "kingdom", "republic", "empire", "capital",
    "north", "south", "east", "west",
    "town", "village", "port", "fort", "mount", "lake", "river",
    "island", "peninsula", "valley", "bay", "gulf", "cape",
})

_METHOD_CONFIDENCE_CAPS = {
    GeoMethod.EXACT: 1.00,
    GeoMethod.REGION_TABLE: 0.94,
    GeoMethod.TRIGRAM_FUZZY: 0.88,
    GeoMethod.QUERY: 0.86,
    GeoMethod.REGEX: 0.72,
    GeoMethod.MANUAL: 1.00,
}

_COLS = [
    "geonameid", "name", "asciiname", "altnames", "lat", "lon",
    "feat_class", "feat_code", "country", "cc2",
    "admin1", "admin2", "admin3", "admin4",
    "population", "elevation", "dem", "timezone", "modified",
]


# ── Named historical regions: (lat, lon, radius_km) ──────────────────
NAMED_REGIONS: dict[str, tuple[float, float, float]] = {
    "mesopotamia": (33.5, 44.4, 350),
    "levant": (33.0, 36.0, 300),
    "rhineland": (50.9, 7.0, 150),
    "iberia": (40.0, -4.0, 700),
    "anatolia": (39.0, 35.0, 600),
    "gaul": (46.5, 2.5, 600),
    "magna graecia": (39.0, 16.5, 350),
    "holy land": (31.7, 35.2, 150),
    "bohemia": (49.8, 15.5, 200),
    "transylvania": (46.5, 24.5, 200),
    "pannonia": (47.0, 18.0, 350),
    "dacia": (45.5, 25.0, 300),
    "numidia": (36.0, 6.0, 400),
    "thrace": (41.5, 26.0, 250),
    "bithynia": (40.5, 30.0, 200),
}

# ── Query extraction patterns ────────────────────────────────────────
_SPATIAL_CUES = re.compile(
    r"\b(?:near|around|from|at|within|close to|proximate to|"
    r"surrounding|adjacent to|vicinity of|region of|area of|"
    r"province of|kingdom of)\b\s+",
    re.IGNORECASE,
)
_QUOTED_PLACE = re.compile(r'"([^"]{2,60})"')
_CAP_NP_STRICT = re.compile(
    r"^([A-Z][a-zA-Z\-]{1,}(?:\s+[A-Z][a-zA-Z\-]{1,}){0,2})"
)
_CAP_NP_CHAIN = re.compile(
    r"^([A-Z][a-zA-Z\-]{1,}(?:\s+[A-Z][a-zA-Z\-]{1,}){0,2}"
    r"(?:\s+and\s+[A-Z][a-zA-Z\-]{1,}(?:\s+[A-Z][a-zA-Z\-]{1,}){0,2})*)"
)
_BETWEEN_CHAIN = re.compile(
    r"\b[Bb]etween\s+\"?([A-Z][a-zA-Z\-]{1,}(?:\s+[A-Z][a-zA-Z\-]{1,}){0,2})\"?"
    r"\s+[Aa]nd\s+"
    r"\"?([A-Z][a-zA-Z\-]{1,}(?:\s+[A-Z][a-zA-Z\-]{1,}){0,2})\"?",
)

# ── Country / admin lookup tables ─────────────────────────────────────
_COUNTRY_HINT_TO_CODE: dict[str, str] = {
    "france": "FR", "french": "FR",
    "united states": "US", "usa": "US", "us": "US",
    "egypt": "EG", "egyptian": "EG",
    "syria": "SY", "syrian": "SY",
    "turkey": "TR", "turkish": "TR",
    "germany": "DE", "german": "DE",
    "italy": "IT", "italian": "IT",
    "spain": "ES", "spanish": "ES",
    "greece": "GR", "greek": "GR",
    "china": "CN", "chinese": "CN",
    "japan": "JP", "japanese": "JP",
    "india": "IN", "indian": "IN",
    "russia": "RU", "russian": "RU",
    "brazil": "BR", "brazilian": "BR",
    "mexico": "MX", "mexican": "MX",
    "canada": "CA", "canadian": "CA",
    "australia": "AU", "australian": "AU",
    "united kingdom": "GB", "uk": "GB", "britain": "GB", "british": "GB",
    "iran": "IR", "iranian": "IR", "persia": "IR", "persian": "IR",
    "iraq": "IQ", "iraqi": "IQ",
    "israel": "IL", "israeli": "IL",
    "poland": "PL", "polish": "PL",
    "portugal": "PT", "portuguese": "PT",
    "netherlands": "NL", "dutch": "NL",
    "sweden": "SE", "swedish": "SE",
    "norway": "NO", "norwegian": "NO",
    "austria": "AT", "austrian": "AT",
    "switzerland": "CH", "swiss": "CH",
}

_US_ADMIN1_HINT_TO_CODE: dict[str, str] = {
    "alabama": "AL", "al": "AL", "alaska": "AK", "ak": "AK",
    "arizona": "AZ", "az": "AZ", "arkansas": "AR", "ar": "AR",
    "california": "CA", "ca": "CA", "colorado": "CO", "co": "CO",
    "connecticut": "CT", "ct": "CT", "delaware": "DE", "de": "DE",
    "florida": "FL", "fl": "FL", "georgia": "GA", "ga": "GA",
    "hawaii": "HI", "hi": "HI", "idaho": "ID", "id": "ID",
    "illinois": "IL", "il": "IL", "indiana": "IN", "in": "IN",
    "iowa": "IA", "ia": "IA", "kansas": "KS", "ks": "KS",
    "kentucky": "KY", "ky": "KY", "louisiana": "LA", "la": "LA",
    "maine": "ME", "me": "ME", "maryland": "MD", "md": "MD",
    "massachusetts": "MA", "ma": "MA", "michigan": "MI", "mi": "MI",
    "minnesota": "MN", "mn": "MN", "mississippi": "MS", "ms": "MS",
    "missouri": "MO", "mo": "MO", "montana": "MT", "mt": "MT",
    "nebraska": "NE", "ne": "NE", "nevada": "NV", "nv": "NV",
    "new hampshire": "NH", "nh": "NH", "new jersey": "NJ", "nj": "NJ",
    "new mexico": "NM", "nm": "NM", "new york": "NY", "ny": "NY",
    "north carolina": "NC", "nc": "NC", "north dakota": "ND", "nd": "ND",
    "ohio": "OH", "oh": "OH", "oklahoma": "OK", "ok": "OK",
    "oregon": "OR", "or": "OR", "pennsylvania": "PA", "pa": "PA",
    "rhode island": "RI", "ri": "RI", "south carolina": "SC", "sc": "SC",
    "south dakota": "SD", "sd": "SD", "tennessee": "TN", "tn": "TN",
    "texas": "TX", "tx": "TX", "utah": "UT", "ut": "UT",
    "vermont": "VT", "vt": "VT", "virginia": "VA", "va": "VA",
    "washington": "WA", "wa": "WA", "west virginia": "WV", "wv": "WV",
    "wisconsin": "WI", "wi": "WI", "wyoming": "WY", "wy": "WY",
    "district of columbia": "DC", "dc": "DC",
}


# ── Dataclasses ───────────────────────────────────────────────────────

@dataclass(frozen=True)
class GeoPlace:
    geonameid: int
    name: str
    asciiname: str
    lat: float
    lon: float
    country: str
    admin1: str
    population: int
    top_aliases: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict:
        return {
            "geonameid": self.geonameid,
            "display": self.name,
            "asciiname": self.asciiname,
            "lat": self.lat,
            "lon": self.lon,
            "country": self.country,
            "admin1": self.admin1,
            "population": self.population,
        }


@dataclass(frozen=True)
class GeoMatch:
    place: GeoPlace
    score: float
    matched_on: str
    method: GeoMethod
    ambiguous: bool = False
    candidates: tuple[GeoPlace, ...] = field(default_factory=tuple)
    confidence_value: Optional[float] = None
    candidate_count: int = 1
    margin_score: Optional[float] = None
    entity_type: Optional[str] = None

    @property
    def confidence(self) -> float:
        """Composite confidence in [0, 1]."""
        if self.confidence_value is not None:
            return max(0.0, min(1.0, float(self.confidence_value)))
        method_prior = {
            GeoMethod.EXACT: 1.00,
            GeoMethod.REGION_TABLE: 0.95,
            GeoMethod.TRIGRAM_FUZZY: 0.90,
            GeoMethod.QUERY: 0.85,
            GeoMethod.REGEX: 0.70,
            GeoMethod.MANUAL: 1.00,
        }.get(self.method, 0.80)
        raw = (self.score / 100.0) * method_prior
        return raw * (0.85 if self.ambiguous else 1.0)


# ── Math / geo utilities ─────────────────────────────────────────────

def _to_unit(lat: float, lon: float) -> np.ndarray:
    """Convert (lat, lon) degrees to 3D unit-sphere vector."""
    phi, lam = math.radians(lat), math.radians(lon)
    cp = math.cos(phi)
    return np.array([cp * math.cos(lam), cp * math.sin(lam), math.sin(phi)])


def _radius_to_chord(radius_km: float) -> float:
    """Great-circle distance (km) → 3D chord length on unit sphere."""
    return 2.0 * math.sin(min(radius_km / _EARTH_RADIUS_KM, math.pi) / 2.0)


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Exact great-circle distance in km."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = (
        math.sin(dphi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2.0) ** 2
    )
    return 2.0 * _EARTH_RADIUS_KM * math.asin(math.sqrt(min(1.0, a)))


def distance_decay_boost(
    dist_km: float,
    max_boost: float = _GEO_BOOST_MAX,
    half_km: float = _GEO_DECAY_HALF_KM,
) -> float:
    """Exponential decay: max_boost × 2^(−dist / half_km)."""
    return max_boost * math.exp(-math.log(2) * dist_km / half_km)


def compute_geo_boost(
    chunk_lat: float,
    chunk_lon: float,
    query_lat: float,
    query_lon: float,
    radius_km: float,
    geo_confidence: float = 1.0,
) -> float:
    """Confidence-scaled, distance-decaying additive score bonus."""
    dist = haversine_km(chunk_lat, chunk_lon, query_lat, query_lon)
    if dist > radius_km * 2.0:
        return 0.0
    return distance_decay_boost(dist, half_km=radius_km / 2.0) * min(1.0, geo_confidence)


# ── Unicode / text normalization ──────────────────────────────────────

def _strip_diacritics(text: str) -> str:
    """Remove combining marks (accents) via NFKD decomposition.

    'São Paulo' → 'Sao Paulo', 'Zürich' → 'Zurich', etc.
    """
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(c for c in decomposed if unicodedata.category(c) != "Mn")


def _normalize_for_index(name: str) -> str:
    """Normalize a place name for index insertion: lowercase, stripped diacritics."""
    return _strip_diacritics(name).lower().strip()


def _normalize_query(place_name: str) -> str:
    """Normalize user input for lookup: lowercase, strip diacritics, trim articles."""
    query = _strip_diacritics(place_name).strip().lower()
    query = re.sub(r"\s+", " ", query)
    if query.startswith("the "):
        query = query[4:]
    return query


# ── Trigram helpers ───────────────────────────────────────────────────

def _trigrams(s: str) -> set[str]:
    padded = f"  {s}  "
    return {padded[i : i + _NGRAM_SIZE] for i in range(len(padded) - _NGRAM_SIZE + 1)}


# ── File utilities ────────────────────────────────────────────────────

def _file_checksum(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while buf := f.read(1 << 20):
            h.update(buf)
    return h.hexdigest()[:16]


def _check_stale(path: str) -> None:
    age_days = (time.time() - os.path.getmtime(path)) / 86400
    if age_days > _STALE_DAYS:
        log.warning(
            "cities500.txt is %.0f days old (>%d). Re-download: "
            "https://download.geonames.org/export/dump/",
            age_days,
            _STALE_DAYS,
        )


def save_version_info(path: str, extra: dict | None = None) -> None:
    info = {
        "path": path,
        "checksum": _file_checksum(path),
        "mtime": os.path.getmtime(path),
        "size_mb": round(os.path.getsize(path) / 1e6, 1),
        "age_days": round((time.time() - os.path.getmtime(path)) / 86400, 1),
        "build_timestamp": time.time(),
        "attribution": "GeoNames (https://www.geonames.org) - CC BY 4.0",
        **(extra or {}),
    }
    Path(_VERSION_PATH).write_text(json.dumps(info, indent=2), encoding="utf-8")


def load_version_info() -> dict | None:
    try:
        return json.loads(Path(_VERSION_PATH).read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None


# ── Country-code resolution ──────────────────────────────────────────

def _country_code_for_query(query: str) -> Optional[str]:
    """Resolve a string to an ISO-3166 alpha-2 country code, or None."""
    if not query:
        return None
    direct = _COUNTRY_HINT_TO_CODE.get(query)
    if direct:
        return direct
    if _pycountry is None:
        return None
    candidate = query.upper()
    if len(candidate) == 2:
        country = _pycountry.countries.get(alpha_2=candidate)
        if country is not None:
            return candidate
    try:
        country = _pycountry.countries.lookup(query)
        code = getattr(country, "alpha_2", None)
        if isinstance(code, str) and len(code) == 2:
            return code.upper()
    except LookupError:
        pass
    return None


# ── Cache sentinel ────────────────────────────────────────────────────
_SENTINEL = object()

WeightedTerm = tuple[str, float]


# ══════════════════════════════════════════════════════════════════════
#  OfflineGeocoder
# ══════════════════════════════════════════════════════════════════════

class OfflineGeocoder:
    """Thread-safe offline geocoder backed by GeoNames cities500.txt.

    The data structures are immutable after ``_load()`` completes, so
    concurrent reads from ``forward`` / ``find_near`` / ``reverse`` are
    safe without per-query locks.  The ``_ready`` event acts as the
    publication barrier.
    """

    def __init__(self, geonames_path: str = _GEONAMES_PATH) -> None:
        self._path = geonames_path
        self._state = GeocoderState.COLD
        self._error: Optional[str] = None
        self._build_ts: Optional[float] = None
        self._ready = threading.Event()
        self._load_lock = threading.Lock()

        # Core indexes (immutable after load)
        self.places_by_id: dict[int, GeoPlace] = {}
        self.alias_to_ids: dict[str, list[int]] = {}
        self._id_to_aliases: dict[int, list[str]] = {}
        self._ngram_to_ids: dict[str, set[int]] = {}
        self._country_top_place: dict[str, int] = {}
        self.kdtree: Optional[cKDTree] = None
        self._idx_to_id: list[int] = []
        self._id_to_idx: dict[int, int] = {}

        # Pre-built per-gid lowercase alias lists for fuzzy matching
        self._gid_lower_aliases: dict[int, list[str]] = {}

        # Per-place normalized alias token sets for disambiguation
        self._gid_alias_tokens: dict[int, frozenset[str]] = {}

        self._place_count = 0
        self._alias_count = 0

        # Forward-lookup cache keyed on (normalized_query, threshold).
        # Context-dependent refinement is applied *after* cache lookup,
        # so the cache is effective even when context varies.
        self._fwd_cache: dict[tuple[str, int], GeoMatch | None] = {}

    # ── Lifecycle ─────────────────────────────────────────────────────

    def warm(self, background: bool = False) -> None:
        if self._ready.is_set():
            return
        if background:
            threading.Thread(
                target=self._load_safe, name="geocoder-warm", daemon=True
            ).start()
            return
        self._load_safe()

    def _ensure_loaded(self) -> bool:
        if self._ready.is_set():
            return True
        self._load_safe()
        return self._ready.is_set()

    def is_available(self) -> bool:
        return self._ready.is_set()

    def status(self) -> dict:
        return {
            "state": self._state.value,
            "error": self._error,
            "places_count": self._place_count,
            "alias_count": self._alias_count,
            "build_ts": self._build_ts,
            "version_info": load_version_info(),
            "attribution": "GeoNames (https://www.geonames.org) - CC BY 4.0",
        }

    def _load_safe(self) -> None:
        with self._load_lock:
            if self._ready.is_set():
                return
            self._state = GeocoderState.WARMING
            try:
                self._load()
                self._state = GeocoderState.READY
                self._build_ts = time.time()
                self._ready.set()
                log.info(
                    "Geocoder ready: %d places, %d aliases.",
                    self._place_count,
                    self._alias_count,
                )
            except Exception as exc:
                self._state = GeocoderState.FAILED
                self._error = str(exc)
                log.error("GeoNames load failed – geocoding disabled: %s", exc)

    # ── Data loading ──────────────────────────────────────────────────

    def _load(self) -> None:
        path = Path(self._path)
        if not path.exists():
            raise FileNotFoundError(
                f"GeoNames data not found at {self._path}.\n"
                "Run: curl -L https://download.geonames.org/export/dump/"
                "cities500.zip -o data/cities500.zip && "
                "unzip data/cities500.zip -d data/"
            )

        _check_stale(self._path)
        log.info("Loading GeoNames from %s …", self._path)
        t0 = time.time()

        df = pd.read_csv(
            self._path,
            sep="\t",
            header=None,
            names=_COLS,
            low_memory=False,
            dtype={"geonameid": "Int64", "lat": float, "lon": float, "population": "Int64"},
        ).dropna(subset=["lat", "lon", "geonameid"])

        places_by_id: dict[int, GeoPlace] = {}
        alias_to_ids: dict[str, list[int]] = defaultdict(list)
        id_to_aliases: dict[int, list[str]] = defaultdict(list)
        ngram_to_ids: dict[str, set[int]] = defaultdict(set)
        gid_lower_aliases: dict[int, list[str]] = {}
        gid_alias_tokens: dict[int, frozenset[str]] = {}

        for row in df.itertuples(index=False, name=None):
            gid = int(row[0])   # geonameid
            name = str(row[1])  # name
            ascii_ = str(row[2])  # asciiname
            alt_raw = row[3]    # altnames
            lat = float(row[4])
            lon = float(row[5])
            country = str(row[8])
            admin1 = str(row[10])
            pop = int(row[14]) if pd.notna(row[14]) else 0

            raw_alts: list[str] = []
            if pd.notna(alt_raw):
                raw_alts = [a.strip() for a in str(alt_raw).split(",") if a.strip()]

            all_names = {name, ascii_} | set(raw_alts)
            top_aliases = tuple(sorted(all_names, key=len)[:10])

            places_by_id[gid] = GeoPlace(
                geonameid=gid,
                name=name,
                asciiname=ascii_,
                lat=lat,
                lon=lon,
                country=country,
                admin1=admin1,
                population=pop,
                top_aliases=top_aliases,
            )

            lower_aliases: list[str] = []
            alias_token_set: set[str] = set()

            for original in all_names:
                original = original.strip()
                if not original:
                    continue

                # Index under both the raw lowercase and diacritic-stripped form
                keys_to_index: set[str] = set()
                raw_lower = original.lower()
                keys_to_index.add(raw_lower)
                stripped = _normalize_for_index(original)
                if stripped and stripped != raw_lower:
                    keys_to_index.add(stripped)

                for key in keys_to_index:
                    alias_to_ids[key].append(gid)
                    for tg in _trigrams(key):
                        ngram_to_ids[tg].add(gid)

                if original not in id_to_aliases[gid]:
                    id_to_aliases[gid].append(original)

                lower_aliases.append(raw_lower)
                alias_token_set.update(
                    t for t in re.findall(r"[a-z]{2,}", raw_lower)
                )

            gid_lower_aliases[gid] = lower_aliases[:12]
            gid_alias_tokens[gid] = frozenset(alias_token_set)

        # Sort each alias bucket by descending population
        for key in alias_to_ids:
            alias_to_ids[key].sort(
                key=lambda g: places_by_id[g].population, reverse=True
            )

        # Build KDTree
        ordered_ids = list(places_by_id.keys())
        unit_vecs = np.vstack(
            [_to_unit(places_by_id[g].lat, places_by_id[g].lon) for g in ordered_ids]
        )

        # Build country → top-population place
        country_top: dict[str, int] = {}
        for gid, place in places_by_id.items():
            cc = place.country.upper()
            cur = country_top.get(cc)
            if cur is None or places_by_id[cur].population < place.population:
                country_top[cc] = gid

        # Publish (all assignments below this point; _ready event is the barrier)
        self.places_by_id = places_by_id
        self.alias_to_ids = dict(alias_to_ids)
        self._id_to_aliases = dict(id_to_aliases)
        self._ngram_to_ids = dict(ngram_to_ids)
        self._gid_lower_aliases = gid_lower_aliases
        self._gid_alias_tokens = gid_alias_tokens
        self.kdtree = cKDTree(unit_vecs)
        self._idx_to_id = ordered_ids
        self._id_to_idx = {g: i for i, g in enumerate(ordered_ids)}
        self._country_top_place = country_top
        self._place_count = len(places_by_id)
        self._alias_count = sum(len(v) for v in alias_to_ids.values())

        log.info("GeoNames loaded in %.1fs.", time.time() - t0)

        try:
            save_version_info(
                self._path,
                {"place_count": self._place_count, "alias_count": self._alias_count},
            )
        except Exception as exc:
            log.debug("Could not write geo version info: %s", exc)

    # ── Internal helpers ──────────────────────────────────────────────

    def _trigram_candidates(self, query: str) -> list[int]:
        """Return top geonameids ranked by trigram overlap with *query*."""
        score: dict[int, int] = defaultdict(int)
        for tg in _trigrams(query):
            for gid in self._ngram_to_ids.get(tg, ()):
                score[gid] += 1
        return sorted(score, key=score.__getitem__, reverse=True)[:_NGRAM_CANDIDATES]

    def _flat_aliases_for(self, gids: list[int]) -> tuple[list[str], list[int]]:
        """Build parallel (alias, gid) lists for rapidfuzz from candidate gids."""
        aliases: list[str] = []
        gid_map: list[int] = []
        for gid in gids:
            for alias in self._gid_lower_aliases.get(gid, ()):
                aliases.append(alias)
                gid_map.append(gid)
        return aliases, gid_map

    @staticmethod
    def _is_generic_place_like(query: str) -> bool:
        tokens = re.findall(r"[a-z]{2,}", query)
        if not tokens:
            return False
        if len(tokens) == 1:
            return tokens[0] in _GENERIC_PLACE_TERMS
        if len(tokens) <= 3:
            return all(t in _GENERIC_PLACE_TERMS for t in tokens)
        return False

    # ── Disambiguation ────────────────────────────────────────────────

    def _disambiguate(
        self,
        gids: list[int],
        context_words: tuple[str, ...],
    ) -> int:
        """Pick best geonameid using context tokens, geographic proximity, and population."""
        if not context_words:
            return max(gids, key=lambda g: self.places_by_id[g].population)

        ctx_tokens: set[str] = set()
        ctx_phrases: set[str] = set()
        for word in context_words:
            tokens = [t.lower() for t in re.findall(r"[A-Za-z]{2,}", word)]
            if tokens:
                ctx_tokens.update(tokens)
                ctx_phrases.add(" ".join(tokens))

        hint_terms = ctx_tokens | ctx_phrases
        ctx_country_codes = {
            _COUNTRY_HINT_TO_CODE[t] for t in hint_terms if t in _COUNTRY_HINT_TO_CODE
        }
        ctx_us_admin1_codes = {
            _US_ADMIN1_HINT_TO_CODE[t]
            for t in hint_terms
            if t in _US_ADMIN1_HINT_TO_CODE
        }

        # Resolve context words to coordinates for proximity scoring
        ctx_coords: list[tuple[float, float]] = []
        for word in context_words:
            key = word.strip().lower()
            bucket = self.alias_to_ids.get(key)
            if bucket:
                ref = self.places_by_id[bucket[0]]
                # Only use high-confidence context coords (pop > 100K)
                # to avoid obscure hamlet matches dragging results
                if ref.population >= 100_000:
                    ctx_coords.append((ref.lat, ref.lon))

        # Find max population across candidates for normalization
        max_pop = max(self.places_by_id[g].population for g in gids)

        def _score(gid: int) -> tuple[float, int]:
            place = self.places_by_id[gid]
            s = 0.0

            # ── Population prior (linear-scaled, dominant signal) ─────
            # Uses sqrt to preserve large population differences.
            # Dublin IE (1M) vs Dublin OH (49K): 1000 vs 221 = 4.5× gap
            if max_pop > 0:
                s += 5.0 * math.sqrt(place.population / max(max_pop, 1))

            # ── Country match from explicit context ───────────────────
            if place.country.upper() in ctx_country_codes:
                s += 4.0
            elif place.country.lower() in hint_terms:
                s += 3.0

            # ── US state match ────────────────────────────────────────
            if (
                ctx_us_admin1_codes
                and place.country.upper() == "US"
                and place.admin1.upper() in ctx_us_admin1_codes
            ):
                s += 4.5

            # ── Admin1 token overlap ──────────────────────────────────
            if set(place.admin1.lower().split()) & ctx_tokens:
                s += 2.0

            # ── Alias token overlap ───────────────────────────────────
            if self._gid_alias_tokens.get(gid, frozenset()) & ctx_tokens:
                s += 1.5

            # ── Geographic proximity (capped, weaker than population) ─
            if ctx_coords:
                min_dist = min(
                    haversine_km(place.lat, place.lon, clat, clon)
                    for clat, clon in ctx_coords
                )
                # Max 1.5 points, decays over 150km — not enough to
                # override a major population difference alone
                s += 1.5 * math.exp(-min_dist / 150.0)

            return s, place.population

        return max(gids, key=_score)

    def _disambiguate_batch(
        self,
        candidates: dict[str, list[int]],
    ) -> dict[str, int]:
        """
        Two-pass disambiguation using geographic coherence across
        co-mentioned places.

        candidates: {normalised_name: [geonameid, ...]}
        Returns:    {normalised_name: best_geonameid}
        """
        # ── Pass 1: rank by population alone ──────────────────────
        pop_best: dict[str, int] = {}
        for name, gids in candidates.items():
            pop_best[name] = max(
                gids, key=lambda g: self.places_by_id[g].population
            )

        # Nothing to cross-reference with a single mention
        if len(candidates) <= 1:
            return pop_best

        # ── Pass 2: re-score ambiguous names with coherence ───────
        result: dict[str, int] = {}
        for name, gids in candidates.items():
            # Unambiguous — keep as-is
            if len(gids) == 1:
                result[name] = gids[0]
                continue

            # Anchor set = pass-1 picks for every OTHER mention
            anchors: list[tuple[float, float]] = []
            for other_name, other_gid in pop_best.items():
                if other_name != name:
                    p = self.places_by_id[other_gid]
                    anchors.append((p.lat, p.lon))

            def _score(gid: int) -> float:
                p = self.places_by_id[gid]

                # Population remains the dominant signal (sqrt-scaled)
                s = math.sqrt(max(p.population, 1))

                # Coherence: multiply by a bonus based on mean distance
                # to anchors.  Close to the cluster → up to +50%.
                # Far from the cluster → no bonus, but no penalty either.
                if anchors:
                    mean_dist = sum(
                        haversine_km(p.lat, p.lon, alat, alon)
                        for alat, alon in anchors
                    ) / len(anchors)
                    s *= 1.0 + 0.5 * math.exp(-mean_dist / 500.0)

                return s

            result[name] = max(gids, key=_score)

        return result

    # ── Confidence ────────────────────────────────────────────────────

    def _compute_confidence(
        self,
        *,
        score: float,
        method: GeoMethod,
        ambiguous: bool,
        query: str,
        entity_type: Optional[str],
        candidate_count: int,
        margin_score: Optional[float],
    ) -> float:
        """Hardened composite confidence in [0, cap]."""
        cap = _METHOD_CONFIDENCE_CAPS.get(method, 0.80)
        conf = min(max(0.0, min(100.0, score)) / 100.0, cap)

        if ambiguous:
            conf *= 0.85

        if (
            method == GeoMethod.TRIGRAM_FUZZY
            and margin_score is not None
            and candidate_count >= 2
            and GEOTAG_FUZZY_MARGIN_THRESHOLD > 0
        ):
            ratio = max(0.0, min(1.0, margin_score / GEOTAG_FUZZY_MARGIN_THRESHOLD))
            conf *= 0.65 + 0.35 * ratio

        penalty = 0.0
        if method == GeoMethod.TRIGRAM_FUZZY and self._is_generic_place_like(query):
            penalty += GEOTAG_GENERIC_TOKEN_PENALTY
        if (entity_type or "").upper() in {"PERSON", "ORG"}:
            penalty += GEOTAG_ENTITY_TYPE_PENALTY

        return max(0.0, min(cap, conf - penalty))

    # ── Named-region resolution (single path) ─────────────────────────

    def _resolve_named_region(
        self,
        query: str,
        entity_type: Optional[str],
    ) -> GeoMatch | None:
        region = NAMED_REGIONS.get(query)
        if region is None:
            return None
        lat, lon, _ = region
        nearby = self.find_near(lat, lon, radius_km=50.0)
        if not nearby:
            return None
        proxy = max(nearby, key=lambda p: p.population)
        conf = self._compute_confidence(
            score=95.0,
            method=GeoMethod.REGION_TABLE,
            ambiguous=False,
            query=query,
            entity_type=entity_type,
            candidate_count=1,
            margin_score=None,
        )
        return GeoMatch(
            place=proxy,
            score=95.0,
            matched_on=query,
            method=GeoMethod.REGION_TABLE,
            ambiguous=False,
            confidence_value=conf,
            candidate_count=1,
            entity_type=entity_type,
        )

    # ── Country-level resolution ──────────────────────────────────────

    def _resolve_country(
        self,
        query: str,
        context_words: tuple[str, ...],
        entity_type: Optional[str],
    ) -> GeoMatch | None:
        code = _country_code_for_query(query)
        if code is None:
            for word in context_words:
                code = _country_code_for_query(word.strip().lower())
                if code is not None:
                    break
        if code is None:
            return None
        gid = self._country_top_place.get(code)
        if gid is None:
            return None
        conf = self._compute_confidence(
            score=93.0,
            method=GeoMethod.QUERY,
            ambiguous=False,
            query=query,
            entity_type=entity_type,
            candidate_count=1,
            margin_score=None,
        )
        return GeoMatch(
            place=self.places_by_id[gid],
            score=93.0,
            matched_on=query,
            method=GeoMethod.QUERY,
            ambiguous=False,
            confidence_value=conf,
            candidate_count=1,
            entity_type=entity_type,
        )

    # ── Core forward lookup (cached layer) ────────────────────────────

    def _forward_cached(self, query: str, threshold: int) -> GeoMatch | None:
        """Cached context-free lookup.  Exact → region → fuzzy fallback."""
        key = (query, threshold)
        cached = self._fwd_cache.get(key, _SENTINEL)
        if cached is not _SENTINEL:
            return cached
        result = self._forward_core(query, threshold)
        if len(self._fwd_cache) < _FORWARD_CACHE_SIZE:
            self._fwd_cache[key] = result
        return result

    def _forward_core(self, query: str, threshold: int) -> GeoMatch | None:
        """Expensive lookup: exact match → named region → trigram fuzzy."""

        # ── Exact match ───────────────────────────────────────────────
        if query in self.alias_to_ids:
            gids = self.alias_to_ids[query]
            primary = self.places_by_id[gids[0]]
            runner_up = self.places_by_id.get(gids[1]) if len(gids) > 1 else None
            ambiguous = (
                runner_up is not None
                and runner_up.population > primary.population * 0.05
            )
            return GeoMatch(
                place=primary,
                score=100.0,
                matched_on=query,
                method=GeoMethod.EXACT,
                ambiguous=ambiguous,
                candidates=tuple(self.places_by_id[g] for g in gids[:5]),
                candidate_count=len(gids),
            )

        # ── Named region ──────────────────────────────────────────────
        if query in NAMED_REGIONS:
            return self._resolve_named_region(query, entity_type=None)

        # ── Trigram fuzzy ─────────────────────────────────────────────
        candidate_gids = self._trigram_candidates(query)
        if not candidate_gids:
            return None
        aliases, gid_map = self._flat_aliases_for(candidate_gids)
        if not aliases:
            return None

        result = process.extractOne(
            query, aliases, scorer=fuzz.token_sort_ratio, score_cutoff=threshold
        )
        if result is None:
            return None

        matched_alias, score, local_idx = result
        best_gid = gid_map[local_idx]
        score_f = float(score)

        # Gather nearby-scoring alternatives for ambiguity detection
        alt_cutoff = max(float(threshold), score_f - _AMBIGUITY_WINDOW)
        alt_results = process.extract(
            query, aliases, scorer=fuzz.token_sort_ratio,
            score_cutoff=alt_cutoff, limit=20,
        )
        candidate_scores: dict[int, float] = {best_gid: score_f}
        for _, alt_score, idx in alt_results:
            gid = gid_map[idx]
            existing = candidate_scores.get(gid)
            if existing is None or float(alt_score) > existing:
                candidate_scores[gid] = float(alt_score)

        sorted_scores = sorted(candidate_scores.values(), reverse=True)
        margin = (
            sorted_scores[0] - sorted_scores[1] if len(sorted_scores) > 1 else None
        )
        alt_gids = [g for g in candidate_scores if g != best_gid]
        ambiguous = bool(alt_gids)

        return GeoMatch(
            place=self.places_by_id[best_gid],
            score=score_f,
            matched_on=matched_alias,
            method=GeoMethod.TRIGRAM_FUZZY,
            ambiguous=ambiguous,
            candidates=tuple(
                self.places_by_id[g]
                for g in sorted(
                    alt_gids,
                    key=lambda g: self.places_by_id[g].population,
                    reverse=True,
                )[:4]
            ),
            candidate_count=len(candidate_scores),
            margin_score=margin,
        )

    # ── Post-cache refinement (context + confidence) ──────────────────

    def _refine_match(
        self,
        raw: GeoMatch,
        query: str,
        context_words: tuple[str, ...],
        entity_type: Optional[str],
    ) -> GeoMatch | None:
        """Apply disambiguation, hardened filters, and confidence to a cached raw match."""

        place = raw.place
        ambiguous = raw.ambiguous
        score = raw.score
        method = raw.method
        margin = raw.margin_score
        n_candidates = raw.candidate_count

        # ── Disambiguation ────────────────────────────────────────────
        if ambiguous and context_words and method == GeoMethod.EXACT:
            gids = self.alias_to_ids.get(query, [])
            if len(gids) > 1:
                chosen = self._disambiguate_batch({query: gids[:10]})
                best_gid = chosen.get(query, gids[0])
                place = self.places_by_id[best_gid]

        if ambiguous and context_words and method == GeoMethod.TRIGRAM_FUZZY:
            all_gids = [raw.place.geonameid] + [c.geonameid for c in raw.candidates]
            if len(all_gids) > 1:
                chosen = self._disambiguate_batch({query: all_gids})
                best_gid = chosen.get(query, all_gids[0])
                place = self.places_by_id[best_gid]

        # ── Hardened filters (fuzzy path only) ────────────────────────
        if USE_HARDENED_GEOCODER and method == GeoMethod.TRIGRAM_FUZZY:
            floor = max(float(_DEFAULT_THRESHOLD), float(GEOTAG_FUZZY_SCORE_FLOOR))
            if score < floor:
                return None
            if (
                n_candidates >= 2
                and margin is not None
                and margin < GEOTAG_FUZZY_MARGIN_THRESHOLD
            ):
                return None
            if self._is_generic_place_like(query) and score < 90.0:
                return None

        # ── Confidence ────────────────────────────────────────────────
        conf = self._compute_confidence(
            score=score,
            method=method,
            ambiguous=ambiguous,
            query=query,
            entity_type=entity_type,
            candidate_count=n_candidates,
            margin_score=margin,
        )

        return GeoMatch(
            place=place,
            score=score,
            matched_on=raw.matched_on,
            method=method,
            ambiguous=ambiguous,
            candidates=raw.candidates,
            confidence_value=conf,
            candidate_count=n_candidates,
            margin_score=margin,
            entity_type=entity_type,
        )

    # ── Public forward API ────────────────────────────────────────────

    def forward(
        self,
        place_name: str,
        threshold: int = _DEFAULT_THRESHOLD,
        context_words: tuple[str, ...] = (),
        entity_type: Optional[str] = None,
    ) -> GeoMatch | None:
        """Forward-geocode a place name.  Exact → region → country → fuzzy."""
        if not self._ensure_loaded():
            return None

        query = _normalize_query(place_name)
        if not query:
            return None

        # Cached core lookup (context-independent)
        raw = self._forward_cached(query, threshold)

        # If the core lookup missed, try country-level resolution
        if raw is None:
            return self._resolve_country(query, context_words, entity_type)

        # Apply context-dependent disambiguation + confidence
        return self._refine_match(raw, query, context_words, entity_type)

    def forward_batch(
        self,
        place_names: list[str],
        query: str = "",
        threshold: int = _DEFAULT_THRESHOLD,
        entity_types: list[str | None] | None = None,
    ) -> list[GeoMatch | None]:
        """Batch forward-geocode with shared context and deduplication."""
        if not self._ensure_loaded():
            return [None] * len(place_names)

        if entity_types is None:
            entity_types = [None] * len(place_names)

        all_ctx = tuple(re.findall(r"[A-Z][a-zA-Z]{2,}", query))
        results: list[GeoMatch | None] = []
        for name, etype in zip(place_names, entity_types):
            ctx = tuple(w for w in all_ctx if w.lower() not in name.lower())
            results.append(self.forward(name, threshold, ctx, etype))
        return results

    def resolve_all(
        self,
        place_names: list[str],
        query: str = "",
        threshold: int = _DEFAULT_THRESHOLD,
        entity_types: list[str | None] | None = None,
    ) -> list[GeoMatch]:
        """Resolve all candidate place names, returning only successful matches."""
        batch = self.forward_batch(place_names, query, threshold, entity_types)
        return [m for m in batch if m is not None]

    # ── Spatial queries ───────────────────────────────────────────────

    def find_near(
        self,
        lat: float,
        lon: float,
        radius_km: float = 50.0,
        max_results: int = 200,
    ) -> list[GeoPlace]:
        """Return places within *radius_km* of (lat, lon), distance-sorted."""
        if not self._ensure_loaded() or self.kdtree is None:
            return []
        chord = _radius_to_chord(radius_km)
        qvec = _to_unit(lat, lon)
        raw_idxs = self.kdtree.query_ball_point(qvec, chord)

        results: list[tuple[float, GeoPlace]] = []
        for idx in raw_idxs:
            place = self.places_by_id[self._idx_to_id[idx]]
            dist = haversine_km(lat, lon, place.lat, place.lon)
            if dist <= radius_km:
                results.append((dist, place))

        results.sort(key=lambda pair: pair[0])
        return [p for _, p in results[:max_results]]

    def reverse(
        self, lat: float, lon: float, k: int = 1
    ) -> list[tuple[float, GeoPlace]]:
        """Nearest-neighbour reverse geocode → [(distance_km, GeoPlace)]."""
        if not self._ensure_loaded() or self.kdtree is None:
            return []
        qvec = _to_unit(lat, lon)
        _, idxs = self.kdtree.query(qvec, k=k)
        if k == 1:
            idxs = [idxs]
        return [
            (
                haversine_km(lat, lon, self.places_by_id[self._idx_to_id[i]].lat,
                             self.places_by_id[self._idx_to_id[i]].lon),
                self.places_by_id[self._idx_to_id[i]],
            )
            for i in idxs
        ]

    def get_aliases(self, geonameid: int) -> list[str]:
        """O(1) alias lookup by geonameid."""
        return self._id_to_aliases.get(geonameid, [])

    def spatial_center(
        self, place_name: str
    ) -> tuple[float, float, float] | None:
        """Return (lat, lon, radius_km) for a place name or named region."""
        query = place_name.lower().strip()
        if query in NAMED_REGIONS:
            return NAMED_REGIONS[query]
        match = self.forward(query)
        return (match.place.lat, match.place.lon, 50.0) if match else None


# ══════════════════════════════════════════════════════════════════════
#  Query extraction
# ══════════════════════════════════════════════════════════════════════

def extract_places_from_query(query: str) -> list[str]:
    """Extract all candidate place names from a user query string."""
    found: list[str] = []
    seen_lower: set[str] = set()

    def _add(value: str) -> None:
        cleaned = value.strip()
        key = cleaned.lower()
        if key and key not in seen_lower:
            found.append(cleaned)
            seen_lower.add(key)

    for m in _QUOTED_PLACE.finditer(query):
        _add(m.group(1))

    for m in _SPATIAL_CUES.finditer(query):
        after = query[m.end():]
        chain = _CAP_NP_CHAIN.match(after)
        if chain:
            for part in re.split(r"\s+and\s+", chain.group(1), maxsplit=4):
                _add(part)
            continue
        np_match = _CAP_NP_STRICT.match(after)
        if np_match:
            _add(np_match.group(1))

    for m in _BETWEEN_CHAIN.finditer(query):
        _add(m.group(1))
        _add(m.group(2))

    lower_q = query.lower()
    for region in NAMED_REGIONS:
        if re.search(r"\b" + re.escape(region) + r"\b", lower_q):
            _add(region.title())

    return found


# ══════════════════════════════════════════════════════════════════════
#  Query expansion
# ══════════════════════════════════════════════════════════════════════

def build_geo_query_expansion(
    match: GeoMatch,
    geocoder: OfflineGeocoder,
    max_terms: int = _MAX_EXPANSION_TERMS,
) -> list[WeightedTerm]:
    """Build weighted expansion terms for BM25/FTS query augmentation."""
    canonical = match.place.name
    all_aliases = geocoder.get_aliases(match.place.geonameid)
    result: list[WeightedTerm] = [(canonical, 1.0)]
    seen: set[str] = {canonical.lower()}

    scored: list[tuple[str, float]] = []
    for raw in all_aliases:
        alias = raw.strip()
        if not alias or alias.lower() in seen:
            continue
        sim = fuzz.ratio(canonical.lower(), alias.lower()) / 100.0
        scored.append((alias, round(0.4 + 0.3 * sim, 2)))

    scored.sort(key=lambda item: item[1], reverse=True)
    for alias, weight in scored[: max_terms - 1]:
        result.append((alias, weight))
        seen.add(alias.lower())
    return result


# ══════════════════════════════════════════════════════════════════════
#  Singleton
# ══════════════════════════════════════════════════════════════════════

_instance: Optional[OfflineGeocoder] = None
_instance_lock = threading.Lock()


def get_geocoder(path: str = _GEONAMES_PATH) -> OfflineGeocoder:
    """Return the singleton geocoder instance."""
    global _instance
    if _instance is None:
        with _instance_lock:
            if _instance is None:
                _instance = OfflineGeocoder(path)
    return _instance