"""Shared geo enums used by geocoder, metadata, and API layers."""
from enum import Enum


class GeoMethod(str, Enum):
    EXACT = "exact"
    TRIGRAM_FUZZY = "trigram_fuzzy"
    REGION_TABLE = "region_table"
    REGEX = "regex"
    MANUAL = "manual"
    QUERY = "query"


class GeocoderState(str, Enum):
    COLD = "cold"
    WARMING = "warming"
    READY = "ready"
    FAILED = "failed"
