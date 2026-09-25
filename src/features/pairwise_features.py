"""
pairwise_features.py

Computes similarity features between a Source 1 record and a candidate
Source 2/3 record. Used both for training the matching model and for
scoring candidates at inference time.
"""

from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein

from src.preprocessing.normalize import (
    normalize_name, normalize_address, normalize_country,
    strip_legal_suffix, extract_legal_suffix,
)


def _token_jaccard(a: str, b: str) -> float:
    set_a, set_b = set(a.split()), set(b.split())
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def _levenshtein_ratio(a: str, b: str) -> float:
    if not a and not b:
        return 1.0
    return Levenshtein.normalized_similarity(a, b)


def compute_pair_features(record1: dict, record2: dict) -> dict:
    """
    record1: Source 1 record (dict with business_name, business_address, country)
    record2: candidate Source 2/3 record (same shape)
    Returns a flat dict of numeric features.
    """
    name1_raw, name2_raw = record1.get("business_name", ""), record2.get("business_name", "")
    addr1_raw, addr2_raw = record1.get("business_address", ""), record2.get("business_address", "")
    country1, country2 = normalize_country(record1.get("country", "")), normalize_country(record2.get("country", ""))

    name1, name2 = normalize_name(name1_raw), normalize_name(name2_raw)
    core1, core2 = strip_legal_suffix(name1_raw), strip_legal_suffix(name2_raw)
    suffix1, suffix2 = extract_legal_suffix(name1_raw), extract_legal_suffix(name2_raw)

    addr1, addr2 = normalize_address(addr1_raw), normalize_address(addr2_raw)

    features = {
        # --- name features ---
        "name_exact_match": float(name1 == name2 and name1 != ""),
        "name_core_exact_match": float(core1 == core2 and core1 != ""),
        "name_jaccard_token": _token_jaccard(name1, name2),
        "name_levenshtein_ratio": _levenshtein_ratio(name1, name2),
        "name_jaro_winkler": fuzz.WRatio(name1, name2) / 100.0,
        "name_token_sort_ratio": fuzz.token_sort_ratio(name1, name2) / 100.0,
        "name_partial_ratio": fuzz.partial_ratio(name1, name2) / 100.0,
        "suffix_match_flag": float(suffix1 == suffix2 and suffix1 != ""),

        # --- address features ---
        "address_jaccard_token": _token_jaccard(addr1, addr2),
        "address_levenshtein_ratio": _levenshtein_ratio(addr1, addr2),
        "address_token_sort_ratio": fuzz.token_sort_ratio(addr1, addr2) / 100.0,
        "address_partial_ratio": fuzz.partial_ratio(addr1, addr2) / 100.0,

        # --- meta features ---
        "country_match_flag": float(country1 == country2 and country1 != ""),
        "country_known_flag": float(bool(country1) and bool(country2)),
        "name_length_diff": abs(len(name1) - len(name2)),
        "address_length_diff": abs(len(addr1) - len(addr2)),
        "name1_empty": float(name1 == ""),
        "name2_empty": float(name2 == ""),
        "address1_empty": float(addr1 == ""),
        "address2_empty": float(addr2 == ""),
    }
    return features


FEATURE_NAMES = list(compute_pair_features(
    {"business_name": "a", "business_address": "b", "country": "c"},
    {"business_name": "a", "business_address": "b", "country": "c"},
).keys())
