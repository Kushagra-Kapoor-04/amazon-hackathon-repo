"""
parse_address.py

Stage 2 — Practical address parser / normalizer.

Goals
-----
* Extract structured components from a free-form address string so that
  pairwise feature engineering can compare them at the field level
  (e.g. postal-code exact match, city token overlap).
* No external geocoding, no internet lookups, no external databases.
* Country-agnostic heuristics that degrade gracefully on unseen formats.
* All functions accept None / NaN safely.

Parsed fields (all strings, empty string if not found)
-------------------------------------------------------
  house_number  : leading numeric token(s) before the street name
  street        : main street name portion
  unit          : apartment / suite / floor / unit designator
  city          : city / locality
  state         : state / province / region
  postal_code   : ZIP / PIN / postal code (digits + optional letters)
  country_hint  : country fragment if present at the end of the string

Usage
-----
    from src.preprocessing.parse_address import parse_address, normalize_address_fields

    parsed = parse_address("42 Baker St, Suite 3A, London W1A 1AA, UK")
    # -> AddressRecord(house_number='42', street='baker street',
    #                  unit='suite 3a', city='london', state='',
    #                  postal_code='w1a1aa', country_hint='uk')
"""

import re
from dataclasses import dataclass, field, asdict
from typing import Optional

from src.preprocessing.normalize import (
    basic_clean,
    normalize_address,
    normalize_country,
    extract_numeric_tokens,
    _safe_str,
    _RE_WHITESPACE,
)

# ---------------------------------------------------------------------------
# Compiled patterns
# ---------------------------------------------------------------------------

# Postal code patterns (order: most-specific first)
_POSTAL_PATTERNS = [
    # Indian PIN: exactly 6 digits
    re.compile(r"\b(\d{6})\b"),
    # US ZIP or ZIP+4
    re.compile(r"\b(\d{5}(?:-\d{4})?)\b"),
    # UK postcode: e.g. W1A 1AA, SW1W 0NY (allow no space)
    re.compile(r"\b([A-Z]{1,2}\d[A-Z\d]?\s*\d[A-Z]{2})\b", re.IGNORECASE),
    # Canadian: A1A 1A1
    re.compile(r"\b([A-Z]\d[A-Z]\s*\d[A-Z]\d)\b", re.IGNORECASE),
    # German / EU 5-digit
    re.compile(r"\b(\d{5})\b"),
    # 4-digit (NL, AU, etc.)
    re.compile(r"\b(\d{4})\b"),
]

# Unit / floor / suite designators
_UNIT_PATTERN = re.compile(
    r"\b(apt|apartment|suite|ste|floor|fl|flr|unit|room|rm|#)\s*\.?\s*([A-Z0-9\-]+)",
    re.IGNORECASE,
)

# Building / block prefixes (Indian / SE-Asian addresses)
_BLOCK_PATTERN = re.compile(
    r"\b(plot|block|blk|sector|phase|bldg|building|flat|door|no\.?|plot\s*no\.?)\s*[:\-]?\s*([A-Z0-9\-/]+)",
    re.IGNORECASE,
)

# Leading house number at start of address
_HOUSE_NUMBER_PATTERN = re.compile(r"^\s*(\d+[\w\-/]*)\b")

# Common street-type words (used to locate the street segment)
_STREET_TYPES = (
    "street", "road", "avenue", "boulevard", "lane", "drive", "court",
    "place", "square", "parkway", "highway", "expressway", "freeway",
    "way", "walk", "trail", "circle", "close", "marg", "nagar", "vihar",
    "chowk", "gali", "colony", "layout", "extension", "enclave",
)

# Separators used in multi-component addresses
_SEP_PATTERN = re.compile(r"[,;|]+")

# ---------------------------------------------------------------------------
# Data class for parsed address
# ---------------------------------------------------------------------------

@dataclass
class AddressRecord:
    """Structured representation of a parsed address."""
    house_number: str = ""
    street: str = ""
    unit: str = ""
    city: str = ""
    state: str = ""
    postal_code: str = ""
    country_hint: str = ""
    # Full normalized string (for fallback similarity comparison)
    normalized_full: str = ""

    def as_dict(self) -> dict:
        return asdict(self)

    def is_empty(self) -> bool:
        return not any([
            self.house_number, self.street, self.unit,
            self.city, self.state, self.postal_code,
        ])


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _strip_postal(text: str) -> tuple[str, str]:
    """
    Scan text for the first matching postal code pattern.
    Returns (text_with_postal_removed, postal_code_string).
    postal_code is normalized to uppercase, no spaces.
    """
    for pat in _POSTAL_PATTERNS:
        m = pat.search(text)
        if m:
            raw = m.group(1)
            postal = re.sub(r"\s+", "", raw).upper()
            cleaned = text[:m.start()] + text[m.end():]
            return cleaned.strip(), postal
    return text, ""


def _strip_unit(text: str) -> tuple[str, str]:
    """Remove unit/suite/floor designation, return (remainder, unit_str)."""
    m = _UNIT_PATTERN.search(text)
    if m:
        designator = m.group(1).lower()
        number = m.group(2).upper()
        unit_str = f"{designator} {number}"
        cleaned = text[:m.start()] + text[m.end():]
        return cleaned.strip(), unit_str
    return text, ""


def _strip_block(text: str) -> tuple[str, str]:
    """Remove block/plot/sector designation, return (remainder, block_str)."""
    m = _BLOCK_PATTERN.search(text)
    if m:
        block_str = m.group(0).strip().lower()
        cleaned = text[:m.start()] + text[m.end():]
        return cleaned.strip(), block_str
    return text, ""


def _split_segments(text: str) -> list:
    """Split address into comma/semicolon-delimited segments, cleaned."""
    parts = _SEP_PATTERN.split(text)
    return [p.strip() for p in parts if p.strip()]


def _looks_like_country(segment: str) -> bool:
    """
    Heuristic: a final short segment (1–3 words, no digits) that is likely
    a country name / code.
    """
    words = segment.split()
    if len(words) > 3 or len(words) == 0:
        return False
    if any(ch.isdigit() for ch in segment):
        return False
    # If it's a known country alias or a 2-letter code, very likely
    normed = normalize_country(segment)
    if normed and len(normed) >= 2:
        return True
    return False


def _looks_like_postal(segment: str) -> bool:
    return bool(re.fullmatch(r"[A-Z0-9\s\-]{4,10}", segment.strip(), re.IGNORECASE))


def _find_street_in_segment(segment: str) -> str:
    """
    Try to identify the street portion of a single segment.
    Returns the full segment normalized as the street if a street-type keyword
    is found; otherwise returns the segment as-is.
    """
    seg_lower = normalize_address(segment)
    for st in _STREET_TYPES:
        if st in seg_lower.split():
            return seg_lower
    return seg_lower


# ---------------------------------------------------------------------------
# Main public function
# ---------------------------------------------------------------------------

def parse_address(address) -> AddressRecord:
    """
    Parse a free-form address string into structured components.

    Strategy (heuristic pipeline):
    1. Normalize raw text (unicode, lower, expand abbreviations).
    2. Extract & remove postal code.
    3. Extract & remove unit/floor designator.
    4. Split on commas/semicolons into segments.
    5. Heuristically assign:
       - Last segment → country if it looks like one.
       - Second-to-last (or last remaining) → state/city split.
       - First segment → house number + street.
    6. Anything not classified goes into the street field.

    All fields are lowercase strings. Postal code is uppercase (standard).
    Returns AddressRecord with normalized_full always populated.
    """
    raw = _safe_str(address)
    record = AddressRecord()

    if not raw:
        return record

    # Full normalized address (always available as fallback)
    record.normalized_full = normalize_address(raw)

    # Work on a mutable copy
    text = raw

    # Step 1 — extract postal code
    text, record.postal_code = _strip_postal(text)

    # Step 2 — extract unit/suite
    text, record.unit = _strip_unit(text)
    if not record.unit:
        text, block_str = _strip_block(text)
        if block_str:
            record.unit = block_str  # treat block as unit-like designator

    # Step 3 — normalize remaining text for segmentation
    # (keep punctuation for splitting, don't call normalize_address yet)
    text = re.sub(r"\s+", " ", text).strip()

    # Step 4 — split into segments
    segments = _split_segments(text)

    if not segments:
        return record

    # Step 5 — assign country from last segment
    if len(segments) >= 2 and _looks_like_country(segments[-1]):
        record.country_hint = normalize_country(segments[-1])
        segments = segments[:-1]

    # Step 6 — assign state/city from tail segments
    # Pattern: [..., city, state] or [..., city_state_combined]
    if len(segments) >= 2:
        # Last remaining → state candidate
        last = normalize_address(segments[-1])
        second_last = normalize_address(segments[-2])

        # If last is very short (likely state abbreviation or city), treat as state
        last_words = last.split()
        if len(last_words) <= 2 and not any(ch.isdigit() for ch in last):
            record.state = last
            segments = segments[:-1]
            # Now last segment → city
            if segments:
                record.city = normalize_address(segments[-1])
                segments = segments[:-1]
        else:
            # Try to split "City State" from a single combined segment
            record.city = last
            segments = segments[:-1]
    elif len(segments) == 1:
        # Only one segment left — could be just a street or city+street
        pass

    # Step 7 — street + house number from first remaining segment
    if segments:
        first_seg = segments[0]
        norm_first = normalize_address(first_seg)

        # Extract leading house number
        m = _HOUSE_NUMBER_PATTERN.match(norm_first)
        if m:
            record.house_number = m.group(1)
            record.street = norm_first[m.end():].strip()
        else:
            record.street = norm_first

        # Remaining middle segments → append to street (building name etc.)
        for mid in segments[1:]:
            extra = normalize_address(mid)
            if extra:
                record.street = (record.street + " " + extra).strip()

    return record


# ---------------------------------------------------------------------------
# Batch normalizer: returns a flat dict of normalized fields for a row
# ---------------------------------------------------------------------------

def normalize_address_fields(address_str, country_str=None) -> dict:
    """
    High-level function used by the pipeline.
    Returns a dict with keys:
      addr_full, addr_house_number, addr_street, addr_unit,
      addr_city, addr_state, addr_postal, addr_country_hint
    """
    parsed = parse_address(address_str)
    # Prefer the explicit country field from the dataset over the hint
    country_norm = normalize_country(country_str) if country_str else parsed.country_hint
    return {
        "addr_full":         parsed.normalized_full,
        "addr_house_number": parsed.house_number,
        "addr_street":       parsed.street,
        "addr_unit":         parsed.unit,
        "addr_city":         parsed.city,
        "addr_state":        parsed.state,
        "addr_postal":       parsed.postal_code,
        "addr_country_hint": country_norm,
    }


# ---------------------------------------------------------------------------
# Self-contained demo / internal tests
# (run: python -m src.preprocessing.parse_address)
# ---------------------------------------------------------------------------

def _run_tests():
    PASS = "\033[32mPASS\033[0m"
    FAIL = "\033[31mFAIL\033[0m"
    results = []

    def check(label, got, expected):
        ok = (got == expected)
        results.append(ok)
        status = PASS if ok else FAIL
        print(f"  [{status}] {label}")
        if not ok:
            print(f"         got      : {got!r}")
            print(f"         expected : {expected!r}")

    def check_field(label, address_str, field_name, expected):
        parsed = parse_address(address_str)
        got = getattr(parsed, field_name)
        check(label, got, expected)

    print("\n=== parse_address.py — internal tests ===\n")

    # --- postal code extraction ---
    check_field("US ZIP",        "123 Main St, Springfield, IL 62701", "postal_code", "62701")
    check_field("Indian PIN",    "42 MG Road, Bengaluru 560001, India", "postal_code", "560001")
    check_field("UK postcode",   "10 Downing St, London SW1A 2AA, UK", "postal_code", "SW1A2AA")
    check_field("No postal",     "Some Street, Some City",             "postal_code", "")

    # --- country hint ---
    check_field("Country: India", "42 MG Road, Bengaluru 560001, India", "country_hint", "india")
    check_field("Country: UK",    "10 Downing St, London SW1A 2AA, UK", "country_hint", "united kingdom")
    check_field("Country: USA",   "123 Main St, NY 10001, USA",         "country_hint", "united states")

    # --- house number ---
    check_field("House num: US", "42 Baker Street, London", "house_number", "42")
    check_field("House num: none", "Baker Street, London",  "house_number", "")

    # --- unit extraction ---
    check_field("Unit: Apt",     "123 Main St, Apt 4B, Chicago",    "unit", "apt 4B")
    check_field("Unit: Suite",   "500 Park Ave, Suite 200, NY",     "unit", "suite 200")
    check_field("Unit: Floor",   "1 Corp Plaza, Floor 3, Mumbai",   "unit", "floor 3")

    # --- normalized_full always populated ---
    r = parse_address("123 Main St, Apt 4B, Springfield, IL 62701, USA")
    ok = bool(r.normalized_full)
    results.append(ok)
    status = PASS if ok else FAIL
    print(f"  [{status}] normalized_full populated")

    # --- None / empty safe ---
    r_none = parse_address(None)
    check("None -> empty record", r_none.is_empty(), True)
    r_empty = parse_address("")
    check("Empty string -> empty record", r_empty.is_empty(), True)

    # --- normalize_address_fields dict keys ---
    d = normalize_address_fields("42 Baker St, London W1A 1AA, UK", "United Kingdom")
    expected_keys = {
        "addr_full", "addr_house_number", "addr_street", "addr_unit",
        "addr_city", "addr_state", "addr_postal", "addr_country_hint"
    }
    check("normalize_address_fields keys", set(d.keys()), expected_keys)
    check("normalize_address_fields country", d["addr_country_hint"], "united kingdom")

    # Summary
    n_pass = sum(results)
    n_fail = len(results) - n_pass
    print(f"\n  {n_pass}/{len(results)} passed, {n_fail} failed\n")
    return n_fail == 0


if __name__ == "__main__":
    import sys
    ok = _run_tests()
    sys.exit(0 if ok else 1)
