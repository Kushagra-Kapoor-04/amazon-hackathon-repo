"""
normalize.py

Stage 2 — Preprocessing / Normalization Foundation.

Text normalization for business names and addresses.
No external lookups — pure stdlib + dictionary-based cleaning only,
per competition fair-play rules.

Key design decisions
--------------------
* Unicode / accent normalization via unicodedata.normalize (NFD → ASCII)
  so that "Müller" and "Muller" land on the same token.
* Legal suffix table is larger than the stub; covers common forms across
  US, UK, India, EU, and several other regions seen in typical ML challenge data.
* Suffix expansion is done ONLY for normalization equality; the raw suffix
  is preserved separately so feature engineering can use match/mismatch.
* Address abbreviations cover both US postal standards and Indian/EU street
  terminology frequently seen in multi-country business data.
* Country normalization is open-set: no hard-coding beyond a small alias map.
* All functions accept None / float(nan) safely.
"""

import re
import unicodedata

# ---------------------------------------------------------------------------
# Compiled regex constants (build once, reuse)
# ---------------------------------------------------------------------------

_RE_WHITESPACE = re.compile(r"\s+")
_RE_PUNCT_EXCEPT_AMPERSAND = re.compile(r"[^\w\s&]")
_RE_DIGITS_ONLY = re.compile(r"^\d+$")

# ---------------------------------------------------------------------------
# Lookup tables
# ---------------------------------------------------------------------------

# Legal / entity-type suffixes (post-normalization tokens).
# Order matters: longer / more-specific entries should be listed before
# shorter ambiguous ones.  The set is used for O(1) lookup.
LEGAL_SUFFIXES_SET = {
    # Full English words
    "incorporated", "corporation", "company", "limited", "private",
    "partnership", "association", "foundation", "enterprise", "enterprises",
    "group", "holding", "holdings", "trust", "syndicate", "cooperative",
    # Common abbreviations (already lower-cased)
    "inc", "corp", "co", "ltd", "pvt", "llc", "llp", "lp", "plc",
    "gmbh", "ag", "sa", "sas", "srl", "bv", "nv", "ab", "oy", "as",
    # Indian variants
    "pvt", "pte",
    # Joint-stock / others
    "jsc", "ojsc", "zao", "ooo",
}

# Canonical expansion for abbreviations found IN THE NAME (not just suffix).
# Applied before suffix detection so that "ABC Corp." → "ABC corporation"
# and "Co." → "company".  & is expanded last (after punctuation removal).
_NAME_ABBREV_PATTERNS = [
    # longer patterns first to avoid partial matches
    (r"\bllc\b",   "limited liability company"),
    (r"\bllp\b",   "limited liability partnership"),
    (r"\bplc\b",   "public limited company"),
    (r"\bcorp\b",  "corporation"),
    (r"\binc\b",   "incorporated"),
    (r"\bltd\b",   "limited"),
    (r"\bpvt\b",   "private"),
    (r"\bpte\b",   "private"),
    (r"\bco\b",    "company"),
    (r"\bgmbh\b",  "gesellschaft mit beschrankter haftung"),
    # & is NOT a word-char so \b won't fire around it; use a plain literal
    (r"&",         " and "),
]
_NAME_ABBREV_COMPILED = [
    (re.compile(p, re.IGNORECASE), r) for p, r in _NAME_ABBREV_PATTERNS
]

# Address abbreviations: both US postal abbreviations and international ones.
_ADDR_ABBREV_PATTERNS = [
    # Street types — & fix same as NAME
    (r"&",           " and "),
    (r"\bblvd\b",    "boulevard"),
    (r"\bave\b",     "avenue"),
    (r"\bav\b",      "avenue"),
    (r"\bst\b",      "street"),
    (r"\brd\b",      "road"),
    (r"\bdr\b",      "drive"),
    (r"\bln\b",      "lane"),
    (r"\bct\b",      "court"),
    (r"\bpl\b",      "place"),
    (r"\bsq\b",      "square"),
    (r"\bpkwy\b",    "parkway"),
    (r"\bfwy\b",     "freeway"),
    (r"\bhwy\b",     "highway"),
    (r"\bexpy\b",    "expressway"),
    # Unit types
    (r"\bapt\b",     "apartment"),
    (r"\bste\b",     "suite"),
    (r"\bfl\b",      "floor"),
    (r"\bflr\b",     "floor"),
    (r"\bunit\b",    "unit"),
    (r"\brm\b",      "room"),
    (r"\bbldg\b",    "building"),
    (r"\bblk\b",     "block"),
    # Directionals
    (r"\bn\b",       "north"),
    (r"\bs\b",       "south"),
    (r"\be\b",       "east"),
    (r"\bw\b",       "west"),
    (r"\bne\b",      "northeast"),
    (r"\bnw\b",      "northwest"),
    (r"\bse\b",      "southeast"),
    (r"\bsw\b",      "southwest"),
    # Indian / South-Asian
    (r"\bnr\b",      "near"),
    (r"\bopp\b",     "opposite"),
    (r"\bnagar\b",   "nagar"),
    (r"\bmarg\b",    "marg"),
    # Generic
    (r"\b&\b",       "and"),
]
_ADDR_ABBREV_COMPILED = [
    (re.compile(p, re.IGNORECASE), r) for p, r in _ADDR_ABBREV_PATTERNS
]

# Multi-word legal suffixes to strip from the END of a normalized name.
# Ordered longest-first so the greedy match fires first.
_MULTI_WORD_SUFFIX_RE = re.compile(
    r"\s+(limited liability company"
    r"|limited liability partnership"
    r"|public limited company"
    r"|gesellschaft mit beschrankter haftung"
    r")$",
    re.IGNORECASE,
)

# Country alias map: maps common alternative spellings / abbreviations
# to a canonical name.  Open-set: unknown values pass through unchanged.
_COUNTRY_ALIASES = {
    "us": "united states",
    "usa": "united states",
    "u.s.": "united states",
    "u.s.a.": "united states",
    "united states of america": "united states",
    "uk": "united kingdom",
    "u.k.": "united kingdom",
    "great britain": "united kingdom",
    "britain": "united kingdom",
    "in": "india",
    "ind": "india",
    "de": "germany",
    "deutschland": "germany",
    "fr": "france",
    "cn": "china",
    "prc": "china",
    "jp": "japan",
    "ca": "canada",
    "au": "australia",
    "br": "brazil",
    "mx": "mexico",
    "sg": "singapore",
    "ae": "united arab emirates",
    "uae": "united arab emirates",
}

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _safe_str(value) -> str:
    """Convert any input (None, float NaN, int, str) to a plain string."""
    if value is None:
        return ""
    s = str(value).strip()
    # float NaN lands as "nan" after str()
    if s.lower() in {"nan", "none", "null", "na", "n/a", ""}:
        return ""
    return s


def _unicode_to_ascii(text: str) -> str:
    """
    NFD-decompose then drop combining characters so accented letters
    become their ASCII base (e.g. é→e, ü→u, ñ→n).
    Characters with no ASCII equivalent are kept as-is (CJK stays CJK).
    """
    nfd = unicodedata.normalize("NFD", text)
    return "".join(ch for ch in nfd if unicodedata.category(ch) != "Mn")


def _apply_compiled(text: str, patterns) -> str:
    for regex, replacement in patterns:
        text = regex.sub(replacement, text)
    return text


# ---------------------------------------------------------------------------
# Public API — name normalization
# ---------------------------------------------------------------------------

def basic_clean(text) -> str:
    """
    Foundation cleaner used by all normalizers:
    - None / NaN safe
    - Unicode accent stripping (NFD)
    - Lowercase
    - Remove punctuation except '&' (expanded separately)
    - Collapse whitespace
    """
    text = _safe_str(text)
    if not text:
        return ""
    text = _unicode_to_ascii(text)
    text = text.lower()
    text = _RE_PUNCT_EXCEPT_AMPERSAND.sub(" ", text)
    text = _RE_WHITESPACE.sub(" ", text).strip()
    return text


def normalize_name(name) -> str:
    """
    Normalize a business name for comparison:
      1. basic_clean (unicode, lowercase, punct removal)
      2. expand known legal abbreviations
      3. final whitespace collapse

    Legal suffix is NOT stripped here — use strip_legal_suffix() separately
    since suffix match/mismatch is itself a pairwise feature.
    """
    text = basic_clean(name)
    if not text:
        return ""
    text = _apply_compiled(text, _NAME_ABBREV_COMPILED)
    text = _RE_WHITESPACE.sub(" ", text).strip()
    return text


def extract_legal_suffix(name) -> str:
    """
    Return the last token of the normalized name if it is a recognised
    legal/entity-type suffix, else return ''.

    Examples
    --------
    >>> extract_legal_suffix("Acme Corp.")  -> "corporation"
    >>> extract_legal_suffix("Acme Inc")    -> "incorporated"
    >>> extract_legal_suffix("Foo & Bar")   -> ""
    """
    norm = normalize_name(name)
    tokens = norm.split()
    if not tokens:
        return ""
    # After expansion, "corp" → "corporation", so check expanded token
    last = tokens[-1]
    if last in LEGAL_SUFFIXES_SET:
        return last
    return ""


def strip_legal_suffix(name) -> str:
    """
    Return the normalized name with trailing legal suffix removed.
    Used for core-name-only comparison in feature engineering.

    Handles both single-token suffixes ("ltd" -> "limited")
    and multi-token expansions ("llc" -> "limited liability company").

    Examples
    --------
    >>> strip_legal_suffix("Acme Corporation")  -> "acme"
    >>> strip_legal_suffix("Foo & Bar Ltd")     -> "foo and bar"
    >>> strip_legal_suffix("Smith & Jones LLC") -> "smith and jones"
    """
    norm = normalize_name(name)
    if not norm:
        return ""
    # Strip multi-word suffixes first (longest-match via alternation order)
    stripped = _MULTI_WORD_SUFFIX_RE.sub("", norm).strip()
    norm = stripped
    # Then strip any remaining single-token suffix
    tokens = norm.split()
    if tokens and tokens[-1] in LEGAL_SUFFIXES_SET:
        tokens = tokens[:-1]
    return " ".join(tokens)


def tokenize_name(name) -> list:
    """Return sorted list of non-trivial tokens from a normalized name (no suffix)."""
    core = strip_legal_suffix(name)
    tokens = core.split()
    # drop pure-numeric tokens of length 1 (noise)
    tokens = [t for t in tokens if not (len(t) == 1 and t.isdigit())]
    return tokens


# ---------------------------------------------------------------------------
# Public API — address normalization
# ---------------------------------------------------------------------------

def normalize_address(address) -> str:
    """
    Normalize an address string for fuzzy comparison:
      1. basic_clean
      2. expand address abbreviations
      3. final whitespace collapse

    Does NOT parse into components — see parse_address.py.
    Country-agnostic: abbreviation table covers US, IN, EU common terms.
    """
    text = basic_clean(address)
    if not text:
        return ""
    text = _apply_compiled(text, _ADDR_ABBREV_COMPILED)
    text = _RE_WHITESPACE.sub(" ", text).strip()
    return text


def extract_numeric_tokens(text) -> list:
    """
    Extract all digit sequences from raw text (PIN, ZIP, building numbers).
    Used by feature engineering for exact numeric component matching.

    Example: "123 Main St, 110001" -> ["123", "110001"]
    """
    return re.findall(r"\b\d+\b", _safe_str(text))


# ---------------------------------------------------------------------------
# Public API — country normalization
# ---------------------------------------------------------------------------

def normalize_country(country) -> str:
    """
    Light normalization: lowercase, strip, apply alias map.
    Open-set — unknown country labels (e.g. 'Senegal') pass through unchanged
    rather than being dropped or replaced.
    """
    text = _safe_str(country).lower().strip()
    if not text:
        return ""
    # Remove trailing periods (e.g. "U.S.A." → after basic strip)
    text = text.replace(".", "")
    text = _RE_WHITESPACE.sub(" ", text).strip()
    return _COUNTRY_ALIASES.get(text, text)


# ---------------------------------------------------------------------------
# Public API — phone / misc
# ---------------------------------------------------------------------------

def normalize_phone(phone) -> str:
    """
    Strip all non-digit characters and drop leading country-code 0/+.
    Returns a digit-only string for exact / suffix matching.
    """
    raw = _safe_str(phone)
    digits = re.sub(r"\D", "", raw)
    # Drop leading zeros or '00' country prefix heuristic
    digits = digits.lstrip("0")
    return digits


# ---------------------------------------------------------------------------
# Self-contained demo / internal tests
# (run: python -m src.preprocessing.normalize)
# ---------------------------------------------------------------------------

def _run_tests():
    """
    Mini test suite — no external test framework required.
    Covers representative patterns from the challenge dataset types.
    """
    PASS = "\033[32mPASS\033[0m"
    FAIL = "\033[31mFAIL\033[0m"
    results = []

    def check(label, got, expected):
        ok = (got == expected)
        status = PASS if ok else FAIL
        results.append(ok)
        print(f"  [{status}] {label}")
        if not ok:
            print(f"         got      : {got!r}")
            print(f"         expected : {expected!r}")

    print("\n=== normalize.py — internal tests ===\n")

    # --- basic_clean ---
    check("basic_clean: None",            basic_clean(None),           "")
    check("basic_clean: NaN string",      basic_clean("nan"),          "")
    check("basic_clean: float nan",       basic_clean(float("nan")),   "")
    check("basic_clean: accent é",        basic_clean("Café"),         "cafe")
    check("basic_clean: accent ü",        basic_clean("Müller GmbH"),  "muller gmbh")
    check("basic_clean: extra spaces",    basic_clean("  Foo   Bar  "), "foo bar")
    check("basic_clean: punct stripped",  basic_clean("Foo, Bar. Inc!"), "foo bar inc")

    # --- normalize_name ---
    check("normalize_name: Corp.",        normalize_name("Acme Corp."),           "acme corporation")
    check("normalize_name: Inc",          normalize_name("Global Inc"),            "global incorporated")
    check("normalize_name: LLC",          normalize_name("Smith & Jones LLC"),     "smith and jones limited liability company")
    check("normalize_name: LLP",          normalize_name("Deloitte LLP"),          "deloitte limited liability partnership")
    check("normalize_name: Pvt Ltd",      normalize_name("Infosys Pvt Ltd"),       "infosys private limited")
    check("normalize_name: PLC",          normalize_name("BP PLC"),                "bp public limited company")
    check("normalize_name: GmbH",         normalize_name("Siemens GmbH"),          "siemens gesellschaft mit beschrankter haftung")
    check("normalize_name: None",         normalize_name(None),                    "")
    check("normalize_name: empty string", normalize_name(""),                      "")

    # --- extract_legal_suffix ---
    check("suffix: Corporation",          extract_legal_suffix("Acme Corporation"), "corporation")
    check("suffix: Corp.",                extract_legal_suffix("Acme Corp."),       "corporation")
    check("suffix: Inc",                  extract_legal_suffix("Global Inc"),       "incorporated")
    check("suffix: Ltd",                  extract_legal_suffix("Foo Ltd"),          "limited")
    check("suffix: no suffix",            extract_legal_suffix("Google"),           "")
    check("suffix: None",                 extract_legal_suffix(None),               "")

    # --- strip_legal_suffix ---
    check("strip: Acme Corp",             strip_legal_suffix("Acme Corp"),          "acme")
    check("strip: Smith & Jones LLC",     strip_legal_suffix("Smith & Jones LLC"),  "smith and jones")
    check("strip: no suffix",             strip_legal_suffix("Amazon"),             "amazon")
    check("strip: None",                  strip_legal_suffix(None),                 "")

    # --- normalize_address ---
    check("addr: US abbreviations",
          normalize_address("123 Main St, Apt 4B"),
          "123 main street apartment 4b")
    check("addr: Blvd, Ave",
          normalize_address("500 Park Ave Blvd"),
          "500 park avenue boulevard")
    check("addr: Indian opp/nr",
          normalize_address("Opp. Railway Station, Nr. Post Office"),
          "opposite railway station near post office")
    check("addr: None",
          normalize_address(None),
          "")
    check("addr: empty",
          normalize_address(""),
          "")

    # --- extract_numeric_tokens ---
    check("nums: PIN + building",
          extract_numeric_tokens("42 Baker St, London 110001"),
          ["42", "110001"])
    check("nums: empty",
          extract_numeric_tokens(None),
          [])

    # --- normalize_country ---
    check("country: USA",           normalize_country("USA"),            "united states")
    check("country: U.S.A.",        normalize_country("U.S.A."),         "united states")
    check("country: uk",            normalize_country("UK"),             "united kingdom")
    check("country: Great Britain", normalize_country("Great Britain"),  "united kingdom")
    check("country: IN",            normalize_country("IN"),             "india")
    check("country: india",         normalize_country("india"),          "india")
    check("country: France",        normalize_country("France"),         "france")   # open-set pass-through
    check("country: None",          normalize_country(None),             "")
    check("country: empty",         normalize_country(""),               "")

    # --- normalize_phone ---
    check("phone: US format",  normalize_phone("+1-800-555-1234"),  "18005551234")
    check("phone: Indian",     normalize_phone("022-12345678"),      "2212345678")
    check("phone: None",       normalize_phone(None),                "")

    # Summary
    n_pass = sum(results)
    n_fail = len(results) - n_pass
    print(f"\n  {n_pass}/{len(results)} passed, {n_fail} failed\n")
    return n_fail == 0


if __name__ == "__main__":
    import sys
    ok = _run_tests()
    sys.exit(0 if ok else 1)
