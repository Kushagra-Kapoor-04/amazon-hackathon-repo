"""
normalize.py

Text normalization for business names and addresses.
No external lookups — pure string/dictionary-based cleaning only,
per competition fair-play rules.
"""

import re

NAME_ABBREVIATIONS = {
    r"\bcorp\b": "corporation",
    r"\bco\b": "company",
    r"\bltd\b": "limited",
    r"\bpvt\b": "private",
    r"\binc\b": "incorporated",
    r"\bllc\b": "limited liability company",
    r"\bllp\b": "limited liability partnership",
    r"\b&\b": "and",
}

ADDRESS_ABBREVIATIONS = {
    r"\brd\b": "road",
    r"\bst\b": "street",
    r"\bave\b": "avenue",
    r"\bblvd\b": "boulevard",
    r"\bln\b": "lane",
    r"\bdr\b": "drive",
    r"\bapt\b": "apartment",
    r"\bfl\b": "floor",
    r"\bnr\b": "near",
    r"\bopp\b": "opposite",
}

LEGAL_SUFFIXES = [
    "incorporated", "corporation", "company", "limited",
    "private", "llc", "llp", "inc", "corp", "ltd", "pvt", "co",
]


def _apply_replacements(text: str, mapping: dict) -> str:
    for pattern, replacement in mapping.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    return text


def basic_clean(text: str) -> str:
    """Lowercase, strip extra whitespace and punctuation noise (keep alphanumerics + spaces)."""
    if text is None:
        return ""
    text = str(text).lower().strip()
    text = re.sub(r"[^\w\s&]", " ", text)  # keep & before expansion
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_name(name: str) -> str:
    """
    Normalize a business name: lowercase, expand abbreviations, strip punctuation.
    Returns normalized string. Legal suffix is NOT stripped here — use
    extract_legal_suffix() separately since suffix match/mismatch is itself a feature.
    """
    text = basic_clean(name)
    text = _apply_replacements(text, NAME_ABBREVIATIONS)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_legal_suffix(name: str) -> str:
    """Return the legal suffix token found at the end of a normalized name, or ''."""
    norm = normalize_name(name)
    tokens = norm.split()
    if tokens and tokens[-1] in LEGAL_SUFFIXES:
        return tokens[-1]
    return ""


def strip_legal_suffix(name: str) -> str:
    """Return normalized name with trailing legal suffix removed, for core-name comparison."""
    norm = normalize_name(name)
    tokens = norm.split()
    if tokens and tokens[-1] in LEGAL_SUFFIXES:
        tokens = tokens[:-1]
    return " ".join(tokens)


def normalize_address(address: str) -> str:
    """
    Normalize an address string: lowercase, expand common abbreviations,
    strip punctuation. Does NOT parse into components — see parse_address.py.
    """
    text = basic_clean(address)
    text = _apply_replacements(text, ADDRESS_ABBREVIATIONS)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_country(country: str) -> str:
    """
    Light normalization only — country is an OPEN set of labels.
    Do not hardcode to {US, India}; unseen labels (e.g. France) must pass through unchanged.
    """
    if country is None:
        return ""
    return str(country).strip().lower()
