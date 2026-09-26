"""
src/preprocessing/__init__.py

Public API for the preprocessing package.
Import from here so callers are insulated from internal module layout.

Note: parse_address is imported with an alias to avoid shadowing the
      `parse_address` module with a function of the same name.
"""

from src.preprocessing.normalize import (
    basic_clean,
    normalize_name,
    extract_legal_suffix,
    strip_legal_suffix,
    tokenize_name,
    normalize_address,
    extract_numeric_tokens,
    normalize_country,
    normalize_phone,
    LEGAL_SUFFIXES_SET,
)

from src.preprocessing.parse_address import (
    parse_address as parse_address_record,
    normalize_address_fields,
    AddressRecord,
)

# Keep the canonical names accessible too
parse_address = parse_address_record

__all__ = [
    # normalize
    "basic_clean",
    "normalize_name",
    "extract_legal_suffix",
    "strip_legal_suffix",
    "tokenize_name",
    "normalize_address",
    "extract_numeric_tokens",
    "normalize_country",
    "normalize_phone",
    "LEGAL_SUFFIXES_SET",
    # parse_address
    "parse_address",
    "parse_address_record",
    "normalize_address_fields",
    "AddressRecord",
]
