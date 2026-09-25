"""
Business Address Normalization module.

Deterministic, pure, null-safe normalization of physical business addresses.
Handles thoroughfare abbreviations (Street -> st, Avenue -> ave, etc.),
unit and floor abbreviations, numeric tokens (1st -> 1, 2nd -> 2),
punctuation removal, casing, and diacritics removal.
"""

from typing import Optional
import re
from .text_utils import clean_whitespace, is_null_or_empty, to_ascii

# Thoroughfare and street type abbreviations
# Standardized to short canonical tokens
STREET_TYPE_MAP = [
    (r"\bstreet\b", "st"),
    (r"\bstr\b", "st"),
    (r"\bavenue\b", "ave"),
    (r"\bav\b", "ave"),
    (r"\broad\b", "rd"),
    (r"\bdrive\b", "dr"),
    (r"\bboulevard\b", "blvd"),
    (r"\blane\b", "ln"),
    (r"\bcourt\b", "ct"),
    (r"\bcircle\b", "cir"),
    (r"\bparkway\b", "pkwy"),
    (r"\bhighway\b", "hwy"),
    (r"\bplace\b", "pl"),
    (r"\bterrace\b", "ter"),
    (r"\bsquare\b", "sq"),
    (r"\btrail\b", "trl"),
    (r"\broute\b", "rte"),
    (r"\bsaint\b", "st"),
]

# Secondary unit and floor designations
UNIT_TYPE_MAP = [
    (r"\bapartment\b", "apt"),
    (r"\bapartments\b", "apt"),
    (r"\bsuite\b", "ste"),
    (r"\bfloor\b", "fl"),
    (r"\bflr\b", "fl"),
    (r"\betage\b", "fl"),
    (r"\bbuilding\b", "bldg"),
    (r"\broom\b", "rm"),
    (r"\bdepartment\b", "dept"),
    (r"\bdistrict\b", "dist"),
    (r"\bcare\s+of\b", "c/o"),
    (r"\bc\s*/\s*o\b", "c/o"),
    (r"\bpost\s+office\s+box\b", "po box"),
    (r"\bp\s*\.?\s*o\s*\.?\s*box\b", "po box"),
    (r"\bpobox\b", "po box"),
]

# Cardinal directions when isolated
DIRECTION_MAP = [
    (r"\bnorth\b", "n"),
    (r"\bsouth\b", "s"),
    (r"\beast\b", "e"),
    (r"\bwest\b", "w"),
    (r"\bnortheast\b", "ne"),
    (r"\bnorthwest\b", "nw"),
    (r"\bsoutheast\b", "se"),
    (r"\bsouthwest\b", "sw"),
]

# Pre-compiled compiled regexes for address terms
COMPILED_STREET_REPLACEMENTS = [
    (re.compile(p, flags=re.IGNORECASE), repl) for p, repl in STREET_TYPE_MAP
]
COMPILED_UNIT_REPLACEMENTS = [
    (re.compile(p, flags=re.IGNORECASE), repl) for p, repl in UNIT_TYPE_MAP
]
COMPILED_DIR_REPLACEMENTS = [
    (re.compile(p, flags=re.IGNORECASE), repl) for p, repl in DIRECTION_MAP
]

# Numeric token normalization: 1st -> 1, 2nd -> 2, 3rd -> 3, 4th -> 4, etc.
RE_ORDINAL_NUMBERS = re.compile(r"\b(\d+)(?:st|nd|rd|th)\b", flags=re.IGNORECASE)

# Number prefix normalization: '# 162', '#162', 'no. 162' -> 'no 162'
RE_HASH_NUMBER = re.compile(r"#\s*(\d+)")
RE_NO_NUMBER = re.compile(r"\bno\s*\.?\s*(\d+)", flags=re.IGNORECASE)

# Address punctuation: replace with space to avoid token concatenation
RE_ADDR_PUNCT = re.compile(r"['\"`\.,\-_/\\#\*\(\)\[\]\{\}:;!\?\|~<>@\+=]")

# Repeated consecutive words (e.g. 'unit unit 2' or 'buildstructure buildstructure')
RE_DUPLICATE_WORDS = re.compile(r"\b(\w+)\s+\1\b", flags=re.IGNORECASE)


def normalize_numeric_tokens(text: str) -> str:
    """Normalize numeric expressions like ordinals ('1st' -> '1', '14th' -> '14')."""
    text = RE_ORDINAL_NUMBERS.sub(r"\1", text)
    text = RE_HASH_NUMBER.sub(r"no \1", text)
    text = RE_NO_NUMBER.sub(r"no \1", text)
    return text


def standardize_address_abbreviations(text: str) -> str:
    """Standardize street types, unit designations, and directions to canonical tokens."""
    for pattern, repl in COMPILED_STREET_REPLACEMENTS:
        text = pattern.sub(repl, text)
    for pattern, repl in COMPILED_UNIT_REPLACEMENTS:
        text = pattern.sub(repl, text)
    for pattern, repl in COMPILED_DIR_REPLACEMENTS:
        text = pattern.sub(repl, text)
    return text


def normalize_address(address: Optional[str]) -> str:
    """
    Standardize a physical business address deterministically:
    1. Null-safe handling (returns '' for null/empty/NaN).
    2. Convert Unicode diacritics, quotes, and symbols to clean ASCII.
    3. Convert to lowercase.
    4. Normalize numeric tokens (e.g. '11th floor' -> '11 fl', '#162' -> 'no 162').
    5. Clean punctuation to spaces.
    6. Standardize abbreviations (e.g. 'Street' -> 'st', 'Avenue' -> 'ave', 'Road' -> 'rd').
    7. Deduplicate immediately repeated tokens (e.g. 'unit unit' -> 'unit').
    8. Collapse whitespace.
    9. Fully idempotent: normalize_address(normalize_address(x)) == normalize_address(x).
    """
    if is_null_or_empty(address):
        return ""
    
    # Unicode / ASCII conversion
    text = to_ascii(address).lower()
    
    # Normalize numeric ordinals before stripping letters attached to digits
    text = normalize_numeric_tokens(text)
    
    # Standardize abbreviations with periods before stripping punctuation (e.g. 'st.' vs 'st')
    text = re.sub(r"\bst\s*\.", "st ", text)
    text = re.sub(r"\bave\s*\.", "ave ", text)
    text = re.sub(r"\brd\s*\.", "rd ", text)
    text = re.sub(r"\bdr\s*\.", "dr ", text)
    text = re.sub(r"\bblvd\s*\.", "blvd ", text)
    text = re.sub(r"\bfl\s*\.", "fl ", text)
    text = re.sub(r"\bapt\s*\.", "apt ", text)
    text = re.sub(r"\bste\s*\.", "ste ", text)
    
    # Replace punctuation with spaces
    text = RE_ADDR_PUNCT.sub(" ", text)
    text = clean_whitespace(text)
    
    # Standardize address terms
    text = standardize_address_abbreviations(text)
    
    # Deduplicate repeated adjacent tokens (e.g. 'unit unit')
    text = RE_DUPLICATE_WORDS.sub(r"\1", text)
    
    return clean_whitespace(text)
