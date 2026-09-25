"""
Business Name Normalization module.

Deterministic, pure, null-safe normalization of company/business names.
Handles legal entity suffix standardization, punctuation cleaning,
casing, diacritics removal, and local rule-based transliteration.
"""

from typing import Optional
import re
from .text_utils import clean_whitespace, is_null_or_empty, to_ascii

# Legal suffixes to standardize to canonical forms
# Multi-word suffixes must be matched before single-word suffixes
LEGAL_SUFFIX_MAP = [
    # Indian / Commonwealth
    (r"\bprivate\s+limited\b", "pvt ltd"),
    (r"\bpvt\s+limited\b", "pvt ltd"),
    (r"\bprivate\s+ltd\b", "pvt ltd"),
    (r"\bp\s+ltd\b", "pvt ltd"),
    (r"\bpvtltd\b", "pvt ltd"),
    (r"\bpvt\s+l\b", "pvt ltd"),
    (r"\bpvt\b", "pvt"),
    
    # General / US
    (r"\blimited\s+liability\s+company\b", "llc"),
    (r"\blimited\s+liability\s+partnership\b", "llp"),
    (r"\bprofessional\s+limited\s+liability\s+company\b", "pllc"),
    (r"\bprofessional\s+corporation\b", "pc"),
    (r"\bpublic\s+limited\s+company\b", "plc"),
    (r"\bincorporated\b", "inc"),
    (r"\bcorporation\b", "corp"),
    (r"\bllc\b", "llc"),
    (r"\bllp\b", "llp"),
    (r"\binc\b", "inc"),
    (r"\bcorp\b", "corp"),
    (r"\bltd\b", "ltd"),
    (r"\blimited\b", "ltd"),
    
    # France / European (Open-set support for France in test)
    (r"\bsociete\s+anonyme\b", "sa"),
    (r"\bsociete\s+a\s+responsabilite\s+limitee\b", "sarl"),
    (r"\bsociete\s+par\s+actions\s+simplifiee\s+unipersonnelle\b", "sasu"),
    (r"\bsociete\s+par\s+actions\s+simplifiee\b", "sas"),
    (r"\bentreprise\s+unipersonnelle\s+a\s+responsabilite\s+limitee\b", "eurl"),
    (r"\bsociete\s+civile\s+immobiliere\b", "sci"),
    (r"\bgesellschaft\s+mit\s+beschrankter\s+haftung\b", "gmbh"),
    (r"\bbesloten\s+vennootschap\b", "bv"),
    (r"\bnaamloze\s+vennootschap\b", "nv"),
    (r"\bgmbh\b", "gmbh"),
    (r"\bsarl\b", "sarl"),
    (r"\bsas\b", "sas"),
    (r"\bsasu\b", "sasu"),
    (r"\beurl\b", "eurl"),
    (r"\bsci\b", "sci"),
    (r"\bsa\b", "sa"),
]

# Pre-compiled compiled patterns for suffix normalization
COMPILED_SUFFIX_REPLACEMENTS = [
    (re.compile(pattern, flags=re.IGNORECASE), repl)
    for pattern, repl in LEGAL_SUFFIX_MAP
]

# Patterns for stripping legal suffixes when extracting the core entity name stem
# These match only at the very end of the string (or leading for prefix anomalies)
CORE_STRIP_SUFFIXES = [
    r"pvt ltd",
    r"private limited",
    r"pvt",
    r"ltd",
    r"limited",
    r"inc",
    r"incorporated",
    r"corp",
    r"corporation",
    r"llc",
    r"llp",
    r"pllc",
    r"plc",
    r"gmbh",
    r"sarl",
    r"sas",
    r"sasu",
    r"eurl",
    r"sci",
    r"sa",
    r"co",
    r"company",
]

RE_TRAILING_LEGAL_SUFFIX = re.compile(
    r"\s+(?:co|company|" + "|".join(CORE_STRIP_SUFFIXES) + r")$",
    flags=re.IGNORECASE
)

RE_LEADING_LEGAL_SUFFIX = re.compile(
    r"^(?:llc|inc|corp|ltd|sarl|sa|sas)\s+",
    flags=re.IGNORECASE
)

# Punctuation to clean in company names: replace with space to prevent accidental token merging
RE_NAME_PUNCT = re.compile(r"['\"`\.,\-_/\\#\*\(\)\[\]\{\}:;!\?\|~<>@\+=]")


def standardize_legal_suffixes(text: str) -> str:
    """Standardize verbose and variant legal entity forms to canonical abbreviations."""
    for pattern, repl in COMPILED_SUFFIX_REPLACEMENTS:
        text = pattern.sub(repl, text)
    
    # Handle 'company' / 'co' only when at the end of the company name
    # e.g., 'Zander Blue Company' -> 'Zander Blue co', but preserve 'Company of Drivers'
    text = re.sub(r"\bcompany\b$", "co", text, flags=re.IGNORECASE)
    return text


def normalize_name(name: Optional[str]) -> str:
    """
    Standardize a business name deterministically:
    1. Null-safe handling (returns '' for null/empty).
    2. Normalize Unicode diacritics, quotes, and ligatures to clean ASCII.
    3. Convert to lowercase.
    4. Clean noise punctuation (brackets, slashes, dashes, commas, dots) to space.
    5. Standardize legal suffixes (e.g. 'Private Limited' -> 'pvt ltd', 'Inc.' -> 'inc').
    6. Collapse whitespace.
    7. Fully idempotent: normalize_name(normalize_name(x)) == normalize_name(x).
    """
    if is_null_or_empty(name):
        return ""
    
    # Unicode / ASCII conversion
    text = to_ascii(name).lower()
    
    # Standardize dots in abbreviations before stripping punctuation (e.g. 'p.v.t. l.t.d.' or 'inc.')
    text = re.sub(r"\bp\s*\.\s*v\s*\.\s*t\s*\.\s*l\s*\.\s*t\s*\.\s*d\s*\.?", "pvt ltd", text)
    text = re.sub(r"\bl\s*\.\s*l\s*\.\s*c\s*\.?", "llc", text)
    text = re.sub(r"\bp\s*\.\s*l\s*\.\s*t\s*\.\s*d\s*\.?", "pvt ltd", text)
    text = re.sub(r"\bl\s*\.\s*t\s*\.\s*d\s*\.?", "ltd", text)
    text = re.sub(r"\bi\s*\.\s*n\s*\.\s*c\s*\.?", "inc", text)
    
    # Replace noise punctuation with spaces
    text = RE_NAME_PUNCT.sub(" ", text)
    text = clean_whitespace(text)
    
    # Standardize legal suffixes
    text = standardize_legal_suffixes(text)
    
    return clean_whitespace(text)


def strip_legal_suffix(name: Optional[str]) -> str:
    """
    Strip trailing (or anomalous leading) legal suffixes to extract the core business stem.
    e.g. 'custom wealth services llc' -> 'custom wealth services'
         'systel buildstructure india pvt ltd' -> 'systel buildstructure india'
         'llc orellana investments' -> 'orellana investments'
    If stripping leaves the string empty (e.g. 'LLC'), returns the normalized string.
    """
    if is_null_or_empty(name):
        return ""
    
    norm = normalize_name(name)
    stripped = RE_TRAILING_LEGAL_SUFFIX.sub("", norm).strip()
    stripped = RE_LEADING_LEGAL_SUFFIX.sub("", stripped).strip()
    
    return stripped if stripped else norm
