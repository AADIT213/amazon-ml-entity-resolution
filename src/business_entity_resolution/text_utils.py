"""
Common text normalization utilities for Business Entity Resolution.

Deterministic, pure, null-safe functions for character-level and token-level cleaning.
Uses only Python standard library (unicodedata, re). No external API calls.
"""

from typing import Any, Optional
import re
import unicodedata

# Pre-compiled regex patterns for performance across millions of rows
RE_WHITESPACE = re.compile(r"[\s\u00a0\u1680\u2000-\u200b\u2028\u2029\u202f\u205f\u3000\ufeff]+")
RE_AMPERSAND = re.compile(r"\s*&\s*")
RE_PUNCTUATION = re.compile(r"[^\w\s]")
RE_CONSECUTIVE_DASHES = re.compile(r"-{2,}")

# Specific Unicode replacements for common transliteration/ligatures
LIGATURE_MAP = {
    "æ": "ae",
    "Æ": "AE",
    "œ": "oe",
    "Œ": "OE",
    "ß": "ss",
    "ø": "o",
    "Ø": "O",
    "ł": "l",
    "Ł": "L",
    "đ": "d",
    "Đ": "D",
    "ð": "d",
    "Ð": "D",
    "þ": "th",
    "Þ": "TH",
}

# Unicode quotation marks and dashes to ASCII
CHAR_SUBSTITUTIONS = {
    "\u2018": "'",  # ‘
    "\u2019": "'",  # ’
    "\u201a": "'",  # ‚
    "\u201b": "'",  # ‛
    "\u201c": '"',  # “
    "\u201d": '"',  # ”
    "\u201e": '"',  # „
    "\u201f": '"',  # ‟
    "\u00ab": '"',  # «
    "\u00bb": '"',  # »
    "\u2013": "-",  # – en dash
    "\u2014": "-",  # — em dash
    "\u2015": "-",  # ― horizontal bar
    "\u2212": "-",  # − minus sign
    "\u2022": " ",  # • bullet
    "\u00b7": " ",  # · middle dot
    "\u2219": " ",  # ∙ bullet operator
    "/": " ",       # slash to space to avoid token merge
    "\\": " ",      # backslash to space
    "_": " ",       # underscore to space
}


def is_null_or_empty(val: Any) -> bool:
    """Check if value is None, NaN, or an empty/whitespace-only string."""
    if val is None:
        return True
    # Check for float NaN without importing numpy
    if isinstance(val, float) and val != val:
        return True
    if isinstance(val, str):
        return len(val.strip()) == 0
    return False


def clean_whitespace(text: Optional[str]) -> str:
    """Collapse consecutive whitespace and strip leading/trailing spaces. Null-safe."""
    if is_null_or_empty(text):
        return ""
    return RE_WHITESPACE.sub(" ", str(text)).strip()


def remove_diacritics(text: Optional[str]) -> str:
    """
    Decompose Unicode accents/diacritics (e.g. 'é' -> 'e', 'ü' -> 'u', 'ç' -> 'c').
    Preserves base Latin characters and common transliteration. Null-safe.
    """
    if is_null_or_empty(text):
        return ""
    
    text = str(text)
    
    # Replace special ligatures first
    for char, replacement in LIGATURE_MAP.items():
        if char in text:
            text = text.replace(char, replacement)
    
    # NFKD decomposition separates base letters from diacritical marks
    decomposed = unicodedata.normalize("NFKD", text)
    # Strip combining diacritical marks (category 'Mn')
    stripped = "".join(c for c in decomposed if unicodedata.category(c) != "Mn")
    return stripped


DEVANAGARI_TO_LATIN = {
    "क": "k", "ख": "kh", "ग": "g", "घ": "gh", "ङ": "ng",
    "च": "ch", "छ": "chh", "ज": "j", "झ": "jh", "ञ": "ny",
    "ट": "t", "ठ": "th", "ड": "d", "ढ": "dh", "ण": "n",
    "त": "t", "थ": "th", "द": "d", "ध": "dh", "न": "n",
    "प": "p", "फ": "ph", "ब": "b", "भ": "bh", "म": "m",
    "य": "y", "र": "r", "ल": "l", "व": "v", "श": "sh", "ष": "sh", "स": "s", "ह": "h",
    "अ": "a", "आ": "a", "इ": "i", "ई": "i", "उ": "u", "ऊ": "u", "ऋ": "ri",
    "ए": "e", "ऐ": "ai", "ओ": "o", "औ": "au",
    "ा": "a", "ि": "i", "ी": "i", "ु": "u", "ू": "u", "ृ": "ri",
    "े": "e", "ै": "ai", "ो": "o", "ौ": "au",
    "ं": "n", "ँ": "n", "्": "", "ः": "h", "़": "",
    "ज़": "z", "फ़": "f", "क़": "q", "ख़": "kh", "ग़": "gh",
}

RE_DEVANAGARI = re.compile(r"[\u0900-\u097f]")


def transliterate_indic(text: Optional[str]) -> str:
    """
    Deterministic rule-based transliteration of Devanagari script to Latin ASCII.
    Pure local implementation (no external API calls) adhering to TRD §3.
    """
    if is_null_or_empty(text) or not RE_DEVANAGARI.search(str(text)):
        return str(text) if text is not None else ""
    
    text = str(text)
    out = []
    i = 0
    n = len(text)
    while i < n:
        if i + 1 < n and text[i:i+2] in DEVANAGARI_TO_LATIN:
            out.append(DEVANAGARI_TO_LATIN[text[i:i+2]])
            i += 2
        elif text[i] in DEVANAGARI_TO_LATIN:
            out.append(DEVANAGARI_TO_LATIN[text[i]])
            i += 1
        else:
            out.append(text[i])
            i += 1
    return "".join(out)


def to_ascii(text: Optional[str]) -> str:
    """
    Normalize Unicode quotes, dashes, ligatures, Indic script, and diacritics to clean ASCII.
    Expands '&' to ' and '. Null-safe.
    """
    if is_null_or_empty(text):
        return ""
    
    text = str(text)
    
    # Transliterate Indic characters if present
    if RE_DEVANAGARI.search(text):
        text = transliterate_indic(text)
    
    # Replace specific Unicode characters and brackets/slashes
    for k, v in CHAR_SUBSTITUTIONS.items():
        if k in text:
            text = text.replace(k, v)
    
    # Standardize ampersands to ' and '
    text = RE_AMPERSAND.sub(" and ", text)
    
    # Remove diacritics
    text = remove_diacritics(text)
    
    return clean_whitespace(text)


def strip_punctuation(text: Optional[str], replace_with: str = " ") -> str:
    """
    Replace non-alphanumeric punctuation with replacement character (default: space)
    to prevent unintended token concatenations. Null-safe.
    """
    if is_null_or_empty(text):
        return ""
    
    text = str(text)
    cleaned = RE_PUNCTUATION.sub(replace_with, text)
    return clean_whitespace(cleaned)
