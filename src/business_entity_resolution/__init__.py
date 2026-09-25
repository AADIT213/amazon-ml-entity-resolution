"""
Business Entity Resolution Package
Amazon ML Challenge 2026
"""

from .normalize import (
    normalize_name,
    normalize_address,
    normalize_country,
    normalize_record,
    strip_legal_suffix,
)
from .text_utils import (
    clean_whitespace,
    remove_diacritics,
    strip_punctuation,
    to_ascii,
)

__all__ = [
    "clean_whitespace",
    "remove_diacritics",
    "strip_punctuation",
    "to_ascii",
    "normalize_name",
    "strip_legal_suffix",
    "normalize_address",
    "normalize_country",
    "normalize_record",
]
