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
from .blocking import (
    build_default_strategies,
    build_indexes_from_lists,
    get_candidates,
    generate_candidates,
    ExactNormalizedNameBlock,
    NameStemBlock,
    NameTokenBlock,
    CountryExactNameBlock,
    AddressTokenBlock,
    CountryNameStemBlock,
)
from .features import (
    LIGHTWEIGHT_FEATURE_NAMES,
    compute_lightweight_batch_features,
    compute_lightweight_pair_features,
    jaccard_similarity,
    extract_numeric_tokens,
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
    "build_default_strategies",
    "build_indexes_from_lists",
    "get_candidates",
    "generate_candidates",
    "ExactNormalizedNameBlock",
    "NameStemBlock",
    "NameTokenBlock",
    "CountryExactNameBlock",
    "AddressTokenBlock",
    "CountryNameStemBlock",
    "LIGHTWEIGHT_FEATURE_NAMES",
    "compute_lightweight_batch_features",
    "compute_lightweight_pair_features",
    "jaccard_similarity",
    "extract_numeric_tokens",
]
