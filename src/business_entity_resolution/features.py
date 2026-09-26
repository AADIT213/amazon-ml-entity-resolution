"""
Phase 4: Pair Feature Engineering Module.

Provides vectorized, lightweight, deterministic, side-effect-free feature computation
for business entity resolution pairs (TRD.md §5).

Lightweight Feature Set (12 features):
- Name Features:
  1. exact_name_match: 1.0 if normalized names are identical and non-empty, else 0.0
  2. name_token_jaccard: Word token set Jaccard similarity [0, 1]
  3. name_char_similarity: Normalized Levenshtein character similarity [0, 1]
  4. name_len_diff: Absolute difference in string length
  5. name_token_count_diff: Absolute difference in word token counts

- Address Features:
  6. exact_addr_match: 1.0 if normalized addresses are identical and non-empty, else 0.0
  7. addr_token_jaccard: Word token set Jaccard similarity on address [0, 1]
  8. addr_numeric_overlap: Jaccard overlap of numeric digit groups (house/pin numbers)
  9. addr_len_diff: Absolute difference in address string length

- Cross-Field Features:
  10. country_match: 1.0 if normalized countries are identical and non-empty, else 0.0
  11. missing_name_flag: 1.0 if either entity is missing name, else 0.0
  12. missing_addr_flag: 1.0 if either entity is missing address, else 0.0
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np
from rapidfuzz import distance

LIGHTWEIGHT_FEATURE_NAMES = [
    "exact_name_match",
    "name_token_jaccard",
    "name_char_similarity",
    "name_len_diff",
    "name_token_count_diff",
    "exact_addr_match",
    "addr_token_jaccard",
    "addr_numeric_overlap",
    "addr_len_diff",
    "country_match",
    "missing_name_flag",
    "missing_addr_flag",
]


def jaccard_similarity(set_a: Set[str], set_b: Set[str]) -> float:
    """Compute Jaccard similarity between two token sets."""
    if not set_a or not set_b:
        return 0.0
    intersection = len(set_a.intersection(set_b))
    union = len(set_a.union(set_b))
    return float(intersection / union) if union > 0 else 0.0


def extract_numeric_tokens(text: str) -> Set[str]:
    """Extract sequence of numbers (e.g. house number, zip codes) from address."""
    if not text:
        return set()
    return set(re.findall(r"\d+", text))


def compute_lightweight_pair_features(
    name1: str,
    addr1: str,
    ctry1: str,
    name2: str,
    addr2: str,
    ctry2: str,
) -> Dict[str, float]:
    """
    Compute lightweight 12-feature dictionary for a single entity pair.
    Deterministic, pure, and side-effect free.
    """
    # Name features
    missing_name = 1.0 if (not name1 or not name2) else 0.0
    exact_name = 1.0 if (name1 and name2 and name1 == name2) else 0.0
    
    toks1 = set(name1.split()) if name1 else set()
    toks2 = set(name2.split()) if name2 else set()
    name_jaccard = jaccard_similarity(toks1, toks2) if not missing_name else 0.0

    name_char_sim = float(distance.Levenshtein.normalized_similarity(name1, name2)) if (name1 and name2) else 0.0
    name_len_diff = float(abs(len(name1) - len(name2)))
    name_tok_diff = float(abs(len(toks1) - len(toks2)))

    # Address features
    missing_addr = 1.0 if (not addr1 or not addr2) else 0.0
    exact_addr = 1.0 if (not missing_addr and addr1 == addr2) else 0.0

    a_toks1 = set(addr1.split()) if addr1 else set()
    a_toks2 = set(addr2.split()) if addr2 else set()
    addr_jaccard = jaccard_similarity(a_toks1, a_toks2) if not missing_addr else 0.0

    nums1 = extract_numeric_tokens(addr1)
    nums2 = extract_numeric_tokens(addr2)
    if not missing_addr:
        if nums1 and nums2:
            addr_num_overlap = jaccard_similarity(nums1, nums2)
        elif not nums1 and not nums2:
            addr_num_overlap = 1.0  # Both have no numbers
        else:
            addr_num_overlap = 0.0
    else:
        addr_num_overlap = 0.0

    addr_len_diff = float(abs(len(addr1) - len(addr2)))

    # Cross-field features
    ctry_match = 1.0 if (ctry1 and ctry2 and ctry1 == ctry2) else 0.0

    return {
        "exact_name_match": exact_name,
        "name_token_jaccard": name_jaccard,
        "name_char_similarity": name_char_sim,
        "name_len_diff": name_len_diff,
        "name_token_count_diff": name_tok_diff,
        "exact_addr_match": exact_addr,
        "addr_token_jaccard": addr_jaccard,
        "addr_numeric_overlap": addr_num_overlap,
        "addr_len_diff": addr_len_diff,
        "country_match": ctry_match,
        "missing_name_flag": missing_name,
        "missing_addr_flag": missing_addr,
    }


def compute_lightweight_batch_features(
    s1_ids: Sequence[str],
    cand_ids: Sequence[str],
    entity_lookup: Mapping[str, Tuple[str, str, str]],  # eid -> (norm_name, norm_addr, norm_ctry)
) -> np.ndarray:
    """
    Vectorized batch feature computation over arrays of (S1, Candidate) ID pairs.
    Returns 2D numpy array of shape (N_pairs, len(LIGHTWEIGHT_FEATURE_NAMES)) with dtype float32.
    """
    n_pairs = len(s1_ids)
    if n_pairs == 0:
        return np.empty((0, len(LIGHTWEIGHT_FEATURE_NAMES)), dtype=np.float32)

    feats = np.zeros((n_pairs, len(LIGHTWEIGHT_FEATURE_NAMES)), dtype=np.float32)
    empty_tuple = ("", "", "")

    for i in range(n_pairs):
        e1 = s1_ids[i]
        e2 = cand_ids[i]

        rec1 = entity_lookup.get(e1, empty_tuple)
        rec2 = entity_lookup.get(e2, empty_tuple)

        n1, a1, c1 = rec1
        n2, a2, c2 = rec2

        # Missing flags
        missing_n = 1.0 if (not n1 or not n2) else 0.0
        missing_a = 1.0 if (not a1 or not a2) else 0.0
        feats[i, 10] = missing_n
        feats[i, 11] = missing_a

        # Name metrics
        feats[i, 0] = 1.0 if (n1 and n2 and n1 == n2) else 0.0
        
        toks1 = set(n1.split()) if n1 else set()
        toks2 = set(n2.split()) if n2 else set()
        feats[i, 1] = jaccard_similarity(toks1, toks2) if not missing_n else 0.0

        feats[i, 2] = float(distance.Levenshtein.normalized_similarity(n1, n2)) if (n1 and n2) else 0.0
        feats[i, 3] = float(abs(len(n1) - len(n2)))
        feats[i, 4] = float(abs(len(toks1) - len(toks2)))

        # Address metrics
        feats[i, 5] = 1.0 if (not missing_a and a1 == a2) else 0.0

        a_toks1 = set(a1.split()) if a1 else set()
        a_toks2 = set(a2.split()) if a2 else set()
        feats[i, 6] = jaccard_similarity(a_toks1, a_toks2) if not missing_a else 0.0

        if not missing_a:
            nums1 = extract_numeric_tokens(a1)
            nums2 = extract_numeric_tokens(a2)
            if nums1 and nums2:
                feats[i, 7] = jaccard_similarity(nums1, nums2)
            elif not nums1 and not nums2:
                feats[i, 7] = 1.0
            else:
                feats[i, 7] = 0.0
        else:
            feats[i, 7] = 0.0

        feats[i, 8] = float(abs(len(a1) - len(a2)))

        # Country metric
        feats[i, 9] = 1.0 if (c1 and c2 and c1 == c2) else 0.0

    return feats
