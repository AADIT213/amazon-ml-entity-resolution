"""
Unit tests for Phase 4 lightweight pair feature engineering functions.

Tests cover:
- Exact name matching and missing name flag
- Token Jaccard overlap
- Character similarity (Levenshtein normalized)
- Name length & token count differences
- Exact address matching and missing address flag
- Address token Jaccard & numeric token overlap
- Address length difference
- Country equality
- Batch vs single pair parity
"""

import sys
import os
import pytest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from business_entity_resolution.features import (
    LIGHTWEIGHT_FEATURE_NAMES,
    compute_lightweight_batch_features,
    compute_lightweight_pair_features,
    jaccard_similarity,
    extract_numeric_tokens,
)


def test_jaccard_similarity():
    assert jaccard_similarity({"apple", "banana"}, {"apple", "banana"}) == 1.0
    assert jaccard_similarity({"apple", "banana"}, {"banana", "cherry"}) == 1 / 3
    assert jaccard_similarity(set(), {"apple"}) == 0.0
    assert jaccard_similarity(set(), set()) == 0.0


def test_extract_numeric_tokens():
    assert extract_numeric_tokens("123 Main Street, Suite 400") == {"123", "400"}
    assert extract_numeric_tokens("No numbers here") == set()
    assert extract_numeric_tokens("") == set()
    assert extract_numeric_tokens("Pin 560001, 11th Floor") == {"560001", "11"}


def test_identical_pair_lightweight():
    feats = compute_lightweight_pair_features(
        name1="acme logistics corp",
        addr1="123 main st peoria il",
        ctry1="us",
        name2="acme logistics corp",
        addr2="123 main st peoria il",
        ctry2="us",
    )

    assert feats["exact_name_match"] == 1.0
    assert feats["name_token_jaccard"] == 1.0
    assert feats["name_char_similarity"] == 1.0
    assert feats["name_len_diff"] == 0.0
    assert feats["name_token_count_diff"] == 0.0

    assert feats["exact_addr_match"] == 1.0
    assert feats["addr_token_jaccard"] == 1.0
    assert feats["addr_numeric_overlap"] == 1.0
    assert feats["addr_len_diff"] == 0.0

    assert feats["country_match"] == 1.0
    assert feats["missing_name_flag"] == 0.0
    assert feats["missing_addr_flag"] == 0.0


def test_partial_match_pair():
    feats = compute_lightweight_pair_features(
        name1="global trade pvt ltd",
        addr1="45 park road bangalore",
        ctry1="india",
        name2="global trade limited",
        addr2="45 park rd bangalore",
        ctry2="india",
    )

    assert feats["exact_name_match"] == 0.0
    assert feats["name_token_jaccard"] == 2 / 5  # 'global', 'trade' in common; union size 5
    assert feats["name_char_similarity"] >= 0.7
    assert feats["country_match"] == 1.0
    assert feats["addr_numeric_overlap"] == 1.0  # '45'
    assert feats["missing_name_flag"] == 0.0
    assert feats["missing_addr_flag"] == 0.0


def test_null_address_pair():
    feats = compute_lightweight_pair_features(
        name1="tech solutions inc",
        addr1="",
        ctry1="us",
        name2="tech solutions inc",
        addr2="",
        ctry2="us",
    )

    assert feats["missing_addr_flag"] == 1.0
    assert feats["exact_addr_match"] == 0.0
    assert feats["addr_token_jaccard"] == 0.0
    assert feats["addr_numeric_overlap"] == 0.0
    assert feats["exact_name_match"] == 1.0


def test_missing_name_pair():
    feats = compute_lightweight_pair_features(
        name1="",
        addr1="10 high st",
        ctry1="us",
        name2="acme corp",
        addr2="10 high st",
        ctry2="us",
    )

    assert feats["missing_name_flag"] == 1.0
    assert feats["exact_name_match"] == 0.0
    assert feats["name_token_jaccard"] == 0.0
    assert feats["exact_addr_match"] == 1.0


def test_country_mismatch_pair():
    feats = compute_lightweight_pair_features(
        name1="bistro paris",
        addr1="12 rue de la paix paris",
        ctry1="france",
        name2="bistro paris",
        addr2="12 rue de la paix paris",
        ctry2="us",
    )

    assert feats["exact_name_match"] == 1.0
    assert feats["country_match"] == 0.0


def test_batch_vs_single_parity_lightweight():
    lookup = {
        "S1-1": ("alpha beta corp", "10 high st", "us"),
        "S2-1": ("alpha beta inc", "10 high street", "us"),
        "S2-2": ("gamma delta llc", "99 low ave", "india"),
    }

    batch_feats = compute_lightweight_batch_features(["S1-1", "S1-1"], ["S2-1", "S2-2"], lookup)

    assert batch_feats.shape == (2, len(LIGHTWEIGHT_FEATURE_NAMES))

    single_1 = compute_lightweight_pair_features(
        name1="alpha beta corp",
        addr1="10 high st",
        ctry1="us",
        name2="alpha beta inc",
        addr2="10 high street",
        ctry2="us",
    )

    for f_idx, f_name in enumerate(LIGHTWEIGHT_FEATURE_NAMES):
        np.testing.assert_almost_equal(
            batch_feats[0, f_idx], single_1[f_name], decimal=4, err_msg=f"Mismatch on {f_name}"
        )
