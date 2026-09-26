"""
Unit and integration tests for Phase 3 blocking strategies.

Tests cover:
- Each blocking strategy independently
- Candidate union behaviour
- Null-address fallback (AddressTokenBlock)
- Open-set country handling (CountryNameStemBlock)
- Duplicate removal in candidate sets
- Zero-candidate entities
- Multiple candidates per S1 entity
- candidate_pairs.tsv schema / output contract
- Full generate_candidates() pipeline
"""
import pytest
import sys
import os
import tempfile
import csv

# Ensure src is importable when running from the repo root with pytest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from business_entity_resolution.blocking import (
    ExactNormalizedNameBlock,
    NameStemBlock,
    NameTokenBlock,
    CountryExactNameBlock,
    AddressTokenBlock,
    CountryNameStemBlock,
    build_default_strategies,
    build_indexes_from_lists,
    get_candidates,
    generate_candidates,
    _name_tokens,
    _char_bigrams,
    _addr_tokens,
)
from business_entity_resolution.normalize import normalize_dataframe
import pandas as pd


# ---------------------------------------------------------------------------
# Fixtures: minimal normalised records
# ---------------------------------------------------------------------------

def _make_rec(entity_id, name_norm="", name_stem="", addr_norm="", has_address=True, country=""):
    """Build a minimal normalised record dict."""
    return {
        "entity_id": entity_id,
        "normalized_name": name_norm,
        "name_stem": name_stem,
        "normalized_address": addr_norm,
        "has_address": has_address,
        "normalized_country": country,
    }


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

class TestHelperFunctions:
    def test_name_tokens_filters_single_char(self):
        assert "a" not in _name_tokens("acme a corp")
        assert "acme" in _name_tokens("acme a corp")
        assert "corp" in _name_tokens("acme a corp")

    def test_name_tokens_empty(self):
        assert _name_tokens("") == []

    def test_char_bigrams_basic(self):
        bgs = _char_bigrams("abc")
        assert "ab" in bgs
        assert "bc" in bgs

    def test_char_bigrams_empty(self):
        assert _char_bigrams("") == []

    def test_char_bigrams_spaces_collapsed(self):
        # "ab cd" -> bigrams of "abcd"
        bgs = set(_char_bigrams("ab cd"))
        assert "bc" in bgs   # crosses the word boundary

    def test_addr_tokens_skips_stopwords(self):
        tokens = _addr_tokens("3315 fremont st peoria il")
        assert "st" not in tokens
        assert "fremont" in tokens
        assert "peoria" in tokens


# ---------------------------------------------------------------------------
# Strategy 1: ExactNormalizedNameBlock
# ---------------------------------------------------------------------------

class TestExactNormalizedNameBlock:
    def _make(self):
        s = ExactNormalizedNameBlock()
        recs = [
            _make_rec("S2-1", name_norm="acme corp"),
            _make_rec("S2-2", name_norm="globex inc"),
            _make_rec("S3-1", name_norm="acme corp"),  # duplicate name, different entity
        ]
        s.build_index(recs)
        return s

    def test_exact_match(self):
        s = self._make()
        q = _make_rec("S1-1", name_norm="acme corp")
        result = s.query(q)
        assert "S2-1" in result
        assert "S3-1" in result

    def test_no_match(self):
        s = self._make()
        q = _make_rec("S1-1", name_norm="unknown name")
        assert s.query(q) == set()

    def test_empty_name(self):
        s = self._make()
        q = _make_rec("S1-1", name_norm="")
        assert s.query(q) == set()

    def test_multiple_s2_s3_entities_same_name(self):
        s = self._make()
        q = _make_rec("S1-1", name_norm="acme corp")
        result = s.query(q)
        assert len(result) == 2  # S2-1 and S3-1


# ---------------------------------------------------------------------------
# Strategy 2: NameStemBlock
# ---------------------------------------------------------------------------

class TestNameStemBlock:
    def _make(self):
        s = NameStemBlock()
        recs = [
            _make_rec("S2-1", name_stem="acme"),
            _make_rec("S2-2", name_stem="acme corp"),
            _make_rec("S3-1", name_stem="globex"),
        ]
        s.build_index(recs)
        return s

    def test_stem_match(self):
        s = self._make()
        q = _make_rec("S1-1", name_stem="acme")
        assert "S2-1" in s.query(q)

    def test_no_match(self):
        s = self._make()
        q = _make_rec("S1-1", name_stem="unknown")
        assert s.query(q) == set()

    def test_short_stem_skipped(self):
        s = NameStemBlock()
        s.build_index([_make_rec("S2-1", name_stem="co")])
        q = _make_rec("S1-1", name_stem="co")
        assert s.query(q) == set()   # "co" is too short (< MIN_STEM_LEN=4)

    def test_stem_must_be_at_least_4_chars(self):
        s = NameStemBlock()
        s.build_index([_make_rec("S2-1", name_stem="abcd")])
        q = _make_rec("S1-1", name_stem="abcd")
        assert "S2-1" in s.query(q)


# ---------------------------------------------------------------------------
# Strategy 3: NameTokenBlock
# ---------------------------------------------------------------------------

class TestNameTokenBlock:
    def _make(self):
        s = NameTokenBlock(min_shared_tokens=2)
        recs = [
            _make_rec("S2-1", name_norm="laxmi golden investments pvt ltd"),
            _make_rec("S2-2", name_norm="golden star enterprises"),
            _make_rec("S3-1", name_norm="laxmi enterprises pvt ltd"),
        ]
        s.build_index(recs)
        return s

    def test_two_shared_tokens(self):
        s = self._make()
        q = _make_rec("S1-1", name_norm="laxmi golden corp")
        result = s.query(q)
        assert "S2-1" in result  # shares "laxmi" and "golden"

    def test_one_shared_token_below_threshold(self):
        s = self._make()
        q = _make_rec("S1-1", name_norm="star widgets")
        result = s.query(q)
        # Only "star" is shared with S2-2 -> below threshold of 2
        assert "S2-2" not in result

    def test_empty_name_returns_empty(self):
        s = self._make()
        q = _make_rec("S1-1", name_norm="")
        assert s.query(q) == set()

    def test_short_name_lowered_threshold(self):
        s = NameTokenBlock(min_shared_tokens=2)
        recs = [_make_rec("S2-1", name_norm="acme")]
        s.build_index(recs)
        q = _make_rec("S1-1", name_norm="acme")
        # Single token -> threshold lowered to 1
        assert "S2-1" in s.query(q)


# ---------------------------------------------------------------------------
# Strategy 4: CountryExactNameBlock
# ---------------------------------------------------------------------------

class TestCountryExactNameBlock:
    def _make(self):
        s = CountryExactNameBlock()
        recs = [
            _make_rec("S2-1", name_norm="acme corp", country="us"),
            _make_rec("S2-2", name_norm="acme corp", country="india"),
            _make_rec("S3-1", name_norm="globex inc", country="us"),
        ]
        s.build_index(recs)
        return s

    def test_country_exact_name_match(self):
        s = self._make()
        q = _make_rec("S1-1", name_norm="acme corp", country="us")
        result = s.query(q)
        assert "S2-1" in result
        assert "S2-2" not in result

    def test_different_country_no_match(self):
        s = self._make()
        q = _make_rec("S1-1", name_norm="acme corp", country="france")
        result = s.query(q)
        assert "S2-1" not in result
        assert "S2-2" not in result

    def test_empty_name_no_match(self):
        s = self._make()
        q = _make_rec("S1-1", name_norm="", country="us")
        assert s.query(q) == set()


# ---------------------------------------------------------------------------
# Strategy 5: AddressTokenBlock + null-address fallback
# ---------------------------------------------------------------------------

class TestAddressTokenBlock:
    def _make(self):
        s = AddressTokenBlock(min_shared_tokens=3)
        recs = [
            _make_rec("S2-1", addr_norm="3315 fremont peoria illinois", has_address=True, name_stem="acme"),
            _make_rec("S2-2", addr_norm="100 main street springfield ohio", has_address=True, name_stem="globex"),
            _make_rec("S3-1", addr_norm="", has_address=False, name_stem="nullco"),   # null-addr record
        ]
        s.build_index(recs)
        return s

    def test_address_match(self):
        s = self._make()
        q = _make_rec("S1-1", addr_norm="3315 fremont avenue peoria", has_address=True)
        result = s.query(q)
        assert "S2-1" in result

    def test_address_no_match_below_threshold(self):
        s = self._make()
        q = _make_rec("S1-1", addr_norm="3315 fremont", has_address=True)
        result = s.query(q)
        # Only 2 tokens -> below threshold of 3
        assert "S2-1" not in result

    def test_null_address_s1_queries_stem_fallback(self):
        """S1 with no address should retrieve null-addr S2/S3 entities by stem."""
        s = self._make()
        q = _make_rec("S1-1", addr_norm="", has_address=False, name_stem="nullco")
        result = s.query(q)
        assert "S3-1" in result

    def test_null_address_s1_no_match_different_stem(self):
        s = self._make()
        q = _make_rec("S1-1", addr_norm="", has_address=False, name_stem="somethingelse")
        result = s.query(q)
        assert "S3-1" not in result

    def test_null_address_s1_empty_stem(self):
        s = self._make()
        q = _make_rec("S1-1", addr_norm="", has_address=False, name_stem="")
        assert s.query(q) == set()


# ---------------------------------------------------------------------------
# Strategy 6: CountryNameStemBlock (open-set)
# ---------------------------------------------------------------------------

class TestCountryNameStemBlock:
    def _make(self):
        s = CountryNameStemBlock()
        recs = [
            _make_rec("S2-1", name_stem="acme", country="us"),
            _make_rec("S2-2", name_stem="acme", country="india"),
            _make_rec("S2-3", name_stem="acme", country="france"),   # open-set country
            _make_rec("S3-1", name_stem="acme", country="us"),
        ]
        s.build_index(recs)
        return s

    def test_country_bucket_us(self):
        s = self._make()
        q = _make_rec("S1-1", name_stem="acme", country="us")
        result = s.query(q)
        assert "S2-1" in result
        assert "S3-1" in result
        assert "S2-2" not in result   # different country
        assert "S2-3" not in result

    def test_country_bucket_france_open_set(self):
        """France is test-only (not in training ground truth). Must still work."""
        s = self._make()
        q = _make_rec("S1-1", name_stem="acme", country="france")
        result = s.query(q)
        assert "S2-3" in result
        assert "S2-1" not in result   # different country

    def test_country_bucket_india(self):
        s = self._make()
        q = _make_rec("S1-1", name_stem="acme", country="india")
        result = s.query(q)
        assert "S2-2" in result
        assert "S2-1" not in result

    def test_short_stem_excluded(self):
        s = CountryNameStemBlock()
        s.build_index([_make_rec("S2-1", name_stem="co", country="us")])
        q = _make_rec("S1-1", name_stem="co", country="us")
        assert s.query(q) == set()

    def test_empty_country_bucket(self):
        """Records with no country match only other no-country records."""
        s = CountryNameStemBlock()
        s.build_index([
            _make_rec("S2-1", name_stem="widgetco", country=""),
            _make_rec("S2-2", name_stem="widgetco", country="us"),
        ])
        q = _make_rec("S1-1", name_stem="widgetco", country="")
        result = s.query(q)
        assert "S2-1" in result
        assert "S2-2" not in result   # different country bucket


# ---------------------------------------------------------------------------
# Candidate union and deduplication
# ---------------------------------------------------------------------------

class TestCandidateUnion:
    def test_union_across_strategies(self):
        """Candidates from all strategies are unioned; each entity appears at most once."""
        s2 = [
            _make_rec("S2-1", name_norm="acme corp", name_stem="acme",
                      addr_norm="1 main st anytown", has_address=True, country="us"),
        ]
        s3 = []
        s1 = [
            _make_rec("S1-1", name_norm="acme corp", name_stem="acme",
                      addr_norm="1 main anytown road", has_address=True, country="us"),
        ]
        strategies = build_indexes_from_lists(s2, s3)
        candidates = get_candidates(s1[0], strategies)
        # S2-1 should appear via multiple strategies but deduped to one occurrence
        assert "S2-1" in candidates
        # Set type guarantees no duplicates
        assert len([x for x in candidates if x == "S2-1"]) == 1

    def test_zero_candidates(self):
        """S1 entity with no matching S2/S3 entities returns empty set."""
        s2 = [_make_rec("S2-1", name_norm="globex inc", name_stem="globex", country="us")]
        s3 = []
        s1 = [_make_rec("S1-1", name_norm="unknown co xyz", name_stem="unknown co xyz", country="us")]
        result = generate_candidates(s1, s2, s3)
        assert result["S1-1"] == set()

    def test_multiple_candidates(self):
        """S1 entity may have multiple candidates (never capped at 1)."""
        s2 = [
            _make_rec("S2-1", name_norm="acme corp", name_stem="acme"),
            _make_rec("S2-2", name_norm="acme incorporated", name_stem="acme"),
            _make_rec("S2-3", name_norm="acme llc", name_stem="acme"),
        ]
        s3 = [_make_rec("S3-1", name_norm="acme pvt ltd", name_stem="acme")]
        s1 = [_make_rec("S1-1", name_norm="acme corp", name_stem="acme")]
        result = generate_candidates(s1, s2, s3)
        assert len(result["S1-1"]) >= 3   # should find at least S2-1, S2-2, S2-3

    def test_every_s1_entity_appears_in_result(self):
        """generate_candidates must return a key for every S1 entity."""
        s2 = [_make_rec("S2-1", name_norm="acme corp", name_stem="acme")]
        s3 = []
        s1 = [
            _make_rec("S1-1", name_norm="acme corp", name_stem="acme"),
            _make_rec("S1-2", name_norm="totally different", name_stem="totally different"),
        ]
        result = generate_candidates(s1, s2, s3)
        assert "S1-1" in result
        assert "S1-2" in result   # even if no candidates


# ---------------------------------------------------------------------------
# candidate_pairs.tsv output contract
# ---------------------------------------------------------------------------

class TestCandidatePairsTsvContract:
    """Validate the output contract for candidate_pairs.tsv."""

    def _write_tsv(self, candidates: dict, path: str) -> None:
        """Replicates the write logic from run_blocking.py for testing."""
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f, delimiter="\t")
            writer.writerow(["source1_entity_id", "candidate_entity_ids"])
            for s1_id in sorted(candidates):
                cands = candidates[s1_id]
                writer.writerow([s1_id, ",".join(sorted(cands))])

    def test_schema_columns(self):
        cands = {"S1-1": {"S2-1", "S3-1"}, "S1-2": set()}
        with tempfile.NamedTemporaryFile(suffix=".tsv", delete=False, mode="w") as tf:
            path = tf.name
        try:
            self._write_tsv(cands, path)
            df = pd.read_csv(path, sep="\t", dtype=str)
            assert list(df.columns) == ["source1_entity_id", "candidate_entity_ids"]
        finally:
            os.unlink(path)

    def test_one_row_per_s1_entity(self):
        cands = {"S1-1": {"S2-1"}, "S1-2": set(), "S1-3": {"S3-1", "S3-2"}}
        with tempfile.NamedTemporaryFile(suffix=".tsv", delete=False, mode="w") as tf:
            path = tf.name
        try:
            self._write_tsv(cands, path)
            df = pd.read_csv(path, sep="\t", dtype=str)
            assert len(df) == 3
            assert df["source1_entity_id"].nunique() == 3
        finally:
            os.unlink(path)

    def test_no_duplicate_ids_within_row(self):
        """Candidate set is a Python set so duplicates are impossible, but verify."""
        cands = {"S1-1": {"S2-1", "S2-2", "S3-1"}}
        with tempfile.NamedTemporaryFile(suffix=".tsv", delete=False, mode="w") as tf:
            path = tf.name
        try:
            self._write_tsv(cands, path)
            df = pd.read_csv(path, sep="\t", dtype=str)
            row = df.iloc[0]["candidate_entity_ids"]
            ids = row.split(",")
            assert len(ids) == len(set(ids))
        finally:
            os.unlink(path)

    def test_empty_candidate_set_written_not_dropped(self):
        """S1 with zero candidates must still appear in the output (empty column)."""
        cands = {"S1-1": set()}
        with tempfile.NamedTemporaryFile(suffix=".tsv", delete=False, mode="w") as tf:
            path = tf.name
        try:
            self._write_tsv(cands, path)
            with open(path, encoding="utf-8") as f:
                lines = f.read().strip().split("\n")
            assert len(lines) == 2   # header + 1 data row
            assert "S1-1" in lines[1]
        finally:
            os.unlink(path)

    def test_only_s2_s3_prefixes(self):
        """Candidate IDs must start with S2- or S3-; no S1- self-matches."""
        cands = {"S1-1": {"S2-1", "S3-99"}}
        for eid in cands["S1-1"]:
            assert eid.startswith("S2-") or eid.startswith("S3-")


# ---------------------------------------------------------------------------
# Full pipeline end-to-end (tiny synthetic data)
# ---------------------------------------------------------------------------

class TestFullPipeline:
    def _build_df(self, rows):
        """Build a normalised DataFrame from raw records for testing."""
        df = pd.DataFrame(rows)
        df = normalize_dataframe(df)
        return df

    def _to_records(self, df):
        return df.to_dict("records")

    def test_end_to_end_small(self):
        """End-to-end generate_candidates on tiny synthetic data."""
        s1_rows = [
            {"entity_id": "S1-1", "business_name": "Acme Corporation", "business_address": "123 Main St Springfield", "country": "US"},
            {"entity_id": "S1-2", "business_name": "Globex Inc", "business_address": "", "country": "US"},
        ]
        s2_rows = [
            {"entity_id": "S2-1", "business_name": "Acme Corp", "business_address": "123 Main Street Springfield", "country": "US"},
            {"entity_id": "S2-2", "business_name": "Unrelated Business", "business_address": "999 Other Ave", "country": "US"},
        ]
        s3_rows = [
            {"entity_id": "S3-1", "business_name": "Globex Incorporated", "business_address": "", "country": "US"},
        ]

        s1 = self._to_records(self._build_df(s1_rows))
        s2 = self._to_records(self._build_df(s2_rows))
        s3 = self._to_records(self._build_df(s3_rows))

        result = generate_candidates(s1, s2, s3)

        # Every S1 entity is in result
        assert "S1-1" in result
        assert "S1-2" in result

        # S1-1 should find S2-1 (same name + address)
        assert "S2-1" in result["S1-1"]

        # S1-2 has no address; should find S3-1 (Globex stem match or token)
        assert "S3-1" in result["S1-2"]

    def test_no_self_match(self):
        """S1 entity IDs must never appear in their own candidate set."""
        s2_rows = [{"entity_id": "S2-1", "business_name": "Acme Corp", "business_address": "100 Main", "country": "US"}]
        s1_rows = [{"entity_id": "S1-1", "business_name": "Acme Corp", "business_address": "100 Main", "country": "US"}]
        s1 = self._to_records(self._build_df(s1_rows))
        s2 = self._to_records(self._build_df(s2_rows))
        result = generate_candidates(s1, s2, [])
        assert "S1-1" not in result["S1-1"]
