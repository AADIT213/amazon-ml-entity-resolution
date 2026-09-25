"""
Candidate Generation / Blocking - Phase 3.

Lightweight, scalable inverted-index blocking strategies union-ed per S1 entity:
  1. ExactNormalizedNameBlock   - exact normalized name match
  2. NameStemBlock              - legal-suffix-stripped stem match
  3. NameTokenBlock             - inverted token index on name (selective / thresholded)
  4. AddressTokenBlock          - address-token index with null-address fallback to name stem
  5. CountryNameStemBlock       - (country, name_stem) bucket, fully open-set country
  6. CountryExactNameBlock      - (country, normalized_name) bucket

Design guarantees:
- Fast & lightweight memory footprint. No O(n^2) cross-join.
- Open-set country: country value derived dynamically from data, never hard-coded.
- Null-address fallback: entities with no address fall back to name stem indexing.
- Candidate counts never capped arbitrarily; empty candidate sets written.
- Modular strategies can be added, tested, or swapped independently.
"""
from __future__ import annotations

import re
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Set

from .normalize import normalize_country
from .name_norm import normalize_name, strip_legal_suffix
from .address_norm import normalize_address


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _name_tokens(text: str) -> List[str]:
    """Whitespace-split tokens; drop single-char tokens."""
    return [t for t in text.split() if len(t) > 1]


def _char_bigrams(text: str) -> List[str]:
    """Overlapping character bigrams (spaces collapsed)."""
    clean = re.sub(r"\s+", "", text)
    return [clean[i : i + 2] for i in range(len(clean) - 1)]


_ADDR_SKIP = {
    "st", "rd", "dr", "ave", "blvd", "ln", "ct", "cir", "pl",
    "n", "s", "e", "w", "no", "fl", "apt", "ste", "rm",
    "bldg", "po", "box", "ne", "nw", "se", "sw",
}


def _addr_tokens(text: str) -> List[str]:
    """Address tokens with common short stopwords removed."""
    return [t for t in text.split() if len(t) > 2 and t not in _ADDR_SKIP]


def _norm_country(country: Optional[str]) -> str:
    """Normalize country for use as a bucket key (open-set, no enum)."""
    return normalize_country(country).lower().strip()


# ---------------------------------------------------------------------------
# Strategy 1: Exact normalized name
# ---------------------------------------------------------------------------

class ExactNormalizedNameBlock:
    """Exact match on the full normalized business name."""

    name = "exact_normalized_name"

    def __init__(self) -> None:
        self._index: Dict[str, Set[str]] = defaultdict(set)

    def build_index(self, records: Iterable[Dict[str, Any]]) -> None:
        for rec in records:
            key = rec.get("normalized_name", "")
            if key:
                self._index[key].add(rec["entity_id"])

    def query(self, record: Dict[str, Any]) -> Set[str]:
        key = record.get("normalized_name", "")
        if not key:
            return set()
        return set(self._index.get(key, set()))


# ---------------------------------------------------------------------------
# Strategy 2: Name stem (legal suffix stripped)
# ---------------------------------------------------------------------------

class NameStemBlock:
    """Exact match on the legal-suffix-stripped name stem.
    Catches 'Acme Inc' vs 'Acme LLC' -- same core stem, different suffix."""

    name = "name_stem"
    MIN_STEM_LEN = 4

    def __init__(self) -> None:
        self._index: Dict[str, Set[str]] = defaultdict(set)

    def build_index(self, records: Iterable[Dict[str, Any]]) -> None:
        for rec in records:
            key = rec.get("name_stem", "")
            if key and len(key) >= self.MIN_STEM_LEN:
                self._index[key].add(rec["entity_id"])

    def query(self, record: Dict[str, Any]) -> Set[str]:
        key = record.get("name_stem", "")
        if not key or len(key) < self.MIN_STEM_LEN:
            return set()
        return set(self._index.get(key, set()))


# ---------------------------------------------------------------------------
# Strategy 3: Name token overlap
# ---------------------------------------------------------------------------

class NameTokenBlock:
    """Inverted token index on normalized name."""

    name = "name_token"

    def __init__(self, min_shared_tokens: int = 2) -> None:
        self._min = min_shared_tokens
        self._index: Dict[str, Set[str]] = defaultdict(set)

    def build_index(self, records: Iterable[Dict[str, Any]]) -> None:
        for rec in records:
            tokens = _name_tokens(rec.get("normalized_name", ""))
            eid = rec["entity_id"]
            for tok in set(tokens):
                self._index[tok].add(eid)

    def query(self, record: Dict[str, Any]) -> Set[str]:
        tokens = _name_tokens(record.get("normalized_name", ""))
        if not tokens:
            return set()
        if len(tokens) < self._min:
            return set(self._index.get(tokens[0], set()))
        counter: Dict[str, int] = defaultdict(int)
        for tok in set(tokens):
            for eid in self._index.get(tok, set()):
                counter[eid] += 1
        return {eid for eid, cnt in counter.items() if cnt >= self._min}


# ---------------------------------------------------------------------------
# Strategy 4: Address token overlap with null-address fallback
# ---------------------------------------------------------------------------

class AddressTokenBlock:
    """Inverted address-token index.
    Null-address S2/S3 records are indexed under name stem (fallback bucket).
    S1 records with no address query the fallback bucket by name stem."""

    name = "address_token"

    def __init__(self, min_shared_tokens: int = 3) -> None:
        self._min = min_shared_tokens
        self._addr_index: Dict[str, Set[str]] = defaultdict(set)
        self._stem_fallback: Dict[str, Set[str]] = defaultdict(set)

    def build_index(self, records: Iterable[Dict[str, Any]]) -> None:
        for rec in records:
            eid = rec["entity_id"]
            if rec.get("has_address", False):
                for tok in set(_addr_tokens(rec.get("normalized_address", ""))):
                    self._addr_index[tok].add(eid)
            else:
                stem = rec.get("name_stem", "")
                if stem:
                    self._stem_fallback[stem].add(eid)

    def query(self, record: Dict[str, Any]) -> Set[str]:
        if not record.get("has_address", False):
            stem = record.get("name_stem", "")
            return set(self._stem_fallback.get(stem, set())) if stem else set()
        tokens = _addr_tokens(record.get("normalized_address", ""))
        if len(tokens) < self._min:
            return set()
        counter: Dict[str, int] = defaultdict(int)
        for tok in tokens:
            for eid in self._addr_index.get(tok, set()):
                counter[eid] += 1
        return {eid for eid, cnt in counter.items() if cnt >= self._min}


# ---------------------------------------------------------------------------
# Strategy 5: Country x name-stem bucket (open-set)
# ---------------------------------------------------------------------------

class CountryNameStemBlock:
    """(country, name_stem) bucket for high-precision candidates.
    Country is dynamically derived from data -- fully open-set (US, India, France, etc.)."""

    name = "country_name_stem"
    MIN_STEM_LEN = 4

    def __init__(self) -> None:
        self._index: Dict[tuple, Set[str]] = defaultdict(set)

    def build_index(self, records: Iterable[Dict[str, Any]]) -> None:
        for rec in records:
            stem = rec.get("name_stem", "")
            if not stem or len(stem) < self.MIN_STEM_LEN:
                continue
            country = _norm_country(rec.get("normalized_country", ""))
            self._index[(country, stem)].add(rec["entity_id"])

    def query(self, record: Dict[str, Any]) -> Set[str]:
        stem = record.get("name_stem", "")
        if not stem or len(stem) < self.MIN_STEM_LEN:
            return set()
        country = _norm_country(record.get("normalized_country", ""))
        return set(self._index.get((country, stem), set()))


# ---------------------------------------------------------------------------
# Strategy 6: Country x exact normalized name (open-set)
# ---------------------------------------------------------------------------

class CountryExactNameBlock:
    """(country, normalized_name) bucket for precise country-aware matching."""

    name = "country_exact_name"

    def __init__(self) -> None:
        self._index: Dict[tuple, Set[str]] = defaultdict(set)

    def build_index(self, records: Iterable[Dict[str, Any]]) -> None:
        for rec in records:
            name_norm = rec.get("normalized_name", "")
            if not name_norm:
                continue
            country = _norm_country(rec.get("normalized_country", ""))
            self._index[(country, name_norm)].add(rec["entity_id"])

    def query(self, record: Dict[str, Any]) -> Set[str]:
        name_norm = record.get("normalized_name", "")
        if not name_norm:
            return set()
        country = _norm_country(record.get("normalized_country", ""))
        return set(self._index.get((country, name_norm), set()))


# ---------------------------------------------------------------------------
# Engine / Pipeline interface
# ---------------------------------------------------------------------------

def build_default_strategies() -> List[Any]:
    """Return default strategy instances."""
    return [
        ExactNormalizedNameBlock(),
        NameStemBlock(),
        NameTokenBlock(min_shared_tokens=2),
        AddressTokenBlock(min_shared_tokens=3),
        CountryNameStemBlock(),
        CountryExactNameBlock(),
    ]


def build_indexes_from_lists(
    s2_records: List[Dict[str, Any]],
    s3_records: List[Dict[str, Any]],
    strategies: Optional[List[Any]] = None,
) -> List[Any]:
    if strategies is None:
        strategies = build_default_strategies()
    all_records = s2_records + s3_records
    for strategy in strategies:
        strategy.build_index(all_records)
    return strategies


def get_candidates(
    s1_record: Dict[str, Any],
    strategies: List[Any],
) -> Set[str]:
    candidates: Set[str] = set()
    for strategy in strategies:
        candidates |= strategy.query(s1_record)
    return candidates


def generate_candidates(
    s1_records: List[Dict[str, Any]],
    s2_records: List[Dict[str, Any]],
    s3_records: List[Dict[str, Any]],
    strategies: Optional[List[Any]] = None,
) -> Dict[str, Set[str]]:
    built = build_indexes_from_lists(s2_records, s3_records, strategies)
    result: Dict[str, Set[str]] = {}
    for rec in s1_records:
        eid = rec["entity_id"]
        result[eid] = get_candidates(rec, built)
    return result

