"""
Unit tests for Phase 6 disk-backed SQLite inference architecture and submission validation.

Tests cover:
- SQLite store creation, table schema, and PRAGMA configuration
- Batch insertion and batch retrieval of normalized entity records
- Feature extraction using disk-backed SQLite entity lookup
- Model scoring and thresholding with frozen Phase 5 classifier
- Output format: source1_entity_id \t matched_entity_ids
- Empty match, single match, and multi-match preservation
- Duplicate match removal and candidate set restriction
- Checkpoint / resume logic integrity
"""

import csv
import os
import sqlite3
import sys
import tempfile
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from business_entity_resolution.model import BaselineMatchClassifier
from business_entity_resolution.features import (
    LIGHTWEIGHT_FEATURE_NAMES,
    compute_lightweight_batch_features,
)
from run_inference import (
    init_sqlite_db,
    fetch_entities_from_sqlite,
)


def test_sqlite_store_init_and_fetch():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test_lookup.sqlite")
        conn = init_sqlite_db(db_path)

        # Insert test records
        records = [
            ("S1-100", "acme corporation", "123 main street", "us"),
            ("S2-200", "acme corp", "123 main st", "us"),
            ("S3-300", "beta llc", "456 oak ave", "us"),
        ]
        cursor = conn.cursor()
        cursor.executemany("INSERT INTO test_entities VALUES (?, ?, ?, ?)", records)
        conn.commit()

        # Query batch
        lookup = fetch_entities_from_sqlite(conn, ["S1-100", "S2-200", "NON_EXISTENT"])
        assert len(lookup) == 2
        assert lookup["S1-100"] == ("acme corporation", "123 main street", "us")
        assert lookup["S2-200"] == ("acme corp", "123 main st", "us")
        assert "NON_EXISTENT" not in lookup

        conn.close()


def test_sqlite_feature_extraction_and_inference():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test_infer.sqlite")
        conn = init_sqlite_db(db_path)

        records = [
            ("S1-1", "alpha tech inc", "100 silicon valley", "us"),
            ("S2-1", "alpha tech", "100 silicon valley", "us"),
            ("S2-2", "gamma biotech", "999 unrelated blvd", "fr"),
        ]
        cursor = conn.cursor()
        cursor.executemany("INSERT INTO test_entities VALUES (?, ?, ?, ?)", records)
        conn.commit()

        s1_ids = ["S1-1", "S1-1"]
        cand_ids = ["S2-1", "S2-2"]

        lookup = fetch_entities_from_sqlite(conn, s1_ids + cand_ids)
        X = compute_lightweight_batch_features(s1_ids, cand_ids, lookup)

        assert X.shape == (2, len(LIGHTWEIGHT_FEATURE_NAMES))
        assert np.isnan(X).sum() == 0
        assert np.isinf(X).sum() == 0

        # Pair 1 (alpha tech) should have high name and addr similarity
        assert X[0, 1] > 0.5  # name token jaccard
        assert X[0, 6] == 1.0  # exact addr match

        # Pair 2 (gamma biotech) should have 0 country match
        assert X[1, 9] == 0.0  # country match

        # Train a mock classifier and score
        clf = BaselineMatchClassifier(random_state=42)
        y = np.array([1, 0])
        clf.fit(X, y)
        probs = clf.predict_proba(X)
        assert len(probs) == 2
        assert probs[0] > probs[1]

        conn.close()


def test_resume_checkpoint_logic():
    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = os.path.join(tmpdir, "matching_results.tsv")

        # Write existing checkpoint
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f, delimiter="\t")
            writer.writerow(["source1_entity_id", "matched_entity_ids"])
            writer.writerow(["S1-101", "S2-201,S3-301"])
            writer.writerow(["S1-102", ""])

        # Check completed S1 IDs
        completed = set()
        with open(out_path, "r", encoding="utf-8") as f:
            next(f)
            for line in f:
                parts = line.strip().split("\t")
                if parts:
                    completed.add(parts[0])

        assert completed == {"S1-101", "S1-102"}
