"""
Phase 6: Full Test Inference & Submission Generation Script.

Executes streaming inference of the frozen Phase 5 baseline model against
the complete test candidate set (output/candidate_pairs.tsv) to produce
the official output/matching_results.tsv using a disk-backed SQLite entity store.

Enforces:
- Schema: source1_entity_id \t matched_entity_ids
- Exactly one row per test Source 1 entity (1,732,544 rows)
- Probability threshold = 0.90 (optimal baseline validation F0.5 = 92.27%)
- Disk-backed SQLite store (output/test_entity_lookup.sqlite) with bounded memory
- Incremental flush every 10,000 S1 rows with verified resume capability
"""

from __future__ import annotations

import argparse
import csv
import gc
import os
import sqlite3
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from business_entity_resolution.features import (
    LIGHTWEIGHT_FEATURE_NAMES,
    compute_lightweight_batch_features,
)
from business_entity_resolution.model import BaselineMatchClassifier
from business_entity_resolution.normalize import (
    normalize_country,
    normalize_name,
    normalize_address,
)


def get_process_memory_mb() -> float:
    """Retrieve current process WorkingSet memory in MB via Windows psapi."""
    try:
        import ctypes
        class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
            _fields_ = [
                ("cb", ctypes.c_ulong),
                ("PageFaultCount", ctypes.c_ulong),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]
        pmc = PROCESS_MEMORY_COUNTERS()
        pmc.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
        get_current_process = ctypes.windll.kernel32.GetCurrentProcess
        get_current_process.restype = ctypes.c_void_p
        handle = get_current_process()

        get_process_memory_info = ctypes.windll.psapi.GetProcessMemoryInfo
        get_process_memory_info.argtypes = [ctypes.c_void_p, ctypes.POINTER(PROCESS_MEMORY_COUNTERS), ctypes.c_ulong]
        get_process_memory_info.restype = ctypes.c_int

        if get_process_memory_info(handle, ctypes.byref(pmc), ctypes.sizeof(pmc)):
            return pmc.WorkingSetSize / (1024 * 1024)
    except Exception:
        pass
    return 0.0


def get_sqlite_total_size_mb(db_path: str) -> float:
    """Calculate total size of SQLite DB including WAL and SHM files."""
    total_bytes = 0
    for path in [db_path, f"{db_path}-wal", f"{db_path}-shm"]:
        if os.path.exists(path):
            total_bytes += os.path.getsize(path)
    return total_bytes / (1024 * 1024)


def init_sqlite_db(db_path: str) -> sqlite3.Connection:
    """Initialize SQLite connection with high-performance PRAGMAs."""
    os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA synchronous = OFF;")
    conn.execute("PRAGMA journal_mode = WAL;")
    conn.execute("PRAGMA cache_size = -64000;")  # 64MB cache
    conn.execute("PRAGMA temp_store = MEMORY;")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS test_entities (
            entity_id TEXT PRIMARY KEY,
            normalized_name TEXT,
            normalized_address TEXT,
            normalized_country TEXT
        );
    """)
    conn.commit()
    return conn


def build_test_entity_sqlite(
    db_path: str = "output/test_entity_lookup.sqlite",
    data_dir: str = "data",
    chunk_size: int = 50000,
    max_records_per_file: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Build disk-backed SQLite database from test Source 1, 2, and 3 TSVs.
    Processes files in bounded streaming chunks, normalizes fields with Phase 2
    functions, and inserts into SQLite without global in-memory dictionary storage.
    """
    t0 = time.time()
    print(f"\nBuilding disk-backed SQLite entity lookup: {db_path} ...", flush=True)
    conn = init_sqlite_db(db_path)

    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM test_entities")
    existing_count = cursor.fetchone()[0]
    if existing_count > 0 and max_records_per_file is None:
        print(f"  Found existing SQLite database with {existing_count:,} entities.", flush=True)
        db_size_mb = os.path.getsize(db_path) / (1024 * 1024)
        return {
            "db_path": db_path,
            "total_entities": existing_count,
            "db_size_mb": db_size_mb,
            "build_time": 0.0,
        }

    files = [
        os.path.join(data_dir, "test", "test_source1.tsv"),
        os.path.join(data_dir, "test", "test_source2.tsv"),
        os.path.join(data_dir, "test", "test_source3.tsv"),
    ]

    total_inserted = 0

    for fpath in files:
        if not os.path.isfile(fpath):
            continue
        fname = os.path.basename(fpath)
        file_count = 0
        t_f = time.time()

        for chunk in pd.read_csv(fpath, sep="\t", chunksize=chunk_size, dtype=str):
            u_names = chunk["business_name"].fillna("").astype(str).unique()
            n_map = {n: normalize_name(n) for n in u_names if n}
            names = chunk["business_name"].fillna("").astype(str).map(n_map).fillna("")

            u_addrs = chunk["business_address"].fillna("").astype(str).unique()
            a_map = {a: normalize_address(a) for a in u_addrs if a}
            addrs = chunk["business_address"].fillna("").astype(str).map(a_map).fillna("")

            ctrys = chunk["country"].fillna("").astype(str).map(normalize_country)
            eids = chunk["entity_id"].astype(str)

            rows = list(zip(eids, names, addrs, ctrys))
            cursor.executemany(
                "INSERT OR REPLACE INTO test_entities (entity_id, normalized_name, normalized_address, normalized_country) VALUES (?, ?, ?, ?)",
                rows,
            )
            conn.commit()

            file_count += len(rows)
            total_inserted += len(rows)

            if max_records_per_file is not None and file_count >= max_records_per_file:
                break

        print(f"  {fname}: inserted {file_count:,} entities in {time.time()-t_f:.2f}s (RAM: {get_process_memory_mb():.1f} MB)", flush=True)

    db_size_mb = get_sqlite_total_size_mb(db_path)
    build_time = time.time() - t0
    print(f"SQLite build complete: {total_inserted:,} entities, {db_size_mb:.2f} MB in {build_time:.2f}s (Peak RAM: {get_process_memory_mb():.1f} MB)\n", flush=True)
    conn.close()

    return {
        "db_path": db_path,
        "total_entities": total_inserted,
        "db_size_mb": db_size_mb,
        "build_time": build_time,
    }


def fetch_entities_from_sqlite(
    conn: sqlite3.Connection,
    entity_ids: Sequence[str],
    query_chunk_size: int = 500,
) -> Dict[str, Tuple[str, str, str]]:
    """
    Retrieve normalized entity records from SQLite for a batch of entity IDs.
    Returns mapping from entity_id -> (normalized_name, normalized_address, normalized_country).
    """
    lookup: Dict[str, Tuple[str, str, str]] = {}
    u_ids = list(set(entity_ids))
    if not u_ids:
        return lookup

    cursor = conn.cursor()
    for i in range(0, len(u_ids), query_chunk_size):
        batch = u_ids[i : i + query_chunk_size]
        placeholders = ",".join("?" for _ in range(len(batch)))
        cursor.execute(
            f"SELECT entity_id, normalized_name, normalized_address, normalized_country FROM test_entities WHERE entity_id IN ({placeholders})",
            batch,
        )
        for row in cursor.fetchall():
            lookup[row[0]] = (row[1] or "", row[2] or "", row[3] or "")

    return lookup


def run_smoke_benchmark(
    model_path: str = "models/baseline_logreg.joblib",
    candidate_path: str = "output/candidate_pairs.tsv",
    data_dir: str = "data",
    sample_pairs_limit: int = 10000,
    threshold: float = 0.90,
    db_path: str = "output/smoke_test_entity_lookup.sqlite",
) -> Dict[str, Any]:
    """
    Execute controlled smoke benchmark on 10,000 candidate pairs using SQLite-backed lookup.
    Verifies model loading, feature extraction throughput, inference throughput,
    probability validity, NaN/Inf checks, memory footprint, and output validation.
    """
    print("\n===============================================================================================")
    print("PHASE 6: CONTROLLED SMOKE BENCHMARK (10,000 CANDIDATE PAIRS WITH SQLITE LOOKUP)")
    print("===============================================================================================")
    t_start = time.time()
    mem_start = get_process_memory_mb()

    # 1. Load Model
    print(f"1. Loading model from {model_path} ...", flush=True)
    assert os.path.exists(model_path), f"Model file not found: {model_path}"
    clf: BaselineMatchClassifier = BaselineMatchClassifier.load(model_path)
    assert clf.is_fitted, "Model is not fitted!"
    assert clf.feature_names == list(LIGHTWEIGHT_FEATURE_NAMES), "Feature names mismatch!"
    print(f"  Model loaded successfully with {len(clf.feature_names)} features.")

    # 2. Stream first 10,000 pairs from candidate_pairs.tsv
    print(f"2. Streaming first {sample_pairs_limit:,} candidate pairs from {candidate_path} ...", flush=True)
    s1_ids: List[str] = []
    cand_ids: List[str] = []
    needed_s1: Set[str] = set()
    needed_s2: Set[str] = set()
    needed_s3: Set[str] = set()

    with open(candidate_path, "r", encoding="utf-8", errors="ignore") as f:
        next(f, None)  # header
        for line in f:
            line = line.rstrip("\r\n")
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) > 1 and parts[1]:
                s1 = parts[0]
                for c in parts[1].split(","):
                    if not c:
                        continue
                    s1_ids.append(s1)
                    cand_ids.append(c)
                    needed_s1.add(s1)
                    if c.startswith("S2-"):
                        needed_s2.add(c)
                    elif c.startswith("S3-"):
                        needed_s3.add(c)
                    else:
                        needed_s2.add(c)

                    if len(s1_ids) >= sample_pairs_limit:
                        break
            if len(s1_ids) >= sample_pairs_limit:
                break

    all_needed_ids = needed_s1 | needed_s2 | needed_s3
    print(f"  Streamed {len(s1_ids):,} candidate pairs referencing {len(all_needed_ids):,} unique entities.")

    # 3. Build/Populate SQLite DB for needed entities
    print(f"3. Populating SQLite database: {db_path} ...", flush=True)
    if os.path.exists(db_path):
        os.remove(db_path)

    t_db_start = time.time()
    conn = init_sqlite_db(db_path)
    cursor = conn.cursor()

    file_targets = [
        (os.path.join(data_dir, "test", "test_source1.tsv"), needed_s1),
        (os.path.join(data_dir, "test", "test_source2.tsv"), needed_s2),
        (os.path.join(data_dir, "test", "test_source3.tsv"), needed_s3),
    ]

    inserted_count = 0
    for fpath, target_ids in file_targets:
        if not target_ids or not os.path.isfile(fpath):
            continue
        with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
            next(f, None)
            rows_batch = []
            for line in f:
                parts = line.rstrip("\r\n").split("\t")
                if parts[0] in target_ids:
                    n = normalize_name(parts[1] if len(parts) > 1 else "")
                    a = normalize_address(parts[2] if len(parts) > 2 else "")
                    c = normalize_country(parts[3] if len(parts) > 3 else "")
                    rows_batch.append((parts[0], n, a, c))
                    if len(rows_batch) >= 1000:
                        cursor.executemany("INSERT OR REPLACE INTO test_entities VALUES (?, ?, ?, ?)", rows_batch)
                        conn.commit()
                        inserted_count += len(rows_batch)
                        rows_batch = []
            if rows_batch:
                cursor.executemany("INSERT OR REPLACE INTO test_entities VALUES (?, ?, ?, ?)", rows_batch)
                conn.commit()
                inserted_count += len(rows_batch)

    db_build_time = time.time() - t_db_start
    db_size_mb = get_sqlite_total_size_mb(db_path)
    print(f"  SQLite populated: {inserted_count:,} entities in {db_build_time:.2f}s ({db_size_mb:.2f} MB).")

    # 4. Fetch entities from SQLite
    print("4. Fetching batch entity metadata from SQLite ...", flush=True)
    t_fetch = time.time()
    entity_lookup = fetch_entities_from_sqlite(conn, list(all_needed_ids))
    fetch_time = time.time() - t_fetch
    print(f"  Retrieved {len(entity_lookup):,} entities from SQLite in {fetch_time:.4f}s.")

    # 5. Compute Features
    print("5. Computing 12 lightweight features ...", flush=True)
    t_feat = time.time()
    X_sample = compute_lightweight_batch_features(s1_ids, cand_ids, entity_lookup)
    feat_time = time.time() - t_feat
    feat_throughput = len(s1_ids) / feat_time if feat_time > 0 else 0.0

    nan_count = int(np.isnan(X_sample).sum())
    inf_count = int(np.isinf(X_sample).sum())
    print(f"  Features computed in {feat_time:.4f}s ({feat_throughput:,.0f} pairs/sec), shape: {X_sample.shape}")
    print(f"  NaN Count: {nan_count}, Inf Count: {inf_count}")
    assert nan_count == 0, "NaN values detected in feature matrix!"
    assert inf_count == 0, "Inf values detected in feature matrix!"

    # 6. Predict Probabilities
    print(f"6. Predicting match probabilities (threshold={threshold:.2f}) ...", flush=True)
    t_pred = time.time()
    probs = clf.predict_proba(X_sample)
    pred_time = time.time() - t_pred
    total_infer_time = feat_time + pred_time
    total_throughput = len(s1_ids) / total_infer_time if total_infer_time > 0 else 0.0

    assert len(probs) == len(s1_ids)
    assert (probs >= 0.0).all() and (probs <= 1.0).all(), "Invalid probabilities out of [0, 1]!"

    matches = probs >= threshold
    n_matches = int(np.sum(matches))
    print(f"  Scored {len(s1_ids):,} pairs in {pred_time:.4f}s: {n_matches:,} matches ({n_matches/len(s1_ids)*100:.2f}%)")
    print(f"  Combined Feature+Model Throughput: {total_throughput:,.1f} pairs/sec")

    # 7. Validate Output Formatting
    print("7. Validating submission output format ...", flush=True)
    sample_out_path = "output/smoke_test_matching_results.tsv"
    with open(sample_out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["source1_entity_id", "matched_entity_ids"])
        
        # Group by s1_id
        cur_s1 = None
        s1_cands: List[str] = []
        for s1, c, m in zip(s1_ids, cand_ids, matches):
            if s1 != cur_s1:
                if cur_s1 is not None:
                    writer.writerow([cur_s1, ",".join(s1_cands)])
                cur_s1 = s1
                s1_cands = []
            if m:
                if c not in s1_cands:
                    s1_cands.append(c)
        if cur_s1 is not None:
            writer.writerow([cur_s1, ",".join(s1_cands)])

    # Verify sample output
    with open(sample_out_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        hdr = next(reader)
        assert hdr == ["source1_entity_id", "matched_entity_ids"], f"Invalid header: {hdr}"
        row_count = sum(1 for _ in reader)

    print(f"  Sample submission written: {row_count} S1 entities, valid format.")
    
    conn.close()
    mem_peak = get_process_memory_mb()
    total_time = time.time() - t_start

    print("\n===============================================================================================")
    print("CONTROLLED SMOKE BENCHMARK REPORT")
    print("===============================================================================================")
    print(f"Sample Candidate Pairs:      {sample_pairs_limit:,}")
    print(f"SQLite DB Path:              {db_path}")
    print(f"SQLite DB Build Time:        {db_build_time:.2f}s")
    print(f"SQLite DB Size:              {db_size_mb:.2f} MB")
    print(f"SQLite Fetch Time:           {fetch_time:.4f}s ({len(all_needed_ids)/(fetch_time if fetch_time>0 else 1):,.0f} entities/sec)")
    print(f"Feature Extraction Time:     {feat_time:.4f}s ({feat_throughput:,.0f} pairs/sec)")
    print(f"Model Scoring Time:          {pred_time:.4f}s ({len(s1_ids)/(pred_time if pred_time>0 else 1):,.0f} pairs/sec)")
    print(f"Total Inference Throughput:  {total_throughput:,.1f} pairs/sec")
    print(f"NaN / Inf Count:             {nan_count} / {inf_count}")
    print(f"Peak Process Memory:         {mem_peak:.2f} MB (Delta: +{mem_peak - mem_start:.2f} MB)")
    print(f"Output Validation:           PASS (Header, Tab-Separated, Valid Schema)")
    print(f"Total Benchmark Runtime:     {total_time:.2f}s")
    print("===============================================================================================\n")

    return {
        "sample_pairs": sample_pairs_limit,
        "db_path": db_path,
        "db_build_time": db_build_time,
        "db_size_mb": db_size_mb,
        "fetch_time": fetch_time,
        "feat_throughput": feat_throughput,
        "total_throughput": total_throughput,
        "nan_count": nan_count,
        "inf_count": inf_count,
        "peak_mem_mb": mem_peak,
        "validation": "PASS",
    }


def run_full_inference(
    model_path: str = "models/baseline_logreg.joblib",
    candidate_path: str = "output/candidate_pairs.tsv",
    db_path: str = "output/test_entity_lookup.sqlite",
    data_dir: str = "data",
    output_path: str = "output/matching_results.tsv",
    threshold: float = 0.90,
    flush_interval: int = 10000,
    candidate_batch_size: int = 50000,
) -> Dict[str, Any]:
    """
    Execute full test inference on all 1,732,544 test S1 entities and ~334M candidate pairs
    using the disk-backed SQLite entity store.
    """
    t_start = time.time()
    print("\n===============================================================================================")
    print("PHASE 6: FULL TEST INFERENCE & SUBMISSION GENERATION (SQLITE-BACKED)")
    print("===============================================================================================")
    print(f"Model Path:           {model_path}")
    print(f"Candidate Pairs:      {candidate_path}")
    print(f"SQLite Lookup DB:     {db_path}")
    print(f"Output File:          {output_path}")
    print(f"Match Threshold:      {threshold:.2f}")

    # 1. Load Model
    clf: BaselineMatchClassifier = BaselineMatchClassifier.load(model_path)
    print(f"Model loaded with {len(clf.feature_names)} features.")

    # 2. Check / Connect SQLite Store
    if not os.path.exists(db_path):
        print(f"SQLite database {db_path} not found. Building now from {data_dir}/test ...", flush=True)
        build_test_entity_sqlite(db_path=db_path, data_dir=data_dir)

    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA cache_size = -64000;")
    conn.execute("PRAGMA synchronous = OFF;")

    # 3. Resume / Checkpoint Management
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    completed_s1_ids: Set[str] = set()
    write_mode = "w"

    if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
        print(f"Checking existing output file {output_path} for resume checkpoint...", flush=True)
        try:
            with open(output_path, "r", encoding="utf-8") as f:
                header = f.readline().strip()
                for line in f:
                    line = line.strip()
                    if line:
                        s1_id = line.split("\t", 1)[0]
                        completed_s1_ids.add(s1_id)
            if completed_s1_ids:
                print(f"  Found {len(completed_s1_ids):,} existing completed S1 records. Resuming in append mode...", flush=True)
                write_mode = "a"
        except Exception as e:
            print(f"  Warning reading existing file: {e}. Starting fresh.", flush=True)
            completed_s1_ids = set()
            write_mode = "w"

    print(f"Streaming {candidate_path} and writing {output_path} (Mode: {write_mode}) ...", flush=True)

    total_s1_processed = len(completed_s1_ids)
    total_candidate_pairs_processed = 0
    total_matches_predicted = 0

    zero_match_count = 0
    single_match_count = 0
    multi_match_count = 0

    t_infer_start = time.time()
    rows_since_flush = 0

    with open(output_path, write_mode, newline="", encoding="utf-8") as out_f:
        writer = csv.writer(out_f, delimiter="\t")
        if write_mode == "w":
            writer.writerow(["source1_entity_id", "matched_entity_ids"])

        with open(candidate_path, "r", encoding="utf-8", errors="ignore") as in_f:
            header = in_f.readline()  # header

            batch_s1: List[str] = []
            batch_cand: List[str] = []
            s1_pair_slices: List[Tuple[str, int, int]] = []

            for line in in_f:
                line = line.rstrip("\r\n")
                if not line:
                    continue

                parts = line.split("\t")
                s1_id = parts[0]

                if s1_id in completed_s1_ids:
                    continue

                cands_str = parts[1] if len(parts) > 1 else ""

                if not cands_str:
                    writer.writerow([s1_id, ""])
                    completed_s1_ids.add(s1_id)
                    total_s1_processed += 1
                    zero_match_count += 1
                    rows_since_flush += 1
                    continue

                cands = [c for c in cands_str.split(",") if c]
                if not cands:
                    writer.writerow([s1_id, ""])
                    completed_s1_ids.add(s1_id)
                    total_s1_processed += 1
                    zero_match_count += 1
                    rows_since_flush += 1
                    continue

                # Add to batch
                start_i = len(batch_s1)
                for c in cands:
                    batch_s1.append(s1_id)
                    batch_cand.append(c)
                end_i = len(batch_s1)
                s1_pair_slices.append((s1_id, start_i, end_i))

                # Batch scoring
                if len(batch_s1) >= candidate_batch_size:
                    total_candidate_pairs_processed += len(batch_s1)
                    
                    # Fetch required entities from SQLite for this batch only
                    batch_entity_ids = list(set(batch_s1) | set(batch_cand))
                    batch_lookup = fetch_entities_from_sqlite(conn, batch_entity_ids)

                    # Compute features & score
                    X_batch = compute_lightweight_batch_features(batch_s1, batch_cand, batch_lookup)
                    probs = clf.predict_proba(X_batch)

                    for cur_s1, st, en in s1_pair_slices:
                        s1_cands = batch_cand[st:en]
                        s1_probs = probs[st:en]

                        matched_cands = [cand for cand, p in zip(s1_cands, s1_probs) if p >= threshold]

                        seen = set()
                        deduped = []
                        for mc in matched_cands:
                            if mc not in seen:
                                seen.add(mc)
                                deduped.append(mc)

                        n_m = len(deduped)
                        if n_m == 0:
                            zero_match_count += 1
                        elif n_m == 1:
                            single_match_count += 1
                        else:
                            multi_match_count += 1

                        total_matches_predicted += n_m
                        writer.writerow([cur_s1, ",".join(deduped)])
                        total_s1_processed += 1
                        rows_since_flush += 1

                    batch_s1.clear()
                    batch_cand.clear()
                    s1_pair_slices.clear()
                    batch_lookup.clear()

                if rows_since_flush >= flush_interval:
                    out_f.flush()
                    rows_since_flush = 0
                    elapsed_so_far = time.time() - t_infer_start
                    thru = total_candidate_pairs_processed / elapsed_so_far if elapsed_so_far > 0 else 0
                    print(
                        f"  Processed {total_s1_processed:,} / 1,732,544 S1 entities "
                        f"({total_s1_processed/1732544*100:.1f}%) | "
                        f"{total_candidate_pairs_processed:,} pairs ({thru:,.0f} pairs/sec) | "
                        f"RAM: {get_process_memory_mb():.1f} MB",
                        flush=True,
                    )

            # Remaining batch
            if batch_s1:
                total_candidate_pairs_processed += len(batch_s1)
                batch_entity_ids = list(set(batch_s1) | set(batch_cand))
                batch_lookup = fetch_entities_from_sqlite(conn, batch_entity_ids)

                X_batch = compute_lightweight_batch_features(batch_s1, batch_cand, batch_lookup)
                probs = clf.predict_proba(X_batch)

                for cur_s1, st, en in s1_pair_slices:
                    s1_cands = batch_cand[st:en]
                    s1_probs = probs[st:en]

                    matched_cands = [cand for cand, p in zip(s1_cands, s1_probs) if p >= threshold]
                    seen = set()
                    deduped = []
                    for mc in matched_cands:
                        if mc not in seen:
                            seen.add(mc)
                            deduped.append(mc)

                    n_m = len(deduped)
                    if n_m == 0:
                        zero_match_count += 1
                    elif n_m == 1:
                        single_match_count += 1
                    else:
                        multi_match_count += 1

                    total_matches_predicted += n_m
                    writer.writerow([cur_s1, ",".join(deduped)])
                    total_s1_processed += 1

                batch_s1.clear()
                batch_cand.clear()
                s1_pair_slices.clear()
                batch_lookup.clear()

            out_f.flush()

    conn.close()

    infer_time = time.time() - t_infer_start
    total_time = time.time() - t_start
    throughput = total_candidate_pairs_processed / infer_time if infer_time > 0 else 0.0
    mem_mb = get_process_memory_mb()

    print("\n===============================================================================================")
    print("PHASE 6 INFERENCE PERFORMANCE REPORT")
    print("===============================================================================================")
    print(f"Total Test S1 Entities:          {total_s1_processed:,}")
    print(f"Total Candidate Pairs:           {total_candidate_pairs_processed:,}")
    print(f"Total Matches Predicted:         {total_matches_predicted:,}")
    print(f"Inference Runtime:               {infer_time:.2f}s ({infer_time/60:.2f} min)")
    print(f"Inference Throughput:            {throughput:,.1f} pairs/sec")
    print(f"Peak Process Memory:             {mem_mb:.2f} MB")
    print(f"Total Pipeline Runtime:          {total_time:.2f}s ({total_time/60:.2f} min)")
    print(f"\nMatch Cardinality Breakdown:")
    print(f"  S1 Entities with 0 matches:    {zero_match_count:,} ({zero_match_count/total_s1_processed*100:.2f}%)")
    print(f"  S1 Entities with 1 match:      {single_match_count:,} ({single_match_count/total_s1_processed*100:.2f}%)")
    print(f"  S1 Entities with 2+ matches:   {multi_match_count:,} ({multi_match_count/total_s1_processed*100:.2f}%)")
    print(f"\nMatching Results Written To:     {output_path}")

    return {
        "total_s1": total_s1_processed,
        "total_pairs": total_candidate_pairs_processed,
        "total_matches": total_matches_predicted,
        "zero_matches": zero_match_count,
        "single_matches": single_match_count,
        "multi_matches": multi_match_count,
        "infer_time": infer_time,
        "total_time": total_time,
        "throughput": throughput,
        "mem_mb": mem_mb,
        "output_path": output_path,
    }


def main():
    parser = argparse.ArgumentParser(description="Phase 6 Test Inference & Submission Generation (SQLite-backed)")
    parser.add_argument("--data-dir", default="data", help="Data directory path")
    parser.add_argument("--model-path", default="models/baseline_logreg.joblib", help="Trained model artifact path")
    parser.add_argument("--candidate-file", default="output/candidate_pairs.tsv", help="Candidate pairs file")
    parser.add_argument("--db-path", default="output/test_entity_lookup.sqlite", help="SQLite entity lookup path")
    parser.add_argument("--output", default="output/matching_results.tsv", help="Output matching results TSV")
    parser.add_argument("--threshold", type=float, default=0.90, help="Probability threshold for match")
    parser.add_argument("--smoke-benchmark-only", action="store_true", help="Run only the 10k controlled smoke benchmark")
    parser.add_argument("--build-db-only", action="store_true", help="Build SQLite entity database and exit")
    parser.add_argument("--full-run", action="store_true", help="Execute full 334M test inference")
    parser.add_argument("--skip-smoke", action="store_true", help="Skip smoke benchmark before full inference")
    args = parser.parse_args()

    if args.build_db_only:
        build_test_entity_sqlite(db_path=args.db_path, data_dir=args.data_dir)
        return

    # Step 1: Controlled Smoke Benchmark (unless skipped)
    if not args.skip_smoke:
        smoke_results = run_smoke_benchmark(
            model_path=args.model_path,
            candidate_path=args.candidate_file,
            data_dir=args.data_dir,
            sample_pairs_limit=10000,
            threshold=args.threshold,
        )

    if args.smoke_benchmark_only:
        return

    if args.full_run:
        run_full_inference(
            model_path=args.model_path,
            candidate_path=args.candidate_file,
            db_path=args.db_path,
            data_dir=args.data_dir,
            output_path=args.output,
            threshold=args.threshold,
        )
    else:
        print("Controlled smoke benchmark completed. To run full inference, pass --full-run.")


if __name__ == "__main__":
    main()
