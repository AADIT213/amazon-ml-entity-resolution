"""
Phase 4: Lightweight Sample Feature Computation (10,000 Pairs).

Executes fast streaming feature extraction on the first 10,000 candidate pairs:
1. Streams candidate_pairs.tsv until exactly 10,000 pairs are collected
2. Reads only the required entity records from test_source1/2/3.tsv
3. Precomputes normalized entity attributes
4. Computes 12 lightweight features:
   - exact_name_match, name_token_jaccard, name_char_similarity, name_len_diff, name_token_count_diff
   - exact_addr_match, addr_token_jaccard, addr_numeric_overlap, addr_len_diff
   - country_match, missing_name_flag, missing_addr_flag
5. Saves output to output/features_sample_10000.tsv
6. Evaluates data integrity (NaN/Inf counts, value ranges, memory footprint, runtime)
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from business_entity_resolution.features import (
    LIGHTWEIGHT_FEATURE_NAMES,
    compute_lightweight_batch_features,
)
from business_entity_resolution.normalize import (
    normalize_country,
    normalize_name,
    normalize_address,
)


def load_needed_entities_streaming(
    data_dir: str,
    needed_s1: Set[str],
    needed_s2: Set[str],
    needed_s3: Set[str],
) -> Dict[str, Tuple[str, str, str]]:
    """
    Stream source files and load normalized representations for only the needed entity IDs.
    Early-stops reading each file as soon as all required IDs for that source are found.
    """
    lookup: Dict[str, Tuple[str, str, str]] = {}
    
    file_targets = [
        (os.path.join(data_dir, "test", "test_source1.tsv"), needed_s1),
        (os.path.join(data_dir, "test", "test_source2.tsv"), needed_s2),
        (os.path.join(data_dir, "test", "test_source3.tsv"), needed_s3),
    ]

    for fpath, target_ids in file_targets:
        if not target_ids or not os.path.isfile(fpath):
            continue
        
        fname = os.path.basename(fpath)
        found_in_file = 0
        total_in_target = len(target_ids)
        t0 = time.time()

        with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
            header = f.readline()
            for line in f:
                line = line.rstrip("\r\n")
                if not line:
                    continue
                parts = line.split("\t")
                eid = parts[0]
                if eid in target_ids:
                    raw_name = parts[1] if len(parts) > 1 else ""
                    raw_addr = parts[2] if len(parts) > 2 else ""
                    raw_ctry = parts[3] if len(parts) > 3 else ""

                    norm_n = normalize_name(raw_name)
                    norm_a = normalize_address(raw_addr)
                    norm_c = normalize_country(raw_ctry)

                    lookup[eid] = (norm_n, norm_a, norm_c)
                    found_in_file += 1

                    if found_in_file >= total_in_target:
                        break

        print(f"  {fname}: loaded {found_in_file:,} / {total_in_target:,} needed entities in {time.time()-t0:.2f}s", flush=True)

    return lookup


def run_sample_feature_generation(
    data_dir: str = "data",
    candidate_path: str = "output/candidate_pairs.tsv",
    output_sample_path: str = "output/features_sample_10000.tsv",
    sample_limit: int = 10000,
) -> None:
    t_start = time.time()
    print("========================================================")
    print(f"PHASE 4: LIGHTWEIGHT FEATURE BENCHMARK ({sample_limit:,} PAIRS)")
    print("========================================================")

    # 1. Stream candidate_pairs.tsv until sample_limit pairs are gathered
    print(f"\n1. Streaming first {sample_limit:,} candidate pairs from {candidate_path} ...", flush=True)
    t_stream = time.time()

    s1_ids: List[str] = []
    cand_ids: List[str] = []
    needed_s1: Set[str] = set()
    needed_s2: Set[str] = set()
    needed_s3: Set[str] = set()

    with open(candidate_path, "r", encoding="utf-8", errors="ignore") as f:
        header = f.readline()
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

                    if len(s1_ids) >= sample_limit:
                        break
            if len(s1_ids) >= sample_limit:
                break

    actual_pairs = len(s1_ids)
    print(f"  Streamed {actual_pairs:,} pairs in {time.time()-t_stream:.3f}s")
    print(f"  Unique entity references: S1={len(needed_s1):,}, S2={len(needed_s2):,}, S3={len(needed_s3):,}")

    # 2. Fast load needed entities
    print("\n2. Loading and normalizing referenced entity records ...", flush=True)
    entity_lookup = load_needed_entities_streaming(data_dir, needed_s1, needed_s2, needed_s3)
    print(f"  Total entity lookup table size: {len(entity_lookup):,} entities")

    # 3. Compute lightweight features in batch
    print("\n3. Computing 12 lightweight features ...", flush=True)
    t_compute = time.time()
    feats_array = compute_lightweight_batch_features(s1_ids, cand_ids, entity_lookup)
    compute_time = time.time() - t_compute
    throughput = actual_pairs / compute_time if compute_time > 0 else 0.0

    print(f"  Computed feature matrix of shape {feats_array.shape} in {compute_time:.4f}s ({throughput:,.1f} pairs/sec)")

    # 4. Construct DataFrame and check integrity
    df_feats = pd.DataFrame(feats_array, columns=LIGHTWEIGHT_FEATURE_NAMES)
    df_feats.insert(0, "candidate_entity_id", cand_ids)
    df_feats.insert(0, "source1_entity_id", s1_ids)

    # 5. Integrity checks
    nan_counts = df_feats[LIGHTWEIGHT_FEATURE_NAMES].isna().sum()
    inf_counts = np.isinf(feats_array).sum(axis=0)
    all_valid = (nan_counts.sum() == 0) and (inf_counts.sum() == 0)

    # 6. Save sample to output
    os.makedirs(os.path.dirname(os.path.abspath(output_sample_path)), exist_ok=True)
    df_feats.to_csv(output_sample_path, sep="\t", index=False)
    file_size_bytes = os.path.getsize(output_sample_path)
    file_size_kb = file_size_bytes / 1024.0

    total_time = time.time() - t_start

    # Measure current memory usage
    try:
        import psutil
        process = psutil.Process(os.getpid())
        mem_mb = process.memory_info().rss / (1024 * 1024)
    except Exception:
        mem_mb = sys.getsizeof(df_feats) / (1024 * 1024)

    # 7. Print comprehensive report
    print("\n========================================================")
    print("PHASE 4 SAMPLE BENCHMARK RESULTS")
    print("========================================================")
    print(f"Number of pairs processed:       {actual_pairs:,}")
    print(f"Feature count:                   {len(LIGHTWEIGHT_FEATURE_NAMES)} features")
    print(f"Feature computation runtime:     {compute_time:.4f}s ({throughput:,.1f} pairs/sec)")
    print(f"Total script runtime:            {total_time:.2f}s")
    print(f"Peak / process memory usage:     ~{mem_mb:.2f} MB")
    print(f"Feature output file:             {output_sample_path} ({file_size_kb:.1f} KB)")
    print(f"All feature values valid:        {all_valid}")
    print("\nMissing / NaN values per feature:")
    for col, n_nan in nan_counts.items():
        print(f"  {col:<25}: {n_nan}")

    print("\nFirst 5 feature rows:")
    print(df_feats.head(5).to_string(index=False))
    print("========================================================")


def main():
    parser = argparse.ArgumentParser(description="Phase 4 Sample Feature Generation")
    parser.add_argument("--data-dir", default="data", help="Data directory")
    parser.add_argument("--candidate-file", default="output/candidate_pairs.tsv", help="Path to candidate pairs")
    parser.add_argument("--output", default="output/features_sample_10000.tsv", help="Output TSV for sample features")
    parser.add_argument("--sample-limit", type=int, default=10000, help="Number of pairs to sample")
    args = parser.parse_args()

    run_sample_feature_generation(
        data_dir=args.data_dir,
        candidate_path=args.candidate_file,
        output_sample_path=args.output,
        sample_limit=args.sample_limit,
    )


if __name__ == "__main__":
    main()
