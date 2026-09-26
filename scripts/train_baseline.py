"""
Phase 5: Baseline Matching Model Training & Validation Script.

Implements the memory-safe 10,000-S1 training benchmark:
1. Deterministic uniform random sample of 10,000 train Source 1 entities (seed 42)
2. Stream ground truth for ONLY the 10,000 sampled S1 entities
3. Compute 7 improved Phase 3 blocking keys for the 10,000 S1 entities
4. Streaming scan of train_source2.tsv and train_source3.tsv:
   - Evaluates keys on the fly
   - Retains ONLY target records relevant to the 10,000 S1 entities
   - Discards all unmatched records immediately (zero global index memory)
5. Generates candidate pairs (all generated positives + capped negatives)
6. Computes 12 lightweight features
7. 80/20 entity-grouped train/validation split
8. Trains Logistic Regression baseline (class_weight='balanced')
9. Evaluates threshold grid [0.10, 0.20, ..., 0.90] for F0.5 optimization
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from collections import defaultdict
from typing import Any, Dict, List, Set, Tuple

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from business_entity_resolution.features import (
    LIGHTWEIGHT_FEATURE_NAMES,
    compute_lightweight_batch_features,
)
from business_entity_resolution.model import (
    BaselineMatchClassifier,
    compute_classification_metrics,
)
from business_entity_resolution.normalize import (
    normalize_country,
    normalize_name,
    normalize_address,
    strip_legal_suffix,
)

STOPWORDS = {
    "and", "of", "the", "in", "for", "on", "at", "to", "a", "an", "is", "by", "with",
    "co", "corp", "inc", "ltd", "llc", "services", "center", "solutions",
    "technologies", "group", "enterprises", "holdings", "private", "limited",
}


def sorted_stem_key(stem: str) -> str:
    """Order-invariant token stem key."""
    parts = [t for t in stem.split() if t not in STOPWORDS]
    if len(parts) <= 1:
        parts = stem.split()
    if not parts:
        return ""
    parts.sort()
    res = " ".join(parts)
    return res if len(res) >= 4 else ""


def addr_prefix_key(addr: str) -> str:
    """14-char address prefix key."""
    return addr[:14].strip() if len(addr) >= 14 else ""


def select_sampled_s1_entities(
    s1_path: str,
    n_sample: int = 10000,
    total_rows: int = 2206821,
    random_seed: int = 42,
) -> Dict[str, Tuple[str, str, str]]:
    """
    Deterministically sample exactly n_sample S1 entities using uniform random index selection.
    Returns: {s1_id: (norm_name, norm_addr, norm_country)}
    """
    print(f"\n1. Sampling exactly {n_sample:,} S1 entities from {s1_path} (seed={random_seed}) ...", flush=True)
    t0 = time.time()
    np.random.seed(random_seed)
    sampled_indices: Set[int] = set(np.random.choice(total_rows, size=n_sample, replace=False))

    s1_records: Dict[str, Tuple[str, str, str]] = {}
    with open(s1_path, "r", encoding="utf-8", errors="ignore") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader, None)  # header
        for row_idx, row in enumerate(reader):
            if row_idx in sampled_indices:
                eid = row[0]
                n = normalize_name(row[1] if len(row) > 1 else "")
                a = normalize_address(row[2] if len(row) > 2 else "")
                c = normalize_country(row[3] if len(row) > 3 else "")
                s1_records[eid] = (n, a, c)
                if len(s1_records) >= n_sample:
                    break

    print(f"  Selected {len(s1_records):,} S1 entities in {time.time()-t0:.2f}s", flush=True)
    return s1_records


def load_sampled_ground_truth(
    gt_path: str,
    sampled_s1_ids: Set[str],
) -> Dict[str, Set[str]]:
    """
    Stream ground truth and keep only rows for the sampled S1 IDs.
    """
    print(f"\n2. Streaming ground truth from {gt_path} for {len(sampled_s1_ids):,} sampled S1 entities ...", flush=True)
    t0 = time.time()
    gt_map: Dict[str, Set[str]] = {}
    total_true_matches = 0

    with open(gt_path, "r", encoding="utf-8", errors="ignore") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader, None)  # header
        for row in reader:
            if len(row) >= 2:
                s1_id = row[0]
                if s1_id in sampled_s1_ids:
                    m_ids = set(row[1].split(",")) if row[1] else set()
                    gt_map[s1_id] = m_ids
                    total_true_matches += len(m_ids)
                    if len(gt_map) >= len(sampled_s1_ids):
                        break

    print(f"  Loaded ground truth: {len(gt_map):,} S1 entities, {total_true_matches:,} true positive matches in {time.time()-t0:.2f}s")
    return gt_map


def build_s1_blocking_keys(
    s1_records: Dict[str, Tuple[str, str, str]],
) -> Tuple[
    Dict[str, Set[str]],
    Dict[Tuple[str, str], Set[str]],
    Dict[str, Set[str]],
    Dict[Tuple[str, str], Set[str]],
    Dict[Tuple[str, str], Set[str]],
    Dict[Tuple[str, str], Set[str]],
]:
    """
    Build inverted key -> Set[s1_id] maps for the improved Phase 3 blocking strategies:
    1. exact_name
    2. country_stem
    3. exact_addr
    4. country_sorted_stem
    5. country_addr_prefix
    6. null_addr_fallback
    """
    print("\n3. Building improved Phase 3 blocking keys for sampled S1 entities ...", flush=True)
    t0 = time.time()

    exact_name_map: Dict[str, Set[str]] = defaultdict(set)
    country_stem_map: Dict[Tuple[str, str], Set[str]] = defaultdict(set)
    exact_addr_map: Dict[str, Set[str]] = defaultdict(set)
    country_sorted_stem_map: Dict[Tuple[str, str], Set[str]] = defaultdict(set)
    country_addr_prefix_map: Dict[Tuple[str, str], Set[str]] = defaultdict(set)
    null_addr_map: Dict[Tuple[str, str], Set[str]] = defaultdict(set)

    for s1_id, (n, a, c) in s1_records.items():
        if n:
            exact_name_map[n].add(s1_id)
            stem = strip_legal_suffix(n)
            if stem and c and len(stem) >= 4:
                country_stem_map[(c, stem)].add(s1_id)
                s_key = sorted_stem_key(stem)
                if s_key:
                    country_sorted_stem_map[(c, s_key)].add(s1_id)

        if a and len(a) >= 8:
            exact_addr_map[a].add(s1_id)
            if c:
                ap_key = addr_prefix_key(a)
                if ap_key:
                    country_addr_prefix_map[(c, ap_key)].add(s1_id)

        if not a and n and c:
            stem = strip_legal_suffix(n)
            if stem and len(stem) >= 4:
                null_addr_map[(c, stem)].add(s1_id)

    print(f"  Built keys in {time.time()-t0:.2f}s:")
    print(f"    exact_name keys:           {len(exact_name_map):,}")
    print(f"    country_stem keys:         {len(country_stem_map):,}")
    print(f"    exact_addr keys:           {len(exact_addr_map):,}")
    print(f"    country_sorted_stem keys:  {len(country_sorted_stem_map):,}")
    print(f"    country_addr_prefix keys:  {len(country_addr_prefix_map):,}")
    print(f"    null_addr keys:            {len(null_addr_map):,}")

    return (
        exact_name_map,
        country_stem_map,
        exact_addr_map,
        country_sorted_stem_map,
        country_addr_prefix_map,
        null_addr_map,
    )


def stream_source_file_match_candidates(
    fpath: str,
    exact_name_map: Dict[str, Set[str]],
    country_stem_map: Dict[Tuple[str, str], Set[str]],
    exact_addr_map: Dict[str, Set[str]],
    country_sorted_stem_map: Dict[Tuple[str, str], Set[str]],
    country_addr_prefix_map: Dict[Tuple[str, str], Set[str]],
    null_addr_map: Dict[Tuple[str, str], Set[str]],
    s1_candidates: Dict[str, Set[str]],
    retained_entities: Dict[str, Tuple[str, str, str]],
) -> float:
    """
    Streaming scan of one source file.
    Evaluates keys on the fly and retains ONLY target records matching any sampled S1 key.
    """
    fname = os.path.basename(fpath)
    print(f"  Streaming {fname} ...", flush=True)
    t0 = time.time()
    n_scanned = 0
    n_matched = 0

    with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader, None)  # header
        for row in reader:
            n_scanned += 1
            if len(row) < 4:
                continue

            eid = row[0]
            raw_n = row[1]
            raw_a = row[2]
            raw_c = row[3]

            norm_n = normalize_name(raw_n)
            norm_a = normalize_address(raw_a)
            norm_c = normalize_country(raw_c)

            matched_s1_for_this_record: Set[str] = set()

            # 1. Exact Name Strategy
            if norm_n and norm_n in exact_name_map:
                matched_s1_for_this_record.update(exact_name_map[norm_n])

            # 2. Country Stem & Sorted Stem Strategies
            if norm_n and norm_c:
                stem_n = strip_legal_suffix(norm_n)
                if stem_n and len(stem_n) >= 4:
                    c_stem = (norm_c, stem_n)
                    if c_stem in country_stem_map:
                        matched_s1_for_this_record.update(country_stem_map[c_stem])

                    s_key = sorted_stem_key(stem_n)
                    if s_key:
                        c_skey = (norm_c, s_key)
                        if c_skey in country_sorted_stem_map:
                            matched_s1_for_this_record.update(country_sorted_stem_map[c_skey])

                    if not norm_a and c_stem in null_addr_map:
                        matched_s1_for_this_record.update(null_addr_map[c_stem])

            # 3. Exact Address & Address Prefix Strategies
            if norm_a and len(norm_a) >= 8:
                if norm_a in exact_addr_map:
                    matched_s1_for_this_record.update(exact_addr_map[norm_a])
                if norm_c:
                    ap_key = addr_prefix_key(norm_a)
                    if ap_key:
                        c_ap = (norm_c, ap_key)
                        if c_ap in country_addr_prefix_map:
                            matched_s1_for_this_record.update(country_addr_prefix_map[c_ap])

            # If matched, retain candidate entity and link to matched S1 entities
            if matched_s1_for_this_record:
                n_matched += 1
                retained_entities[eid] = (norm_n, norm_a, norm_c)
                for s1_id in matched_s1_for_this_record:
                    s1_candidates[s1_id].add(eid)

    elapsed = time.time() - t0
    print(f"    Scanned {n_scanned:,} rows from {fname} in {elapsed:.2f}s ({n_matched:,} relevant candidates retained)")
    return elapsed


def run_phase5_benchmark(
    data_dir: str = "data",
    n_sample_s1: int = 10000,
    max_negatives_per_s1: int = 15,
    model_output_path: str = "models/baseline_logreg.joblib",
    random_seed: int = 42,
) -> Dict[str, Any]:
    t_start = time.time()
    print("===============================================================================================")
    print("PHASE 5: BASELINE MATCHING MODEL (10,000-S1 STREAMING BENCHMARK)")
    print("===============================================================================================")

    s1_path = os.path.join(data_dir, "train", "train_source1.tsv")
    s2_path = os.path.join(data_dir, "train", "train_source2.tsv")
    s3_path = os.path.join(data_dir, "train", "train_source3.tsv")
    gt_path = os.path.join(data_dir, "train", "train_ground_truth.tsv")

    # Step 1: Select deterministic S1 sample
    s1_records = select_sampled_s1_entities(s1_path, n_sample=n_sample_s1, random_seed=random_seed)
    sampled_s1_ids = set(s1_records.keys())

    # Step 2: Stream ground truth for only sampled S1 entities
    gt_map = load_sampled_ground_truth(gt_path, sampled_s1_ids)

    # Step 3: Build blocking keys for only the 10,000 S1 entities
    (
        exact_name_map,
        country_stem_map,
        exact_addr_map,
        country_sorted_stem_map,
        country_addr_prefix_map,
        null_addr_map,
    ) = build_s1_blocking_keys(s1_records)

    # Step 4: Stream Source 2 and Source 3 row-by-row
    print("\n4. Streaming Source 2 and Source 3 to collect candidates on the fly ...", flush=True)
    t_scan_start = time.time()
    s1_candidates: Dict[str, Set[str]] = defaultdict(set)
    retained_target_entities: Dict[str, Tuple[str, str, str]] = {}

    s2_time = stream_source_file_match_candidates(
        s2_path, exact_name_map, country_stem_map, exact_addr_map,
        country_sorted_stem_map, country_addr_prefix_map, null_addr_map,
        s1_candidates, retained_target_entities,
    )

    s3_time = stream_source_file_match_candidates(
        s3_path, exact_name_map, country_stem_map, exact_addr_map,
        country_sorted_stem_map, country_addr_prefix_map, null_addr_map,
        s1_candidates, retained_target_entities,
    )
    scan_time = time.time() - t_scan_start

    # Step 5: Generate candidate pairs (all positives + capped negatives)
    print("\n5. Assembling candidate pairs & labeling from ground truth ...", flush=True)
    t_pair_gen = time.time()
    s1_pair_list: List[str] = []
    cand_pair_list: List[str] = []
    labels: List[int] = []

    pos_count = 0
    neg_count = 0
    np.random.seed(random_seed)

    for s1_id in sampled_s1_ids:
        cands = s1_candidates.get(s1_id, set())
        true_set = gt_map.get(s1_id, set())

        positives = [c for c in cands if c in true_set]
        negatives = [c for c in cands if c not in true_set]

        # Preserve ALL generated positives
        for p in positives:
            s1_pair_list.append(s1_id)
            cand_pair_list.append(p)
            labels.append(1)
            pos_count += 1

        # Deterministically cap negatives per S1
        if len(negatives) > max_negatives_per_s1:
            negatives = list(np.random.choice(negatives, size=max_negatives_per_s1, replace=False))

        for n in negatives:
            s1_pair_list.append(s1_id)
            cand_pair_list.append(n)
            labels.append(0)
            neg_count += 1

    cand_gen_time = time.time() - t_pair_gen
    total_candidate_pairs = len(labels)
    pos_neg_ratio = pos_count / neg_count if neg_count > 0 else 0.0

    print(f"  Candidate Generation Summary:")
    print(f"    Positive Candidates:      {pos_count:,}")
    print(f"    Negative Candidates:      {neg_count:,}")
    print(f"    Total Candidate Pairs:    {total_candidate_pairs:,}")
    print(f"    Pos/Neg Ratio:            1 : {1/pos_neg_ratio:.2f}")
    print(f"    Unique Targets Retained:  {len(retained_target_entities):,}")

    # Entity lookup table (contains only S1 sample + retained candidates)
    entity_lookup = {**s1_records, **retained_target_entities}

    # Step 6 & 7: Entity-Grouped Train / Validation Split (80% / 20%)
    unique_s1 = sorted(list(sampled_s1_ids))
    np.random.seed(random_seed)
    np.random.shuffle(unique_s1)
    split_idx = int(0.80 * len(unique_s1))
    train_s1_set = set(unique_s1[:split_idx])
    val_s1_set = set(unique_s1[split_idx:])

    train_indices = [i for i, s1 in enumerate(s1_pair_list) if s1 in train_s1_set]
    val_indices = [i for i, s1 in enumerate(s1_pair_list) if s1 in val_s1_set]

    y_array = np.array(labels, dtype=int)
    y_train = y_array[train_indices]
    y_val = y_array[val_indices]

    print(f"\n6. Entity-Grouped Split (80/20):")
    print(f"  Train:      {len(train_s1_set):,} S1 entities -> {len(train_indices):,} pairs ({np.sum(y_train==1):,} pos / {np.sum(y_train==0):,} neg)")
    print(f"  Validation: {len(val_s1_set):,} S1 entities -> {len(val_indices):,} pairs ({np.sum(y_val==1):,} pos / {np.sum(y_val==0):,} neg)")

    # Step 8: Vectorized Feature Extraction
    print(f"\n7. Computing 12 lightweight features on {total_candidate_pairs:,} pairs ...", flush=True)
    t_feat = time.time()
    X_all = compute_lightweight_batch_features(s1_pair_list, cand_pair_list, entity_lookup)
    feat_time = time.time() - t_feat
    print(f"  Computed feature matrix {X_all.shape} in {feat_time:.4f}s ({total_candidate_pairs/feat_time:,.1f} pairs/sec)")

    X_train = X_all[train_indices]
    X_val = X_all[val_indices]

    # Step 9: Train Baseline Logistic Regression
    print("\n8. Training Baseline Logistic Regression Classifier ...", flush=True)
    t_train = time.time()
    clf = BaselineMatchClassifier(c=1.0, class_weight="balanced", max_iter=1000, random_state=random_seed)
    clf.fit(X_train, y_train)
    train_time = time.time() - t_train
    print(f"  Model trained in {train_time:.4f}s")

    # Learned feature coefficients
    print("\n  Learned Feature Weights (Logistic Regression):")
    weights = clf.get_feature_importances()
    for feat, w in sorted(weights.items(), key=lambda x: abs(x[1]), reverse=True):
        print(f"    {feat:<25}: {w:+.4f}")

    # Step 10: Evaluate Threshold Grid on Validation Set
    print("\n===============================================================================================")
    print("THRESHOLD / F0.5 EVALUATION TABLE (VALIDATION SPLIT)")
    print("===============================================================================================")
    thresholds = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
    df_eval = clf.evaluate_thresholds(X_val, y_val, thresholds=thresholds, beta=0.5)

    print(f"{'Threshold':<10} | {'Precision':<10} | {'Recall':<10} | {'F0.5':<10} | {'Predicted Matches':<18} | {'TP':<7} | {'FP':<7} | {'FN':<7} | {'TN':<7}")
    print("-" * 105)
    for _, row in df_eval.iterrows():
        print(f"{row['threshold']:<10.2f} | {row['precision']:<10.4f} | {row['recall']:<10.4f} | {row['f_beta']:<10.4f} | {int(row['predicted_matches']):<18,} | {int(row['tp']):<7,} | {int(row['fp']):<7,} | {int(row['fn']):<7,} | {int(row['tn']):<7,}")

    best_thresh = clf.best_threshold
    best_metrics = df_eval.loc[df_eval["threshold"] == best_thresh].iloc[0].to_dict()

    print(f"\n--- OPTIMAL THRESHOLD SUMMARY ---")
    print(f"Optimal Probability Threshold: {best_thresh:.2f}")
    print(f"  Validation Precision:        {best_metrics['precision']*100:.2f}%")
    print(f"  Validation Recall:           {best_metrics['recall']*100:.2f}%")
    print(f"  Validation F0.5 Score:       {best_metrics['f_beta']*100:.2f}%")
    print(f"  Confusion Matrix:            TP={int(best_metrics['tp']):,}, FP={int(best_metrics['fp']):,}, FN={int(best_metrics['fn']):,}, TN={int(best_metrics['tn']):,}")

    # Save Model Artifact
    os.makedirs(os.path.dirname(os.path.abspath(model_output_path)), exist_ok=True)
    clf.save(model_output_path)

    total_time = time.time() - t_start

    # Memory calculation
    try:
        import psutil
        process = psutil.Process(os.getpid())
        mem_mb = process.memory_info().rss / (1024 * 1024)
    except Exception:
        mem_mb = sys.getsizeof(X_all) / (1024 * 1024)

    print(f"\nTotal Pipeline Runtime:       {total_time:.2f}s")
    print(f"Process Memory:               ~{mem_mb:.2f} MB")
    print(f"Saved Model Artifact:         {model_output_path}")

    return {
        "sampled_s1_count": n_sample_s1,
        "pos_count": pos_count,
        "neg_count": neg_count,
        "total_pairs": total_candidate_pairs,
        "pos_neg_ratio": pos_neg_ratio,
        "unique_targets_retained": len(retained_target_entities),
        "s2_scan_time": s2_time,
        "s3_scan_time": s3_time,
        "cand_gen_time": cand_gen_time,
        "feat_time": feat_time,
        "train_time": train_time,
        "total_time": total_time,
        "mem_mb": mem_mb,
        "df_eval": df_eval,
        "best_threshold": best_thresh,
        "best_precision": best_metrics["precision"],
        "best_recall": best_metrics["recall"],
        "best_f_beta": best_metrics["f_beta"],
    }


def main():
    parser = argparse.ArgumentParser(description="Phase 5 Baseline Matching Model Benchmark")
    parser.add_argument("--data-dir", default="data", help="Data directory path")
    parser.add_argument("--sample-s1", type=int, default=10000, help="Number of S1 entities to sample")
    parser.add_argument("--max-negatives", type=int, default=15, help="Max negative candidates per S1")
    parser.add_argument("--output-model", default="models/baseline_logreg.joblib", help="Path to save trained model")
    args = parser.parse_args()

    run_phase5_benchmark(
        data_dir=args.data_dir,
        n_sample_s1=args.sample_s1,
        max_negatives_per_s1=args.max_negatives,
        model_output_path=args.output_model,
        random_seed=42,
    )


if __name__ == "__main__":
    main()
