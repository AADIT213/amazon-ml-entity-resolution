"""
Evaluate Train Validation Slice Script.

Runs end-to-end blocking + feature extraction + scoring using the FROZEN
baseline model (models/baseline_logreg.joblib) on a held-out slice of train S1 entities.
Does NOT modify output/matching_results.tsv, test sets, or retrain the model.
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

# pyrefly: ignore [missing-import]
from business_entity_resolution.features import (
    compute_lightweight_batch_features,
)
# pyrefly: ignore [missing-import]
from business_entity_resolution.model import (
    BaselineMatchClassifier,
    compute_classification_metrics,
)
# pyrefly: ignore [missing-import]
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


def run_validation_slice(
    data_dir: str = "data",
    model_path: str = "models/baseline_logreg.joblib",
    n_sample_s1: int = 1000,
    random_seed: int = 9999,
    threshold: float = 0.90,
    total_train_rows: int = 2206821,
) -> Dict[str, Any]:
    """
    Run blocking + feature extraction + scoring on a slice of train S1 entities.
    """
    t_start = time.time()
    print("\n===============================================================================================")
    print(f"EVALUATING TRAIN VALIDATION SLICE ({n_sample_s1:,} HELD-OUT TRAIN S1 ENTITIES)")
    print("===============================================================================================")
    print(f"Model Path:         {model_path}")
    print(f"Data Directory:     {data_dir}")
    print(f"Slice Size:         {n_sample_s1:,} S1 entities")
    print(f"Random Seed:        {random_seed}")
    print(f"Threshold:          {threshold:.2f}")

    # 1. Load Frozen Model
    print(f"\n1. Loading frozen model from {model_path} ...", flush=True)
    assert os.path.exists(model_path), f"Model artifact not found: {model_path}"
    clf: BaselineMatchClassifier = BaselineMatchClassifier.load(model_path)
    assert clf.is_fitted, "Model is not fitted!"
    print(f"  Model loaded with {len(clf.feature_names)} features.")

    # 2. Select Held-out Train S1 Entities
    s1_path = os.path.join(data_dir, "train", "train_source1.tsv")
    s2_path = os.path.join(data_dir, "train", "train_source2.tsv")
    s3_path = os.path.join(data_dir, "train", "train_source3.tsv")
    gt_path = os.path.join(data_dir, "train", "train_ground_truth.tsv")

    print(f"\n2. Sampling {n_sample_s1:,} held-out train S1 entities from {s1_path} ...", flush=True)
    t_sample_start = time.time()
    np.random.seed(random_seed)
    sampled_indices: Set[int] = set(np.random.choice(total_train_rows, size=n_sample_s1, replace=False))

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
                if len(s1_records) >= n_sample_s1:
                    break

    sample_time = time.time() - t_sample_start
    sampled_s1_ids = set(s1_records.keys())
    print(f"  Sampled {len(s1_records):,} S1 entities in {sample_time:.2f}s.")

    # 3. Stream Ground Truth for the sampled slice
    print(f"\n3. Loading ground truth from {gt_path} for the slice ...", flush=True)
    t_gt_start = time.time()
    gt_map: Dict[str, Set[str]] = {}
    total_true_positives = 0

    with open(gt_path, "r", encoding="utf-8", errors="ignore") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader, None)  # header
        for row in reader:
            if len(row) >= 2:
                s1_id = row[0]
                if s1_id in sampled_s1_ids:
                    m_ids = set(row[1].split(",")) if row[1] else set()
                    gt_map[s1_id] = m_ids
                    total_true_positives += len(m_ids)
                    if len(gt_map) >= len(sampled_s1_ids):
                        break

    gt_time = time.time() - t_gt_start
    print(f"  Loaded ground truth: {len(gt_map):,} S1 entities with {total_true_positives:,} true matches in {gt_time:.2f}s.")

    # 4. Build Phase 3 Blocking Keys
    print("\n4. Building Phase 3 blocking keys for slice ...", flush=True)
    t_keys_start = time.time()

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

    keys_time = time.time() - t_keys_start
    print(f"  Keys built in {keys_time:.4f}s.")

    # 5. Stream Source 2 and Source 3 for Candidates
    print("\n5. Streaming Source 2 & Source 3 to generate candidate pairs ...", flush=True)
    t_blocking_start = time.time()
    s1_candidates: Dict[str, Set[str]] = defaultdict(set)
    retained_target_entities: Dict[str, Tuple[str, str, str]] = {}

    for src_path in [s2_path, s3_path]:
        fname = os.path.basename(src_path)
        t_f = time.time()
        n_scanned = 0
        n_retained = 0
        with open(src_path, "r", encoding="utf-8", errors="ignore") as f:
            reader = csv.reader(f, delimiter="\t")
            next(reader, None)  # header
            for row in reader:
                n_scanned += 1
                if len(row) < 4:
                    continue
                eid, raw_n, raw_a, raw_c = row[0], row[1], row[2], row[3]
                norm_n = normalize_name(raw_n)
                norm_a = normalize_address(raw_a)
                norm_c = normalize_country(raw_c)

                matched_s1: Set[str] = set()
                if norm_n and norm_n in exact_name_map:
                    matched_s1.update(exact_name_map[norm_n])

                if norm_n and norm_c:
                    stem_n = strip_legal_suffix(norm_n)
                    if stem_n and len(stem_n) >= 4:
                        c_stem = (norm_c, stem_n)
                        if c_stem in country_stem_map:
                            matched_s1.update(country_stem_map[c_stem])
                        s_key = sorted_stem_key(stem_n)
                        if s_key:
                            c_skey = (norm_c, s_key)
                            if c_skey in country_sorted_stem_map:
                                matched_s1.update(country_sorted_stem_map[c_skey])
                        if not norm_a and c_stem in null_addr_map:
                            matched_s1.update(null_addr_map[c_stem])

                if norm_a and len(norm_a) >= 8:
                    if norm_a in exact_addr_map:
                        matched_s1.update(exact_addr_map[norm_a])
                    if norm_c:
                        ap_key = addr_prefix_key(norm_a)
                        if ap_key:
                            c_ap = (norm_c, ap_key)
                            if c_ap in country_addr_prefix_map:
                                matched_s1.update(country_addr_prefix_map[c_ap])

                if matched_s1:
                    n_retained += 1
                    retained_target_entities[eid] = (norm_n, norm_a, norm_c)
                    for s1_id in matched_s1:
                        s1_candidates[s1_id].add(eid)

        print(f"  {fname}: scanned {n_scanned:,} rows in {time.time()-t_f:.2f}s ({n_retained:,} candidates linked)", flush=True)

    blocking_time = time.time() - t_blocking_start

    # Build candidate pair lists
    s1_pair_list: List[str] = []
    cand_pair_list: List[str] = []
    labels: List[int] = []

    for s1_id in sampled_s1_ids:
        cands = s1_candidates.get(s1_id, set())
        true_set = gt_map.get(s1_id, set())
        for c in cands:
            s1_pair_list.append(s1_id)
            cand_pair_list.append(c)
            labels.append(1 if c in true_set else 0)

    total_candidate_pairs = len(labels)
    y_true = np.array(labels, dtype=int)
    pos_generated = int(np.sum(y_true == 1))
    neg_generated = int(np.sum(y_true == 0))

    blocking_recall = pos_generated / total_true_positives if total_true_positives > 0 else 0.0
    print(f"\n  Blocking Results on Slice:")
    print(f"    Total Candidate Pairs Generated: {total_candidate_pairs:,}")
    print(f"    True Positives Captured:         {pos_generated:,} / {total_true_positives:,} ({blocking_recall*100:.2f}% recall)")
    print(f"    Negative Candidates Generated:   {neg_generated:,}")
    print(f"    Blocking Runtime:                {blocking_time:.2f}s")

    # 6. Feature Extraction
    print(f"\n6. Computing 12 lightweight features on {total_candidate_pairs:,} pairs ...", flush=True)
    t_feat_start = time.time()
    entity_lookup = {**s1_records, **retained_target_entities}
    X_slice = compute_lightweight_batch_features(s1_pair_list, cand_pair_list, entity_lookup)
    feat_time = time.time() - t_feat_start
    feat_throughput = total_candidate_pairs / feat_time if feat_time > 0 else 0.0
    print(f"  Features computed in {feat_time:.4f}s ({feat_throughput:,.0f} pairs/sec), shape: {X_slice.shape}")

    # 7. Model Scoring & Threshold Evaluation
    print(f"\n7. Scoring with frozen Baseline Logistic Regression (threshold={threshold:.2f}) ...", flush=True)
    t_score_start = time.time()
    probs = clf.predict_proba(X_slice)
    score_time = time.time() - t_score_start
    score_throughput = total_candidate_pairs / score_time if score_time > 0 else 0.0
    print(f"  Scored in {score_time:.4f}s ({score_throughput:,.0f} pairs/sec)")

    # Evaluation at threshold
    y_pred = (probs >= threshold).astype(int)
    metrics = compute_classification_metrics(y_true, y_pred, beta=0.5)

    total_wall_clock = time.time() - t_start

    print("\n===============================================================================================")
    print("TRAIN VALIDATION SLICE RESULTS (1,000 S1 ENTITIES)")
    print("===============================================================================================")
    print(f"Total S1 Entities Evaluated:      {n_sample_s1:,}")
    print(f"Total Candidate Pairs Evaluated:  {total_candidate_pairs:,}")
    print(f"True Positive Ground Truth Pairs: {total_true_positives:,}")
    print(f"Model Probability Threshold:      {threshold:.2f}")
    print(f"-----------------------------------------------------------------------------------------------")
    print(f"Precision:                        {metrics['precision']*100:.2f}%")
    print(f"Recall (vs candidate pairs):      {metrics['recall']*100:.2f}%")
    print(f"End-to-End Recall (vs GT):        {metrics['tp']/total_true_positives*100:.2f}% ({metrics['tp']}/{total_true_positives})")
    print(f"F0.5 Score:                       {metrics['f_beta']*100:.2f}%")
    print(f"Confusion Matrix:                 TP={metrics['tp']:,}, FP={metrics['fp']:,}, FN={metrics['fn']:,}, TN={metrics['tn']:,}")
    print(f"-----------------------------------------------------------------------------------------------")
    print(f"TIMING BREAKDOWN:")
    print(f"  S1 Sampling & GT Streaming:     {sample_time + gt_time:.2f}s")
    print(f"  Blocking Keys & Source Scan:    {blocking_time:.2f}s")
    print(f"  Feature Extraction:             {feat_time:.4f}s")
    print(f"  Model Scoring:                  {score_time:.4f}s")
    print(f"  TOTAL EXACT WALL-CLOCK TIME:    {total_wall_clock:.2f}s ({total_wall_clock/60:.2f} min)")
    print("===============================================================================================\n")

    return {
        "n_sample_s1": n_sample_s1,
        "total_pairs": total_candidate_pairs,
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f_beta": metrics["f_beta"],
        "tp": metrics["tp"],
        "fp": metrics["fp"],
        "fn": metrics["fn"],
        "tn": metrics["tn"],
        "blocking_time": blocking_time,
        "feat_time": feat_time,
        "score_time": score_time,
        "total_wall_clock": total_wall_clock,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate train validation slice timing and metrics")
    parser.add_argument("--data-dir", default="data", help="Data directory")
    parser.add_argument("--model-path", default="models/baseline_logreg.joblib", help="Model path")
    parser.add_argument("--sample-s1", type=int, default=1000, help="Number of S1 entities to evaluate")
    parser.add_argument("--threshold", type=float, default=0.90, help="Threshold")
    parser.add_argument("--seed", type=int, default=9999, help="Random seed for held-out sample")
    args = parser.parse_args()

    run_validation_slice(
        data_dir=args.data_dir,
        model_path=args.model_path,
        n_sample_s1=args.sample_s1,
        random_seed=args.seed,
        threshold=args.threshold,
    )


if __name__ == "__main__":
    main()
