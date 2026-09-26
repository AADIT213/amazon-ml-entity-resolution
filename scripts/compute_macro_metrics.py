"""
Compute Macro-Averaged F0.5 on 1,000 Train S1 Entities.

Reuses the exact frozen baseline model (models/baseline_logreg.joblib)
and exact 1,000 S1 entity sample (seed 9999) to compute:
- Per-entity Precision, Recall, and F0.5
- Macro-averaged F0.5 across all 1,000 S1 entities
- Breakdown by Singleton vs Multi-match vs Zero-match
"""

import csv
import functools
import os
import sys
import time
from collections import defaultdict
from typing import Dict, List, Set, Tuple

import joblib
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))


# pyrefly: ignore [missing-import]
from business_entity_resolution.features import compute_lightweight_batch_features
# pyrefly: ignore [missing-import]
from business_entity_resolution.model import BaselineMatchClassifier
# pyrefly: ignore [missing-import]
from business_entity_resolution.normalize import (
    normalize_address,
    normalize_country,
    normalize_name,
    strip_legal_suffix,
)

STOPWORDS = {
    "and", "of", "the", "in", "for", "on", "at", "to", "a", "an", "is", "by", "with",
    "co", "corp", "inc", "ltd", "llc", "services", "center", "solutions",
    "technologies", "group", "enterprises", "holdings", "private", "limited",
}


@functools.lru_cache(maxsize=500_000)
def cached_normalize_name(name: str) -> str:
    return normalize_name(name)


@functools.lru_cache(maxsize=500_000)
def cached_normalize_address(addr: str) -> str:
    return normalize_address(addr)


@functools.lru_cache(maxsize=50_000)
def cached_normalize_country(country: str) -> str:
    return normalize_country(country)


@functools.lru_cache(maxsize=500_000)
def cached_strip_legal_suffix(name: str) -> str:
    return strip_legal_suffix(name)


@functools.lru_cache(maxsize=500_000)
def sorted_stem_key(stem: str) -> str:
    parts = [t for t in stem.split() if t not in STOPWORDS]
    if len(parts) <= 1:
        parts = stem.split()
    if not parts:
        return ""
    parts.sort()
    res = " ".join(parts)
    return res if len(res) >= 4 else ""


def addr_prefix_key(addr: str) -> str:
    return addr[:14].strip() if len(addr) >= 14 else ""


def calculate_per_entity_f05(gt_set: Set[str], pred_set: Set[str]) -> Tuple[float, float, float]:
    """
    Computes (precision, recall, F0.5) for a single entity.
    F0.5 = (1 + 0.5^2) * P * R / (0.5^2 * P + R) = 1.25 * P * R / (0.25 * P + R)
    """
    if len(gt_set) == 0:
        if len(pred_set) == 0:
            return 1.0, 1.0, 1.0
        else:
            return 0.0, 1.0, 0.0

    if len(pred_set) == 0:
        return 0.0, 0.0, 0.0

    tp = len(pred_set & gt_set)
    fp = len(pred_set - gt_set)
    fn = len(gt_set - pred_set)

    precision = tp / len(pred_set)
    recall = tp / len(gt_set)

    if tp == 0:
        return precision, recall, 0.0

    f05 = 1.25 * precision * recall / (0.25 * precision + recall)
    return precision, recall, f05


def main():
    data_dir = "data"
    model_path = "models/baseline_logreg.joblib"
    n_sample_s1 = 1000
    random_seed = 9999
    threshold = 0.90
    total_train_rows = 2206821

    print("Loading model...", flush=True)
    clf = BaselineMatchClassifier.load(model_path)

    s1_path = os.path.join(data_dir, "train", "train_source1.tsv")
    s2_path = os.path.join(data_dir, "train", "train_source2.tsv")
    s3_path = os.path.join(data_dir, "train", "train_source3.tsv")
    gt_path = os.path.join(data_dir, "train", "train_ground_truth.tsv")

    print(f"Sampling {n_sample_s1} S1 entities (seed={random_seed})...", flush=True)
    np.random.seed(random_seed)
    sampled_indices = set(np.random.choice(total_train_rows, size=n_sample_s1, replace=False))

    s1_records: Dict[str, Tuple[str, str, str]] = {}
    with open(s1_path, "r", encoding="utf-8", errors="ignore") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader, None)
        for row_idx, row in enumerate(reader):
            if row_idx in sampled_indices:
                eid = row[0]
                n = normalize_name(row[1] if len(row) > 1 else "")
                a = normalize_address(row[2] if len(row) > 2 else "")
                c = normalize_country(row[3] if len(row) > 3 else "")
                s1_records[eid] = (n, a, c)
                if len(s1_records) >= n_sample_s1:
                    break

    sampled_s1_ids = set(s1_records.keys())

    print("Loading ground truth...", flush=True)
    gt_map: Dict[str, Set[str]] = {eid: set() for eid in sampled_s1_ids}
    with open(gt_path, "r", encoding="utf-8", errors="ignore") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader, None)
        for row in reader:
            if len(row) >= 2 and row[0] in gt_map:
                m_ids = set(row[1].split(",")) if row[1] else set()
                gt_map[row[0]] = m_ids

    # Build blocking keys
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

    target_countries = {c for c, _ in country_stem_map} | {c for c, _ in country_addr_prefix_map} | {c for c, _ in null_addr_map}

    # Fast scan of target sources
    print(f"Generating candidate pairs across {len(target_countries)} active slice countries...", flush=True)
    s1_candidates: Dict[str, Set[str]] = defaultdict(set)
    retained_target_entities: Dict[str, Tuple[str, str, str]] = {}

    for src_path in [s2_path, s3_path]:
        t0 = time.time()
        n_scanned = 0
        n_matched = 0
        print(f"Scanning {os.path.basename(src_path)}...", flush=True)
        with open(src_path, "r", encoding="utf-8", errors="ignore") as f:
            reader = csv.reader(f, delimiter="\t")
            next(reader, None)
            for row in reader:
                n_scanned += 1
                if len(row) < 4:
                    continue
                eid, raw_n, raw_a, raw_c = row[0], row[1], row[2], row[3]
                
                norm_c = cached_normalize_country(raw_c) if raw_c else ""
                has_country = norm_c in target_countries
                
                matched_s1: Set[str] = set()
                norm_n = ""
                if raw_n:
                    norm_n = cached_normalize_name(raw_n)
                    if norm_n in exact_name_map:
                        matched_s1.update(exact_name_map[norm_n])

                    if has_country:
                        stem_n = cached_strip_legal_suffix(norm_n)
                        if stem_n and len(stem_n) >= 4:
                            c_stem = (norm_c, stem_n)
                            if c_stem in country_stem_map:
                                matched_s1.update(country_stem_map[c_stem])
                            s_key = sorted_stem_key(stem_n)
                            if s_key:
                                c_skey = (norm_c, s_key)
                                if c_skey in country_sorted_stem_map:
                                    matched_s1.update(country_sorted_stem_map[c_skey])
                            if not raw_a and c_stem in null_addr_map:
                                matched_s1.update(null_addr_map[c_stem])

                norm_a = ""
                if raw_a:
                    norm_a = cached_normalize_address(raw_a)
                    if norm_a and len(norm_a) >= 8:
                        if norm_a in exact_addr_map:
                            matched_s1.update(exact_addr_map[norm_a])
                        if has_country:
                            ap_key = addr_prefix_key(norm_a)
                            if ap_key:
                                c_ap = (norm_c, ap_key)
                                if c_ap in country_addr_prefix_map:
                                    matched_s1.update(country_addr_prefix_map[c_ap])

                if matched_s1:
                    n_matched += 1
                    if not norm_n and raw_n:
                        norm_n = cached_normalize_name(raw_n)
                    if not norm_a and raw_a:
                        norm_a = cached_normalize_address(raw_a)
                    retained_target_entities[eid] = (norm_n, norm_a, norm_c)
                    for s1_id in matched_s1:
                        s1_candidates[s1_id].add(eid)

        print(f"  {os.path.basename(src_path)}: scanned {n_scanned:,} in {time.time()-t0:.2f}s, matched {n_matched:,}", flush=True)

    # Feature extraction & scoring
    s1_pair_list: List[str] = []
    cand_pair_list: List[str] = []
    for s1_id in sampled_s1_ids:
        for c in s1_candidates.get(s1_id, set()):
            s1_pair_list.append(s1_id)
            cand_pair_list.append(c)

    print(f"Total candidate pairs: {len(s1_pair_list):,}", flush=True)
    entity_lookup = {**s1_records, **retained_target_entities}
    X_slice = compute_lightweight_batch_features(s1_pair_list, cand_pair_list, entity_lookup)
    probs = clf.predict_proba(X_slice)

    # Gather predicted matches per S1 entity
    s1_predictions: Dict[str, Set[str]] = {eid: set() for eid in sampled_s1_ids}
    for (s1_id, cand_id), prob in zip(zip(s1_pair_list, cand_pair_list), probs):
        if prob >= threshold:
            s1_predictions[s1_id].add(cand_id)

    # Compute macro metrics
    per_entity_metrics = []
    zero_match_entities = []
    singleton_entities = []
    multi_match_entities = []

    for s1_id in sampled_s1_ids:
        gt_set = gt_map[s1_id]
        pred_set = s1_predictions[s1_id]
        p, r, f05 = calculate_per_entity_f05(gt_set, pred_set)
        
        entry = {
            "s1_id": s1_id,
            "gt_count": len(gt_set),
            "pred_count": len(pred_set),
            "tp": len(pred_set & gt_set),
            "precision": p,
            "recall": r,
            "f05": f05,
        }
        per_entity_metrics.append(entry)

        if len(gt_set) == 0:
            zero_match_entities.append(entry)
        elif len(gt_set) == 1:
            singleton_entities.append(entry)
        else:
            multi_match_entities.append(entry)

    all_f05 = [m["f05"] for m in per_entity_metrics]
    all_p = [m["precision"] for m in per_entity_metrics]
    all_r = [m["recall"] for m in per_entity_metrics]

    zero_f05 = [m["f05"] for m in zero_match_entities]
    zero_p = [m["precision"] for m in zero_match_entities]
    zero_r = [m["recall"] for m in zero_match_entities]

    single_f05 = [m["f05"] for m in singleton_entities]
    single_p = [m["precision"] for m in singleton_entities]
    single_r = [m["recall"] for m in singleton_entities]

    multi_f05 = [m["f05"] for m in multi_match_entities]
    multi_p = [m["precision"] for m in multi_match_entities]
    multi_r = [m["recall"] for m in multi_match_entities]

    print("\n" + "=" * 90, flush=True)
    print("CORRECTED MACRO-AVERAGED VALIDATION METRICS (1,000 TRAIN S1 ENTITIES)", flush=True)
    print("=" * 90, flush=True)
    print(f"Total Evaluated S1 Entities:       {len(sampled_s1_ids):,}", flush=True)
    print(f"Overall Macro-Averaged F0.5:       {np.mean(all_f05)*100:.2f}%", flush=True)
    print(f"Overall Macro-Averaged Precision:  {np.mean(all_p)*100:.2f}%", flush=True)
    print(f"Overall Macro-Averaged Recall:     {np.mean(all_r)*100:.2f}%", flush=True)
    print("-" * 90, flush=True)
    print(f"BREAKDOWN BY GROUND TRUTH CATEGORY:", flush=True)
    print(f"  1. Zero-Match Entities (GT = 0):", flush=True)
    print(f"     - Count:                      {len(zero_match_entities):,} ({len(zero_match_entities)/len(sampled_s1_ids)*100:.1f}%)", flush=True)
    print(f"     - Avg Precision:              {np.mean(zero_p)*100 if zero_p else 0.0:.2f}%", flush=True)
    print(f"     - Avg Recall:                 {np.mean(zero_r)*100 if zero_r else 0.0:.2f}%", flush=True)
    print(f"     - Avg F0.5:                   {np.mean(zero_f05)*100 if zero_f05 else 0.0:.2f}%", flush=True)
    print(f"  2. Singleton Entities (GT = 1):", flush=True)
    print(f"     - Count:                      {len(singleton_entities):,} ({len(singleton_entities)/len(sampled_s1_ids)*100:.1f}%)", flush=True)
    print(f"     - Avg Precision:              {np.mean(single_p)*100 if single_p else 0.0:.2f}%", flush=True)
    print(f"     - Avg Recall:                 {np.mean(single_r)*100 if single_r else 0.0:.2f}%", flush=True)
    print(f"     - Avg F0.5:                   {np.mean(single_f05)*100 if single_f05 else 0.0:.2f}%", flush=True)
    print(f"  3. Multi-Match Entities (GT > 1):", flush=True)
    print(f"     - Count:                      {len(multi_match_entities):,} ({len(multi_match_entities)/len(sampled_s1_ids)*100:.1f}%)", flush=True)
    print(f"     - Avg Precision:              {np.mean(multi_p)*100 if multi_p else 0.0:.2f}%", flush=True)
    print(f"     - Avg Recall:                 {np.mean(multi_r)*100 if multi_r else 0.0:.2f}%", flush=True)
    print(f"     - Avg F0.5:                   {np.mean(multi_f05)*100 if multi_f05 else 0.0:.2f}%", flush=True)
    print("=" * 90, flush=True)


if __name__ == "__main__":
    main()
