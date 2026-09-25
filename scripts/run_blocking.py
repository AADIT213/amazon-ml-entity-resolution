#!/usr/bin/env python3
"""
Phase 3: Blocking / Candidate Generation Script.

Executes candidate generation on Amazon ML Challenge 2026 dataset:
- Builds fast lightweight inverted indexes over Source 2 and Source 3
- Queries Source 1 entities and unions candidates from complementary strategies
- Evaluates candidate recall on training ground truth
- Analyzes French test entities for open-set validation
- Generates official candidate_pairs.tsv adhering strictly to output contract
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from collections import defaultdict
from typing import Dict, List, Set, Tuple

import pandas as pd

# Add src to python path if needed
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from business_entity_resolution.normalize import (
    normalize_country,
    normalize_name,
    normalize_address,
    strip_legal_suffix,
)


def _addr_tokens(text: str) -> List[str]:
    _ADDR_SKIP = {
        "st", "rd", "dr", "ave", "blvd", "ln", "ct", "cir", "pl",
        "n", "s", "e", "w", "no", "fl", "apt", "ste", "rm",
        "bldg", "po", "box", "ne", "nw", "se", "sw",
    }
    return [t for t in text.split() if len(t) > 2 and t not in _ADDR_SKIP]


class FastBlockingIndex:
    """Compact, high-speed inverted index for S2 and S3 records."""

    def __init__(self) -> None:
        self.exact_name: Dict[str, Set[str]] = defaultdict(set)
        self.name_stem: Dict[str, Set[str]] = defaultdict(set)
        self.country_stem: Dict[Tuple[str, str], Set[str]] = defaultdict(set)
        self.country_name: Dict[Tuple[str, str], Set[str]] = defaultdict(set)
        self.exact_addr: Dict[str, Set[str]] = defaultdict(set)
        self.country_addr: Dict[Tuple[str, str], Set[str]] = defaultdict(set)
        self.null_addr_fallback: Dict[Tuple[str, str], Set[str]] = defaultdict(set)
        self.total_records = 0

    def add_file(self, file_path: str, chunksize: int = 500000) -> None:
        print(f"  Indexing {file_path} ...", flush=True)
        t0 = time.time()
        count = 0

        for chunk in pd.read_csv(file_path, sep="\t", chunksize=chunksize, dtype=str):
            names = (
                chunk["business_name"]
                .fillna("")
                .astype(str)
                .str.lower()
                .str.replace(r"[^\w\s]", " ", regex=True)
                .str.replace(r"\s+", " ", regex=True)
                .str.strip()
            )
            addrs = (
                chunk["business_address"]
                .fillna("")
                .astype(str)
                .str.lower()
                .str.replace(r"[^\w\s]", " ", regex=True)
                .str.replace(r"\s+", " ", regex=True)
                .str.strip()
            )
            ctrys = chunk["country"].fillna("").astype(str).str.lower().str.strip()
            eids = chunk["entity_id"].astype(str)

            # Unique name stem map for speed within chunk
            u_names = names.unique()
            stem_map = {n: strip_legal_suffix(n) for n in u_names if n}

            for eid, norm_n, norm_a, ctry_norm in zip(eids, names, addrs, ctrys):
                count += 1

                if norm_n:
                    self.exact_name[norm_n].add(eid)
                    if ctry_norm:
                        self.country_name[(ctry_norm, norm_n)].add(eid)

                    stem_n = stem_map.get(norm_n, "")
                    if stem_n and len(stem_n) >= 4:
                        self.name_stem[stem_n].add(eid)
                        if ctry_norm:
                            self.country_stem[(ctry_norm, stem_n)].add(eid)

                if norm_a and len(norm_a) >= 8:
                    self.exact_addr[norm_a].add(eid)
                    if ctry_norm:
                        self.country_addr[(ctry_norm, norm_a)].add(eid)

                # Null address fallback
                if not norm_a and norm_n:
                    stem_n = stem_map.get(norm_n, "")
                    if stem_n and len(stem_n) >= 4:
                        self.null_addr_fallback[(ctry_norm, stem_n)].add(eid)

        self.total_records += count
        print(f"  Indexed {count:,} rows from {os.path.basename(file_path)} in {time.time()-t0:.2f}s", flush=True)

    def query(self, norm_n: str, stem_n: str, norm_a: str, ctry_norm: str) -> Set[str]:
        cands: Set[str] = set()

        if norm_n:
            if norm_n in self.exact_name:
                cands.update(self.exact_name[norm_n])
            if ctry_norm and (ctry_norm, norm_n) in self.country_name:
                cands.update(self.country_name[(ctry_norm, norm_n)])

        if stem_n and len(stem_n) >= 4:
            if ctry_norm and (ctry_norm, stem_n) in self.country_stem:
                cands.update(self.country_stem[(ctry_norm, stem_n)])
            if stem_n in self.name_stem:
                cands.update(self.name_stem[stem_n])

        if norm_a and len(norm_a) >= 8:
            if ctry_norm and (ctry_norm, norm_a) in self.country_addr:
                cands.update(self.country_addr[(ctry_norm, norm_a)])
            if norm_a in self.exact_addr:
                cands.update(self.exact_addr[norm_a])

        # Null address query fallback
        if not norm_a and stem_n and len(stem_n) >= 4:
            if (ctry_norm, stem_n) in self.null_addr_fallback:
                cands.update(self.null_addr_fallback[(ctry_norm, stem_n)])

        return cands


def process_train(data_dir: str) -> Dict[str, Any]:
    print("\n========================================================")
    print("PHASE 3: TRAINING CANDIDATE GENERATION & EVALUATION")
    print("========================================================")
    t_start = time.time()

    s2_path = os.path.join(data_dir, "train", "train_source2.tsv")
    s3_path = os.path.join(data_dir, "train", "train_source3.tsv")
    s1_path = os.path.join(data_dir, "train", "train_source1.tsv")
    gt_path = os.path.join(data_dir, "train", "train_ground_truth.tsv")

    index = FastBlockingIndex()
    index.add_file(s2_path)
    index.add_file(s3_path)

    print(f"\nLoading Ground Truth from {gt_path} ...", flush=True)
    gt_map: Dict[str, Set[str]] = {}
    with open(gt_path, "r", encoding="utf-8", errors="ignore") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader, None)
        for row in reader:
            if len(row) >= 2:
                s1_id = row[0]
                m_ids = set(row[1].split(",")) if row[1] else set()
                gt_map[s1_id] = m_ids

    total_gt_matches = sum(len(v) for v in gt_map.values())
    print(f"Loaded GT: {len(gt_map):,} S1 entities, {total_gt_matches:,} total true matches.")

    print("\nGenerating Candidates & Measuring Recall for Train S1 ...", flush=True)
    t_eval = time.time()

    retained_gt_matches = 0
    candidate_counts: List[int] = []
    total_candidate_pairs = 0

    for chunk in pd.read_csv(s1_path, sep="\t", chunksize=500000, dtype=str):
        names = (
            chunk["business_name"]
            .fillna("")
            .astype(str)
            .str.lower()
            .str.replace(r"[^\w\s]", " ", regex=True)
            .str.replace(r"\s+", " ", regex=True)
            .str.strip()
        )
        addrs = (
            chunk["business_address"]
            .fillna("")
            .astype(str)
            .str.lower()
            .str.replace(r"[^\w\s]", " ", regex=True)
            .str.replace(r"\s+", " ", regex=True)
            .str.strip()
        )
        ctrys = chunk["country"].fillna("").astype(str).str.lower().str.strip()
        eids = chunk["entity_id"].astype(str)

        u_names = names.unique()
        stem_map = {n: strip_legal_suffix(n) for n in u_names if n}

        for eid, norm_n, norm_a, ctry_norm in zip(eids, names, addrs, ctrys):
            stem_n = stem_map.get(norm_n, "")
            cands = index.query(norm_n, stem_n, norm_a, ctry_norm)

            gt_set = gt_map.get(eid, set())
            retained_gt_matches += len(gt_set.intersection(cands))
            c_len = len(cands)
            candidate_counts.append(c_len)
            total_candidate_pairs += c_len

    eval_time = time.time() - t_eval
    total_time = time.time() - t_start

    recall = retained_gt_matches / total_gt_matches if total_gt_matches > 0 else 0.0
    s_cands = pd.Series(candidate_counts)
    total_possible_pairs = len(candidate_counts) * index.total_records
    reduction_ratio = 1.0 - (total_candidate_pairs / total_possible_pairs)

    stats = {
        "split": "train",
        "total_s1_entities": len(candidate_counts),
        "total_index_entities": index.total_records,
        "total_gt_matches": total_gt_matches,
        "retained_gt_matches": retained_gt_matches,
        "candidate_recall": recall,
        "total_candidate_pairs": total_candidate_pairs,
        "avg_candidates_per_s1": float(s_cands.mean()),
        "median_candidates_per_s1": float(s_cands.median()),
        "p90_candidates": float(s_cands.quantile(0.90)),
        "p95_candidates": float(s_cands.quantile(0.95)),
        "p99_candidates": float(s_cands.quantile(0.99)),
        "max_candidates": int(s_cands.max()),
        "zero_candidate_count": int((s_cands == 0).sum()),
        "reduction_ratio": float(reduction_ratio),
        "runtime_seconds": float(total_time),
    }

    print("\n--- TRAIN EVALUATION RESULTS ---")
    print(f"Candidate Recall:              {recall*100:.2f}% ({retained_gt_matches:,} / {total_gt_matches:,})")
    print(f"Total Candidate Pairs:         {total_candidate_pairs:,}")
    print(f"Average Candidates / S1:       {stats['avg_candidates_per_s1']:.2f}")
    print(f"Median Candidates / S1:        {stats['median_candidates_per_s1']:.0f}")
    print(f"P90 Candidates:                {stats['p90_candidates']:.0f}")
    print(f"P95 Candidates:                {stats['p95_candidates']:.0f}")
    print(f"P99 Candidates:                {stats['p99_candidates']:.0f}")
    print(f"Max Candidates:                {stats['max_candidates']}")
    print(f"Zero-candidate Count:          {stats['zero_candidate_count']:,}")
    print(f"Candidate Reduction Ratio:     {reduction_ratio*100:.6f}%")
    print(f"Total Runtime:                 {total_time:.2f}s")

    return stats


def process_test(data_dir: str, output_path: str) -> Dict[str, Any]:
    print("\n========================================================")
    print("PHASE 3: TEST CANDIDATE GENERATION & FRANCE VALIDATION")
    print("========================================================")
    t_start = time.time()

    s2_path = os.path.join(data_dir, "test", "test_source2.tsv")
    s3_path = os.path.join(data_dir, "test", "test_source3.tsv")
    s1_path = os.path.join(data_dir, "test", "test_source1.tsv")

    index = FastBlockingIndex()
    index.add_file(s2_path)
    index.add_file(s3_path)

    print(f"\nGenerating Candidates for Test S1 & Writing {output_path} ...", flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    candidate_counts: List[int] = []
    country_counts: Dict[str, List[int]] = defaultdict(list)
    total_candidate_pairs = 0
    s1_seen: Set[str] = set()

    with open(output_path, "w", newline="", encoding="utf-8") as out_f:
        writer = csv.writer(out_f, delimiter="\t")
        writer.writerow(["source1_entity_id", "candidate_entity_ids"])

        for chunk in pd.read_csv(s1_path, sep="\t", chunksize=500000, dtype=str):
            names = (
                chunk["business_name"]
                .fillna("")
                .astype(str)
                .str.lower()
                .str.replace(r"[^\w\s]", " ", regex=True)
                .str.replace(r"\s+", " ", regex=True)
                .str.strip()
            )
            addrs = (
                chunk["business_address"]
                .fillna("")
                .astype(str)
                .str.lower()
                .str.replace(r"[^\w\s]", " ", regex=True)
                .str.replace(r"\s+", " ", regex=True)
                .str.strip()
            )
            ctrys = chunk["country"].fillna("").astype(str).str.lower().str.strip()
            eids = chunk["entity_id"].astype(str)

            u_names = names.unique()
            stem_map = {n: strip_legal_suffix(n) for n in u_names if n}

            for eid, norm_n, norm_a, ctry_norm in zip(eids, names, addrs, ctrys):
                s1_seen.add(eid)
                stem_n = stem_map.get(norm_n, "")
                cands = index.query(norm_n, stem_n, norm_a, ctry_norm)

                cand_str = ",".join(sorted(cands))
                writer.writerow([eid, cand_str])

                c_len = len(cands)
                candidate_counts.append(c_len)
                total_candidate_pairs += c_len
                country_counts[ctry_norm].append(c_len)

    total_time = time.time() - t_start
    s_cands = pd.Series(candidate_counts)
    total_possible_pairs = len(candidate_counts) * index.total_records
    reduction_ratio = 1.0 - (total_candidate_pairs / total_possible_pairs)

    # Validate output file contract
    print("\nValidating Candidate Pairs Output Contract ...", flush=True)
    assert len(s1_seen) == len(candidate_counts), "Duplicate S1 IDs processed!"
    assert os.path.exists(output_path), "candidate_pairs.tsv was not written!"
    assert os.path.getsize(output_path) > 0, "candidate_pairs.tsv is empty!"

    # Breakdown by country (specifically France check)
    france_cands = pd.Series(country_counts.get("france", []))
    us_cands = pd.Series(country_counts.get("us", []))
    india_cands = pd.Series(country_counts.get("india", []))

    france_stats = {
        "count": len(france_cands),
        "zero_cands": int((france_cands == 0).sum()) if len(france_cands) > 0 else 0,
        "avg_cands": float(france_cands.mean()) if len(france_cands) > 0 else 0.0,
        "median_cands": float(france_cands.median()) if len(france_cands) > 0 else 0.0,
        "max_cands": int(france_cands.max()) if len(france_cands) > 0 else 0,
    }

    stats = {
        "split": "test",
        "total_s1_entities": len(candidate_counts),
        "total_candidate_pairs": total_candidate_pairs,
        "avg_candidates_per_s1": float(s_cands.mean()),
        "median_candidates_per_s1": float(s_cands.median()),
        "p90_candidates": float(s_cands.quantile(0.90)),
        "p95_candidates": float(s_cands.quantile(0.95)),
        "p99_candidates": float(s_cands.quantile(0.99)),
        "max_candidates": int(s_cands.max()),
        "zero_candidate_count": int((s_cands == 0).sum()),
        "reduction_ratio": float(reduction_ratio),
        "runtime_seconds": float(total_time),
        "france_stats": france_stats,
    }

    print("\n--- TEST EVALUATION & FRANCE SANITY RESULTS ---")
    print(f"Total Test S1 Entities:       {len(candidate_counts):,}")
    print(f"Total Candidate Pairs:         {total_candidate_pairs:,}")
    print(f"Average Candidates / S1:       {stats['avg_candidates_per_s1']:.2f}")
    print(f"Median Candidates / S1:        {stats['median_candidates_per_s1']:.0f}")
    print(f"P90 Candidates:                {stats['p90_candidates']:.0f}")
    print(f"Max Candidates:                {stats['max_candidates']}")
    print(f"Zero-candidate Count:          {stats['zero_candidate_count']:,}")
    print(f"Candidate Reduction Ratio:     {reduction_ratio*100:.6f}%")

    print("\n--- PER-COUNTRY TEST CANDIDATE BREAKDOWN ---")
    print(f"France S1 Entities:            {france_stats['count']:,}")
    print(f"France Avg Candidates:         {france_stats['avg_cands']:.2f}")
    print(f"France Median Candidates:      {france_stats['median_cands']:.0f}")
    print(f"France Zero-candidate Count:   {france_stats['zero_cands']:,} ({(france_stats['zero_cands']/france_stats['count']*100):.2f}%)")

    print(f"US S1 Entities:                {len(us_cands):,}, Avg Cands: {us_cands.mean():.2f}, Median: {us_cands.median():.0f}")
    print(f"India S1 Entities:             {len(india_cands):,}, Avg Cands: {india_cands.mean():.2f}, Median: {india_cands.median():.0f}")

    if france_stats['avg_cands'] < 0.1 * us_cands.mean():
        print("\nWARNING: French candidate volume is suspiciously low compared to US!")
    else:
        print("\nSANITY CHECK PASSED: French candidates generated appropriately (Open-Set functional).")

    print(f"\nTotal Runtime:                 {total_time:.2f}s")
    print(f"Candidate Pairs Written To:    {output_path}")

    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 3 Candidate Generation")
    parser.add_argument("--data_dir", type=str, default="data", help="Directory containing train/ and test/")
    parser.add_argument("--split", type=str, choices=["train", "test", "all"], default="all", help="Split to execute")
    parser.add_argument("--output", type=str, default="output/candidate_pairs.tsv", help="Output file path for test candidates")

    args = parser.parse_args()

    if args.split in ["train", "all"]:
        process_train(args.data_dir)

    if args.split in ["test", "all"]:
        process_test(args.data_dir, args.output)
        # Also copy candidate_pairs.tsv to root if output is inside output/
        root_target = "candidate_pairs.tsv"
        if args.output != root_target and os.path.exists(args.output):
            import shutil
            shutil.copy(args.output, root_target)
            print(f"Copied candidate_pairs.tsv to root: {root_target}")


if __name__ == "__main__":
    main()
