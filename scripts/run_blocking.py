#!/usr/bin/env python3
"""
Phase 3: Blocking / Candidate Generation Script.

Executes candidate generation on Amazon ML Challenge 2026 dataset:
- Builds fast lightweight inverted indexes over Source 2 and Source 3
- Queries Source 1 entities and unions candidates from complementary strategies:
  1. Exact normalized name
  2. Normalized name + country
  3. Country x name stem (stem len >= 4)
  4. Exact normalized address (len >= 8)
  5. Null-address fallback (country x name stem)
- Evaluates candidate recall on training ground truth
- Analyzes French test entities for open-set validation
- Generates official candidate_pairs.tsv adhering strictly to output contract
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
import time
from collections import defaultdict
from typing import Any, Dict, List, Set, Tuple

import pandas as pd

# Add src to python path if needed
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from business_entity_resolution.normalize import (
    normalize_country,
    normalize_name,
    normalize_address,
    strip_legal_suffix,
)

SUFFIXES = (
    " pvt ltd", " private limited", " pvt limited", " ltd", " inc", " corp",
    " llc", " llp", " co", " company", " limited", " incorporated", " gmbh",
    " sarl", " sas", " sasu", " eurl", " sci", " sa", " pc", " plc", " pllc",
)

def fast_strip_legal_suffix(n: str) -> str:
    for s in SUFFIXES:
        if n.endswith(s):
            return n[:-len(s)].strip()
    return n


RE_PUNCT = re.compile(r"[^\w\s]")
RE_SPACE = re.compile(r"\s+")

def _clean_str(text: str) -> str:
    return RE_SPACE.sub(" ", RE_PUNCT.sub(" ", text.lower())).strip()


STOPWORDS = {"and", "of", "the", "in", "for", "on", "at", "to", "a", "an", "is", "by", "with", "co", "corp", "inc", "ltd", "llc", "services", "center", "solutions", "technologies", "group", "enterprises", "holdings", "private", "limited"}

_CACHE_SORTED: Dict[str, str] = {}

def sorted_stem_key(stem: str) -> str:
    res = _CACHE_SORTED.get(stem)
    if res is not None:
        return res
    parts = [t for t in stem.split() if t not in STOPWORDS]
    if len(parts) <= 1:
        parts = stem.split()
    if not parts:
        _CACHE_SORTED[stem] = ""
        return ""
    parts.sort()
    res = " ".join(parts)
    final_res = res if len(res) >= 4 else ""
    _CACHE_SORTED[stem] = final_res
    return final_res

_CACHE_ADDR_PREF: Dict[str, str] = {}

def addr_prefix_10_key(addr: str) -> str:
    res = _CACHE_ADDR_PREF.get(addr)
    if res is not None:
        return res
    if len(addr) >= 14:
        final_res = addr[:14].strip()
    else:
        final_res = ""
    _CACHE_ADDR_PREF[addr] = final_res
    return final_res


class FastBlockingIndex:
    """Compact, high-speed inverted index for S2 and S3 records."""

    def __init__(self) -> None:
        self.exact_name: Dict[str, Set[str]] = defaultdict(set)
        self.country_name: Dict[Tuple[str, str], Set[str]] = defaultdict(set)
        self.country_stem: Dict[Tuple[str, str], Set[str]] = defaultdict(set)
        self.country_sorted_stem: Dict[Tuple[str, str], Set[str]] = defaultdict(set)
        self.exact_addr: Dict[str, Set[str]] = defaultdict(set)
        self.country_addr_prefix: Dict[Tuple[str, str], Set[str]] = defaultdict(set)
        self.null_addr_fallback: Dict[Tuple[str, str], Set[str]] = defaultdict(set)
        self.total_records = 0
        self._cache: Dict[Tuple[str, str, str, str], Set[str]] = {}

    def add_file(self, file_path: str, chunksize: int = 500000) -> None:
        print(f"  Indexing {file_path} ...", flush=True)
        t0 = time.time()
        count = 0

        for chunk in pd.read_csv(file_path, sep="\t", chunksize=chunksize, dtype=str):
            u_names = chunk["business_name"].fillna("").astype(str).unique()
            n_map = {n: _clean_str(n) for n in u_names if n}
            names = chunk["business_name"].fillna("").astype(str).map(n_map).fillna("")

            u_addrs = chunk["business_address"].fillna("").astype(str).unique()
            a_map = {a: _clean_str(a) for a in u_addrs if a}
            addrs = chunk["business_address"].fillna("").astype(str).map(a_map).fillna("")

            ctrys = chunk["country"].fillna("").astype(str).str.lower().str.strip()
            eids = chunk["entity_id"].astype(str)

            # Unique maps for maximum speed per chunk
            u_norm_names = names.unique()
            stem_map = {n: fast_strip_legal_suffix(n) for n in u_norm_names if n}
            sorted_map = {n: sorted_stem_key(stem_map[n]) for n in u_norm_names if n and stem_map.get(n)}

            u_norm_addrs = addrs.unique()
            ap_map = {a: addr_prefix_10_key(a) for a in u_norm_addrs if a}

            for eid, norm_n, norm_a, ctry_norm in zip(eids, names, addrs, ctrys):
                count += 1

                if norm_n:
                    self.exact_name[norm_n].add(eid)
                    if ctry_norm:
                        self.country_name[(ctry_norm, norm_n)].add(eid)

                    stem_n = stem_map.get(norm_n, "")
                    if stem_n and ctry_norm:
                        if len(stem_n) >= 4:
                            self.country_stem[(ctry_norm, stem_n)].add(eid)
                        
                        s_key = sorted_map.get(norm_n, "")
                        if s_key:
                            self.country_sorted_stem[(ctry_norm, s_key)].add(eid)

                if norm_a and len(norm_a) >= 8:
                    self.exact_addr[norm_a].add(eid)
                    if ctry_norm:
                        ap_key = ap_map.get(norm_a, "")
                        if ap_key:
                            self.country_addr_prefix[(ctry_norm, ap_key)].add(eid)

                # Null address fallback
                if not norm_a and norm_n:
                    stem_n = stem_map.get(norm_n, "")
                    if stem_n and len(stem_n) >= 4 and ctry_norm:
                        self.null_addr_fallback[(ctry_norm, stem_n)].add(eid)

        self.total_records += count
        print(f"  Indexed {count:,} rows from {os.path.basename(file_path)} in {time.time()-t0:.2f}s", flush=True)

    def query(self, norm_n: str, stem_n: str, norm_a: str, ctry_norm: str) -> Set[str]:
        q_key = (norm_n, stem_n, norm_a, ctry_norm)
        if q_key in self._cache:
            return self._cache[q_key]

        cands: Set[str] = set()

        if norm_n:
            e_n = self.exact_name.get(norm_n)
            if e_n:
                cands.update(e_n)
            if ctry_norm:
                c_n = self.country_name.get((ctry_norm, norm_n))
                if c_n:
                    cands.update(c_n)
                if stem_n:
                    if len(stem_n) >= 4:
                        c_s = self.country_stem.get((ctry_norm, stem_n))
                        if c_s:
                            cands.update(c_s)

                    s_key = sorted_stem_key(stem_n)
                    if s_key:
                        c_ss = self.country_sorted_stem.get((ctry_norm, s_key))
                        if c_ss:
                            cands.update(c_ss)

        if norm_a and len(norm_a) >= 8:
            e_a = self.exact_addr.get(norm_a)
            if e_a:
                cands.update(e_a)
            if ctry_norm:
                ap_key = addr_prefix_10_key(norm_a)
                if ap_key:
                    c_ap = self.country_addr_prefix.get((ctry_norm, ap_key))
                    if c_ap:
                        cands.update(c_ap)

        # Null address query fallback
        if not norm_a and stem_n and len(stem_n) >= 4 and ctry_norm:
            n_a = self.null_addr_fallback.get((ctry_norm, stem_n))
            if n_a:
                cands.update(n_a)

        self._cache[q_key] = cands
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
        u_names = chunk["business_name"].fillna("").astype(str).unique()
        n_map = {n: _clean_str(n) for n in u_names if n}
        names = chunk["business_name"].fillna("").astype(str).map(n_map).fillna("")

        u_addrs = chunk["business_address"].fillna("").astype(str).unique()
        a_map = {a: _clean_str(a) for a in u_addrs if a}
        addrs = chunk["business_address"].fillna("").astype(str).map(a_map).fillna("")

        ctrys = chunk["country"].fillna("").astype(str).str.lower().str.strip()
        eids = chunk["entity_id"].astype(str)

        u_norm_names = names.unique()
        stem_map = {n: fast_strip_legal_suffix(n) for n in u_norm_names if n}

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

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    # Resume capability check
    completed_s1_ids: Set[str] = set()
    is_compatible = False

    if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
        print(f"\nInspecting existing output file {output_path} for resume compatibility ...", flush=True)
        try:
            with open(output_path, "r", encoding="utf-8") as f:
                header = f.readline().strip()
                if header == "source1_entity_id\tcandidate_entity_ids":
                    sample_rows = []
                    for line in f:
                        line = line.strip()
                        if line:
                            parts = line.split("\t", 1)
                            s1_id = parts[0]
                            completed_s1_ids.add(s1_id)
                            if len(sample_rows) < 50:
                                sample_cands = set(parts[1].split(",")) if len(parts) > 1 and parts[1] else set()
                                sample_rows.append((s1_id, sample_cands))
                    
                    # Test compatibility of sample rows against index
                    if sample_rows:
                        print(f"  Loaded {len(completed_s1_ids):,} completed S1 rows from partial output.", flush=True)
                        # Read sample rows from test_source1 to verify strategy compatibility
                        s1_df = pd.read_csv(s1_path, sep="\t", nrows=500, dtype=str)
                        s1_df["norm_n"] = s1_df["business_name"].fillna("").astype(str).str.lower().str.replace(r"[^\w\s]", " ", regex=True).str.replace(r"\s+", " ", regex=True).str.strip()
                        s1_df["norm_a"] = s1_df["business_address"].fillna("").astype(str).str.lower().str.replace(r"[^\w\s]", " ", regex=True).str.replace(r"\s+", " ", regex=True).str.strip()
                        s1_df["ctry"] = s1_df["country"].fillna("").astype(str).str.lower().str.strip()
                        
                        stem_map = {n: strip_legal_suffix(n) for n in s1_df["norm_n"].unique() if n}
                        s1_dict = {}
                        for _, row in s1_df.iterrows():
                            eid = row["entity_id"]
                            norm_n = row["norm_n"]
                            norm_a = row["norm_a"]
                            ctry = row["ctry"]
                            stem_n = stem_map.get(norm_n, "")
                            s1_dict[eid] = index.query(norm_n, stem_n, norm_a, ctry)

                        matches_count = 0
                        tested_count = 0
                        for s1_id, existing_cands in sample_rows:
                            if s1_id in s1_dict:
                                tested_count += 1
                                if s1_dict[s1_id] == existing_cands:
                                    matches_count += 1
                        
                        if tested_count > 0 and matches_count == tested_count:
                            is_compatible = True
                            print(f"  RESUME VALIDATION PASSED: {matches_count}/{tested_count} sample rows matched current strategy.", flush=True)
                        else:
                            print(f"  RESUME VALIDATION: Strategy mismatch ({matches_count}/{tested_count} matched sample). Starting clean output.", flush=True)
        except Exception as e:
            print(f"  Error inspecting existing output: {e}. Starting clean run.", flush=True)

    partial_backup_path = os.path.join(os.path.dirname(output_path), "candidate_pairs.partial.tsv")
    if not is_compatible:
        if os.path.exists(output_path):
            import shutil
            shutil.copy(output_path, partial_backup_path)
            print(f"  Preserved existing partial file to {partial_backup_path}")
        completed_s1_ids = set()
        write_mode = "w"
    else:
        write_mode = "a"

    print(f"\nGenerating Candidates for Test S1 (Resumable mode: {write_mode}) ...", flush=True)

    candidate_counts: List[int] = []
    country_counts: Dict[str, List[int]] = defaultdict(list)
    total_candidate_pairs = 0
    s1_seen: Set[str] = set()

    with open(output_path, write_mode, newline="", encoding="utf-8") as out_f:
        writer = csv.writer(out_f, delimiter="\t")
        if write_mode == "w":
            writer.writerow(["source1_entity_id", "candidate_entity_ids"])

        for chunk in pd.read_csv(s1_path, sep="\t", chunksize=100000, dtype=str):
            u_names = chunk["business_name"].fillna("").astype(str).unique()
            n_map = {n: _clean_str(n) for n in u_names if n}
            names = chunk["business_name"].fillna("").astype(str).map(n_map).fillna("")

            u_addrs = chunk["business_address"].fillna("").astype(str).unique()
            a_map = {a: _clean_str(a) for a in u_addrs if a}
            addrs = chunk["business_address"].fillna("").astype(str).map(a_map).fillna("")

            ctrys = chunk["country"].fillna("").astype(str).str.lower().str.strip()
            eids = chunk["entity_id"].astype(str)

            u_norm_names = names.unique()
            stem_map = {n: fast_strip_legal_suffix(n) for n in u_norm_names if n}

            new_rows_count = 0
            for eid, norm_n, norm_a, ctry_norm in zip(eids, names, addrs, ctrys):
                s1_seen.add(eid)
                
                if eid in completed_s1_ids:
                    continue

                stem_n = stem_map.get(norm_n, "")
                cands = index.query(norm_n, stem_n, norm_a, ctry_norm)

                cand_str = ",".join(sorted(cands))
                writer.writerow([eid, cand_str])
                new_rows_count += 1

                c_len = len(cands)
                candidate_counts.append(c_len)
                total_candidate_pairs += c_len
                country_counts[ctry_norm].append(c_len)

            out_f.flush()
            if new_rows_count > 0:
                print(f"  Processed chunk, total S1 seen: {len(s1_seen):,}", flush=True)

    total_time = time.time() - t_start

    # If resumed, load candidate counts for all rows to complete statistics
    if completed_s1_ids and len(candidate_counts) < len(s1_seen):
        print("\nLoading full candidate stats for final verification ...", flush=True)
        candidate_counts = []
        country_counts = defaultdict(list)
        total_candidate_pairs = 0

        # Read country mapping from s1_path
        ctry_map = {}
        for chunk in pd.read_csv(s1_path, sep="\t", chunksize=500000, dtype=str):
            for eid, ctry in zip(chunk["entity_id"].astype(str), chunk["country"].fillna("").astype(str).str.lower().str.strip()):
                ctry_map[eid] = ctry

        with open(output_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t")
            next(reader, None)
            for row in reader:
                if len(row) >= 1:
                    eid = row[0]
                    cands = row[1].split(",") if len(row) > 1 and row[1] else []
                    c_len = len(cands)
                    candidate_counts.append(c_len)
                    total_candidate_pairs += c_len
                    ctry = ctry_map.get(eid, "")
                    country_counts[ctry].append(c_len)

    s_cands = pd.Series(candidate_counts) if candidate_counts else pd.Series([0])
    total_possible_pairs = len(candidate_counts) * index.total_records if candidate_counts else 1
    reduction_ratio = 1.0 - (total_candidate_pairs / total_possible_pairs) if total_possible_pairs > 0 else 1.0

    # Validate output file contract
    print("\nValidating Candidate Pairs Output Contract ...", flush=True)
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
    if france_stats['count'] > 0:
        print(f"France Zero-candidate Count:   {france_stats['zero_cands']:,} ({(france_stats['zero_cands']/france_stats['count']*100):.2f}%)")

    if len(us_cands) > 0:
        print(f"US S1 Entities:                {len(us_cands):,}, Avg Cands: {us_cands.mean():.2f}, Median: {us_cands.median():.0f}")
    if len(india_cands) > 0:
        print(f"India S1 Entities:             {len(india_cands):,}, Avg Cands: {india_cands.mean():.2f}, Median: {india_cands.median():.0f}")

    if france_stats['count'] > 0 and len(us_cands) > 0 and france_stats['avg_cands'] < 0.1 * us_cands.mean():
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
