#!/usr/bin/env python3
import os
import sys
import time
import re
import pandas as pd

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
    if not isinstance(text, str) or not text:
        return ""
    return RE_SPACE.sub(" ", RE_PUNCT.sub(" ", text.lower())).strip()

STOPWORDS = {"and", "of", "the", "in", "for", "on", "at", "to", "a", "an", "is", "by", "with", "co", "corp", "inc", "ltd", "llc", "services", "center", "solutions", "technologies", "group", "enterprises", "holdings", "private", "limited"}

def sorted_stem_key(stem: str) -> str:
    parts = [t for t in stem.split() if t not in STOPWORDS]
    if len(parts) <= 1:
        parts = stem.split()
    if not parts:
        return ""
    parts.sort()
    res = " ".join(parts)
    return res if len(res) >= 4 else ""

def first_2_tokens_min6_key(stem: str) -> str:
    parts = [t for t in stem.split() if t not in STOPWORDS]
    if len(parts) >= 2:
        p12 = parts[0] + " " + parts[1]
        if len(p12) >= 6:
            return p12
    elif len(parts) == 1 and len(parts[0]) >= 5:
        return parts[0]
    return ""

def addr_prefix_14_key(addr: str) -> str:
    if len(addr) >= 14:
        return addr[:14].strip()
    return ""

def run_test():
    t0 = time.time()
    gt_path = "data/train/train_ground_truth.tsv"
    s1_path = "data/train/train_source1.tsv"
    s2_path = "data/train/train_source2.tsv"
    s3_path = "data/train/train_source3.tsv"

    gt_pairs = []
    matched_ids_needed = set()
    s1_ids_needed = set()

    gt_df = pd.read_csv(gt_path, sep="\t", dtype=str)
    for s1_id, m_str in zip(gt_df["source1_entity_id"], gt_df["matched_entity_ids"]):
        s1_id = str(s1_id)
        if pd.notna(m_str) and m_str:
            for m_id in str(m_str).split(","):
                m_id = m_id.strip()
                if m_id:
                    gt_pairs.append((s1_id, m_id))
                    s1_ids_needed.add(s1_id)
                    matched_ids_needed.add(m_id)

    print(f"Loaded {len(gt_pairs):,} positive GT pairs.", flush=True)

    s1_attrs = {}
    for chunk in pd.read_csv(s1_path, sep="\t", chunksize=500000, dtype=str):
        eids = chunk["entity_id"].astype(str)
        raw_names = chunk["business_name"].fillna("").astype(str)
        raw_addrs = chunk["business_address"].fillna("").astype(str)
        ctrys = chunk["country"].fillna("").astype(str)

        for eid, rn, ra, ct in zip(eids, raw_names, raw_addrs, ctrys):
            if eid in s1_ids_needed:
                nn = _clean_str(rn)
                na = _clean_str(ra)
                cn = ct.lower().strip()
                st = fast_strip_legal_suffix(nn)
                s1_attrs[eid] = {
                    "norm_n": nn,
                    "stem_n": st,
                    "sorted_st": sorted_stem_key(st),
                    "first2_st": first_2_tokens_min6_key(st),
                    "norm_a": na,
                    "addr_pref": addr_prefix_14_key(na),
                    "ctry": cn
                }

    match_attrs = {}
    for path in [s2_path, s3_path]:
        for chunk in pd.read_csv(path, sep="\t", chunksize=500000, dtype=str):
            eids = chunk["entity_id"].astype(str)
            raw_names = chunk["business_name"].fillna("").astype(str)
            raw_addrs = chunk["business_address"].fillna("").astype(str)
            ctrys = chunk["country"].fillna("").astype(str)

            for eid, rn, ra, ct in zip(eids, raw_names, raw_addrs, ctrys):
                if eid in matched_ids_needed:
                    nn = _clean_str(rn)
                    na = _clean_str(ra)
                    cn = ct.lower().strip()
                    st = fast_strip_legal_suffix(nn)
                    match_attrs[eid] = {
                        "norm_n": nn,
                        "stem_n": st,
                        "sorted_st": sorted_stem_key(st),
                        "first2_st": first_2_tokens_min6_key(st),
                        "norm_a": na,
                        "addr_pref": addr_prefix_14_key(na),
                        "ctry": cn
                    }

    total_gt = len(gt_pairs)
    hits_exact_name = []
    hits_country_stem = []
    hits_exact_addr = []
    hits_null_addr = []
    hits_sorted_stem = []
    hits_first2_stem = []
    hits_addr_pref = []

    for s1_id, m_id in gt_pairs:
        s1 = s1_attrs.get(s1_id)
        m = match_attrs.get(m_id)
        if not s1 or not m:
            continue

        h_ename = bool(s1["norm_n"] and s1["norm_n"] == m["norm_n"])
        h_cstem = bool(s1["norm_n"] and m["norm_n"] and s1["ctry"] and s1["ctry"] == m["ctry"] and s1["stem_n"] and s1["stem_n"] == m["stem_n"] and len(s1["stem_n"]) >= 4)
        h_eaddr = bool(s1["norm_a"] and m["norm_a"] and len(s1["norm_a"]) >= 8 and s1["norm_a"] == m["norm_a"])
        h_nulla = bool(not s1["norm_a"] and not m["norm_a"] and s1["norm_n"] and m["norm_n"] and s1["ctry"] and s1["ctry"] == m["ctry"] and s1["stem_n"] and s1["stem_n"] == m["stem_n"] and len(s1["stem_n"]) >= 4)

        h_csort = bool(s1["ctry"] and s1["ctry"] == m["ctry"] and s1["sorted_st"] and s1["sorted_st"] == m["sorted_st"])
        h_cfirst2 = bool(s1["ctry"] and s1["ctry"] == m["ctry"] and s1["first2_st"] and s1["first2_st"] == m["first2_st"])
        h_caddrp = bool(s1["ctry"] and s1["ctry"] == m["ctry"] and s1["addr_pref"] and s1["addr_pref"] == m["addr_pref"])

        hits_exact_name.append(h_ename)
        hits_country_stem.append(h_cstem)
        hits_exact_addr.append(h_eaddr)
        hits_null_addr.append(h_nulla)
        hits_sorted_stem.append(h_csort)
        hits_first2_stem.append(h_cfirst2)
        hits_addr_pref.append(h_caddrp)

    strategies = [
        ("1. exact_name", hits_exact_name),
        ("2. country_stem (len>=4)", hits_country_stem),
        ("3. exact_addr (len>=8)", hits_exact_addr),
        ("4. null_addr_fallback", hits_null_addr),
        ("5. country_sorted_stem", hits_sorted_stem),
        ("6. country_addr_prefix (len>=14)", hits_addr_pref),
        ("7. country_first2_stem (min len>=6)", hits_first2_stem),
    ]

    print("\nREFINED STRATEGIES EVALUATION WITH FIRST2_STEM:", flush=True)
    cum = [False] * total_gt
    for name, flags in strategies:
        hits = sum(flags)
        inc = sum(f and not c for f, c in zip(flags, cum))
        for i in range(total_gt):
            if flags[i]:
                cum[i] = True
        u_rec = sum(cum) / total_gt
        print(f"{name:<35} | Hits: {hits:<10,} | Inc: {inc:<10,} | Union: {u_rec*100:.2f}%", flush=True)

if __name__ == "__main__":
    run_test()
