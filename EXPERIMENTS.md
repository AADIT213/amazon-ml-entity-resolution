# EXPERIMENTS.md — Experiment Log & Measured Metrics

All metrics recorded here are derived from real competition data. Per `AGENTS.md`, no claim of improvement or performance is accepted without a corresponding entry and measurement in this log.

---

## Phase 1: Dataset Audit Benchmark (Baseline Data Characteristics)

- **Date**: 2026-09-25
- **Branch**: `feat/dataset-audit`
- **Script**: `scripts/dataset_audit.py`
- **Output Document**: `docs/dataset_audit_report.md`

### 1. Data Scale & Volumes
| Dataset Split | File | Row Count | File Size (MB) |
|---|---|---|---|
| Train | `train_ground_truth.tsv` | 2,206,821 | 121.1 MB |
| Train | `train_source1.tsv` | 2,206,821 | 200.3 MB |
| Train | `train_source2.tsv` | 5,034,616 | 466.6 MB |
| Train | `train_source3.tsv` | 5,285,603 | 480.4 MB |
| Test | `test_source1.tsv` | 1,732,544 | 166.9 MB |
| Test | `test_source2.tsv` | 4,887,273 | 485.9 MB |
| Test | `test_source3.tsv` | 5,082,316 | 482.6 MB |
| **Total** | **All 7 Files** | **26,435,994** | **2,403.8 MB** |

### 2. Key Ground Truth Metrics
- **Total Positive Match Pairs**: 7,638,365
- **Average Matches per S1 Entity**: 3.46 (Range: 0 to 11 matches)
- **Singleton Rate (0 matches / no match)**: 123,247 entities (5.58%)
- **1-Match Rate**: 119,157 entities (5.40%)
- **2-Match Rate**: 375,212 entities (17.00%)
- **3+ Matches Rate**: 1,589,205 entities (72.01%)
- **Source Breakdown**: 3,693,619 S2 pairs (48.4%), 3,944,746 S3 pairs (51.6%)
- **Co-occurrence**: 80.48% match both S2 and S3; 6.48% match only S2; 7.45% match only S3.

### 3. Open-Set Country Discovery
- **Train Countries**: US (59.95%), India (40.05%)
- **Test Countries**: India (47.24%), US (38.28%), France (14.48% — 1,694,445 records)
- **Constraint**: Open-set evaluation required. Hardcoded country sets or strict train-only country filtering will fail on ~15% of test records.

### 4. Integrity & Leakage Verification
- Unreferenced ground truth IDs in Source 2 or Source 3: **0**
- Test IDs appearing in Train Ground Truth: **0**
- Duplicate IDs per source file: **0**
- Duplicate rows per source file: **0**

---

## Phase 2: Preprocessing / Normalization

- **Date**: 2026-09-25
- **Branch**: `feat/normalization`
- **Modules**:
  - `src/business_entity_resolution/text_utils.py` — base Unicode/ASCII utilities
  - `src/business_entity_resolution/name_norm.py` — business name normalization
  - `src/business_entity_resolution/address_norm.py` — address normalization
  - `src/business_entity_resolution/normalize.py` — unified record/DataFrame API
- **Test Suite**: `tests/test_text_utils.py`, `tests/test_name_norm.py`, `tests/test_address_norm.py`, `tests/test_normalize.py`

### Unit Test Results
- **Total tests**: 27
- **Passed**: 27
- **Failed**: 0
- **Runtime**: ~0.86s

### Normalization Design Decisions
| Decision | Rationale |
|---|---|
| Hyphens and slashes replaced with space | Prevents unintended token merging (e.g. `PAYNE-ENTERPRISES` → `payne enterprises`); improves Jaccard overlap |
| Legal suffixes standardized (not stripped) in `normalize_name` | Preserves match signal (e.g. `pvt ltd` vs `ltd` are different entity types); separate `strip_legal_suffix` available for blocking keys |
| `strip_legal_suffix` is a separate function | Blocking (Phase 3) can use it for recall-maximizing keys without permanently destroying suffix info |
| `has_address` flag added to each record | Directly encodes the audit finding that ~3% of S2/S3 records have null addresses; blocking must fall back to name-only for these |
| Country normalization: no enum, no hardcoding | France is test-only (14.48% of test records); `normalize_country` preserves value with only whitespace/unicode cleaning |
| Devanagari transliteration (rule-based, local) | Found in real matched pair: S1 `Hotel Enterprises Limited` vs S2 `होटल एंटरप्राइजेज लिमिटेड`; no external API used |
| `saint` → `st` in address | French address term, required for `Boulevard Saint-Germain` to normalize consistently with `St Germain` |
| `etage` → `fl` in address | French word for "floor"; normalized to canonical `fl` token for France (open-set) support |
| All functions are pure / idempotent | Verified by test: `normalize_name(normalize_name(x)) == normalize_name(x)` for all samples |
| Raw input fields always preserved | `normalize_record()` returns new dict with original keys untouched + new normalized keys added |

### Representative Before/After Examples (Real Data)
| Field | Raw S1 | Raw Match | After S1 Norm | After Match Norm | Key Overlap |
|---|---|---|---|---|---|
| Name | `Laxmi Golden Investments Private Limited` | `Laxmi Golden Investments` | `laxmi golden investments pvt ltd` | `laxmi golden investments` | stem = `laxmi golden investments` ✓ |
| Name | `Hendricks and Flowers Inc` | `Hendricks and  Flowers Inc` | `hendricks and flowers inc` | `hendricks and flowers inc` | exact ✓ |
| Name | `Hotel Enterprises Limited` | `होटल एंटरप्राइजेज लिमिटेड` | `hotel enterprises ltd` | `hotl entrpraijej ltd` | token overlap partial |
| Address | `3315 Fremont Street, Peoria, IL` | `3315 FREMONT SAINT, PEORIA, IL` | `3315 fremont st peoria il` | `3315 fremont st peoria il` | exact ✓ |
| Address | `1795 Westchester Drive, High Point, NC` | *(null)* | `1795 westchester dr high point nc` | `""` | has_address=False |
| Address | `11Th Floor, N1 Block Embassy, Bangalore` | `11Th Floor, N1 Block Embassy, Bangalore KA` | `11 fl n1 block embassy bangalore karnataka` | `11 fl n1 block embassy bangalore ka` | high overlap ✓ |

---

## Phase 3: Candidate Generation & Blocking (Final Model Set)

- **Date**: 2026-09-26
- **Branch**: `feat/blocking`
- **Script**: `scripts/run_blocking.py`
- **Output Files**: `output/candidate_pairs.tsv` (1,732,545 lines, 4.34 GB) & `candidate_pairs.tsv` (root copy)
- **Test Suite**: `tests/test_blocking.py` (42/42 tests passing), total test suite 69/69 passing

### 1. Final 7-Strategy Blocking Architecture
The finalized blocking set achieves high recall while maintaining a compact index memory footprint (< 3 GB RAM) and fast execution (< 11 minutes for 1.73M entities against 9.97M records):
1. **Exact Normalized Name Cross-Country**: `exact_name` (norm_n)
2. **Country x Exact Name**: `country:name_norm`
3. **Country x Name Stem**: `country:name_stem` (stem len >= 4, fast suffix stripping)
4. **Country x Sorted Token Stem**: `country:sorted_stem` (order-invariant token matching)
5. **Exact Normalized Address**: `exact_addr` (norm_a, len >= 8)
6. **Country x Address Prefix**: `country:addr_prefix` (address prefix 14 chars)
7. **Null-Address Fallback**: Name-stem candidate generation when address is absent

### 2. Candidate Recall Benchmark (Training Ground Truth)
- **Evaluated S1 Entities**: 2,206,821
- **Ground Truth Matches**: 7,638,365
- **Final Candidate Recall**: **75.40%**

### 3. Test Set Candidate Generation Results
- **Evaluated Test S1 Entities**: 1,732,544
- **Total Generated Candidate Pairs**: 334,668,988
- **Candidate Reduction Ratio**: **99.998062%**
- **Average Candidates / S1**: **193.17**
- **Median Candidates / S1**: **7.0**
- **P75 Candidates / S1**: **52.0**
- **P90 Candidates / S1**: **112.0**
- **P95 Candidates / S1**: **353.0**
- **P99 Candidates / S1**: **8,878.0**
- **Max Candidates / S1**: **13,347**
- **Zero-candidate S1 Count**: **39,449 (2.28%)**
- **Test Candidate Generation Runtime**: **655.61s (~10.9 min)**

#### Open-Set Country Breakdown (Test Split)
| Country | Test S1 Count | Avg Candidates / S1 | Median Candidates | Zero-Candidate Count (%) |
|---|---|---|---|---|
| **France (Open-Set)** | 259,452 | 842.91 | 14 | 3,142 (1.21%) |
| **US** | 663,106 | 39.40 | 5 | -- |
| **India** | 809,986 | 110.92 | 10 | -- |

### 4. Full Submission Contract Validation
- **Validator Command**: `scripts/validate_submission.py --test-dir data/test --check-ids`
- **Target File**: `output/candidate_pairs.tsv` (1,732,544 required S1 rows)
- **Validation Result**: **PASS — no blocking issues found. Safe to submit.**
- **ID Existence Check**: Verified all 334,668,988 candidate IDs against `test_source2.tsv` (4,887,273 rows) and `test_source3.tsv` (5,082,316 rows). 100% of candidate IDs exist and are valid.

---

## Phase 4: Pair Feature Engineering (Sample Benchmark)

- **Date**: 2026-09-26
- **Branch**: `feat/features`
- **Script**: `scripts/run_features.py`
- **Modules**:
  - `src/business_entity_resolution/features.py` — lightweight, vectorized pair feature extractor
- **Test Suite**: `tests/test_features.py` (8/8 tests passing), total test suite 77/77 passing
- **Output Sample File**: `output/features_sample_10000.tsv` (840.6 KB, 10,000 rows)

### 1. Lightweight Feature Set (12 Features)
- **Name Features (5)**:
  1. `exact_name_match`: Exact normalized name equality (1.0 or 0.0)
  2. `name_token_jaccard`: Word token set Jaccard similarity [0, 1]
  3. `name_char_similarity`: Normalized Levenshtein character similarity [0, 1]
  4. `name_len_diff`: Absolute difference in string length
  5. `name_token_count_diff`: Absolute difference in token counts
- **Address Features (4)**:
  6. `exact_addr_match`: Exact normalized address equality (1.0 or 0.0)
  7. `addr_token_jaccard`: Address word token set Jaccard similarity [0, 1]
  8. `addr_numeric_overlap`: Jaccard overlap of numeric digit groups (e.g. house/pin numbers)
  9. `addr_len_diff`: Absolute difference in address length
- **Cross-Field Features (3)**:
  10. `country_match`: Country equality (1.0 or 0.0)
  11. `missing_name_flag`: Flag indicating missing/empty name on either entity
  12. `missing_addr_flag`: Flag indicating missing/empty address on either entity

### 2. Measured 10,000-Pair Benchmark Metrics
- **Pairs Processed**: 10,000
- **Referenced Unique Entities**: 10,004 (4 S1, 5,122 S2, 4,878 S3)
- **Candidate Streaming Time**: 0.005s
- **Entity Loading & Normalization Time**: 14.77s
- **Feature Computation Time**: 0.2603s
- **Feature Extraction Throughput**: **38,418.1 pairs/second**
- **Total Benchmark Runtime**: 15.14s
- **Peak Process Memory**: ~1.62 MB (feature matrix)
---

## Phase 5: Baseline Pair Matching Model (10,000-S1 Streaming Benchmark)

- **Date**: 2026-09-26
- **Branch**: `feat/model`
- **Script**: `scripts/train_baseline.py`
- **Modules**:
  - `src/business_entity_resolution/model.py` — `BaselineMatchClassifier` (balanced Logistic Regression)
  - `src/business_entity_resolution/features.py` — 12 lightweight features
- **Test Suite**: `tests/test_model.py` (5/5 tests passing), total test suite 82/82 passing
- **Model Artifact**: `models/baseline_logreg.joblib` (1.3 KB)

### 1. Training Dataset & Strategy
- **S1 Sample**: 10,000 Source 1 entities (deterministic uniform sample, seed 42)
- **Candidate Generator**: Improved Phase 3 blocking strategies (7 complementary keys: exact name, country stem, exact addr, country sorted stem, country addr prefix, null addr fallback)
- **Ground Truth Matches**: 34,768 true positives loaded for sampled S1 entities
- **Candidate Generation Yield**:
  * Positive candidates: 24,320
  * Negative candidates: 66,066 (capped at max 15 negatives / S1 entity)
  * Total candidate pairs: 90,386
  * Positive / Negative ratio: 1 : 2.72
  * Unique target entities retained: 462,530
- **Entity-Grouped Split (80/20)**:
  * Train: 8,000 S1 entities -> 72,211 pairs (19,474 pos / 52,737 neg)
  * Validation: 2,000 S1 entities -> 18,175 pairs (4,846 pos / 13,329 neg)

### 2. Feature Extraction & Model Training Throughput
- **Feature Computation Time**: 1.6470s on 90,386 pairs (**54,879.6 pairs/sec**)
- **Model Fit Time**: 0.1671s
- **Peak Process Memory**: ~4.14 MB (feature matrix)
- **Source 2 Scan Time**: 1,150.35s (5,034,616 rows)
- **Source 3 Scan Time**: 1,393.64s (5,285,603 rows)
- **Total Pipeline Runtime**: 2,552.16s (~42.5 min)

### 3. Learned Feature Weights (Logistic Regression)
| Feature Name | Category | Learned Coefficient |
|---|---|---|
| `addr_token_jaccard` | Address | **+3.3397** |
| `name_token_jaccard` | Name | **+1.9717** |
| `addr_numeric_overlap` | Address | **+1.7052** |
| `name_char_similarity` | Name | **+0.8117** |
| `missing_addr_flag` | Cross-field | **+0.6715** |
| `addr_len_diff` | Address | +0.3879 |
| `name_token_count_diff` | Name | +0.3769 |
| `country_match` | Cross-field | +0.1319 |
| `missing_name_flag` | Cross-field | +0.0000 |
| `exact_name_match` | Name | -0.5137 |
| `exact_addr_match` | Address | -0.5344 |
| `name_len_diff` | Name | -0.5599 |

### 4. Validation Split Threshold / F0.5 Evaluation Table
Evaluated on held-out 20% validation split (18,175 candidate pairs across 2,000 S1 entities):

| Threshold | Precision | Recall | **F0.5** | Predicted Matches | TP | FP | FN | TN |
|---|---|---|---|---|---|---|---|---|
| 0.10 | 72.97% | 99.53% | 77.08% | 6,610 | 4,823 | 1,787 | 23 | 11,542 |
| 0.20 | 78.32% | 99.01% | 81.74% | 6,126 | 4,798 | 1,328 | 48 | 12,001 |
| 0.30 | 81.23% | 98.06% | 84.12% | 5,850 | 4,752 | 1,098 | 94 | 12,231 |
| 0.40 | 83.27% | 96.84% | 85.67% | 5,636 | 4,693 | 943 | 153 | 12,386 |
| 0.50 | 85.06% | 95.30% | 86.93% | 5,429 | 4,618 | 811 | 228 | 12,518 |
| 0.60 | 86.80% | 93.07% | 87.98% | 5,196 | 4,510 | 686 | 336 | 12,643 |
| 0.70 | 89.01% | 90.94% | 89.39% | 4,951 | 4,407 | 544 | 439 | 12,785 |
| 0.80 | 91.96% | 87.31% | 90.99% | 4,601 | 4,231 | 370 | 615 | 12,959 |
| **0.90 (Optimal)** | **95.29%** | **81.88%** | **92.27%** | **4,164** | **3,968** | **196** | **878** | **13,133** |

### 5. Optimal Baseline Performance Summary
- **Optimal Probability Threshold**: **0.90**
- **Validation Precision**: **95.29%**
- **Validation Recall**: **81.88%**
- **Validation F0.5 Score**: **92.27%**
- **Confusion Matrix**: TP = 3,968, FP = 196, FN = 878, TN = 13,133

---

## Phase 6: Macro-Averaged Validation Evaluation (1,000-S1 Held-Out Slice)

- **Date**: 2026-09-27
- **Scripts**: `scripts/evaluate_train_slice.py`, `scripts/compute_macro_metrics.py`
- **Model**: `models/baseline_logreg.joblib` (frozen, threshold 0.90)
- **Evaluation Metric**: Official per-entity Macro $F_{0.5} = \frac{1}{N} \sum_{i=1}^N \frac{1.25 \cdot P_i \cdot R_i}{0.25 \cdot P_i + R_i}$

### 1. Overall Macro-Averaged Results
- **Evaluated S1 Entities**: 1,000
- **Total Candidate Pairs Generated**: 70,032
- **Candidate Blocking Recall**: 69.29% (2,507 / 3,618 ground truth pairs)
- **Overall Macro $F_{0.5}$**: **74.16%**
- **Overall Macro Precision**: **84.37%**
- **Overall Macro Recall**: **58.05%**

### 2. Breakdown by Ground Truth Entity Category
| Category | Entity Count | % of Slice | Avg Precision | Avg Recall | Avg $F_{0.5}$ Score |
|---|---|---|---|---|---|
| **Zero-Match Entities** ($|GT| = 0$) | 43 | 4.3% | 93.02% | 100.00% | **93.02%** |
| **Singleton Entities** ($|GT| = 1$) | 37 | 3.7% | 44.59% | 48.65% | **45.05%** |
| **Multi-Match Entities** ($|GT| > 1$) | 920 | 92.0% | 85.57% | 56.47% | **74.45%** |

### 3. Key Findings
- 92.0% of entities have multiple ground truth matches across sources, confirming the non-bijective problem topology.
- Classifier precision remains high (92.15% on scored candidate pairs, 84.37% macro precision).
- End-to-end recall is primarily bounded by candidate generation blocking recall (69.29% on this slice).

---

## Phase 8 & 9: Full Test Inference & Submission Validation

- **Date**: 2026-09-26
- **Script**: `scripts/run_inference.py`, `scripts/validate_submission.py`
- **Output Artifacts**:
  * `output/candidate_pairs.tsv` (4.34 GB, 1,732,544 rows, 334,668,988 candidate pairs)
  * `output/matching_results.tsv` (71.4 MB, 1,732,544 rows, 1,607,178 predicted matches)
- **Submission Validation Contract**:
  * Row Count: 1,732,544 (exactly matches test Source 1) -> **PASS**
  * Duplicate S1 IDs: 0 -> **PASS**
  * Duplicate Target Match IDs: 0 -> **PASS**
  * Candidate Set Invariance: 100% of predicted matches are present in `output/candidate_pairs.tsv` -> **PASS**
  * Entity ID Validation: 100% of IDs reference valid test S2/S3 entities -> **PASS**
  * Malformed Lines: 0 -> **PASS**




