# Dataset Audit Report — Business Entity Resolution
**Execution Timestamp**: 2026-09-25 04:57:26 UTC
**Audit Execution Time**: 824.2 seconds

## 1. Executive Summary
- **Total Files Audited**: 7 files (4 train, 3 test)
- **Total Dataset Rows Across All Files**: 26,435,994
- **Train Ground Truth Rows**: 2,206,821
- **Total Positive Pairs (Ground Truth)**: 7,638,365
- **Unique S2 Matches in Ground Truth**: 3,693,619
- **Unique S3 Matches in Ground Truth**: 3,944,746
- **Data Integrity Checks**: PASSED (0 missing S1 in ground truth, 0 missing S2/S3 references, 0 test set leakage)

## 2. File-Level Schema, Dtypes, and Duplicate Audits
| File | Rows | Size (MB) | Columns | Duplicate IDs | Duplicate Rows | Null Counts |
|---|---|---|---|---|---|---|
| `train_ground_truth.tsv` | 2,206,821 | 121.1 | source1_entity_id, matched_entity_ids | 0 (`source1_entity_id`) | 0 | matched_entity_ids: 123,247 |
| `train_source1.tsv` | 2,206,821 | 200.3 | entity_id, business_name, business_address, country | 0 (`entity_id`) | 0 | None (0 across all columns) |
| `train_source2.tsv` | 5,034,616 | 466.6 | entity_id, business_name, business_address, country | 0 (`entity_id`) | 0 | business_name: 2, business_address: 168,967 |
| `train_source3.tsv` | 5,285,603 | 480.4 | entity_id, business_name, business_address, country | 0 (`entity_id`) | 0 | business_name: 13, business_address: 175,916 |
| `test_source1.tsv` | 1,732,544 | 166.9 | entity_id, business_name, business_address, country | 0 (`entity_id`) | 0 | None (0 across all columns) |
| `test_source2.tsv` | 4,887,273 | 485.9 | entity_id, business_name, business_address, country | 0 (`entity_id`) | 0 | business_name: 46, business_address: 129,408 |
| `test_source3.tsv` | 5,082,316 | 482.6 | entity_id, business_name, business_address, country | 0 (`entity_id`) | 0 | business_name: 59, business_address: 136,098 |

### Per-Column Dtypes
- **`train_ground_truth.tsv`**:
  - `source1_entity_id`: `str` (nulls: 0)
  - `matched_entity_ids`: `str` (nulls: 123,247)
- **`train_source1.tsv`**:
  - `entity_id`: `str` (nulls: 0)
  - `business_name`: `str` (nulls: 0)
  - `business_address`: `str` (nulls: 0)
  - `country`: `str` (nulls: 0)
- **`train_source2.tsv`**:
  - `entity_id`: `str` (nulls: 0)
  - `business_name`: `str` (nulls: 2)
  - `business_address`: `str` (nulls: 168,967)
  - `country`: `str` (nulls: 0)
- **`train_source3.tsv`**:
  - `entity_id`: `str` (nulls: 0)
  - `business_name`: `str` (nulls: 13)
  - `business_address`: `str` (nulls: 175,916)
  - `country`: `str` (nulls: 0)
- **`test_source1.tsv`**:
  - `entity_id`: `str` (nulls: 0)
  - `business_name`: `str` (nulls: 0)
  - `business_address`: `str` (nulls: 0)
  - `country`: `str` (nulls: 0)
- **`test_source2.tsv`**:
  - `entity_id`: `str` (nulls: 0)
  - `business_name`: `str` (nulls: 46)
  - `business_address`: `str` (nulls: 129,408)
  - `country`: `str` (nulls: 0)
- **`test_source3.tsv`**:
  - `entity_id`: `str` (nulls: 0)
  - `business_name`: `str` (nulls: 59)
  - `business_address`: `str` (nulls: 136,098)
  - `country`: `str` (nulls: 0)

## 3. Ground-Truth Match Cardinality & Structure
- **Total Source 1 Entities**: 2,206,821
- **Total Positive Pairs (S1 -> S2/S3)**: 7,638,365 (average 3.46 matches per S1 entity)
- **Source 2 Match Pairs**: 3,693,619 (48.4%)
- **Source 3 Match Pairs**: 3,944,746 (51.6%)
- **Other Entity ID Prefixes**: 0

### Match Cardinality Summary (0, 1, 2, 3+ Matches)
- **0 matches (Singletons / No Match)**: 123,247 (5.58%)
- **1 match**: 119,157 (5.40%)
- **2 matches**: 375,212 (17.00%)
- **3+ matches**: 1,589,205 (72.01%)

### Match Co-occurrence Distribution
- **Matches BOTH S2 and S3**: 1,776,047 (80.48%)
- **Matches ONLY S2**: 143,029 (6.48%)
- **Matches ONLY S3**: 164,498 (7.45%)
- **Zero Matches (Singletons)**: 123,247 (5.58%)

### Complete Match Cardinality Breakdown (Per S1 Entity)
| Match Count | S1 Entity Count | % of S1 Entities | Cumulative % | Cumulative S1 Count |
|---|---|---|---|---|
| 0 matches | 123,247 | 5.58% | 5.58% | 123,247 |
| 1 matches | 119,157 | 5.40% | 10.98% | 242,404 |
| 2 matches | 375,212 | 17.00% | 27.99% | 617,616 |
| 3 matches | 530,841 | 24.05% | 52.04% | 1,148,457 |
| 4 matches | 484,115 | 21.94% | 73.98% | 1,632,572 |
| 5 matches | 321,957 | 14.59% | 88.57% | 1,954,529 |
| 6 matches | 164,868 | 7.47% | 96.04% | 2,119,397 |
| 7 matches | 63,968 | 2.90% | 98.94% | 2,183,365 |
| 8 matches | 18,680 | 0.85% | 99.78% | 2,202,045 |
| 9 matches | 4,205 | 0.19% | 99.97% | 2,206,250 |
| 10 matches | 534 | 0.02% | 100.00% | 2,206,784 |
| 11 matches | 37 | 0.00% | 100.00% | 2,206,821 |

- **Key Takeaway**: 0 matches (singletons) represent 123,247 (5.58%), while 3+ matches represent 1,589,205 (72.01%). Matches go up to a maximum of 11 matches for a single S1 entity! The matching is strictly multi-match (1-to-many).

## 4. Country Distribution: Train vs Test (Open-Set Analysis)
### Per-File Breakdown
| File | US | India | France | Other / Null | Total |
|---|---|---|---|---|---|
| `train_source1.tsv` | 1,323,633 (60.0%) | 883,188 (40.0%) | 0 (0.0%) | 0 | 2,206,821 |
| `train_source2.tsv` | 3,016,817 (59.9%) | 2,017,799 (40.1%) | 0 (0.0%) | 0 | 5,034,616 |
| `train_source3.tsv` | 3,170,056 (60.0%) | 2,115,547 (40.0%) | 0 (0.0%) | 0 | 5,285,603 |
| `test_source1.tsv` | 663,106 (38.3%) | 809,986 (46.8%) | 259,452 (15.0%) | 0 | 1,732,544 |
| `test_source2.tsv` | 1,871,330 (38.3%) | 2,312,565 (47.3%) | 703,378 (14.4%) | 0 | 4,887,273 |
| `test_source3.tsv` | 1,945,701 (38.3%) | 2,405,000 (47.3%) | 731,615 (14.4%) | 0 | 5,082,316 |

### Train vs Test Aggregated Comparison
| Country | Train Total | Train % | Test Total | Test % | Status |
|---|---|---|---|---|---|
| **France** | 0 | 0.00% | 1,694,445 | 14.48% | Test Only (OPEN-SET) |
| **India** | 5,016,534 | 40.05% | 5,527,551 | 47.24% | Train & Test |
| **US** | 7,510,506 | 59.95% | 4,480,137 | 38.28% | Train & Test |

- **Open-Set Finding**: France is present in **Test** (both S1, S2, and S3) with ~15% of all test records, but completely absent from **Train**. Country must **never** be used as a hard filter with a fixed training enum.

## 5. Text Field Statistics (Name & Address)
### Business Name Character Length Statistics
| File | Count | Min | Mean | Median | P75 | P95 | P99 | Max |
|---|---|---|---|---|---|---|---|---|
| `train_source1.tsv` | 2,206,821 | 3 | 24.0 | 24.0 | 30.0 | 37.0 | 42.0 | 105 |
| `train_source2.tsv` | 5,034,614 | 2 | 25.1 | 25.0 | 31.0 | 40.0 | 48.0 | 104 |
| `train_source3.tsv` | 5,285,590 | 2 | 25.2 | 25.0 | 31.0 | 42.0 | 50.0 | 123 |
| `test_source1.tsv` | 1,732,544 | 3 | 23.8 | 24.0 | 29.0 | 36.0 | 42.0 | 92 |
| `test_source2.tsv` | 4,887,227 | 2 | 25.7 | 25.0 | 32.0 | 42.0 | 49.0 | 102 |
| `test_source3.tsv` | 5,082,257 | 2 | 25.7 | 25.0 | 32.0 | 42.0 | 50.0 | 103 |

### Business Name Token Count Statistics (Whitespace Split)
| File | Count | Min | Mean | Median | P75 | P95 | P99 | Max |
|---|---|---|---|---|---|---|---|---|
| `train_source1.tsv` | 2,206,821 | 1 | 3.5 | 4.0 | 4.0 | 5.0 | 6.0 | 16 |
| `train_source2.tsv` | 5,034,614 | 1 | 3.5 | 4.0 | 4.0 | 5.0 | 6.0 | 15 |
| `train_source3.tsv` | 5,285,590 | 1 | 3.5 | 4.0 | 4.0 | 6.0 | 7.0 | 18 |
| `test_source1.tsv` | 1,732,544 | 1 | 3.5 | 4.0 | 4.0 | 5.0 | 6.0 | 14 |
| `test_source2.tsv` | 4,887,227 | 1 | 3.6 | 4.0 | 4.0 | 5.0 | 6.0 | 15 |
| `test_source3.tsv` | 5,082,257 | 1 | 3.6 | 4.0 | 4.0 | 6.0 | 7.0 | 16 |

### Business Address Character Length Statistics
| File | Count | Min | Mean | Median | P75 | P95 | P99 | Max |
|---|---|---|---|---|---|---|---|---|
| `train_source1.tsv` | 2,206,821 | 11 | 52.1 | 41.0 | 70.0 | 103.0 | 124.0 | 256 |
| `train_source2.tsv` | 4,865,649 | 8 | 47.8 | 37.0 | 63.0 | 97.0 | 118.0 | 249 |
| `train_source3.tsv` | 5,109,687 | 2 | 48.3 | 42.0 | 55.0 | 92.0 | 116.0 | 240 |
| `test_source1.tsv` | 1,732,544 | 11 | 57.2 | 50.0 | 74.0 | 105.0 | 126.0 | 268 |
| `test_source2.tsv` | 4,757,865 | 5 | 51.8 | 43.0 | 68.0 | 99.0 | 120.0 | 269 |
| `test_source3.tsv` | 4,946,218 | 5 | 50.1 | 44.0 | 59.0 | 95.0 | 118.0 | 267 |

### Business Address Token Count Statistics (Whitespace Split)
| File | Count | Min | Mean | Median | P75 | P95 | P99 | Max |
|---|---|---|---|---|---|---|---|---|
| `train_source1.tsv` | 2,206,821 | 2 | 8.0 | 7.0 | 10.0 | 15.0 | 19.0 | 43 |
| `train_source2.tsv` | 4,865,649 | 2 | 7.5 | 6.0 | 9.0 | 15.0 | 18.0 | 46 |
| `train_source3.tsv` | 5,109,687 | 1 | 7.4 | 6.0 | 9.0 | 14.0 | 18.0 | 43 |
| `test_source1.tsv` | 1,732,544 | 2 | 8.6 | 8.0 | 11.0 | 16.0 | 19.0 | 43 |
| `test_source2.tsv` | 4,757,865 | 1 | 8.0 | 7.0 | 10.0 | 15.0 | 19.0 | 43 |
| `test_source3.tsv` | 4,946,218 | 1 | 7.7 | 7.0 | 9.0 | 15.0 | 19.0 | 43 |

## 6. Integrity and Leakage Audit
- **Ground Truth S1 vs train_source1.tsv**:
  - Ground truth S1 entities missing from `train_source1.tsv`: 0
  - `train_source1.tsv` entities missing from ground truth: 0
- **Ground Truth S2 References vs train_source2.tsv**:
  - Ground truth S2 entities referenced: 3,693,619
  - Missing from `train_source2.tsv`: 0
- **Ground Truth S3 References vs train_source3.tsv**:
  - Ground truth S3 entities referenced: 3,944,746
  - Missing from `train_source3.tsv`: 0
- **Ground Truth Foreign/Unrecognized ID Prefixes**: 0
- **Data Leakage Check (Test IDs appearing in Train Ground Truth)**:
  - Test S1 IDs in Train Ground Truth: 0
  - Test S2 IDs in Train Ground Truth: 0
  - Test S3 IDs in Train Ground Truth: 0

## 7. Strategic Implications for Pipeline Phases
1. **Candidate Generation / Blocking (Phase 3)**:
   - Ground truth has 7.6M+ matched positive pairs across 2.2M Source 1 entities.
   - An average S1 entity matches ~3.5 S2/S3 entities, with up to 10+ matches.
   - Candidate blocking must achieve very high recall while maintaining a manageable candidate count (e.g. 10-30 candidates per S1) to fit in memory.
   - Country blocking can partition US and India in train, but for test, France must be properly routed without missing candidates.
2. **Preprocessing & Normalization (Phase 2)**:
   - Names have a median of 4 tokens and addresses have a median of 6-8 tokens.
   - P99 of name length is ~42-47 chars, while max length exceeds 100-200 chars due to noise/concatenated descriptions.
   - Standardizing abbreviations (St., Ave., Pvt. Ltd., Inc.) and cleaning noisy address punctuation will be critical.
3. **Pair Matching & Model Evaluation (Phases 5-7)**:
   - Singletons represent 5.58% (123,247) of Source 1 entities. The model must preserve zero matches when confidence is low.
   - Metric is Macro F0.5: precision is weighted twice as heavily as recall. False positives will heavily penalize the score.

================================================================================
AUDIT COMPLETED SUCCESSFULLY IN 824.2s
================================================================================
