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

