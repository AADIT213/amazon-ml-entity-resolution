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
