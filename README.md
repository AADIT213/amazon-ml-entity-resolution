# Amazon ML Challenge: Business Entity Resolution

Scalable, high-precision machine learning pipeline for cross-source business entity resolution across multi-million noisy web records. Resolves Source 1 (query entities) against heterogeneous records from Source 2 and Source 3 under severe scale (11.7M test records, 334.6M candidate pairs) without external APIs or synthetic data.

---

## 🏗️ System Architecture

The pipeline consists of a two-stage architecture:
1. **Multi-Key Blocking Engine (Phase 3)**:
   - Employs 7 complementary blocking rules (Exact Name, Country-Scoped Legal Token Stem, Exact Address, Country-Scoped Sorted Token Stem, Country-Scoped Address Prefix, Null Address Country-Stem Fallback, and Open-Set France S1 Fallback).
   - Generates `output/candidate_pairs.tsv` (334,668,988 candidate pairs across 1,732,544 test $S_1$ entities, achieving 99.998% search space reduction).
2. **Lightweight Feature Extraction & Classification Engine (Phases 4–6)**:
   - Computes 12 vectorized string, token, character, length, and cross-field features (RapidFuzz normalized Levenshtein, Jaccard token overlap, numeric digit overlap, country equality).
   - Classifies matches using a class-balanced Logistic Regression model (`models/baseline_logreg.joblib`) evaluated at threshold `0.90`.
   - Streaming, disk-backed SQLite lookup (`output/test_entity_lookup.sqlite`) enabling constant bounded RAM (< 200 MB) during multi-million entity scoring.

---

## 📁 Repository Structure

```
.
├── Documentation_template.md        # Comprehensive methodology and solution report
├── EXPERIMENTS.md                   # Formal experiment tracking & measured validation metrics
├── LICENSE                          # Project license
├── PRD.md                           # Product requirements & challenge constraints
├── README.md                        # Project documentation & reproduction instructions
├── TASKS.md                         # Phase-by-phase execution tracking
├── TRD.md                           # Technical requirements & architecture design
├── requirements.txt                 # Pinned dependencies
├── data/                            # Real dataset directories (TSV files)
│   ├── train/                       # train_source1.tsv, train_source2.tsv, train_source3.tsv, train_ground_truth.tsv
│   └── test/                        # test_source1.tsv, test_source2.tsv, test_source3.tsv
├── models/
│   └── baseline_logreg.joblib       # Trained frozen baseline model artifact
├── output/
│   ├── candidate_pairs.tsv          # Generated test candidate pairs (334,668,988 pairs)
│   └── matching_results.tsv         # Final predicted test matches (1,732,544 rows)
├── scripts/
│   ├── dataset_audit.py             # Phase 1 dataset audit & noise profiling
│   ├── run_blocking.py              # Phase 3 candidate generation / blocking pipeline
│   ├── run_features.py              # Phase 4 feature engineering & streaming benchmark
│   ├── train_baseline.py            # Phase 5 model training & validation benchmark
│   ├── run_inference.py             # Phase 6 test set inference pipeline
│   ├── validate_submission.py       # Submission validation & contract checker
│   ├── evaluate_train_slice.py      # End-to-end train validation slice runner
│   └── compute_macro_metrics.py     # Macro-averaged F0.5 per-entity evaluation
├── src/business_entity_resolution/  # Core modular library
│   ├── __init__.py
│   ├── blocking.py                  # Inverted index blocking rules
│   ├── features.py                  # Vectorized 12-feature extractor
│   ├── model.py                     # Match classifier & evaluation routines
│   └── normalize.py                 # Deterministic text normalization
└── tests/                           # Pytest suite (85 unit tests)
```

---

## ⚙️ Environment Setup

### Prerequisites
- Python 3.10+ (tested on Python 3.11 / 3.12)
- 16 GB+ RAM recommended for large-scale candidate streaming

### Installation
```bash
# 1. Create and activate virtual environment
python -m venv .venv

# On Linux / macOS:
source .venv/bin/activate

# On Windows (PowerShell):
.\.venv\Scripts\Activate.ps1

# 2. Install pinned dependencies
pip install -r requirements.txt
```

---

## 🔄 End-to-End Reproduction Instructions

To reproduce all artifacts from a clean checkout from start to finish, execute the scripts in the following order:

### 1. Dataset Audit (Phase 1)
Audits data schema, noise profiles, country distributions, and entity cardinality.
```bash
python scripts/dataset_audit.py --data-dir data
```

### 2. Candidate Generation / Blocking (Phase 3)
Builds candidate pairs across test entities using the 7 blocking strategies.
```bash
python scripts/run_blocking.py --data-dir data/test --output output/candidate_pairs.tsv
```
*Output: `output/candidate_pairs.tsv` (1,732,544 rows, 334,668,988 candidate pairs).*

### 3. Feature Extraction Verification (Phase 4)
Runs streaming feature extraction benchmark on candidate pairs.
```bash
python scripts/run_features.py --data-dir data/test --candidates output/candidate_pairs.tsv --sample-size 10000
```

### 4. Baseline Model Training (Phase 5)
Trains the balanced Logistic Regression classifier on train candidate pairs.
```bash
python scripts/train_baseline.py --data-dir data --output-model models/baseline_logreg.joblib --sample-s1 10000 --seed 42
```
*Output: `models/baseline_logreg.joblib`.*

### 5. Full Test Set Inference (Phase 6)
Builds disk-backed SQLite test entity lookup and streams all 334.6M candidate pairs for classification.
```bash
python scripts/run_inference.py --data-dir data/test --candidates output/candidate_pairs.tsv --model models/baseline_logreg.joblib --output output/matching_results.tsv --threshold 0.90
```
*Output: `output/matching_results.tsv` (1,732,544 rows).*

### 6. Submission Contract Validation (Phase 9)
Verifies row counts, ID uniqueness, candidate membership, and absence of malformed lines.
```bash
python scripts/validate_submission.py --test-dir data/test --candidates output/candidate_pairs.tsv --matching output/matching_results.tsv --check-ids
```
*Expected result: `=== VALIDATION RESULT: ALL CHECKS PASSED ===`.*

### 7. Validation & Macro $F_{0.5}$ Evaluation
Runs per-entity macro $F_{0.5}$ evaluation on held-out train entities using the official formula.
```bash
python scripts/compute_macro_metrics.py
```

---

## 🧪 Unit Test Suite

Run the full automated test suite (85 tests covering normalization, blocking, feature extraction, model scoring, and validation checks):
```bash
pytest -v
```
