# TASKS.md — Execution Roadmap

Status legend: `TODO` / `IN PROGRESS` / `DONE` / `BLOCKED`

| # | Phase | Priority | Depends on | Expected output | Status |
|---|-------|----------|------------|------------------|--------|
| 0 | Repository bootstrap (docs, env, git checkpoint) | P0 | — | PRD/TRD/AGENTS/TASKS committed, `.venv` ready | DONE |
| 1 | Dataset audit (real data only) | P0 | 0 | Audit report: row counts, schema, missingness, duplicate IDs, country distribution, name/address noise, ground-truth match cardinality, singleton/one/multi-match distribution, positive pair count, candidate-generation & leakage implications | DONE |
| 2 | Preprocessing / normalization | P0 | 1 | Deterministic name/address normalization functions + unit tests | DONE |
| 3 | Candidate generation (blocking) | P0 | 2 | Blocking implementation, measured candidate recall on train, `candidate_pairs.tsv` contract defined | DONE |
| 4 | Feature engineering | P0 | 3 | Name/address/cross-field pair features + unit tests | DONE |
| 5 | Baseline model | P0 | 4 | First complete, end-to-end match classifier | DONE |
| 6 | Validation | P0 | 5 | Measured macro-averaged per-entity precision/recall/F0.5 on held-out train slice, logged in EXPERIMENTS.md | DONE |
| 7 | Optimization | P1/P2 | 6 | Threshold calibration (0.90) and SQLite streaming engine | DONE |
| 8 | Test inference | P0 | 6 | `output/matching_results.tsv`, `output/candidate_pairs.tsv` on real test set (1,732,544 rows, 334.6M pairs) | DONE |
| 9 | Submission validation | P0 | 8 | `validate_submission.py` → PASS (0 duplicate IDs, 0 outside candidate set) | DONE |
| 10 | Final packaging | P0 | 9 | README.md, Documentation_template.md, pinned requirements.txt, submission structure verified | DONE |

## Final Pipeline Summary
1. **Candidate Generation (Phase 3)**:
   - 7 blocking rules generated `output/candidate_pairs.tsv` (1,732,544 rows, 334,668,988 pairs, 99.99806% reduction ratio).
2. **Feature Extraction & Matching Model (Phases 4–5)**:
   - 12 lightweight pairwise features scored via balanced Logistic Regression (`models/baseline_logreg.joblib`) at threshold 0.90.
3. **Full Test Inference (Phase 8)**:
   - Evaluated all 334.6M candidate pairs using disk-backed SQLite streaming lookup (`output/matching_results.tsv`).
4. **Validation (Phase 6)**:
   - Evaluated on 1,000 held-out train S1 entities using official per-entity formula: **74.16% Macro $F_{0.5}$** (84.37% Macro Precision, 58.05% Macro Recall).
5. **Submission Validation (Phase 9)**:
   - All contract checks passed via `scripts/validate_submission.py`.
6. **Final Packaging (Phase 10)**:
   - `README.md`, `Documentation_template.md`, and `requirements.txt` finalized.
