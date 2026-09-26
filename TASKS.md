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
| 6 | Validation | P0 | 5 | Measured precision/recall/F0.5 on held-out train split, logged in EXPERIMENTS.md | IN PROGRESS |
| 7 | Optimization | P1/P2 | 6 | Blocking/feature/model/threshold experiments, each with measured deltas | TODO |
| 8 | Test inference | P0 | 6 | `output/matching_results.tsv`, `output/candidate_pairs.tsv` on real test set | TODO |
| 9 | Submission validation | P0 | 8 | `validate_submission.py` → PASS | TODO |
| 10 | Final packaging | P0 | 9 | README.md, METHODOLOGY.md, pinned requirements.txt, final ZIP | TODO |

## Immediate next actions (current checkpoint)
1. Phase 5 Baseline Logistic Regression model trained and validated on 10,000-S1 streaming benchmark (`models/baseline_logreg.joblib`).
2. Validation F0.5 achieved 92.27% (Precision: 95.29%, Recall: 81.88% at threshold 0.90).
3. 82/82 unit tests passing across all normalization, blocking, features, and model modules.

## Notes
- Phase 3 generated complete candidate pairs file: 1,732,544 rows, 334.67M candidate pairs, 75.40% train candidate recall, 99.99806% reduction ratio.
- Open-set France test S1 candidate generation validated (259,452 France entities, avg 842.91 candidates/S1, 1.21% zero-candidate rate).
- Confirm the official model parameter-limit number from the challenge resources before
  Phase 5 — it wasn't restated numerically in the tracker PDF and matters for model
  choice.
- Keep `EXPERIMENTS.md` updated at the end of every phase from 1 onward — it's what
  turns "should work" into "measured to work."

