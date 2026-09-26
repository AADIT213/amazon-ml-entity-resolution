# TASKS.md — Execution Roadmap

Status legend: `TODO` / `IN PROGRESS` / `DONE` / `BLOCKED`

| # | Phase | Priority | Depends on | Expected output | Status |
|---|-------|----------|------------|------------------|--------|
| 0 | Repository bootstrap (docs, env, git checkpoint) | P0 | — | PRD/TRD/AGENTS/TASKS committed, `.venv` ready | DONE |
| 1 | Dataset audit (real data only) | P0 | 0 | Audit report: row counts, schema, missingness, duplicate IDs, country distribution, name/address noise, ground-truth match cardinality, singleton/one/multi-match distribution, positive pair count, candidate-generation & leakage implications | DONE |
| 2 | Preprocessing / normalization | P0 | 1 | Deterministic name/address normalization functions + unit tests | DONE |
| 3 | Candidate generation (blocking) | P0 | 2 | Blocking implementation, measured candidate recall on train, `candidate_pairs.tsv` contract defined | DONE |
| 4 | Feature engineering | P0 | 3 | Name/address/cross-field pair features + unit tests | TODO |
| 5 | Baseline model | P0 | 4 | First complete, end-to-end match classifier | TODO |
| 6 | Validation | P0 | 5 | Measured precision/recall/F0.5 on held-out train split, logged in EXPERIMENTS.md | TODO |
| 7 | Optimization | P1/P2 | 6 | Blocking/feature/model/threshold experiments, each with measured deltas | TODO |
| 8 | Test inference | P0 | 6 | `output/matching_results.tsv`, `output/candidate_pairs.tsv` on real test set | TODO |
| 9 | Submission validation | P0 | 8 | `validate_submission.py` → PASS | TODO |
| 10 | Final packaging | P0 | 9 | README.md, METHODOLOGY.md, pinned requirements.txt, final ZIP | TODO |

## Immediate next actions (current checkpoint)
1. Phase 3 Candidate Generation finalized: `output/candidate_pairs.tsv` generated with 75.40% train recall set and validated against full test dataset (1,732,544 test S1 entities).
2. All 334,668,988 generated candidate IDs passed strict ID-existence validation (`validate_submission.py --check-ids`).
3. Proceed to Phase 4 (Feature Engineering).

## Notes
- Phase 3 generated complete candidate pairs file: 1,732,544 rows, 334.67M candidate pairs, 75.40% train candidate recall, 99.99806% reduction ratio.
- Open-set France test S1 candidate generation validated (259,452 France entities, avg 842.91 candidates/S1, 1.21% zero-candidate rate).
- Confirm the official model parameter-limit number from the challenge resources before
  Phase 5 — it wasn't restated numerically in the tracker PDF and matters for model
  choice.
- Keep `EXPERIMENTS.md` updated at the end of every phase from 1 onward — it's what
  turns "should work" into "measured to work."

