# TASKS.md — Execution Roadmap

Status legend: `TODO` / `IN PROGRESS` / `DONE` / `BLOCKED`

| # | Phase | Priority | Depends on | Expected output | Status |
|---|-------|----------|------------|------------------|--------|
| 0 | Repository bootstrap (docs, env, git checkpoint) | P0 | — | PRD/TRD/AGENTS/TASKS committed, `.venv` ready | IN PROGRESS |
| 1 | Dataset audit (real data only) | P0 | 0 | Audit report: row counts, schema, missingness, duplicate IDs, country distribution, name/address noise, ground-truth match cardinality, singleton/one/multi-match distribution, positive pair count, candidate-generation & leakage implications | TODO |
| 2 | Preprocessing / normalization | P0 | 1 | Deterministic name/address normalization functions + unit tests | TODO |
| 3 | Candidate generation (blocking) | P0 | 2 | Blocking implementation, measured candidate recall on train, `candidate_pairs.tsv` contract defined | TODO |
| 4 | Feature engineering | P0 | 3 | Name/address/cross-field pair features + unit tests | TODO |
| 5 | Baseline model | P0 | 4 | First complete, end-to-end match classifier | TODO |
| 6 | Validation | P0 | 5 | Measured precision/recall/F0.5 on held-out train split, logged in EXPERIMENTS.md | TODO |
| 7 | Optimization | P1/P2 | 6 | Blocking/feature/model/threshold experiments, each with measured deltas | TODO |
| 8 | Test inference | P0 | 6 | `output/matching_results.tsv`, `output/candidate_pairs.tsv` on real test set | TODO |
| 9 | Submission validation | P0 | 8 | `validate_submission.py` → PASS | TODO |
| 10 | Final packaging | P0 | 9 | README.md, METHODOLOGY.md, pinned requirements.txt, final ZIP | TODO |

## Immediate next actions (current checkpoint)
1. `git status && git log --oneline --all -5 && git remote -v` — confirm repo state.
2. Commit PRD.md / TRD.md / AGENTS.md / TASKS.md → `docs: establish project context`.
3. Push documentation checkpoint to `main` (docs-only, not a feature branch).
4. Locate the actual challenge dataset (1GB, not stored in this environment) into
   `data/train/` and `data/test/` per TRD §2 — **do not** commit raw competition data to
   git unless the competition explicitly requires it.
5. Open the repo in Antigravity and run **Phase 1 (Dataset audit) only** — see PRD §7 /
   TRD §2 and the audit checklist embedded in TASKS row 1. Do not start modeling.
6. Review the audit output before touching Phase 2.

## Notes
- Confirm the official model parameter-limit number from the challenge resources before
  Phase 5 — it wasn't restated numerically in the tracker PDF and matters for model
  choice.
- Keep `EXPERIMENTS.md` updated at the end of every phase from 1 onward — it's what
  turns "should work" into "measured to work."
