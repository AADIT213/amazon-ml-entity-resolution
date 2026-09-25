# AGENTS.md — Rules for the AI Agent (Antigravity) on this repository

These rules are permanent for the duration of this project. Read PRD.md, TRD.md, and
TASKS.md before writing any code.

## Hard rules (never violate)
1. **Never use synthetic or mock data.** Only the challenge-provided TSV files under
   `data/`. If a file is missing, stop and report it — do not fabricate rows.
2. **Never call external services** to resolve business identity: no geocoding APIs, no
   government business-registry lookups, no internet search, no third-party
   entity-resolution APIs, no external data augmentation of any kind.
3. **Never assume one-to-one matching.** A Source 1 entity may have zero, one, or many
   matches. Do not add logic that caps matches at 1 or forces a match to exist.
4. **Never produce an invalid output file.** Every change touching output generation
   must be checked against: one row per test Source 1 entity, no duplicate IDs, every
   match present in `candidate_pairs.tsv`, IDs reference only S2/S3 test entities.
5. **Never claim a performance improvement without a measured number.** Any statement
   like "this should improve recall" must be followed by an actual measurement recorded
   in `EXPERIMENTS.md` before being treated as fact.
6. **Never make undocumented changes.** Every meaningful change is a git commit with a
   descriptive message (`docs:`, `feat:`, `fix:`, `test:` ...). Do not push directly to
   `main` for implementation work — use a feature branch.

## Working style
- Work **phase-by-phase** per TASKS.md. Do not attempt to implement the entire pipeline
  in one prompt/session.
- Inspect existing files before modifying them. Make the smallest reasonable change to
  achieve the current phase's goal.
- Run relevant tests after any significant change; do not proceed to the next phase on
  a failing test.
- Do not introduce a new dependency without a one-line justification in the commit
  message or PR description.
- A working, validated baseline (simple blocking + simple model, end-to-end) is higher
  priority than an advanced but incomplete solution. Get the full loop closed first.
- `git status`, `git log --oneline --all -5`, and `git remote -v` should be checked
  before the first commit of a session to confirm repo state before acting.
- Do not run `git init` again and do not re-clone — the local repo and remote already
  exist.

## Before every implementation session
1. Read PRD.md, TRD.md, AGENTS.md, TASKS.md.
2. Identify the current phase from TASKS.md's status column.
3. Confirm the dataset audit (Phase 1) has been reviewed before writing modeling code.
4. Implement only that phase, then stop for review.
