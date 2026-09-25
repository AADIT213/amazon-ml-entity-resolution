# TRD — Technical Design: Business Entity Resolution

## 1. Architecture (pipeline)
```
Challenge data (TSV)
  -> Data loading (explicit \t separator)
  -> Data validation (schema, dtypes, null/duplicate audit)
  -> Name/address normalization (deterministic)
  -> Candidate generation / blocking  (recall ceiling — see §4)
  -> Final candidate set  == output/candidate_pairs.tsv
  -> Pair feature engineering
  -> Pair matching model (binary: match / no-match)
  -> Threshold selection (tuned on validation F0.5, not fixed at 0.5)
  -> Aggregation by Source 1 entity
  -> output/matching_results.tsv
  -> validate_submission.py
```
Candidate generation and pair scoring are **separate, independently testable
components**. The system must support swapping blocking strategies and model families
without touching the rest of the pipeline (plain function/interface boundaries, no
tight coupling).

## 2. Data Loading & Validation
- Explicit `sep="\t"` on every read.
- Validate: expected columns present, dtypes, non-null `entity_id`, duplicate row/ID
  detection, per-source row counts logged.
- Fail loudly (raise, don't silently coerce) on schema mismatch.

## 3. Normalization (deterministic, no external calls)
- Name: lowercase, strip punctuation, expand/standardize common legal suffixes (Inc,
  LLC, Ltd, Pvt, Pvt Ltd, GmbH, etc.), collapse whitespace, handle transliteration only
  via local, rule-based normalization (no external transliteration API).
- Address: lowercase, standardize common abbreviations (St/Street, Ave/Avenue, Rd/Road),
  strip punctuation, normalize numeric tokens, preserve country as a separate field
  (open-set — never hard-coded to an enum of training countries).
- All normalization functions must be pure and unit-testable.

## 4. Candidate Generation / Blocking
- Establishes the **recall ceiling** — a true match missing from candidates can never be
  recovered downstream. Blocking recall must be measured on training data before any
  model work.
- Candidate strategies to evaluate: token/n-gram blocking on normalized name, sorted
  neighborhood on name+address, optional country-scoped blocking (only where country is
  reliable — remember it's open-set at test time).
- Output of this stage is exactly what gets written to `candidate_pairs.tsv` — no
  further silent filtering downstream that isn't reflected in that file.

## 5. Pair Feature Engineering
- Name features: exact normalized match, edit similarity, Jaccard/token overlap, TF-IDF
  cosine, token/length differences.
- Address features: exact normalized match, edit similarity, Jaccard/token overlap,
  TF-IDF cosine, numeric-token overlap, length differences.
- Cross-field: country equality, combined name+address agreement signal.
- Feature computation must be deterministic and side-effect-free (no lookups).

## 6. Model Interface
- Contract: `predict_proba(pair_features) -> match_probability`, model-agnostic
  (logistic regression, gradient boosting, etc. are all valid — chosen and justified in
  EXPERIMENTS.md, not fixed here).
- Must respect the challenge's parameter-count/license constraint (verify exact cap from
  official resources).

## 7. Thresholding & Aggregation
- Threshold selected by maximizing **F0.5** on a held-out validation split — never
  assumed at 0.5.
- Aggregation groups predictions by `source1_entity_id`; empty match lists are valid and
  must be written, not dropped.

## 8. Evaluation
- Local F0.5 computed against `train_ground_truth.tsv` using the same aggregation logic
  as test inference, so validation numbers are representative.
- Track precision, recall, F0.5, and candidate recall together — precision-heavy metric
  means recall gains from looser blocking can hurt final score if the model doesn't
  filter well.

## 9. Inference & Output
- Same pipeline code path for train (validation) and test (submission) — no
  test-only branches.
- Write `candidate_pairs.tsv` and `matching_results.tsv` exactly per PRD §5 schema.

## 10. Testing
- Unit tests for normalization functions, feature functions, and output-schema
  validation (duplicate IDs, one-row-per-entity, subset-of-candidates check) live in
  `tests/`.
- Run tests after every significant change (see AGENTS.md).

## 11. Reproducibility
- Pinned `requirements.txt`.
- Deterministic seeds for any stochastic model/training step.
- `README.md` documents exact commands to reproduce candidate generation, training, and
  inference from a clean checkout.
