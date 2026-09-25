# PRD — Business Entity Resolution (Amazon ML Challenge 2026)

## 1. Problem
Business records describing the same real-world business arrive from three independent
sources with no shared identifier:
- **Source 1** — deduplicated reference source.
- **Source 2 / Source 3** — noisy records that may refer to businesses already in Source 1.

Noise includes: abbreviations, legal-suffix differences, typos, transliteration,
punctuation, word-order changes, missing address components, landmarks, and address
formatting differences. `country` is an **open-set** string field — the test set contains
countries (e.g. France) absent from training. Do not hard-code the training country set.

## 2. Objective
For every Source 1 entity, identify **zero, one, or multiple** matching Source 2/Source 3
records. This is a pair-matching / clustering problem, not classification into fixed
labels. **This document does not mandate a specific ML algorithm** — blocking strategy,
feature set, and model family are implementation decisions made and justified in TRD.md /
EXPERIMENTS.md.

## 3. Scope
In scope: normalization, candidate generation (blocking), pair feature engineering, a
match/no-match model, thresholding, aggregation into the required output files, and
submission validation.
Out of scope: anything requiring external data or services (see §6).

## 4. Inputs
- `train_source1.tsv`, `train_source2.tsv`, `train_source3.tsv`, `train_ground_truth.tsv`
- `test_source1.tsv`, `test_source2.tsv`, `test_source3.tsv`
- Fields: `entity_id`, `business_name`, `business_address`, `country`
- All files are **TSV** — must be parsed with an explicit tab separator.
- Dataset is challenge-provided only. Never synthesize or mock data.

## 5. Outputs
- `output/matching_results.tsv` — `source1_entity_id` + comma-separated
  `matched_entity_ids`. Exactly one row per **test** Source 1 entity. Empty list is valid.
- `output/candidate_pairs.tsv` — `source1_entity_id` + comma-separated
  `candidate_entity_ids`. Must be the **exact** final candidate set passed to the matching
  model (not a superset generated for exploration).

## 6. Requirements & Constraints (fair play)
- No external entity-resolution APIs or commercial services.
- No government business-registration lookup.
- No geocoding APIs for address normalization.
- No external internet-based business identity lookup.
- No external data augmentation.
- Only challenge-provided data may be used.
- Final model must be MIT/Apache 2.0 licensed and use at most 8 billion parameters.

## 7. Evaluation
- Official metric: **macro F0.5** (precision-heavy — false merges are costly).
- Correct singleton predictions (no match) receive credit — do not force a match.
- Multi-match behavior must be preserved — matching is **not** one-to-one.
- Every final match must appear in `candidate_pairs.tsv`.
- Every test Source 1 entity appears exactly once in `matching_results.tsv`.
- No duplicate IDs within any list.
- Final IDs reference only Source 2/Source 3 **test** entities.
- `validate_submission.py` (official script) must PASS before submission.

## 8. Success Criteria
1. End-to-end pipeline runs on the real dataset and produces both output files.
2. `validate_submission.py` returns PASS.
3. Measured (not assumed) candidate recall, precision, and F0.5 are recorded in
   `EXPERIMENTS.md`.
4. Baseline is simple and complete before any optimization work begins.

## 9. Deliverables (final package)
- `output/` (matching_results.tsv, candidate_pairs.tsv)
- `code/business_entity_resolution/` (runnable source)
- `README.md` (reproduction steps), `METHODOLOGY.md`, pinned `requirements.txt`
- Packaged as a ZIP per the official submission format.
