# Business Entity Resolution Methodology and Technical Report

## 1. Executive Summary
This project presents an end-to-end, high-precision machine learning solution for resolving noisy, multi-source business entity records at scale for the Amazon ML Challenge. By coupling a 7-key inverted blocking engine with a 12-feature class-balanced Logistic Regression classifier and a disk-backed SQLite streaming architecture, the pipeline indexes 11.7 million test entities and evaluates 334.6 million candidate pairs in bounded memory (< 200 MB). On a held-out train validation slice evaluated strictly via the official per-entity macro formula, the system achieves **74.16% Macro $F_{0.5}$** (**84.37% Macro Precision**, **58.05% Macro Recall**), delivering robust entity disambiguation across singletons, multi-matches, and zero-match records.

---

## 2. Problem Analysis & Phase 1 Audit Findings

Comprehensive analysis of the challenge dataset revealed several fundamental noise patterns and structural characteristics:
1. **Severe Scale & Asymmetric Cardinality**:
   - Training set: 2,206,821 Source 1 query entities, 5,034,616 Source 2 records, 5,285,603 Source 3 records (12.5M records total).
   - Test set: 1,732,544 Source 1 query entities, 4,964,482 Source 2 records, 4,996,183 Source 3 records (11.7M records total).
   - Brute-force cartesian matching would require $> 17.2 \times 10^{12}$ pair comparisons, necessitating aggressive search space reduction.
2. **Multi-Source Match Cardinality**:
   - The match relation is fundamentally non-bijective (zero-to-many): 92.0% of Source 1 entities have multiple true matches across Source 2 and Source 3 (averaging ~3.9 true matches per multi-match entity), while 3.7% are singletons and 4.3% have zero matches. Models assuming 1-to-1 matching fail on this topology.
3. **Pervasive String and Token Noise**:
   - **Legal Suffix Variations**: Extensive discrepancies in legal structure naming (`Ltd`, `Limited`, `LLC`, `Corp`, `S.A.`, `GmbH`, `Pty Ltd`).
   - **Address Truncation & Reordering**: Addresses often suffer from missing postal codes, swapped street/city components, or missing address fields entirely (21.4% missing addresses in some sources).
   - **Open-Set Country Distribution (France Shift)**: While the training set is predominantly US/GB/DE/FR, the test Source 1 query set contains 259,452 France (`FR`) entities with country-specific naming conventions that required dedicated blocking treatment.

---

## 3. Solution Strategy: Two-Stage Pipeline

To balance extreme computational efficiency with high matching precision, we adopted a decoupled two-stage architecture:
1. **Stage 1 — Multi-Key Blocking (Candidate Generation)**:
   - High-recall inverted indices generate a candidate subset of target records for each query entity while eliminating 99.998% of non-matching pairs.
2. **Stage 2 — Pair Feature Extraction & Classification**:
   - Vectorized computation of 12 pairwise similarity metrics followed by scoring with a class-balanced Logistic Regression model calibrated for high precision under the competition's $F_{0.5}$ metric (which weights precision twice as heavily as recall).
3. **Disk-Backed Streaming Inference**:
   - To avoid multi-gigabyte memory footprints, entity records are indexed in a persistent SQLite database (`output/test_entity_lookup.sqlite`), allowing batched candidate processing in constant memory.

---

## 4. Candidate Generation (Blocking Engine)

The blocking engine executes **7 complementary inverted index rules**:
1. **Exact Normalized Name**: Matches entities sharing identical normalized name strings.
2. **Country-Scoped Legal Stem**: Strips legal suffixes and matches on token stems within the same normalized country code.
3. **Exact Normalized Address**: Matches entities with identical normalized addresses (minimum length 8 characters).
4. **Country-Scoped Sorted Stem**: Order-invariant token stem matching within the same country to handle word order permutations.
5. **Country-Scoped Address Prefix**: 14-character normalized address prefix matching within the same country.
6. **Null Address Country-Stem Fallback**: Matches query entities lacking address fields against target entities sharing the same country and name stem.
7. **Open-Set France ($S_1$) Fallback**: Dedicated handling for French entities to ensure high candidate coverage across cross-border records.

### Blocking Performance & Yield
- **Test Set Candidate Pairs Generated**: **334,668,988 pairs** across 1,732,544 test $S_1$ entities (average 193.2 candidates / entity).
- **Reduction Ratio**: **99.99806%** search space reduction from the $1.72 \times 10^{13}$ cartesian space.
- **Training Candidate Recall**: **75.40%** ground-truth match coverage.
- **Zero-Candidate Entity Rate**: < 1.3% across all test entities.

---

## 5. Matching Model & Feature Engineering

### 12 Lightweight Pairwise Features
1. `exact_name_match`: Exact normalized name equality (binary 0/1).
2. `name_token_jaccard`: Word token set Jaccard similarity $[0, 1]$.
3. `name_char_similarity`: RapidFuzz normalized Levenshtein edit similarity $[0, 1]$.
4. `name_len_diff`: Absolute difference in normalized name character length.
5. `name_token_count_diff`: Absolute difference in name token counts.
6. `exact_addr_match`: Exact normalized address equality (binary 0/1).
7. `addr_token_jaccard`: Address word token set Jaccard similarity $[0, 1]$.
8. `addr_numeric_overlap`: Jaccard overlap of numeric digit tokens (house numbers, postal codes).
9. `addr_len_diff`: Absolute difference in normalized address character length.
10. `country_match`: Normalized country equality (binary 0/1).
11. `missing_name_flag`: Binary indicator if either entity has an empty name.
12. `missing_addr_flag`: Binary indicator if either entity has an empty address.

### Model Architecture & Decision Threshold
- **Model**: Scikit-Learn `LogisticRegression` with `class_weight="balanced"`.
- **Primary Coefficients**: Address Jaccard (+3.34), Name Jaccard (+1.97), Address Numeric Overlap (+1.71), and Name Character Similarity (+0.81).
- **Decision Threshold**: Calibrated to **0.90** on held-out validation data to heavily favor precision, maximizing the official $F_{0.5}$ metric.

---

## 6. Validation Results & Metric Analysis

Evaluated strictly using the official macro-averaged per-entity formula:
$$\text{Macro } F_{0.5} = \frac{1}{N} \sum_{i=1}^N F_{0.5}^{(i)} \quad \text{where} \quad F_{0.5}^{(i)} = \frac{1.25 \cdot P_i \cdot R_i}{0.25 \cdot P_i + R_i}$$
*(Zero-match entities receive 1.0 if correctly predicted empty and 0.0 otherwise; entities lost at blocking receive 0.0).*

### Macro Performance Summary (1,000 Held-Out Train $S_1$ Entities)

| Metric | Measured Validation Score |
| :--- | :--- |
| **Overall Macro $F_{0.5}$** | **74.16%** |
| **Overall Macro Precision** | **84.37%** |
| **Overall Macro Recall** | **58.05%** |
| **Total Evaluated $S_1$ Entities** | **1,000** |

### Breakdown by Ground Truth Entity Category

| Category | Entity Count (%) | Avg Precision | Avg Recall | Avg $F_{0.5}$ Score |
| :--- | :--- | :--- | :--- | :--- |
| **Zero-Match Entities ($|GT| = 0$)** | 43 (4.3%) | 93.02% | 100.00% | **93.02%** |
| **Singleton Entities ($|GT| = 1$)** | 37 (3.7%) | 44.59% | 48.65% | **45.05%** |
| **Multi-Match Entities ($|GT| > 1$)** | 920 (92.0%) | 85.57% | 56.47% | **74.45%** |

---

## 7. Error Analysis & Bottleneck Identification

1. **Blocking Recall as the Primary Ceiling**:
   - On the 1,000-entity slice, candidate blocking recall captured **69.29%** of ground truth pairs.
   - The classifier achieved **81.49% recall** on candidate pairs and **92.15% pair precision**, demonstrating that the matching classifier is highly reliable, but overall end-to-end recall (58.05% macro) is bounded primarily by candidates captured during the blocking stage.
2. **Address Discrepancies in Singletons**:
   - Singleton entities experienced lower precision (44.59%) primarily in cases where target records had identical company names in differing regions or branch offices without distinct localized addresses.
3. **Zero-Match Precision**:
   - 40 out of 43 zero-match entities were correctly predicted with empty match sets (93.02% accuracy), verifying low false alarm rates.

---

## 8. Conclusion & Lessons Learned

1. **Scalability Through Decoupled Design**: Separating inverted-index blocking from vectorized pairwise feature evaluation enabled full test set execution across 334.6M pairs with constant memory overhead.
2. **Metric Alignment**: Calibrating the decision threshold to 0.90 properly optimized for the $F_{0.5}$ metric's preference for high precision.
3. **Future Enhancements**: Further gains can be achieved by integrating phonetic/metaphone blocking keys, embedding-based approximate nearest neighbor (ANN) retrieval for non-exact name stems, and cross-source graph clustering to resolve transitive multi-entity clusters.
