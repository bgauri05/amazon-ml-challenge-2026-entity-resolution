# Amazon ML Challenge 2026: Business Entity Resolution Dataset Analysis Summary

## Executive Summary

This document provides a comprehensive exploratory data analysis (EDA) of the **Amazon ML Challenge 2026 Business Entity Resolution** dataset. Entity resolution (ER) requires linking noisy, fragmented business records across 3 independent data sources (`Source 1`, `Source 2`, and `Source 3`) to reference entities in `Source 1`.

The dataset spans over **24.2 million entity records** across training and test splits. Key findings highlight an **open-set country generalization challenge** (unseen country `France` in the test set), **multilingual/transliterated text**, **heterogeneous noise patterns**, and a **precision-heavy evaluation metric ($F_{0.5}$)**.

---

## 1. Dataset Scale & Directory Architecture

The dataset is partitioned into `train/` and `test/` splits stored as tab-separated values (`.tsv`).

| Dataset Split | Source File | Record Count | File Size (MB) | Role / Description |
| :--- | :--- | :--- | :--- | :--- |
| **Train** | `train_source1.tsv` | 2,206,821 | 210.1 MB | Deduplicated reference entities ($S_1$) |
| | `train_source2.tsv` | 5,034,616 | 489.3 MB | Secondary source records ($S_2$) |
| | `train_source3.tsv` | 5,285,603 | 503.7 MB | Tertiary source records ($S_3$) |
| | `train_ground_truth.tsv` | 2,206,821 | 127.0 MB | Ground truth match mappings for $S_1$ entities |
| **Train Total** | **3 Sources + GT** | **12,527,040** | **~1.33 GB** | **Complete Training Set** |
| **Test** | `test_source1.tsv` | 1,732,544 | 175.0 MB | Deduplicated reference entities ($S_1$) |
| | `test_source2.tsv` | 4,887,273 | 509.5 MB | Secondary source records ($S_2$) |
| | `test_source3.tsv` | 5,082,316 | 506.0 MB | Tertiary source records ($S_3$) |
| **Test Total** | **3 Sources** | **11,702,133** | **~1.19 GB** | **Complete Test Set** |
| **Grand Total** | **All Files** | **24,229,173** | **~2.52 GB** | **Full Challenge Corpus** |

---

## 2. Country Breakdown & Open-Set Generalization

A critical discovery is the open-set nature of the dataset. While the **Training Set** contains records from **US** and **India**, the **Test Set** introduces a third country, **France**, which is completely absent from the training split.

### Country Distribution Across Sources

```
Training Set Country Split:
US     :  7,510,506 records (59.95%)
India  :  5,016,534 records (40.05%)

Test Set Country Split:
India  :  5,527,551 records (47.24%)
US     :  4,480,137 records (38.28%)
France :  1,694,445 records (14.48%)  <-- UNSEEN IN TRAIN
```

### Detailed Breakdown per Source File

| Split | File | United States (US) | India (IN) | France (FR) | Total |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Train** | `train_source1.tsv` | 1,323,633 (60.0%) | 883,188 (40.0%) | 0 (0.0%) | 2,206,821 |
| | `train_source2.tsv` | 3,016,817 (59.9%) | 2,017,799 (40.1%) | 0 (0.0%) | 5,034,616 |
| | `train_source3.tsv` | 3,170,056 (60.0%) | 2,115,547 (40.0%) | 0 (0.0%) | 5,285,603 |
| **Test** | `test_source1.tsv` | 663,106 (38.3%) | 809,986 (46.8%) | 259,452 (15.0%) | 1,732,544 |
| | `test_source2.tsv` | 1,871,330 (38.3%) | 2,312,565 (47.3%) | 703,378 (14.4%) | 4,887,273 |
| | `test_source3.tsv` | 1,945,701 (38.3%) | 2,405,000 (47.3%) | 731,615 (14.4%) | 5,082,316 |

> [!IMPORTANT]
> **Pipeline Design Rule**: Feature extraction and candidate generation algorithms must NOT hardcode state abbreviations or country-specific lookup tables (e.g. US state lists or Indian PIN code formats only). All components must handle open-set country strings robustly.

---

## 3. Ground Truth Matching Characteristics

Analyzing `train_ground_truth.tsv` reveals the exact distribution of true entity matches for the reference $S_1$ records.

### Key Match Statistics

- **Total $S_1$ Entities**: 2,206,821
- **Singletons (0 matches)**: 123,247 entities (**5.58%**)
- **Matched $S_1$ Entities**: 2,083,574 entities (**94.42%**)
- **Total Pairwise Match Links**: 7,638,365 pairs
  - Source 2 links ($S_1 \to S_2$): **3,693,619**
  - Source 3 links ($S_1 \to S_3$): **3,944,746**
- **Average Matches per $S_1$ Entity**: **3.46 matches** (or 3.67 for non-singleton entities)
- **Max Matches for Single $S_1$ Entity**: **11 matches**

### Match Distribution by Candidate Source Coverage

| Source Match Category | S1 Entity Count | Percentage of Total GT |
| :--- | :--- | :--- |
| **Singletons** (No matches in $S_2$ or $S_3$) | 123,247 | 5.58% |
| **Matched ONLY in $S_2$** | 143,029 | 6.48% |
| **Matched ONLY in $S_3$** | 164,498 | 7.45% |
| **Matched in BOTH $S_2$ and $S_3$** | 1,776,047 | 80.48% |
| **Total** | **2,206,821** | **100.00%** |

### Match Count Histogram (Matches per $S_1$ Entity)

| Number of Matches | $S_1$ Entity Count | Percentage |
| :---: | :--- | :--- |
| **0** (Singleton) | 123,247 | 5.58% |
| **1** | 119,157 | 5.40% |
| **2** | 375,212 | 17.00% |
| **3** | 530,841 | 24.05% |
| **4** | 484,115 | 21.94% |
| **5** | 321,957 | 14.59% |
| **6** | 164,868 | 7.47% |
| **7** | 63,968 | 2.90% |
| **8** | 18,680 | 0.85% |
| **9** | 4,205 | 0.19% |
| **10** | 534 | 0.02% |
| **11** | 37 | < 0.01% |

> [!NOTE]
> **Cluster Disjointness**: Verification shows `multi_matched_candidates_count = 0`. Each $S_2$ and $S_3$ record is matched to at most **one** $S_1$ entity in the ground truth. This means entities form clean disjoint clusters centered around $S_1$.

---

## 4. Data Quality & Missing Value Analysis

Across all files, the completeness of fields varies by source:

| File Name | Missing `entity_id` | Missing `business_name` | Missing `business_address` | Missing `country` |
| :--- | :---: | :---: | :---: | :---: |
| `train_source1.tsv` | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) |
| `train_source2.tsv` | 0 (0.0%) | 2 (< 0.01%) | 168,967 (3.36%) | 0 (0.0%) |
| `train_source3.tsv` | 0 (0.0%) | 13 (< 0.01%) | 175,916 (3.33%) | 0 (0.0%) |
| `test_source1.tsv` | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) |
| `test_source2.tsv` | 0 (0.0%) | 46 (< 0.01%) | 129,408 (2.65%) | 0 (0.0%) |
| `test_source3.tsv` | 0 (0.0%) | 59 (< 0.01%) | 136,098 (2.68%) | 0 (0.0%) |

- **Source 1** is 100% complete with zero missing names or addresses.
- **Sources 2 & 3** contain approximately **3.3% missing addresses** and rare missing names.

---

## 5. Textual & Field Characteristics

Length distributions for `business_name` and `business_address` columns across sources:

| Field | Source / Split | Mean Chars | Median Chars | Max Chars | Mean Words | Median Words | Max Words |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `business_name` | `train_s1` | 24.03 | 24.0 | 105 | 3.55 | 4.0 | 16 |
| | `train_s2` | 25.10 | 25.0 | 104 | 3.50 | 4.0 | 15 |
| | `train_s3` | 25.20 | 25.0 | 123 | 3.53 | 4.0 | 18 |
| | `test_s1` | 23.84 | 24.0 | 92 | 3.52 | 4.0 | 14 |
| `business_address` | `train_s1` | 52.07 | 41.0 | 256 | 8.03 | 7.0 | 43 |
| | `train_s2` | 46.33 | 37.0 | 249 | 7.32 | 6.0 | 46 |
| | `train_s3` | 46.81 | 42.0 | 240 | 7.21 | 6.0 | 43 |
| | `test_s1` | 57.21 | 50.0 | 268 | 8.59 | 8.0 | 43 |

---

## 6. Observed Noise Patterns & Qualitative Taxonomy

Empirical inspection of true match pairs in the dataset reveals 6 primary categories of noise:

```
                                  ┌─────────────────────────────────────────┐
                                  │       Entity Noise Taxonomy             │
                                  └────────────────────┬────────────────────┘
                                                       │
        ┌───────────────────┬───────────────────┬──────┴────────────┬───────────────────┐
        ▼                   ▼                   ▼                   ▼                   ▼
┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│ Multilingual │    │  Typo & OCR  │    │ Suffix & Legal│   │ Address Swap │    │ Missing Data │
│ Script Noise │    │ Corruption   │    │ Variation    │    │ & Tokens     │    │ & Domain URLs│
└──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘
```

1. **Multilingual & Script Transliteration**:
   - *Example*: `Raj Investments LLP` $\leftrightarrow$ Tamil script `ராஜ் இன்வெஸ்ட்மெண்ட்ஸ் எல்எல்பி`
   - *Example*: `Ss Food Private Limited` $\leftrightarrow$ Devanagari/Hindi `एसएस फूड प्राइवेट लिमिटेड`
   - *Example*: Accent marks (`Payne Enterprises` $\leftrightarrow$ `Payne Élterprises`, `Lumay Boral` $\leftrightarrow$ `Lumay Bóral`).

2. **Typographical & OCR Corruptions**:
   - Substring noise and typos (`Enterprises` $\leftrightarrow$ `Etrepndiels`, `Power` $\leftrightarrow$ `Ponr`).
   - Character insertions/deletions (`Payne` $\leftrightarrow$ `PAYNE-ENRTPRMISES`).

3. **Legal Suffix & Business Naming Variations**:
   - `LLC`, `LLP`, `Inc`, `Corp`, `Private Limited`, `Pvt Ltd`, `Limited`.
   - Structural shifts (`Dahlia Power Reliable Scientific LLC` $\leftrightarrow$ `Dahlia Power Reliable Scientific`).

4. **Address Transpositions & Local Abbreviation Noise**:
   - Component reordering: `630 45th Terrace, Kansas City, MO` $\leftrightarrow$ `KANSAS CITY, MO, 630 45ND TERRACE, null`.
   - Abbreviation shifts: `St` $\leftrightarrow$ `Street`, `Ave` $\leftrightarrow$ `Avenue`, `Rd` $\leftrightarrow$ `Road`.
   - Region expansions: `NY` $\leftrightarrow$ `New York`, `TN` $\leftrightarrow$ `Tamil Nadu`, `IL` $\leftrightarrow$ `Illinois`.

5. **Missing Fields & Domain/URL Ingestion**:
   - Missing address records filled as `NaN` or `null`.
   - Web domain aliases in business names: `Maure Williams Colombier Inc` $\leftrightarrow$ `maurewilliamscolombier.com`.

---

## 7. Evaluation Metric & Architectural Guidelines

### Metric: Macro $F_{0.5}$ Score

The challenge evaluates model performance using **Macro-averaged $F_{0.5}$**:

$$\text{Precision} = \frac{\text{True Matches}}{\text{Predicted Matches}}, \quad \text{Recall} = \frac{\text{True Matches}}{\text{Actual Matches}}$$

$$F_{0.5} = \frac{1.25 \times \text{Precision} \times \text{Recall}}{0.25 \times \text{Precision} + \text{Recall}}$$

> [!TIP]
> **Precision Weighting**: $F_{0.5}$ penalizes **false positives (false merges)** twice as heavily as false negatives. High precision thresholding is critical to maximize leaderboard score.
> Singletons earn a score of **1.0** when correctly predicted empty, but **0.0** if any match is falsely predicted.

---

## Summary of File Artefacts

- Root Dataset Summary File: [`dataset_summary.md`](file:///c:/Users/gauri/Downloads/AmazonML/dataset_summary.md)
- Student Resource Summary File: [`dataset_summary.md`](file:///c:/Users/gauri/Downloads/AmazonML/student_resource/student_resource/dataset_summary.md)
- EDA Analysis Script: [`eda_fast.py`](file:///C:/Users/gauri/.gemini/antigravity-ide/brain/2e40d3b3-b3c7-427f-8695-6985e1ebfe50/scratch/eda_fast.py)
- Raw Results JSON: [`eda_summary.json`](file:///C:/Users/gauri/.gemini/antigravity-ide/brain/2e40d3b3-b3c7-427f-8695-6985e1ebfe50/scratch/eda_summary.json)
