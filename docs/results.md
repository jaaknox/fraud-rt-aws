# Results (Validation Set)

**Model:** XGBoost (binary\:logistic), trained in Amazon SageMaker (Tabular XGBoost v1.3, Pipe + Gzip).
**Data:** PCA-style features with label `class` ∈ {0,1}; features: `time, v1..v28, amount`.
**Split:** 80/20 random via Athena UNLOAD (label first, features next).

---

## Summary

* **Validation size:** 56,866 rows
* **Fraud (positives):** 95 (rate 0.167%)
* **ROC–AUC:** **0.99296**
* **PR–AUC:** **0.83834**
* **Training (SageMaker) metric:** `validation-auc ≈ 0.99606`

These numbers indicate very strong ranking performance on a highly imbalanced set.

---

## Operating Points

### 1) Best‑F1 threshold (single cutoff)

* **Threshold:** 0.7991
* **Confusion matrix:** TP=86, FP=19, TN=56,752, FN=9
* **Precision:** 0.827
* **Recall:** 0.905

This is a balanced operating point that maximizes the F1 score on validation.

### 2) Two‑tier policy (review budget ≈ 0.5%)

Quantile thresholds selected from validation probabilities:

* **BLOCK:** p ≥ 0.8801 (≈ top 0.135% of traffic)
* **HOLD/REVIEW:** 0.4767 ≤ p < 0.8801 (≈ next 0.366%)
* **ALLOW:** p < 0.4767

Aggregate (HOLD+BLOCK considered “flagged”):

* **Review rate:** 0.501% of transactions
* **Block rate:** 0.135% of transactions
* **Precision (reviewed):** 0.326
* **Recall:** **0.979**

> Interpretation: reviewing \~0.5% of traffic captures \~98% of fraud, with \~33% precision among reviewed items
