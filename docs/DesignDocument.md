# Real-Time Fraud Detection System — Design Document

**Completion Status:** MVP trained & validated; offline inference complete; real‑time scoring deferred because keeping an endpoint up costs money...

---

## 1) Overview

This project implements a fraud detection platform for payment transactions using AWS for ingestion, storage, training, and (easily implemented in the future if I ever want to spend $ keeping an endpoint up) real‑time inference.

**Goals:** classify each transaction as *ALLOW / HOLD / BLOCK* within tight latency budgets, maximize fraud capture (recall), and minimize false positives.

---

## 2) Objectives & Non‑Goals

**Objectives**

* High recall at controlled review volume
* Sub‑150ms p95 latency for online scoring
* Scalable to thousands TPS via managed services

---

## 3) Architecture

### 3.1 Full data lifecycle as it would be in a production environment:

1. **Ingestion:** Client → Amazon Kinesis Data Streams
2. **Raw landing:** Kinesis Firehose → S3
3. **Warehouse:** Snowpipe → Snowflake
4. **Feature engineering:**

   * **Batch** features (daily/weekly) in Snowflake
   * **Online** short-window aggregates to DynamoDB (via Glue/Snowpark jobs)
5. **Training (weekly):** SageMaker pipelines train XGBoost + anomaly model; artifacts in S3
6. **Real‑time scoring:** API Gateway → Lambda → SageMaker Endpoint(s);
7. **Actions & feedback:** allow/hold/block; alerts; outcomes flow back to Snowflake labels
8. **Monitoring:** CloudWatch, Model Monitor, QuickSight

### 3.2 Static portfolio project:

* **Data:** One-time dump to an S3 CSV; Glue/Athena table (`fraud-analytics-usw2.raw`)
* **Train/Val split:** Athena `UNLOAD` to S3
* **Training:** SageMaker XGBoost v1.3, early stopping, class weighting
* **Evaluation:** Local scoring with exported model (identical predictions), metrics + thresholding
* **Artifacts:** training screenshots; model.tar.gz; scored CSV; SQL; results

---

## 4) Modeling

* **Classifier (MVP):** XGBoost (`objective=binary:logistic`), class imbalance via `scale_pos_weight`
* **Anomaly detector (roadmap):** Random Cut Forest or autoencoder
* **Fusion (roadmap):** weighted combination of anomaly score, XGB probability, and rules

---

## 5) Training & Validation (MVP)

**Data prep (Athena UNLOAD):** label first then 30 features; all cast to `VARCHAR` to avoid quoting/typing issues. 80/20 random split.

**SageMaker training:**

* **Algorithm:** Tabular – XGBoost v1.3
* **Input channels:** `train`, `validation` via Pipe mode, Compression = Gzip
* **Instance:** `ml.m5.xlarge` (or `ml.m4.xlarge`), volume ≥ 30 GB
* **Key hyperparameters:**

  * `objective=binary:logistic`, `eval_metric=auc`, `num_round=300`
  * `max_depth=6`, `eta=0.1`, `subsample=0.8`, `colsample_bytree=0.8`
  * `scale_pos_weight=578` (≈ neg/pos), `early_stopping_rounds=25`

---

## 6) Evaluation & Thresholds (Validation)

**Validation set:** 56,866 rows; 95 positives; prevalence 0.167%

**Quality:**

* **ROC‑AUC:** 0.99296
* **PR‑AUC:** 0.83834

**Best‑F1 operating point**
Threshold 0.7991 ⇒ TP=86, FP=19, TN=56,752, FN=9
Precision 0.827, Recall 0.905

**Two‑tier policy (review budget ≈ 0.5%)**
Quantile thresholds from validation probabilities:

* **BLOCK:** p ≥ 0.8801 (≈ top 0.135%)
* **HOLD:** 0.4767 ≤ p < 0.8801 (≈ next 0.366%)
* **ALLOW:** p < 0.4767

Aggregate (HOLD+BLOCK as “flagged”):
**Review rate 0.501%**, **Block rate 0.135%**, **Precision 0.326**, **Recall 0.979**

*Observation:* No single threshold achieves ≥95% precision on this slice, but 2‑tier policy successfully ID's 98% of fraudulent transactions.
