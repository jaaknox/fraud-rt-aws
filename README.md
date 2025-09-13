# Real-Time Fraud Detection (AWS + SageMaker + Athena)

(using Kaggle's "Credit Transactions Fraud Detection Dataset")
(https://www.kaggle.com/datasets/kartik2112/fraud-detection)

**Goal:** classify card transactions as fraudulent or legitimate in real time, with a batch/offline path for analytics and model selection.

## Highlights
- Trained XGBoost (binary:logistic) on AWS SageMaker with class imbalance handled via `scale_pos_weight`.
- Data in S3 with Athena SQL to produce separate training/validation CSVs.
- Validation (20% holdout): ROC-AUC ≈ 0.993, PR-AUC ≈ 0.838.
- Operating thresholds refined using data:
  - Best-F1 cutoff ~ 0.799 -> Precision 0.827, Recall 0.905.
  - Tested a two-tier policy (review ~0.5% of traffic): HOLD >= 0.4767, BLOCK >= 0.8801 -> Review precision 0.326, Recall raised to 0.979.

## How to reproduce (high level)
1. **Athena** (see `athena/*.sql`):  
   - `unload_train.sql` and `unload_val.sql` create CSVs with label as first column 
2. **SageMaker training (console):**  
   - Algorithm: Tabular XGBoost v1.3, Input Pipe + Gzip.  
   - Hyperparameters: `objective=binary:logistic`, `eval_metric=auc`, `num_round=300`, `max_depth=6`, `eta=0.1`, `subsample=0.8`, `colsample_bytree=0.8`, `scale_pos_weight=578`, `early_stopping_rounds=25`.
3. **Validation scoring (local)**:  
   - Download `model.tar.gz` (find `xgboost-model` inside) and the val_labeled UNLOAD parts.  
   - Run `python local/score_offline.py` to compute ROC-AUC, PR-AUC, and thresholds, writing to `local/val_scored.csv`.

## Results
See `docs/results.md` for exact numbers, confusion matrices, and the ALLOW/HOLD/BLOCK policy.

## Roadmap
- Add Snowflake ingestion & features (Snowpipe + Snowpark tasks for rolling aggregates).
- Add anomaly detector (RCF/autoencoder) and a fusion layer in Lambda.
- Add Model Monitor / PSI and QuickSight dashboards.
