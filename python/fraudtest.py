import glob, pandas as pd, numpy as np, xgboost as xgb
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix

# ---- EDIT THESE PATHS ----
VAL_PARTS = r"YOUR_PATH\val_labeled\*.gz"     # the Athena UNLOAD parts
MODEL_PATH = r"YOUR_PATH\xgboost-model"       # extracted from model.tar.gz
OUT_CSV    = r"YOUR_PATH\val_scored.csv"

dfs = [pd.read_csv(p, header=None) for p in glob.glob(VAL_PARTS)]
df = pd.concat(dfs, ignore_index=True)
df = df.apply(pd.to_numeric, errors="coerce").dropna()

y = df.iloc[:, 0].astype(int).to_numpy()
X = df.iloc[:, 1:].to_numpy()

booster = xgb.Booster()
booster.load_model(MODEL_PATH)
dtest = xgb.DMatrix(X)
probs = booster.predict(dtest)

roc = roc_auc_score(y, probs)
pr  = average_precision_score(y, probs)
print(f"ROC-AUC={roc:.5f}  PR-AUC={pr:.5f}")

for t in (0.90, 0.98):
    pred = (probs >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
    prec = tp/(tp+fp) if (tp+fp)>0 else 0.0
    rec  = tp/(tp+fn) if (tp+fn)>0 else 0.0
    print(f"Threshold {t:.2f}: TP={tp} FP={fp} TN={tn} FN={fn}  Precision={prec:.3f} Recall={rec:.3f}")

pd.DataFrame({"prob": probs, "label": y}).to_csv(OUT_CSV, index=False)
print("Wrote:", OUT_CSV)
