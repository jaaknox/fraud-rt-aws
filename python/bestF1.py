import numpy as np, pandas as pd
from sklearn.metrics import precision_recall_curve, confusion_matrix, average_precision_score

#load the scored validation
df = pd.read_csv(r"YOUR_PATH\val_scored.csv")  # prob,label
y = df["label"].astype(int).to_numpy()
p = df["prob"].astype(float).to_numpy()

pos = int(y.sum()); neg = len(y)-pos
print(f"Validation size={len(y)}  positives={pos}  negatives={neg}  prevalence={pos/len(y):.4%}")
print(f"PR-AUC={average_precision_score(y,p):.5f}")

#Best F1 threshold
prec, rec, th = precision_recall_curve(y, p)
f1 = 2*prec*rec/(prec+rec+1e-12)
i = np.nanargmax(f1)
thr_f1 = th[i-1] if i>0 else th[0]  # precision_recall_curve gives thresholds of len-1
print(f"\nBest F1 threshold ≈ {thr_f1:.4f}  (Precision={prec[i]:.3f}, Recall={rec[i]:.3f})")

#Confusion at that threshold
pred = (p >= thr_f1).astype(int)
tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
print(f"At {thr_f1:.4f}: TP={tp} FP={fp} TN={tn} FN={fn}")

TARGET_PREC = 0.95
idx = np.where(prec[:-1] >= TARGET_PREC)[0]
if len(idx):
    j = idx[-1]
    thr_p95 = th[j]
    print(f"\nThreshold for >={TARGET_PREC:.0%} precision = {thr_p95:.4f} (precision={prec[j]:.3f}, recall={rec[j]:.3f})")
    tn, fp, fn, tp = confusion_matrix(y, (p >= thr_p95)).ravel()
    print(f"At {thr_p95:.4f}: TP={tp} FP={fp} TN={tn} FN={fn}")
else:
    print(f"\nNo threshold reaches precision {TARGET_PREC:.0%}. Probably lower it.")

q_block = 0.999
q_hold  = 0.995
thr_block = np.quantile(p, q_block)
thr_hold  = np.quantile(p, q_hold)
print(f"\nQuantile thresholds: HOLD≥{thr_hold:.4f}, BLOCK≥{thr_block:.4f}")

def tri_label(prob):
    if prob >= thr_block: return 2   # BLOCK
    if prob >= thr_hold:  return 1   # HOLD
    return 0                         # ALLOW

pred3 = np.array([tri_label(x) for x in p])
pred_bin = (pred3>0).astype(int)
tn, fp, fn, tp = confusion_matrix(y, pred_bin).ravel()
precision = tp/(tp+fp) if (tp+fp)>0 else 0.0
recall    = tp/(tp+fn) if (tp+fn)>0 else 0.0
review_rate = (pred3>0).mean()
block_rate  = (pred3==2).mean()
print(f"Review rate={review_rate:.3%}  Block rate={block_rate:.3%}  Precision={precision:.3f}  Recall={recall:.3f}")
