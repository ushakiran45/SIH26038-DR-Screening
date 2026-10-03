"""
Fix 1: APTOS 2019 Feature Extraction & CNN Benchmark
=====================================================
- Loads APTOS 2019 train.csv + train_images/
- Takes a stratified 400-image held-out validation split (seed=42, never seen by training)
- Applies the SAME preprocessing as ai_ml_pipeline.py (crop -> resize 380x380 -> ImageNet norm)
- Extracts 1536-D EfficientNet-B3 features + CNN predictions
- Saves: aptos_val_features.npy, aptos_val_labels.npy, aptos_cnn_metrics.json
- Also saves aptos_val_split.csv so the exact split is reproducible
"""

import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score

from ai_ml_pipeline import load_model, preprocess_fundus_image, to_tensor, DEVICE

APTOS_DIR  = r"C:\Users\USHA\Downloads\aptos2019-blindness-detection"
IMG_DIR    = os.path.join(APTOS_DIR, "train_images")
CSV_PATH   = os.path.join(APTOS_DIR, "train.csv")
OUT_DIR    = os.path.dirname(os.path.abspath(__file__))  # bc_benchmark/

N_VAL      = 400    # held-out images
SEED       = 42

# ── Load labels ──────────────────────────────────────────────────────────────
df = pd.read_csv(CSV_PATH)
print(f"[APTOS] Full dataset: {len(df)} images")
print(f"[APTOS] Class counts:\n{df['diagnosis'].value_counts().sort_index().to_string()}")

# Stratified split: 400 val, rest "train" (we only use val)
_, df_val = train_test_split(
    df, test_size=N_VAL, stratify=df["diagnosis"], random_state=SEED
)
df_val = df_val.reset_index(drop=True)
df_val.to_csv(os.path.join(OUT_DIR, "aptos_val_split.csv"), index=False)
print(f"\n[APTOS] Val split: {len(df_val)} images")
print(f"[APTOS] Val class distribution:\n{df_val['diagnosis'].value_counts().sort_index().to_string()}")
referable_count = (df_val['diagnosis'] >= 2).sum()
print(f"[APTOS] Referable (>=2): {referable_count}  |  Non-referable (<2): {len(df_val)-referable_count}")

# ── Load model ────────────────────────────────────────────────────────────────
print("\n[APTOS] Loading EfficientNet-B3 ...")
model = load_model()

# ── Extract features ──────────────────────────────────────────────────────────
features_list, labels_list, probs_list = [], [], []
errors = 0
t0 = time.time()

for i, row in df_val.iterrows():
    img_path = os.path.join(IMG_DIR, row["id_code"] + ".png")
    if not os.path.exists(img_path):
        img_path = os.path.join(IMG_DIR, row["id_code"] + ".jpg")
    if not os.path.exists(img_path):
        print(f"  [SKIP] {row['id_code']} — file not found")
        errors += 1
        continue

    try:
        img = preprocess_fundus_image(img_path)
        tensor = to_tensor(img).to(DEVICE)

        with torch.no_grad():
            conv_feats = model.backbone.forward_features(tensor)
            pooled = model.backbone.global_pool(conv_feats)
            logits = model.backbone.classifier(pooled)
            probs = F.softmax(logits, dim=1).cpu().numpy()[0]

        feat_vec = pooled.cpu().numpy()[0]
        features_list.append(feat_vec)
        labels_list.append(row["diagnosis"])
        probs_list.append(probs)

        if (i+1) % 50 == 0:
            elapsed = time.time() - t0
            print(f"  [{i+1}/{len(df_val)}] processed  ({elapsed:.0f}s elapsed)")

    except Exception as e:
        print(f"  [ERR] {row['id_code']}: {e}")
        errors += 1

X = np.array(features_list, dtype=np.float32)
y = np.array(labels_list, dtype=int)
P = np.array(probs_list, dtype=np.float32)

print(f"\n[APTOS] Extracted {len(X)} / {len(df_val)} images  ({errors} skipped/errored)")
print(f"[APTOS] Feature shape: {X.shape}")

# ── Save features ─────────────────────────────────────────────────────────────
np.save(os.path.join(OUT_DIR, "aptos_val_features.npy"), X)
np.save(os.path.join(OUT_DIR, "aptos_val_labels.npy"), y)
print(f"[APTOS] Saved aptos_val_features.npy + aptos_val_labels.npy")

# ── CNN evaluation ────────────────────────────────────────────────────────────
y_pred_5class = P.argmax(axis=1)
acc_5class = accuracy_score(y, y_pred_5class)

# Binary referable: level >= 2
ref_true = (y >= 2).astype(int)
ref_pred_argmax = (y_pred_5class >= 2).astype(int)

# Threshold-based: P(referable) = sum of class 2,3,4 probs
p_referable = P[:, 2:].sum(axis=1)
ref_pred_thresh = (p_referable >= 0.5).astype(int)

tp_a = int(((ref_true==1)&(ref_pred_argmax==1)).sum())
tn_a = int(((ref_true==0)&(ref_pred_argmax==0)).sum())
fn_a = int(((ref_true==1)&(ref_pred_argmax==0)).sum())
fp_a = int(((ref_true==0)&(ref_pred_argmax==1)).sum())
sens_argmax = tp_a / max(tp_a+fn_a, 1)
spec_argmax = tn_a / max(tn_a+fp_a, 1)

tp_t = int(((ref_true==1)&(ref_pred_thresh==1)).sum())
tn_t = int(((ref_true==0)&(ref_pred_thresh==0)).sum())
fn_t = int(((ref_true==1)&(ref_pred_thresh==0)).sum())
fp_t = int(((ref_true==0)&(ref_pred_thresh==1)).sum())
sens_thresh = tp_t / max(tp_t+fn_t, 1)
spec_thresh = tn_t / max(tn_t+fp_t, 1)

try:
    auc = roc_auc_score(ref_true, p_referable)
except Exception:
    auc = float("nan")

# Majority baseline (most common class in val)
majority_class = (ref_true.mean() >= 0.5)   # True if majority referable
majority_acc = max(ref_true.mean(), 1-ref_true.mean())

metrics = {
    "dataset": "APTOS 2019 (held-out val split, N=400, stratified seed=42)",
    "N": int(len(X)),
    "referable_N": int(ref_true.sum()),
    "non_referable_N": int((ref_true==0).sum()),
    "majority_baseline_acc_pct": round(float(majority_acc)*100, 1),
    "five_class_accuracy_pct": round(float(acc_5class)*100, 1),
    "cnn_sensitivity_argmax_pct": round(float(sens_argmax)*100, 1),
    "cnn_specificity_argmax_pct": round(float(spec_argmax)*100, 1),
    "cnn_sensitivity_threshold05_pct": round(float(sens_thresh)*100, 1),
    "cnn_specificity_threshold05_pct": round(float(spec_thresh)*100, 1),
    "roc_auc": round(float(auc), 3),
    "note": (
        "Preprocessing: crop black borders -> resize 380x380 -> ImageNet normalisation. "
        "No Ben Graham applied (preprocess_fundus_image does not apply ben_graham_preprocess). "
        "argmax = argmax over 5 classes; threshold = P(class2+3+4) >= 0.5."
    )
}

out_json = os.path.join(OUT_DIR, "aptos_cnn_metrics.json")
with open(out_json, "w") as f:
    json.dump(metrics, f, indent=2)

print("\n" + "="*60)
print("  APTOS 2019 CNN EVALUATION RESULTS")
print("="*60)
print(f"  N evaluated   : {len(X)}")
print(f"  Referable N   : {ref_true.sum()}  ({ref_true.mean()*100:.1f}%)")
print(f"  Majority baseline acc : {majority_acc*100:.1f}%")
print(f"  5-class accuracy      : {acc_5class*100:.1f}%")
print(f"  ROC-AUC               : {auc:.3f}")
print(f"  Sensitivity (argmax)  : {sens_argmax*100:.1f}%")
print(f"  Specificity (argmax)  : {spec_argmax*100:.1f}%")
print(f"  Sensitivity (thresh)  : {sens_thresh*100:.1f}%  <- sum P(2+3+4) >= 0.5")
print(f"  Specificity (thresh)  : {spec_thresh*100:.1f}%")
print(f"\n  Saved: {out_json}")
print("="*60)
print("\n[NEXT] Run: python aptos_qml_benchmark.py")
print("       This will run the full QML/SVM benchmark on these features.")
