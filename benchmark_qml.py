"""
================================================================================
SIH26139 - STRATIFIED CROSS-VALIDATED QUANTUM ML BENCHMARK & SCALING ENGINE
================================================================================
Runs 5-Fold Stratified Cross-Validation on Quantum (VQC) and Classical (SVM) models.
Computes mean ± std for Accuracy, Referable Sensitivity, Specificity, F1-Score,
Training Time, and Inference Latency.

Saves output to:
  results_summary.csv
  results_summary.png (Scaling & Model Comparison Plot)
================================================================================
"""

import os
import argparse
import time
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FEATURES_PATH = os.path.join(BASE_DIR, "features.npy")
LABELS_PATH = os.path.join(BASE_DIR, "labels.npy")

p = argparse.ArgumentParser()
p.add_argument("--n_samples", type=int, default=None, help="Number of samples to evaluate")
p.add_argument("--n_qubits", type=int, default=4, help="Number of qubits (4, 6, 8)")
p.add_argument("--n_layers", type=int, default=2, help="Number of VQC variational layers")
p.add_argument("--folds", type=int, default=5, help="Number of cross-validation folds")
p.add_argument("--skip_qsvm", action="store_true", help="Skip QSVM evaluation")
p.add_argument("--binary", action="store_true", default=True, help="Use binary referable DR (level >= 2)")
p.add_argument("--output_prefix", default="results_summary", help="Output file prefix")
args, _ = p.parse_known_args()

# ------------------------------------------------------------------------------
# 1. LOAD DATASET & PREPARE SPLITS
# ------------------------------------------------------------------------------
if not (os.path.exists(FEATURES_PATH) and os.path.exists(LABELS_PATH)):
    from extract_features import extract_real_dataset_features
    extract_real_dataset_features()

X_all = np.load(FEATURES_PATH)
y_all = np.load(LABELS_PATH).astype(int)

if args.binary:
    # Referable DR binary target: Level >= 2 -> 1, Level < 2 -> 0
    y_target = (y_all >= 2).astype(int)
    num_classes = 2
else:
    y_target = y_all
    num_classes = 5

if args.n_samples and args.n_samples < len(y_target):
    np.random.seed(42)
    idx = np.random.choice(len(y_target), args.n_samples, replace=False)
    X_data = X_all[idx]
    y_data = y_target[idx]
else:
    X_data = X_all
    y_data = y_target

print(f"[BENCHMARK] Evaluating N={len(y_data)} samples | Qubits={args.n_qubits} | Binary={args.binary} | Folds={args.folds}")

# ------------------------------------------------------------------------------
# 2. VQC MODEL SPECIFICATION
# ------------------------------------------------------------------------------
n_qubits = args.n_qubits
n_layers = args.n_layers
q_dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(q_dev, interface="torch")
def benchmark_vqc_circuit(inputs, weights):
    for i in range(n_qubits):
        qml.RY(inputs[i], wires=i)
    for l in range(weights.shape[0]):
        for i in range(n_qubits):
            qml.RY(weights[l, i, 0], wires=i)
            qml.RZ(weights[l, i, 1], wires=i)
        for i in range(n_qubits):
            qml.CNOT(wires=[i, (i + 1) % n_qubits])
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]


class BenchmarkVQC(nn.Module):
    def __init__(self, n_q=n_qubits, n_l=n_layers, n_c=num_classes):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(n_l, n_q, 2, dtype=torch.float32) * 0.15)
        self.fc = nn.Linear(n_q, n_c)

    def forward(self, x_pca):
        q_outs = []
        for b in range(x_pca.shape[0]):
            res = torch.stack(benchmark_vqc_circuit(x_pca[b], self.weights)).to(torch.float32)
            q_outs.append(res)
        q_feats = torch.stack(q_outs)
        return self.fc(q_feats)


# ------------------------------------------------------------------------------
# 3. 5-FOLD STRATIFIED CROSS-VALIDATION LOOP
# ------------------------------------------------------------------------------
skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=42)

svm_accs, svm_f1s, svm_senss, svm_specs, svm_train_times, svm_inf_times = [], [], [], [], [], []
vqc_accs, vqc_f1s, vqc_senss, vqc_specs, vqc_train_times, vqc_inf_times = [], [], [], [], [], []

fold_idx = 1
for train_idx, test_idx in skf.split(X_data, y_data):
    X_tr_raw, y_tr = X_data[train_idx], y_data[train_idx]
    X_te_raw, y_te = X_data[test_idx], y_data[test_idx]

    # Fit PCA ONLY on train split
    pca = PCA(n_components=n_qubits, random_state=42)
    pca_tr = pca.fit_transform(X_tr_raw)
    min_v, max_v = pca_tr.min(axis=0), pca_tr.max(axis=0)
    range_v = np.maximum(max_v - min_v, 1e-6)
    
    def scale_angles(arr):
        norm = (arr - min_v) / range_v
        return np.clip((norm * 2.0 - 1.0) * np.pi, -np.pi, np.pi)

    X_tr_ang = scale_angles(pca_tr)
    X_te_ang = scale_angles(pca.transform(X_te_raw))

    # --- A. Classical SVM Baseline ---
    t0 = time.time()
    svm = SVC(kernel="rbf", C=2.0, probability=True, random_state=42)
    svm.fit(X_tr_ang, y_tr)
    t_svm_train = time.time() - t0

    t0 = time.time()
    svm_preds = svm.predict(X_te_ang)
    t_svm_inf = (time.time() - t0) * 1000.0 / len(y_te)

    svm_acc = accuracy_score(y_te, svm_preds)
    _, _, svm_f1, _ = precision_recall_fscore_support(y_te, svm_preds, average="weighted", zero_division=0)
    
    # Sensitivity & Specificity
    tp_s = int(((y_te == 1) & (svm_preds == 1)).sum())
    tn_s = int(((y_te == 0) & (svm_preds == 0)).sum())
    fn_s = int(((y_te == 1) & (svm_preds == 0)).sum())
    fp_s = int(((y_te == 0) & (svm_preds == 1)).sum())
    svm_sens = tp_s / max(tp_s + fn_s, 1)
    svm_spec = tn_s / max(tn_s + fp_s, 1)

    svm_accs.append(svm_acc)
    svm_f1s.append(svm_f1)
    svm_senss.append(svm_sens)
    svm_specs.append(svm_spec)
    svm_train_times.append(t_svm_train)
    svm_inf_times.append(t_svm_inf)

    # --- B. Hybrid VQC Model ---
    t0 = time.time()
    vqc = BenchmarkVQC(n_q=n_qubits, n_l=n_layers, n_c=num_classes)
    opt = torch.optim.Adam(vqc.parameters(), lr=0.03)
    crit = nn.CrossEntropyLoss()

    X_tr_t = torch.tensor(X_tr_ang, dtype=torch.float32)
    y_tr_t = torch.tensor(y_tr, dtype=torch.long)
    X_te_t = torch.tensor(X_te_ang, dtype=torch.float32)

    vqc.train()
    for ep in range(20):
        opt.zero_grad()
        loss = crit(vqc(X_tr_t), y_tr_t)
        loss.backward()
        opt.step()
    t_vqc_train = time.time() - t0

    t0 = time.time()
    vqc.eval()
    with torch.no_grad():
        vqc_logits = vqc(X_te_t)
        vqc_preds = torch.argmax(vqc_logits, dim=1).cpu().numpy()
    t_vqc_inf = (time.time() - t0) * 1000.0 / len(y_te)

    vqc_acc = accuracy_score(y_te, vqc_preds)
    _, _, vqc_f1, _ = precision_recall_fscore_support(y_te, vqc_preds, average="weighted", zero_division=0)
    
    tp_v = int(((y_te == 1) & (vqc_preds == 1)).sum())
    tn_v = int(((y_te == 0) & (vqc_preds == 0)).sum())
    fn_v = int(((y_te == 1) & (vqc_preds == 0)).sum())
    fp_v = int(((y_te == 0) & (vqc_preds == 1)).sum())
    vqc_sens = tp_v / max(tp_v + fn_v, 1)
    vqc_spec = tn_v / max(tn_v + fp_v, 1)

    vqc_accs.append(vqc_acc)
    vqc_f1s.append(vqc_f1)
    vqc_senss.append(vqc_sens)
    vqc_specs.append(vqc_spec)
    vqc_train_times.append(t_vqc_train)
    vqc_inf_times.append(t_vqc_inf)

    print(f"  Fold {fold_idx}/{args.folds} | SVM Acc: {svm_acc*100:.1f}% | VQC Acc: {vqc_acc*100:.1f}% | VQC Sens: {vqc_sens*100:.1f}%")
    fold_idx += 1

# ------------------------------------------------------------------------------
# 4. COMPUTE CROSS-VALIDATED SUMMARY STATISTICS (MEAN ± STD)
# ------------------------------------------------------------------------------
majority_class_count = max((y_data == 1).sum(), (y_data == 0).sum())
majority_baseline_acc = float(majority_class_count) / len(y_data) * 100.0

summary_df = pd.DataFrame([
    {
        "Model": "Majority Class Baseline (Always Referable)",
        "N_Samples": len(y_data),
        "Qubits": "N/A",
        "Accuracy": f"{majority_baseline_acc:.1f}% ± 0.0%",
        "Referable_Sensitivity": "100.0% ± 0.0%",
        "Specificity": "0.0% ± 0.0%",
        "F1_Score": f"{(majority_baseline_acc/100.0 * 2.0 / (1.0 + majority_baseline_acc/100.0))*100:.1f}% ± 0.0%",
        "Train_Time_Sec": "0.000s",
        "Inference_Latency_ms": "0.00ms"
    },
    {
        "Model": "Classical SVM (RBF)",
        "N_Samples": len(y_data),
        "Qubits": args.n_qubits,
        "Accuracy": f"{np.mean(svm_accs)*100:.1f}% ± {np.std(svm_accs)*100:.1f}%",
        "Referable_Sensitivity": f"{np.mean(svm_senss)*100:.1f}% ± {np.std(svm_senss)*100:.1f}%",
        "Specificity": f"{np.mean(svm_specs)*100:.1f}% ± {np.std(svm_specs)*100:.1f}%",
        "F1_Score": f"{np.mean(svm_f1s)*100:.1f}% ± {np.std(svm_f1s)*100:.1f}%",
        "Train_Time_Sec": f"{np.mean(svm_train_times):.3f}s",
        "Inference_Latency_ms": f"{np.mean(svm_inf_times):.2f}ms"
    },
    {
        "Model": "Hybrid PennyLane VQC",
        "N_Samples": len(y_data),
        "Qubits": args.n_qubits,
        "Accuracy": f"{np.mean(vqc_accs)*100:.1f}% ± {np.std(vqc_accs)*100:.1f}%",
        "Referable_Sensitivity": f"{np.mean(vqc_senss)*100:.1f}% ± {np.std(vqc_senss)*100:.1f}%",
        "Specificity": f"{np.mean(vqc_specs)*100:.1f}% ± {np.std(vqc_specs)*100:.1f}%",
        "F1_Score": f"{np.mean(vqc_f1s)*100:.1f}% ± {np.std(vqc_f1s)*100:.1f}%",
        "Train_Time_Sec": f"{np.mean(vqc_train_times):.3f}s",
        "Inference_Latency_ms": f"{np.mean(vqc_inf_times):.2f}ms"
    }
])

csv_path = os.path.join(BASE_DIR, f"{args.output_prefix}.csv")
summary_df.to_csv(csv_path, index=False)
print(f"\n[BENCHMARK] Saved Stratified {args.folds}-Fold Summary CSV to {csv_path}")
print(summary_df.to_string(index=False))


# ------------------------------------------------------------------------------
# 5. GENERATE COMPARISON BAR CHART (PNG)
# ------------------------------------------------------------------------------
plt.figure(figsize=(10, 5), dpi=300)
models = ["Majority Baseline", "Classical SVM", "Hybrid VQC"]
acc_means = [majority_baseline_acc, np.mean(svm_accs)*100, np.mean(vqc_accs)*100]
acc_stds = [0.0, np.std(svm_accs)*100, np.std(vqc_accs)*100]

sens_means = [100.0, np.mean(svm_senss)*100, np.mean(vqc_senss)*100]
sens_stds = [0.0, np.std(svm_senss)*100, np.std(vqc_senss)*100]

x = np.arange(len(models))
width = 0.35

plt.bar(x - width/2, acc_means, width, yerr=acc_stds, label="Accuracy (%)", color="#ffb300", capsize=5)
plt.bar(x + width/2, sens_means, width, yerr=sens_stds, label="Referable Sensitivity (%)", color="#00f2fe", capsize=5)

plt.title(f"SIH26139 Stratified {args.folds}-Fold Cross-Validation (N={len(y_data)} Seed Images, Qubits={args.n_qubits})", fontsize=11, fontweight="bold")
plt.ylabel("Score (%)", fontsize=10)
plt.xticks(x, models, fontsize=10, fontweight="bold")
plt.ylim(0, 110)
plt.grid(axis="y", linestyle="--", alpha=0.3)
plt.legend(loc="lower right")
plt.tight_layout()

png_path = os.path.join(BASE_DIR, f"{args.output_prefix}.png")
plt.savefig(png_path)
plt.close()
print(f"[PLOT] Saved Benchmark Plot to {png_path}")

