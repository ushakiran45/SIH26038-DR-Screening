"""
Fix 2: VQC Tuning Study on Breast Cancer (fast, ~1-3 min per experiment)
Sweeps: epochs, learning rate, class weights, threshold tuning, QSVM, 6-qubit
Saves: tuning_results.csv  tuning_results.png
"""
import os, time, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import pennylane as qml
from sklearn.model_selection import StratifiedKFold
from sklearn.decomposition import PCA
from sklearn.svm import SVC, SVR
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.pipeline import Pipeline

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
X_all = np.load(os.path.join(BASE_DIR, "features.npy"))
y_raw = np.load(os.path.join(BASE_DIR, "labels.npy")).astype(int)
y_all = (y_raw >= 2).astype(int)   # 1 = malignant

np.random.seed(42)
idx200 = np.random.choice(len(y_all), 200, replace=False)
X_data = X_all[idx200]
y_data = y_all[idx200]

print(f"Dataset: N=200, malignant={y_data.sum()}, benign={(y_data==0).sum()}")

SKF = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# ─────────────────────────────────────────────────────────────────────────────
def make_vqc_circuit(n_qubits):
    dev = qml.device("default.qubit", wires=n_qubits)
    @qml.qnode(dev, interface="torch")
    def circuit(inputs, weights):
        for i in range(n_qubits):
            qml.RY(inputs[i], wires=i)
        for l in range(weights.shape[0]):
            for i in range(n_qubits):
                qml.RY(weights[l, i, 0], wires=i)
                qml.RZ(weights[l, i, 1], wires=i)
            for i in range(n_qubits):
                qml.CNOT(wires=[i, (i+1) % n_qubits])
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
    return circuit

class VQC(nn.Module):
    def __init__(self, circuit, n_qubits, n_layers):
        super().__init__()
        self.circuit = circuit
        self.weights = nn.Parameter(torch.randn(n_layers, n_qubits, 2) * 0.15)
        self.fc = nn.Linear(n_qubits, 2)
    def forward(self, x):
        outs = [torch.stack(self.circuit(x[b], self.weights)).to(torch.float32)
                for b in range(x.shape[0])]
        return self.fc(torch.stack(outs))

def run_vqc_fold(X_tr_ang, y_tr, X_te_ang, y_te,
                 n_qubits, n_layers, epochs, lr,
                 class_weight=None, threshold=0.5):
    circuit = make_vqc_circuit(n_qubits)
    model = VQC(circuit, n_qubits, n_layers)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    w = None
    if class_weight == "balanced":
        n_pos = y_tr.sum(); n_neg = len(y_tr) - n_pos
        w = torch.tensor([n_pos / len(y_tr), n_neg / len(y_tr)], dtype=torch.float32)
    crit = nn.CrossEntropyLoss(weight=w)

    Xt = torch.tensor(X_tr_ang, dtype=torch.float32)
    yt = torch.tensor(y_tr, dtype=torch.long)
    Xe = torch.tensor(X_te_ang, dtype=torch.float32)

    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        loss = crit(model(Xt), yt)
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        logits = model(Xe)
        probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        preds = (probs >= threshold).astype(int)

    acc = accuracy_score(y_te, preds)
    tp = int(((y_te==1)&(preds==1)).sum())
    tn = int(((y_te==0)&(preds==0)).sum())
    fn = int(((y_te==1)&(preds==0)).sum())
    fp = int(((y_te==0)&(preds==1)).sum())
    sens = tp / max(tp+fn, 1)
    spec = tn / max(tn+fp, 1)
    _, _, f1, _ = precision_recall_fscore_support(y_te, preds, average="weighted", zero_division=0)
    return acc, sens, spec, f1

def run_svm_fold(X_tr, y_tr, X_te, y_te):
    svm = SVC(kernel="rbf", C=2.0, random_state=42)
    svm.fit(X_tr, y_tr)
    preds = svm.predict(X_te)
    acc = accuracy_score(y_te, preds)
    tp = int(((y_te==1)&(preds==1)).sum())
    tn = int(((y_te==0)&(preds==0)).sum())
    fn = int(((y_te==1)&(preds==0)).sum())
    fp = int(((y_te==0)&(preds==1)).sum())
    sens = tp / max(tp+fn, 1)
    spec = tn / max(tn+fp, 1)
    _, _, f1, _ = precision_recall_fscore_support(y_te, preds, average="weighted", zero_division=0)
    return acc, sens, spec, f1

def run_qsvm_fold(X_tr, y_tr, X_te, y_te, n_qubits):
    """Quantum kernel SVM using ZZFeatureMap-style kernel via PennyLane."""
    dev = qml.device("default.qubit", wires=n_qubits)
    @qml.qnode(dev)
    def kernel_circuit(x1, x2):
        qml.AngleEmbedding(x1, wires=range(n_qubits))
        qml.adjoint(qml.AngleEmbedding)(x2, wires=range(n_qubits))
        return qml.probs(wires=range(n_qubits))
    def qkernel(A, B):
        return np.array([[kernel_circuit(a, b)[0] for b in B] for a in A])
    K_tr = qkernel(X_tr, X_tr)
    K_te = qkernel(X_te, X_tr)
    svm = SVC(kernel="precomputed", C=2.0, random_state=42)
    svm.fit(K_tr, y_tr)
    preds = svm.predict(K_te)
    acc = accuracy_score(y_te, preds)
    tp = int(((y_te==1)&(preds==1)).sum())
    tn = int(((y_te==0)&(preds==0)).sum())
    fn = int(((y_te==1)&(preds==0)).sum())
    fp = int(((y_te==0)&(preds==1)).sum())
    sens = tp / max(tp+fn, 1)
    spec = tn / max(tn+fp, 1)
    _, _, f1, _ = precision_recall_fscore_support(y_te, preds, average="weighted", zero_division=0)
    return acc, sens, spec, f1

def cv_experiment(label, n_qubits, n_layers, epochs, lr,
                  class_weight=None, threshold=0.5,
                  run_qsvm=False):
    accs, senss, specs, f1s = [], [], [], []
    t0 = time.time()
    for tr_idx, te_idx in SKF.split(X_data, y_data):
        X_tr_raw, y_tr = X_data[tr_idx], y_data[tr_idx]
        X_te_raw, y_te = X_data[te_idx], y_data[te_idx]
        pca = PCA(n_components=n_qubits, random_state=42)
        pca_tr = pca.fit_transform(X_tr_raw)
        mn, mx = pca_tr.min(0), pca_tr.max(0)
        rng = np.maximum(mx - mn, 1e-6)
        def ang(a): return np.clip((((a-mn)/rng)*2-1)*np.pi, -np.pi, np.pi)
        X_tr_a, X_te_a = ang(pca_tr), ang(pca.transform(X_te_raw))
        if run_qsvm:
            a, s, sp, f = run_qsvm_fold(X_tr_a, y_tr, X_te_a, y_te, n_qubits)
        else:
            a, s, sp, f = run_vqc_fold(X_tr_a, y_tr, X_te_a, y_te,
                                        n_qubits, n_layers, epochs, lr,
                                        class_weight, threshold)
        accs.append(a); senss.append(s); specs.append(sp); f1s.append(f)
    elapsed = time.time() - t0
    row = {
        "Experiment": label,
        "Accuracy": f"{np.mean(accs)*100:.1f}% ± {np.std(accs)*100:.1f}%",
        "Sensitivity": f"{np.mean(senss)*100:.1f}% ± {np.std(senss)*100:.1f}%",
        "Specificity": f"{np.mean(specs)*100:.1f}% ± {np.std(specs)*100:.1f}%",
        "F1": f"{np.mean(f1s)*100:.1f}% ± {np.std(f1s)*100:.1f}%",
        "Time_s": f"{elapsed:.0f}s",
        "_acc_mean": np.mean(accs)*100,
        "_sens_mean": np.mean(senss)*100,
    }
    print(f"  {label:50s}  Acc={row['Accuracy']}  Sens={row['Sensitivity']}  ({elapsed:.0f}s)")
    return row

# ─────────────────────────────────────────────────────────────────────────────
print("\n[TUNING] Running baseline SVM (reference)...")
svm_accs, svm_senss = [], []
for tr_idx, te_idx in SKF.split(X_data, y_data):
    X_tr_raw, y_tr = X_data[tr_idx], y_data[tr_idx]
    X_te_raw, y_te = X_data[te_idx], y_data[te_idx]
    pca = PCA(n_components=4, random_state=42)
    pca_tr = pca.fit_transform(X_tr_raw)
    mn, mx = pca_tr.min(0), pca_tr.max(0)
    rng = np.maximum(mx-mn, 1e-6)
    def ang(a): return np.clip((((a-mn)/rng)*2-1)*np.pi, -np.pi, np.pi)
    a, s, sp, f = run_svm_fold(ang(pca_tr), y_tr, ang(pca.transform(X_te_raw)), y_te)
    svm_accs.append(a); svm_senss.append(s)
svm_row = {
    "Experiment": "[SVM] Classical SVM (RBF) -- reference",
    "Accuracy": f"{np.mean(svm_accs)*100:.1f}% +/- {np.std(svm_accs)*100:.1f}%",
    "Sensitivity": f"{np.mean(svm_senss)*100:.1f}% +/- {np.std(svm_senss)*100:.1f}%",
    "Specificity": "--", "F1": "--", "Time_s": "--",
    "_acc_mean": np.mean(svm_accs)*100, "_sens_mean": np.mean(svm_senss)*100,
}
print(f"  {'[SVM] reference':50s}  Acc={svm_row['Accuracy']}  Sens={svm_row['Sensitivity']}")

results = [svm_row]

print("\n[TUNING] E0 — Baseline VQC (20 epochs, lr=0.03, 4q, 2L)")
results.append(cv_experiment("E0 Baseline (20ep, lr=0.03, 4q, 2L)", 4, 2, 20, 0.03))

print("\n[TUNING] E1 — More epochs (100ep, lr=0.03, 4q, 2L)")
results.append(cv_experiment("E1 More epochs (100ep, lr=0.03, 4q, 2L)", 4, 2, 100, 0.03))

print("\n[TUNING] E2a — LR sweep lr=0.01")
results.append(cv_experiment("E2a LR=0.01 (50ep, 4q, 2L)", 4, 2, 50, 0.01))

print("\n[TUNING] E2b — LR sweep lr=0.05")
results.append(cv_experiment("E2b LR=0.05 (50ep, 4q, 2L)", 4, 2, 50, 0.05))

print("\n[TUNING] E2c — LR sweep lr=0.10")
results.append(cv_experiment("E2c LR=0.10 (50ep, 4q, 2L)", 4, 2, 50, 0.10))

print("\n[TUNING] E3 — 6 qubits, 6 PCA, 3 layers, 50ep")
results.append(cv_experiment("E3 6-qubit 6-PCA 3-layer (50ep, lr=0.03)", 6, 3, 50, 0.03))

print("\n[TUNING] E4 — Class-weighted loss (balanced, 50ep, lr=0.03, 4q)")
results.append(cv_experiment("E4 Class-weighted loss (50ep, lr=0.03, 4q, 2L)", 4, 2, 50, 0.03, class_weight="balanced"))

print("\n[TUNING] E5 — Threshold=0.35 (50ep, lr=0.03, 4q, class-weighted)")
results.append(cv_experiment("E5 Threshold=0.35 + class-weight (50ep, lr=0.03, 4q)", 4, 2, 50, 0.03, class_weight="balanced", threshold=0.35))

print("\n[TUNING] E6 — QSVM (quantum kernel, 4q)")
results.append(cv_experiment("E6 QSVM quantum kernel (4q)", 4, 2, 0, 0, run_qsvm=True))

# Best VQC config (pick best acc from E1-E5 for final run with 150 epochs)
print("\n[TUNING] E7 — Best config: 4q, class-weight, threshold=0.35, 150ep, lr=0.05")
results.append(cv_experiment("E7 Best config (150ep, lr=0.05, class-weight, t=0.35, 4q)", 4, 2, 150, 0.05, class_weight="balanced", threshold=0.35))

# ─────────────────────────────────────────────────────────────────────────────
df = pd.DataFrame(results)
csv_path = os.path.join(BASE_DIR, "tuning_results.csv")
df.drop(columns=["_acc_mean","_sens_mean"]).to_csv(csv_path, index=False)
print(f"\n[TUNING] Saved {csv_path}")
print(df[["Experiment","Accuracy","Sensitivity","Specificity","F1","Time_s"]].to_string(index=False))

# Plot
fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=200)
labels = [r["Experiment"][:40] for r in results]
accs = [r["_acc_mean"] for r in results]
senss = [r["_sens_mean"] for r in results]
colors = ["#00b4d8" if "SVM" in l else "#ffb300" for l in labels]

axes[0].barh(labels, accs, color=colors)
axes[0].axvline(64.5, color="red", linestyle="--", linewidth=1, label="Majority baseline 64.5%")
axes[0].set_xlabel("Accuracy (%)"); axes[0].set_title("Accuracy — VQC Tuning Study"); axes[0].legend(fontsize=7)
axes[0].set_xlim(0, 100)

axes[1].barh(labels, senss, color=colors)
axes[1].set_xlabel("Malignant Sensitivity (%)"); axes[1].set_title("Sensitivity — VQC Tuning Study")
axes[1].set_xlim(0, 100)

plt.tight_layout()
png_path = os.path.join(BASE_DIR, "tuning_results.png")
plt.savefig(png_path, bbox_inches="tight")
plt.close()
print(f"[TUNING] Plot saved {png_path}")
