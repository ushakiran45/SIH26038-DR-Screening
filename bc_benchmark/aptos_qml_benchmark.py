"""
Fix 1: APTOS 2019 QML + SVM Benchmark
======================================
Reads aptos_val_features.npy + aptos_val_labels.npy (output of aptos_extract_features.py)
Runs 5-fold stratified CV: SVM vs VQC (binary referable DR classification)
Saves: aptos_results_summary.csv  aptos_results_summary.png
"""
import os, sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch, torch.nn as nn, time
import pennylane as qml
from sklearn.model_selection import StratifiedKFold
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
X_all = np.load(os.path.join(BASE_DIR, "aptos_val_features.npy"))
y_raw = np.load(os.path.join(BASE_DIR, "aptos_val_labels.npy")).astype(int)
y_all = (y_raw >= 2).astype(int)   # binary: referable vs non-referable

N_QUBITS = 4
N_LAYERS = 2
N_FOLDS  = 5

print(f"[APTOS BENCHMARK] N={len(y_all)}  Referable={y_all.sum()}  Non-referable={(y_all==0).sum()}")
majority_acc = max(y_all.mean(), 1-y_all.mean()) * 100
print(f"[APTOS BENCHMARK] Majority baseline: {majority_acc:.1f}%")

q_dev = qml.device("default.qubit", wires=N_QUBITS)

@qml.qnode(q_dev, interface="torch")
def circuit(inputs, weights):
    for i in range(N_QUBITS):
        qml.RY(inputs[i], wires=i)
    for l in range(weights.shape[0]):
        for i in range(N_QUBITS):
            qml.RY(weights[l, i, 0], wires=i)
            qml.RZ(weights[l, i, 1], wires=i)
        for i in range(N_QUBITS):
            qml.CNOT(wires=[i, (i+1) % N_QUBITS])
    return [qml.expval(qml.PauliZ(i)) for i in range(N_QUBITS)]

class VQC(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(N_LAYERS, N_QUBITS, 2)*0.15)
        self.fc = nn.Linear(N_QUBITS, 2)
    def forward(self, x):
        outs = [torch.stack(circuit(x[b], self.weights)).to(torch.float32)
                for b in range(x.shape[0])]
        return self.fc(torch.stack(outs))

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

svm_accs, svm_senss, svm_specs, svm_f1s = [], [], [], []
vqc_accs, vqc_senss, vqc_specs, vqc_f1s = [], [], [], []

for fold, (tr_idx, te_idx) in enumerate(skf.split(X_all, y_all)):
    X_tr_raw, y_tr = X_all[tr_idx], y_all[tr_idx]
    X_te_raw, y_te = X_all[te_idx], y_all[te_idx]

    pca = PCA(n_components=N_QUBITS, random_state=42)
    pca_tr = pca.fit_transform(X_tr_raw)
    mn, mx = pca_tr.min(0), pca_tr.max(0)
    rng = np.maximum(mx-mn, 1e-6)
    def ang(a): return np.clip((((a-mn)/rng)*2-1)*np.pi, -np.pi, np.pi)
    X_tr_a, X_te_a = ang(pca_tr), ang(pca.transform(X_te_raw))

    # SVM
    svm = SVC(kernel="rbf", C=2.0, random_state=42)
    svm.fit(X_tr_a, y_tr)
    sp = svm.predict(X_te_a)
    def stats(yt, yp):
        tp=int(((yt==1)&(yp==1)).sum()); tn=int(((yt==0)&(yp==0)).sum())
        fn=int(((yt==1)&(yp==0)).sum()); fp=int(((yt==0)&(yp==1)).sum())
        _,_,f1,_=precision_recall_fscore_support(yt,yp,average="weighted",zero_division=0)
        return accuracy_score(yt,yp), tp/max(tp+fn,1), tn/max(tn+fp,1), f1
    a,s,sc,f = stats(y_te, sp)
    svm_accs.append(a); svm_senss.append(s); svm_specs.append(sc); svm_f1s.append(f)

    # VQC (class-weighted, 50 epochs, lr=0.05 — best from tuning study)
    n_pos=y_tr.sum(); n_neg=len(y_tr)-n_pos
    w=torch.tensor([n_pos/len(y_tr), n_neg/len(y_tr)],dtype=torch.float32)
    model=VQC(); opt=torch.optim.Adam(model.parameters(),lr=0.05)
    crit=nn.CrossEntropyLoss(weight=w)
    Xt=torch.tensor(X_tr_a,dtype=torch.float32); yt_t=torch.tensor(y_tr,dtype=torch.long)
    Xe=torch.tensor(X_te_a,dtype=torch.float32)
    model.train()
    for _ in range(50):
        opt.zero_grad(); loss=crit(model(Xt),yt_t); loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        probs=torch.softmax(model(Xe),dim=1)[:,1].cpu().numpy()
    vp=(probs>=0.35).astype(int)
    a,s,sc,f = stats(y_te, vp)
    vqc_accs.append(a); vqc_senss.append(s); vqc_specs.append(sc); vqc_f1s.append(f)

    print(f"  Fold {fold+1}/{N_FOLDS} | SVM Acc={svm_accs[-1]*100:.1f}% Sens={svm_senss[-1]*100:.1f}% | VQC Acc={vqc_accs[-1]*100:.1f}% Sens={vqc_senss[-1]*100:.1f}%")

summary = pd.DataFrame([
    {"Model": f"Majority-class baseline (always {'referable' if y_all.mean()>0.5 else 'non-referable'})",
     "N": len(y_all), "Qubits": "N/A",
     "Accuracy": f"{majority_acc:.1f}% +/- 0.0%",
     "Sensitivity": f"{'100.0' if y_all.mean()>0.5 else '0.0'}% +/- 0.0%",
     "Specificity": f"{'0.0' if y_all.mean()>0.5 else '100.0'}% +/- 0.0%",
     "F1": "--"},
    {"Model": "Classical SVM (RBF)", "N": len(y_all), "Qubits": N_QUBITS,
     "Accuracy": f"{np.mean(svm_accs)*100:.1f}% +/- {np.std(svm_accs)*100:.1f}%",
     "Sensitivity": f"{np.mean(svm_senss)*100:.1f}% +/- {np.std(svm_senss)*100:.1f}%",
     "Specificity": f"{np.mean(svm_specs)*100:.1f}% +/- {np.std(svm_specs)*100:.1f}%",
     "F1": f"{np.mean(svm_f1s)*100:.1f}% +/- {np.std(svm_f1s)*100:.1f}%"},
    {"Model": "Hybrid VQC (class-weighted, thresh=0.35, 50ep, lr=0.05)", "N": len(y_all), "Qubits": N_QUBITS,
     "Accuracy": f"{np.mean(vqc_accs)*100:.1f}% +/- {np.std(vqc_accs)*100:.1f}%",
     "Sensitivity": f"{np.mean(vqc_senss)*100:.1f}% +/- {np.std(vqc_senss)*100:.1f}%",
     "Specificity": f"{np.mean(vqc_specs)*100:.1f}% +/- {np.std(vqc_specs)*100:.1f}%",
     "F1": f"{np.mean(vqc_f1s)*100:.1f}% +/- {np.std(vqc_f1s)*100:.1f}%"},
])

csv_path = os.path.join(BASE_DIR, "aptos_results_summary.csv")
summary.to_csv(csv_path, index=False)
print(f"\n[APTOS BENCHMARK] Results:\n{summary.to_string(index=False)}")
print(f"\nSaved: {csv_path}")

# Plot
fig, ax = plt.subplots(figsize=(9,4), dpi=200)
models = ["Majority\nBaseline", "SVM (RBF)", "Hybrid VQC\n(tuned)"]
accs_m = [majority_acc, np.mean(svm_accs)*100, np.mean(vqc_accs)*100]
sens_m = [100.0 if y_all.mean()>0.5 else 0.0, np.mean(svm_senss)*100, np.mean(vqc_senss)*100]
x=np.arange(3); w=0.35
ax.bar(x-w/2, accs_m, w, label="Accuracy (%)", color="#ffb300", capsize=5)
ax.bar(x+w/2, sens_m, w, label="Referable Sensitivity (%)", color="#00f2fe", capsize=5)
ax.set_title(f"APTOS 2019 Val — 5-Fold CV, N={len(y_all)}, 4 Qubits", fontweight="bold")
ax.set_xticks(x); ax.set_xticklabels(models); ax.set_ylim(0,115)
ax.set_ylabel("Score (%)"); ax.legend(); ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, "aptos_results_summary.png"))
plt.close()
print(f"Saved: aptos_results_summary.png")
