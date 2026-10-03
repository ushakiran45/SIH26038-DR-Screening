"""
Fix 6: Generate VQC circuit diagram from the real QNode (qml_pipeline.py)
Saves: static/vqc_circuit_diagram.png
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import pennylane as qml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

N_QUBITS = 4
N_LAYERS = 2

q_device = qml.device("default.qubit", wires=N_QUBITS)

@qml.qnode(q_device, interface="torch")
def vqc_quantum_circuit(inputs, weights):
    # 1. Feature Encoding: Angle encoding via Ry(inputs[i]) on 4 qubits
    for i in range(N_QUBITS):
        qml.RY(inputs[i], wires=i)
    # 2. Parameterized Variational Layers & Entanglement
    num_layers = weights.shape[0]
    for l in range(num_layers):
        for i in range(N_QUBITS):
            qml.RY(weights[l, i, 0], wires=i)
            qml.RZ(weights[l, i, 1], wires=i)
        # Ring CNOT Entanglement
        qml.CNOT(wires=[0, 1])
        qml.CNOT(wires=[1, 2])
        qml.CNOT(wires=[2, 3])
        qml.CNOT(wires=[3, 0])
    return [qml.expval(qml.PauliZ(i)) for i in range(N_QUBITS)]

# Use representative inputs and weights for the diagram
sample_inputs = torch.tensor([0.5, -0.3, 0.8, -0.1], dtype=torch.float32)
sample_weights = torch.zeros(N_LAYERS, N_QUBITS, 2, dtype=torch.float32)

# Draw with qml.draw_mpl
fig, ax = qml.draw_mpl(vqc_quantum_circuit, decimals=None, style="pennylane")(
    sample_inputs, sample_weights
)

ax.set_title(
    "SIH26139 — 4-Qubit VQC: Angle Encoding + Ry/Rz Layers + Ring CNOT Entanglement\n"
    "Inputs: 4 PCA features  |  Layers: 2  |  Measurements: ⟨Z₀⟩…⟨Z₃⟩  |  Head: Linear(4→5 classes)",
    fontsize=9, pad=10
)

out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "static")
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "vqc_circuit_diagram.png")
fig.savefig(out_path, dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"[FIX-6] Circuit diagram saved to {out_path}")

print("[FIX-6] Done. PNG saved successfully (text draw skipped on Windows cp1252 console).")
