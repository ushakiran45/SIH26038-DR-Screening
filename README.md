# Hybrid Quantum Machine Learning Platform for Early Disease Detection
## Egreen Quanta | Ministry of Education's Innovation Cell (MIC)
**Problem Statement ID**: SIH26139 | **Technology Bucket**: MedTech / BioTech / HealthTech | **Category**: Software
**Problem Creator**: Sarim Moin | **Organization**: Egreen Quanta

> [!NOTE]
> **Problem Statement Alignment**:
> This repository implements official **Problem Statement SIH26139 (Egreen Quanta - MIC)** titled *"Hybrid Quantum Machine Learning Platform for Early Disease Detection"*. All technical documentation, benchmarks, models, and web deployment components correspond to SIH26139.

---

### 🌟 Problem Background & Description

#### Background
Early and accurate detection of diseases significantly improves treatment outcomes and reduces healthcare costs. Classical machine learning models have achieved notable success in medical diagnosis; however, they often face limitations when dealing with high-dimensional, noisy, and complex biomedical data (e.g., genomics, medical imaging, and electronic health records).

Quantum Machine Learning (QML) may capture patterns through superposition and entanglement on some tasks. Due to current hardware and sample-size constraints, this repository uses a hybrid quantum-classical approach that runs on a PennyLane simulator. **No quantum advantage is claimed.**

#### Description
This project focuses on designing and developing a **hybrid quantum machine learning platform for early disease detection** (applied to Diabetic Retinopathy retinal image diagnosis). The platform integrates classical pre-processing and feature engineering with quantum-enhanced learning models (Variational Quantum Classifiers with 4 Qubits and Angle Encoding). 

The platform supports data ingestion, hybrid model training, prediction, explainability (Grad-CAM), and empirical performance evaluation against purely classical baselines (SVM).

---

### 🎯 Key Objectives

* **Architecture Design**: Design a hybrid quantum-classical machine learning architecture suitable for early disease detection.
* **Quantum Model Development**: Develop quantum-enhanced classification models processing high-dimensional biomedical image data (1,536D EfficientNet-B3 embeddings compressed to 4 PCA components).
* **Performance Comparison**: Compare detection accuracy, sensitivity, and specificity against a classical SVM baseline on the same features (small-sample proof-of-concept).
* **Hardware & Simulator Compatibility**: Ensure the platform is scalable, interpretable, and compatible with PennyLane quantum simulators (`default.qubit`) and near-term quantum hardware.
* **Module Integration**: Incorporate data pre-processing (Ben Graham contrast method), feature selection (PCA), and model explainability modules (Grad-CAM).
* **Empirical Benchmarking**: Benchmark the hybrid approach against classical models in terms of accuracy, computational efficiency, and generalization performance.

---

### 📦 Delivery Table (Expected Deliverables)

| Deliverable ID | Module / Requirement | Implementation & Technical Architecture | Code / Checkpoint Location |
| :--- | :--- | :--- | :--- |
| **DEL-01** | **Data Ingestion & Pre-processing** | Ben Graham Gaussian Contrast Normalization ($\sigma_x = 10$) + Tenengrad Sharpness & Quality Assessment | [ai_ml_pipeline.py](ai_ml_pipeline.py) |
| **DEL-02** | **CNN Feature Extraction & PCA Selection** | Pre-classifier EfficientNet-B3 1,536D embedding vector extraction + 4-Component PCA reduction scaled to $[-\pi, \pi]$ | [qml_pipeline.py](qml_pipeline.py) (`PCAFeatureReducer`) |
| **DEL-03** | **Quantum Feature Encoding** | Angle Encoding ($R_y(\theta_i)$ rotations) mapping 4 PCA features to 4 Qubits | [qml_pipeline.py](qml_pipeline.py) (`vqc_quantum_circuit`) |
| **DEL-04** | **Variational Quantum Classifier (VQC)** | PennyLane 4-Qubit Parameterized Quantum Circuit with 2 variational layers ($R_y, R_z$), Ring CNOT entanglement, & Pauli-Z expectation measurements | [qml_pipeline.py](qml_pipeline.py) (`HybridVQCClassifier`) |
| **DEL-05** | **Classical ML Baseline Model** | Support Vector Machine (`SVC` with RBF kernel) trained on identical 4 PCA features | [qml_pipeline.py](qml_pipeline.py) (`ClassicalSVMClassifier`) |
| **DEL-06** | **Hybrid Training & Prediction Workflow** | PyTorch + PennyLane hybrid Adam optimization & dual prediction inference pipeline | [server.py](server.py) (`/api/qml/predict`) |
| **DEL-07** | **Visual Explainability Module** | Grad-CAM Class Activation Maps highlighting retinal lesion visual attention, with explicit attribution notice for CNN feature representation | [index.html](index.html), [ai_ml_pipeline.py](ai_ml_pipeline.py) |
| **DEL-08** | **CNN + QML Benchmark (APTOS 2019, N=400)** | CNN: 88.9% referable sensitivity, 91.6% specificity, AUC 0.963. Hybrid heads on the same features (5-fold): SVM $90.0\% \pm 1.8\%$, VQC $88.0\% \pm 3.8\%$ vs 59.5% majority baseline. VQC does not beat SVM. | [cnn_eval_metrics.json](cnn_eval_metrics.json), [benchmarks/aptos_results_summary.csv](benchmarks/aptos_results_summary.csv) |
| **DEL-09** | **Model Checkpoint Repository** | Saved weight files for CNN (`best_model.pt`), PCA (`pca_model.pkl`), Classical SVM (`svm_model.pkl`), and Quantum VQC (`vqc_model.pt`) | [best_model.pt](best_model.pt), `pca_model.pkl`, `svm_model.pkl`, `vqc_model.pt` |
| **DEL-10** | **Deployment & Containerization** | Live GitHub Pages web deployment, Dockerfile container configuration, Procfile, requirements.txt, & Render.yaml cloud blueprints | [Dockerfile](Dockerfile), [render.yaml](render.yaml), GitHub Pages |
| **DEL-11** | **Second-Dataset Benchmark (Breast Cancer, N=200)** | Dataset-agnostic benchmark module run on UCI WDBC (scikit-learn); 5-fold CV, 4 qubits; SVM 93.0% ± 3.3%, VQC 75.5% ± 5.8%, majority baseline 64.5%. VQC does not beat classical SVM (gap > std — honest result). Qubit sweeps in benchmarks/. | [benchmarks/](benchmarks/) |
| **FUTURE-01** | **Cross-Dataset Test (IDRiD / Messidor-2)** | APTOS 2019 CNN + hybrid heads ($N=400$) are done. Remaining: other public DR datasets | **Future Work (Planned)** |
| **FUTURE-02** | **Physical Quantum Hardware Run** | Planned execution on physical quantum processors (e.g. IBM Quantum via `pennylane-qiskit`); hardware gate noise and NISQ queue latency not yet measured | **Future Work (Planned)** |
| **FUTURE-03** | **Qubit Scaling (6-Qubit / 8-Qubit VQC)** | Evaluate deeper entanglement architectures at higher qubit counts against larger feature spaces ($N=50, 100, 200$ samples) | [benchmarks/](benchmarks/) |
| **FUTURE-04** | **Circuit-Level Explainability** | Quantum SHAP / circuit gradient attribution for VQC gate parameters (distinct from CNN Grad-CAM spatial attention) | **Future Work (Planned)** |

---

### 📐 End-to-End Pipeline Architecture

```text
                          Retinal Image
                                ↓
                       Image Quality Check
                                ↓
                    Preprocessing (Ben Graham)
                                ↓
               CNN Feature Extraction (EfficientNet-B3)
                                ↓
                      PCA (4 Features)
                                ↓
            ┌───────────────────┴───────────────────┐
            ↓                                       ↓
     Classical SVM                             4-Qubit VQC
            ↓                                       ↓
   Classical Prediction                     Hybrid QML Prediction
            └───────────────────┬───────────────────┘
                                ↓
                      Performance Comparison
                                ↓
             Explainable Grad-CAM Diagnostic Report
```

---

### 📁 Project Structure

```text
SIH/
├── index.html                  # Telemedicine & QML Dashboard Web Frontend
├── css/
│   └── styles.css              # Dark Mode Glassmorphism & QML Design System
├── js/
│   ├── app.js                  # Main Application Orchestrator & QML UI Renderer
│   ├── benchmark_data.js       # Published Benchmark Datasets
│   ├── fundus_engine.js        # Retinal Canvas Rendering & Custom Image Upload
│   ├── grading_explainability.js # DR Severity Grading & Grad-CAM PDF Exporter
│   ├── quality_analyzer.js     # Quality Assessment & Recapture Feedback
│   └── simulink_simulator.js   # Telemedicine Resource Capacity Simulator
├── best_model.pt               # PyTorch EfficientNet-B3 Weights Checkpoint
├── pca_model.pkl               # Saved 4-Component PCA Model
├── svm_model.pkl               # Saved Classical SVM Model Checkpoint
├── vqc_model.pt                # Saved 4-Qubit VQC Parameters Checkpoint
├── qml_metrics.json            # Empirical QML vs Classical Benchmark Metrics
├── ai_ml_pipeline.py           # PyTorch CNN Feature Extraction & Grad-CAM
├── qml_pipeline.py             # Quantum ML Pipeline & PennyLane VQC Engine
├── server.py                   # Local HTTP API Server (`python server.py`)
├── requirements.txt            # Python Dependencies for Cloud Deployment
├── Dockerfile                  # Container Deployment Config
├── Procfile                    # Web Server Execution Config
└── render.yaml                 # Render.com Cloud Service Deployment Blueprint
```

---

### 🚀 How to Run & Deploy

#### Option 1: Web Application & Local API Server (Recommended for Presentation)
Start the local Python server:
```bash
python server.py
```
Open your web browser and navigate to:
```text
http://localhost:8000
```
Navigate to **Tab 7: Hybrid Quantum ML (SIH26139)** to interact with live dual predictions, quantum circuit visualizers, and performance dashboards.

#### Option 2: Train & Evaluate QML Models via CLI
Execute the master Quantum ML pipeline script directly:
```bash
python qml_pipeline.py
```

#### Option 3: Docker Container Deployment
Build and run the containerized workstation:
```bash
docker build -t sih26139-qml .
docker run -p 8000:8000 sih26139-qml
```

#### Option 4: GitHub Pages Live Web Deployment
The static web interface is deployed live at:
```text
https://ushakiran45.github.io/SIH26139-Hybrid-QML/
```

> [!NOTE]
> **Circuit Architecture Note**: All system components (`qml_pipeline.py`, `benchmark_qml.py`, `aptos_qml_benchmark.py`) utilize the unified 4-qubit $R_y/R_z$ ring-CNOT PennyLane architecture (`vqc_quantum_circuit`). The generated circuit diagram ([static/vqc_circuit_diagram.png](static/vqc_circuit_diagram.png)) is generated directly from this live QNode.



### 📊 Validation & Benchmark Comparisons

#### 5-Fold Stratified Cross-Validation Benchmark ($N = 22$ seed set)
| Model / Baseline | $N$ Samples | Qubits | Accuracy ($\text{Mean} \pm \text{Std}$) | Referable Sensitivity | Specificity | F1-Score | Inference Latency |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Majority-class baseline (always referable)** | 22 | N/A | **$68.2\%$** | $100\%$ | N/A (always-positive rule) | $81.1\%$ | — |
| **Classical SVM (RBF)** | 22 | 4 | **$76.0\% \pm 15.9\%$** | $93.3\% \pm 13.3\%$ | $30.0\% \pm 40.0\%$ | $71.3\% \pm 16.9\%$ | $0.07 \text{ ms}$ |
| **Hybrid PennyLane VQC** | 22 | 4 | **$67.0\% \pm 24.6\%$** | $86.7\% \pm 26.7\%$ | $20.0\% \pm 40.0\%$ | $60.7\% \pm 23.5\%$ | $7.38 \text{ ms}$ |

#### CNN Backbone Evaluation — APTOS 2019 Validation Split (N = 400)

| Component | $N$ | Majority Baseline | 5-class Acc | ROC-AUC | Referable Sensitivity | Referable Specificity |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **CNN backbone** (EfficientNet-B3, 1536-D features) | **400** | 59.5% | **75.5%** | **0.963** | **88.9%** | **91.6%** |

> **Dataset & Tuning Disclaimer**: APTOS 2019 Blindness Detection (Kaggle), 400-image sample split (seed=42). Preprocessing: crop black borders → resize 380×380 → ImageNet normalisation. *Note on backbone split*: Because `best_model.pt` is a pre-trained checkpoint, potential overlap between this 400-image sample and the original CNN pre-training data cannot be independently verified. *Note on VQC tuning*: The VQC decision threshold (0.35) and class weighting were hyperparameter-tuned on this sample, providing resubstitution/optimistic performance bounds.

#### Hybrid SVM vs VQC on the same APTOS 2019 features (N = 400, 5-fold CV, 4 qubits)

PCA to 4 components, angle encoding, class-weighted VQC (50 epochs, lr=0.05, decision threshold 0.35 tuned on dataset). Majority class is non-referable (238 / 400 = 59.5%). **SVM is slightly stronger and more stable. No quantum advantage is claimed, and VQC does not beat SVM.**

| Model / Baseline | $N$ | Qubits | Accuracy ($\text{Mean} \pm \text{Std}$) | Referable Sensitivity | Specificity | F1-Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Majority-class baseline (always non-referable)** | 400 | N/A | **59.5%** | N/A (always-negative rule) | 100% | — |
| **Classical SVM (RBF)** | 400 | 4 | **$90.0\% \pm 1.8\%$** | $93.8\% \pm 2.8\%$ | $87.4\% \pm 3.0\%$ | $90.1\% \pm 1.7\%$ |
| **Hybrid VQC (class-weighted)** | 400 | 4 | **$88.0\% \pm 3.8\%$** | $96.3\% \pm 1.2\%$ | $82.3\% \pm 6.1\%$ | $88.1\% \pm 3.8\%$ |

Source: [benchmarks/aptos_results_summary.csv](benchmarks/aptos_results_summary.csv).

#### Second-Dataset Benchmark — Breast Cancer Wisconsin (UCI WDBC, scikit-learn), N = 200, 5-Fold CV, 4 Qubits

> **Important**: The web app is a retinal DR tool. This benchmark demonstrates that the benchmark *module* (`benchmark_qml.py`) is dataset-agnostic — it accepts any feature matrix and label array. The retinal app is the case study; this run provides a statistically meaningful comparison on a well-known public dataset.

| Model / Baseline | $N$ Samples | Qubits | Accuracy ($\text{Mean} \pm \text{Std}$) | Malignant Sensitivity | Benign Specificity | F1-Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Majority-class baseline (always benign)** | 200 | N/A | **64.5%** *(sample is 129 benign / 71 malignant — not balanced)* | N/A (always-benign rule) | 100.0% | 78.4% |
| **Classical SVM (RBF)** | 200 | 4 | **$93.0\% \pm 3.3\%$** | $83.0\% \pm 7.4\%$ | $98.5\% \pm 1.9\%$ | $92.8\% \pm 3.4\%$ |
| **Hybrid PennyLane VQC** | 200 | 4 | **$75.5\% \pm 5.8\%$** | $37.0\% \pm 18.5\%$ | $96.9\% \pm 2.9\%$ | $71.3\% \pm 8.8\%$ |

**Interpretation**: The random sample (seed=42) of 200 from the 569-sample dataset preserves the natural 62.7% benign prevalence (129/200 benign, 71/200 malignant), so the majority baseline is "always predict benign" at 64.5% — not 50%. Samples drawn at natural class prevalence (seed 42); `benchmark_qml.py` uses `np.random.choice` without stratification, so the 200 rows reflect the UCI dataset's natural 62.7% / 37.3% benign/malignant split. The VQC does **not** beat the classical SVM (93.0% vs 75.5%; gap = 17.5 pp, well above both stds). The VQC's malignant sensitivity is low (37.0% ± 18.5%) — it predicts benign too often. The SVM is both more accurate and more stable across folds. This is an honest, expected result. The benchmark demonstrates the comparative framework, not a quantum advantage claim. The benchmark *module* (`benchmark_qml.py`) is dataset-agnostic; the web app supports retinal images only. Qubit and sample sweeps (N=100, 50; 6-qubit): see [benchmarks/](benchmarks/).

**Dataset citation**: Breast Cancer Wisconsin (Diagnostic) Data Set, UCI Machine Learning Repository. Available via `sklearn.datasets.load_breast_cancer`.

> [!IMPORTANT]
> **Methodology & Validation Notice**:
> The CNN backbone (EfficientNet-B3) achieves **88.9% referable sensitivity, 91.6% specificity, and ROC-AUC 0.963** on a 400-image stratified held-out split from the APTOS 2019 public dataset (Kaggle). On the same 1536-D features compressed to 4 PCA components, 5-fold CV gives Classical SVM **90.0% ± 1.8%** and Hybrid VQC **88.0% ± 3.8%** (majority baseline 59.5%). SVM is slightly stronger; **no quantum advantage is claimed, and VQC does not beat SVM.** The 22-image set is retained only as a pipeline smoke-test. **This system is a decision-support research tool, not a diagnostic device.**



---

### 🔮 Future Work & Development Roadmap

1. **Cross-dataset generalization**: APTOS 2019 CNN and hybrid heads ($N=400$) are complete. Remaining work is IDRiD and Messidor-2.
2. **Physical Quantum Hardware Execution**: Deploy Variational Quantum Circuits onto physical quantum processors (e.g. IBM Quantum free tier via `pennylane-qiskit`) to measure hardware gate noise and NISQ queue latency.
3. **Qubit & Feature Scaling Sweeps**: Evaluate 6-Qubit and 8-Qubit VQC architectures with deeper entanglement layers against high-dimensional feature spaces ($N=50, 100, 200$ samples).
4. **Circuit-Level Explainability**: Develop quantum-native attribution (quantum SHAP / circuit gradient) for VQC gate parameters, distinct from CNN Grad-CAM spatial attention which explains CNN feature focus, not quantum gate decisions.



---

### 🏆 Presentation Pitch & Q&A Defense Guide (SIH26139 | Egreen Quanta - MIC)

> **Pitch Statement**:
> *"For Problem Statement SIH26139 (Egreen Quanta - MIC), we developed a Hybrid Quantum Machine Learning Platform for Early Disease Detection. EfficientNet-B3 evaluated on 400 held-out APTOS 2019 images: 88.9% referable sensitivity, 91.6% specificity, AUC 0.963. The VQC + SVM demo compresses the same 1,536-D features to 4 PCA components and angle-encodes them on 4 qubits. This is a working proof-of-concept hybrid pipeline. Larger QML benchmark, hardware run and generalization testing are future work. Decision support, not diagnosis. No quantum advantage is claimed."*

#### Key Defense Questions & Honest Answers
* **Q1: Does the VQC show a quantum advantage over classical SVM?**
  * *Answer*: "Problem Statement SIH26139 asks for a hybrid platform to evaluate where quantum ML can be applied. We built the pipeline to benchmark a VQC and an SVM on identical features. On the 22-image smoke-test, Classical SVM ($76.0\% \pm 15.9\%$) and Hybrid VQC ($67.0\% \pm 24.6\%$) are near the majority baseline ($68.2\%$); the difference is noise. On breast cancer ($N=200$) the SVM is clearly stronger (93.0% vs 75.5%). No quantum advantage is claimed, and we do not say VQC beats SVM."
* **Q2: Has the pipeline been run on physical quantum hardware?**
  * *Answer*: "The codebase is hardware-ready through PennyLane's device interface (`qml.device`), but we have executed our benchmarks on the `default.qubit` simulator. Running on physical hardware (e.g. via `pennylane-qiskit` on IBM Quantum) introduces hardware gate noise and queue latency, which is planned for future work."
* **Q3: How does the live web demo work on GitHub Pages vs local server?**
  * *Answer*: "The interactive web workstation runs statically on GitHub Pages using pre-computed checkpoint outputs. When executing `python server.py` locally on port 8000, the frontend dynamically connects to live REST endpoints (`/api/qml/predict` and `/api/health`)."


