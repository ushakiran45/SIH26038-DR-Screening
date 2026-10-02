# Hybrid Quantum Machine Learning Platform for Early Disease Detection
## Egreen Quanta | Ministry of Education's Innovation Cell (MIC)
**Problem Statement ID**: TNSAT | **Technology Bucket**: MedTech / BioTech / HealthTech | **Category**: Software
**Problem Creator**: Sarim Moin | **Organization**: Egreen Quanta

---

### 🌟 Problem Background & Description

#### Background
Early and accurate detection of diseases significantly improves treatment outcomes and reduces healthcare costs. Classical machine learning models have achieved notable success in medical diagnosis; however, they often face limitations when dealing with high-dimensional, noisy, and complex biomedical data (e.g., genomics, medical imaging, and electronic health records).

Quantum Machine Learning (QML) offers the potential to capture intricate patterns through quantum superposition and entanglement. Due to current hardware constraints, a hybrid quantum-classical approach provides a practical pathway to leverage quantum advantages while remaining executable on existing quantum simulators and near-term quantum devices.

#### Description
This project focuses on designing and developing a **hybrid quantum machine learning platform for early disease detection** (applied to Diabetic Retinopathy retinal image diagnosis). The platform integrates classical pre-processing and feature engineering with quantum-enhanced learning models (Variational Quantum Classifiers with 4 Qubits and Angle Encoding). 

The platform supports data ingestion, hybrid model training, prediction, explainability (Grad-CAM), and empirical performance evaluation against purely classical baselines (SVM).

---

### 🎯 Key Objectives

* **Architecture Design**: Design a hybrid quantum-classical machine learning architecture suitable for early disease detection.
* **Quantum Model Development**: Develop quantum-enhanced classification models processing high-dimensional biomedical image data (1,536D EfficientNet-B3 embeddings compressed to 4 PCA components).
* **Performance Improvement**: Improve detection accuracy, sensitivity, and specificity compared with classical ML baselines.
* **Hardware & Simulator Compatibility**: Ensure the platform is scalable, interpretable, and compatible with PennyLane quantum simulators (`default.qubit`) and near-term quantum hardware.
* **Module Integration**: Incorporate data pre-processing (Ben Graham contrast method), feature selection (PCA), and model explainability modules (Grad-CAM).
* **Empirical Benchmarking**: Benchmark the hybrid approach against classical models in terms of accuracy, computational efficiency, and generalization performance.

---

### 📦 Delivery Table (Expected Deliverables)

| Deliverable ID | Module / Requirement | Implementation & Technical Architecture | Code / Checkpoint Location |
| :--- | :--- | :--- | :--- |
| **DEL-01** | **Data Ingestion & Pre-processing** | Ben Graham Gaussian Contrast Normalization ($\sigma_x = 10$) + Tenengrad Sharpness & Illumination Quality Assessment | [ai_ml_pipeline.py](ai_ml_pipeline.py) |
| **DEL-02** | **CNN Feature Extraction & PCA Selection** | Pre-classifier EfficientNet-B3 1,536D embedding vector extraction + 4-Component PCA reduction scaled to $[-\pi, \pi]$ | [qml_pipeline.py](qml_pipeline.py) (`PCAFeatureReducer`) |
| **DEL-03** | **Quantum Feature Encoding** | Angle Encoding ($R_y(\theta_i)$ rotations) mapping 4 PCA features to 4 Qubits | [qml_pipeline.py](qml_pipeline.py) (`vqc_quantum_circuit`) |
| **DEL-04** | **Variational Quantum Classifier (VQC)** | PennyLane 4-Qubit Parameterized Quantum Circuit with 2 variational layers ($R_y, R_z$), Ring CNOT entanglement, & Pauli-Z expectation measurements | [qml_pipeline.py](qml_pipeline.py) (`HybridVQCClassifier`) |
| **DEL-05** | **Classical ML Baseline Model** | Support Vector Machine (`SVC` with RBF kernel & probability calibration) trained on identical 4 PCA features | [qml_pipeline.py](qml_pipeline.py) (`ClassicalSVMClassifier`) |
| **DEL-06** | **Hybrid Training & Prediction Workflow** | PyTorch + PennyLane hybrid Adam optimization & dual prediction inference pipeline | [server.py](server.py) (`/api/qml/predict`) |
| **DEL-07** | **Visual Explainability Module** | Grad-CAM Class Activation Maps highlighting retinal lesion visual attention, with explicit attribution notice for CNN feature representation | [index.html](index.html), [ai_ml_pipeline.py](ai_ml_pipeline.py) |
| **DEL-08** | **Empirical Benchmarking Dashboard** | Comparative evaluation of Accuracy, Precision, Recall, F1-Score, Training Time, Inference Latency, & 5x5 Confusion Matrices | [qml_metrics.json](qml_metrics.json), [index.html](index.html) |
| **DEL-09** | **Model Checkpoint Repository** | Saved weight files for CNN (`best_model.pt`), PCA (`pca_model.pkl`), Classical SVM (`svm_model.pkl`), and Quantum VQC (`vqc_model.pt`) | [best_model.pt](best_model.pt), `pca_model.pkl`, `svm_model.pkl`, `vqc_model.pt` |
| **DEL-10** | **Deployment & Containerization** | Live GitHub Pages web deployment, Dockerfile container configuration, Procfile, requirements.txt, & Render.yaml cloud blueprints | [Dockerfile](Dockerfile), [render.yaml](render.yaml), GitHub Pages |

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
├── matlab/
│   ├── main_pipeline.m         # Master MATLAB Driver Script
│   ├── dr_grading.m            # ICDR Grading Logic & Metrics
│   ├── gradcam_explainability.m # MATLAB Grad-CAM Module
│   ├── quality_assessment.m    # Tenengrad Sharpness & CLAHE
│   ├── retinal_segmentation.m  # Multi-Structure Retinal Segmentation
│   └── simulink_simulation.m   # Telemedicine Resource Capacity Model
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
Navigate to **Tab 7: Hybrid Quantum ML (TNSAT)** to interact with live dual predictions, quantum circuit visualizers, and performance dashboards.

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
https://ushakiran45.github.io/SIH26038-DR-Screening/
```

---

### 📊 Validation & Benchmark Comparisons

#### 5-Fold Stratified Cross-Validation Benchmark ($N = 22$ Seed Images)
| Model / Baseline | $N$ Samples | Qubits | Accuracy ($\text{Mean} \pm \text{Std}$) | Referable Sensitivity | Specificity | F1-Score | Inference Latency |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Majority Class Baseline** | 22 | N/A | **$68.2\% \pm 0.0\%$** | $100.0\%$ | $0.0\%$ | $81.1\%$ | $0.00 \text{ ms}$ |
| **Classical SVM (RBF)** | 22 | 4 | **$69.0\% \pm 7.3\%$** | $100.0\% \pm 0.0\%$ | $0.0\% \pm 0.0\%$ | $56.6\% \pm 9.4\%$ | $0.08 \text{ ms}$ |
| **Hybrid PennyLane VQC** | 22 | 4 | **$73.0\% \pm 6.8\%$** | $100.0\% \pm 0.0\%$ | $10.0\% \pm 20.0\%$ | $63.2\% \pm 10.6\%$ | $8.54 \text{ ms}$ |

> [!IMPORTANT]
> **Methodology & Limitation Notice**:
> Because evaluation was conducted on 22 labeled clinical sample images (68.2% Referable DR prevalence baseline), these results serve as a **QML software pipeline proof-of-concept demonstration** rather than a standalone clinical evaluation. The VQC and SVM perform at comparable baseline levels relative to majority-class prevalence (68.2%), and cross-dataset validation on larger external datasets (e.g. APTOS-2019 / IDRiD) is planned.

---

### 🏆 Team SIH Presentation Pitch Script (TNSAT - Egreen Quanta / MIC)
> *"For Problem Statement TNSAT (Egreen Quanta - MIC), we present a Hybrid Quantum Machine Learning Platform for Early Disease Detection. By compressing 1,536-dimensional EfficientNet-B3 CNN embeddings into 4 angle-encoded quantum features, we run a 4-qubit Variational Quantum Circuit (VQC) with Ring CNOT entanglement alongside a classical SVM baseline. Because we evaluated on 22 labeled sample images, these results represent a QML pipeline proof-of-concept demonstration, and we plan to validate on APTOS-2019 and IDRiD."*

