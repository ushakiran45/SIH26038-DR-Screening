# Hybrid Quantum Machine Learning Platform for Early Disease Detection
## Egreen Quanta Problem Statement SIH26139 | Smart India Hackathon
**Technology Bucket**: MedTech / BioTech / HealthTech | **Category**: Software

---

### 🌟 Project Overview
An integrated, explainable **Hybrid Quantum Machine Learning (QML) Retinal Diagnostic Platform** engineered for early disease detection (Diabetic Retinopathy screening) in rural Primary Health Centers (PHCs) in India. 

The platform bridges classical Deep Learning (PyTorch EfficientNet-B3) with Quantum Variational Circuits (PennyLane 4-Qubit VQC) to perform 5-class severity grading, empirical model benchmark comparison, and Grad-CAM feature attribution.

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

### 🎯 Key Technical & Quantum ML Features

1. **Adaptive Image Quality Assessment (IQA)**:
   - Tenengrad gradient focus sharpness metric ($>3.50$ threshold).
   - Mean illumination uniformity check ($15\% - 85\%$ optimal range).
   - Real-time operator recapture feedback for blurry or sub-optimal fundus acquisitions.

2. **Pre-Classification CNN Feature Extraction (`best_model.pt`)**:
   - Uses pre-trained **EfficientNet-B3** backbone.
   - Extracts a high-dimensional **1,536-feature embedding vector** for every retinal image prior to the classification head.

3. **PCA Dimensionality Reduction & Angle Encoding**:
   - Reduces 1,536-dimensional CNN vectors down to **4 principal components**.
   - Normalizes feature values to $[-\pi, \pi]$ for quantum rotation angle encoding ($[\theta_0, \theta_1, \theta_2, \theta_3]$).
   - Saved checkpoint: `pca_model.pkl`.

4. **Classical ML Baseline (Support Vector Machine)**:
   - RBF-kernel SVM classifier trained on the 4 PCA features across 5 DR classes.
   - Saved checkpoint: `svm_model.pkl`.

5. **Variational Quantum Circuit (VQC)**:
   - Built with **PennyLane 0.45** and **PyTorch autograd**.
   - **4 Qubits** ($q_0, q_1, q_2, q_3$) using Angle Encoding ($R_y(\theta_i)$).
   - **2 Variational Layers** of parameterized single-qubit rotations ($R_y(w), R_z(w)$).
   - **Ring CNOT Entanglement** topology ($CNOT(0,1), CNOT(1,2), CNOT(2,3), CNOT(3,0)$).
   - **Measurement**: Pauli-Z expectation values $\langle Z_0, Z_1, Z_2, Z_3 \rangle$.
   - **Hybrid Classification Head**: Classical linear mapping to 5 DR classes + Softmax probability distribution.
   - Saved checkpoint: `vqc_model.pt`.

6. **Grad-CAM Visual Explainability**:
   - Class Activation Map heatmaps generated from the EfficientNet-B3 feature extraction layer.
   - *Attribution Disclaimer*: Grad-CAM visualizes spatial attention from the CNN feature extractor to highlight lesion regions. It explains visual feature representations rather than quantum gate parameters directly.

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

### 🚀 How to Run & Deploy the Project

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
https://ushakiran45.github.io/SIH26038-DR-Screening/
```

---

### 📊 Validation & Benchmark Comparisons

#### Classical SVM vs Hybrid PennyLane VQC (4 Qubits)
| Evaluation Metric | Classical SVM (Baseline) | Hybrid VQC (Quantum ML) |
| :--- | :---: | :---: |
| **Accuracy** | **87.50%** | **87.50%** |
| **Precision** | **91.02%** | **91.02%** |
| **Recall** | **87.50%** | **87.50%** |
| **F1-Score** | **87.66%** | **87.66%** |
| **Training Time** | `0.008 sec` | `19.32 sec` |
| **Inference Latency** | `0.23 ms` | `6.73 ms` |
| **Feature Representation** | `4 PCA Components` | `4 Qubits (Angle Encoded)` |

---

### 🏆 Team SIH Presentation Pitch Script (SIH26139 - Egreen Quanta)
> *"For Problem Statement SIH26139, Egreen Quanta presents a Hybrid Quantum Machine Learning Platform for Early Disease Detection. By compressing 1,536-dimensional EfficientNet-B3 CNN embeddings into 4 angle-encoded quantum features, we run a 4-qubit Variational Quantum Circuit (VQC) with Ring CNOT entanglement alongside a classical SVM baseline. Combined with Grad-CAM feature attribution and quality assessment, our platform provides clinicians with instant dual predictions, quantum state visualizations, and explainable diagnostic reports under 30 seconds."*
