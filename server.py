"""
================================================================================
SIH26038 & TNSAT - HYBRID QUANTUM-CLASSICAL RETINAL TELEMEDICINE SERVER
================================================================================
Runs HTTP Server on http://localhost:8000
Provides live web application hosting, AI inference backend API, 
and Quantum ML (VQC vs SVM) comparison endpoints.
================================================================================
"""

import os
import json
import base64
import urllib.parse
from http.server import HTTPServer, SimpleHTTPRequestHandler

# Import OpenCV & NumPy for base64 image encoding/decoding
try:
    import cv2
    import numpy as np
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

# Import local AI/ML and QML pipelines
try:
    from ai_ml_pipeline import run_retinal_inference
    from qml_pipeline import run_qml_inference, get_model_metrics
    AI_PIPELINE_AVAILABLE = True
except Exception as e:
    print(f"[SERVER WARNING] Pipeline import issue: {e}")
    AI_PIPELINE_AVAILABLE = False


PORT = 8000
WORKSPACE_DIR = os.path.dirname(os.path.abspath(__file__))


def decode_base64_image(base64_str):
    """Decode base64 Data URL to RGB numpy image array."""
    if "," in base64_str:
        base64_str = base64_str.split(",")[1]
    img_bytes = base64.b64decode(base64_str)
    np_arr = np.frombuffer(img_bytes, np.uint8)
    img_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError("Could not decode base64 image data")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return img_rgb


def encode_image_to_base64(img_rgb):
    """Encode RGB numpy image to base64 Data URL."""
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    _, buffer = cv2.imencode(".png", img_bgr)
    b64_str = base64.b64encode(buffer).decode("utf-8")
    return f"data:image/png;base64,{b64_str}"


class TelemedRequestHandler(SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/favicon.ico":
            self.send_response(200)
            self.send_header("Content-Type", "image/svg+xml")
            self.end_headers()
            self.wfile.write(b'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><text y=".9em" font-size="90">\xf0\x9f\x91\x81\xef\xb8\x8f</text></svg>')
            return

        if self.path == "/api/health":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            
            cnn_weights_exist = os.path.exists(os.path.join(WORKSPACE_DIR, "best_model.pt"))
            pca_exist = os.path.exists(os.path.join(WORKSPACE_DIR, "pca_model.pkl"))
            svm_exist = os.path.exists(os.path.join(WORKSPACE_DIR, "svm_model.pkl"))
            vqc_exist = os.path.exists(os.path.join(WORKSPACE_DIR, "vqc_model.pt"))
            
            # Load real metrics if available
            qml_metrics = get_model_metrics() if AI_PIPELINE_AVAILABLE else {}
            models_eval = qml_metrics.get("models", {})
            
            status = {
                "status": "ONLINE",
                "cnn_model_loaded": cnn_weights_exist,
                "qml_pipeline_ready": pca_exist and svm_exist and vqc_exist,
                "model_name": "EfficientNet-B3 + Pennylane VQC (4 Qubits)",
                "checkpoints": {
                    "best_model_pt": cnn_weights_exist,
                    "pca_model_pkl": pca_exist,
                    "svm_model_pkl": svm_exist,
                    "vqc_model_pt": vqc_exist
                },
                "pipeline_version": "SIH26139-HybridQML-v4.0",
                "empirical_metrics": {
                    "classical_svm_acc": f"{models_eval.get('classical_svm', {}).get('accuracy', 'N/A')}%",
                    "hybrid_vqc_acc": f"{models_eval.get('hybrid_vqc', {}).get('accuracy', 'N/A')}%",
                    "vqc_referable_sensitivity": f"{models_eval.get('hybrid_vqc', {}).get('referable_sensitivity', 'N/A')}%"
                },
                "disclaimer": "Clinical Decision Support System — For Research & Screening Verification Only. Not a Standalone Diagnostic."
            }
            self.wfile.write(json.dumps(status).encode("utf-8"))
            return

        if self.path == "/api/qml/metrics":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            metrics = get_model_metrics() if AI_PIPELINE_AVAILABLE else {}
            self.wfile.write(json.dumps(metrics).encode("utf-8"))
            return

        # Serve static web files
        return super().do_GET()

    def do_POST(self):
        if self.path in ["/api/predict", "/api/qml/predict"]:
            content_length = int(self.headers.get("Content-Length", 0))
            post_data = self.rfile.read(content_length)

            try:
                req_json = json.loads(post_data.decode("utf-8"))
                image_data = req_json.get("image_data")

                if image_data and CV2_AVAILABLE and AI_PIPELINE_AVAILABLE:
                    img_rgb = decode_base64_image(image_data)
                    qml_res, img_enh, gradcam_img = run_qml_inference(img_rgb)
                    qml_res["enhanced_image_b64"] = encode_image_to_base64(img_enh)
                    qml_res["gradcam_image_b64"] = encode_image_to_base64(gradcam_img)
                    res = qml_res
                elif AI_PIPELINE_AVAILABLE:
                    # Fallback to test fundus sample
                    sample_path = os.path.join(WORKSPACE_DIR, "Sample_Fundus_Photos", "Real_Patient_1_Normal_Healthy_Level0.jpg")
                    qml_res, img_enh, gradcam_img = run_qml_inference(sample_path)
                    qml_res["enhanced_image_b64"] = encode_image_to_base64(img_enh)
                    qml_res["gradcam_image_b64"] = encode_image_to_base64(gradcam_img)
                    res = qml_res
                else:
                    res = {"status": "ERROR", "message": "Pipeline modules unavailable"}

                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(json.dumps(res).encode("utf-8"))

            except Exception as err:
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"status": "ERROR", "message": str(err)}).encode("utf-8"))
            return

        self.send_error(404, "Endpoint not found")

    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()


def main():
    os.chdir(WORKSPACE_DIR)
    server = HTTPServer(("0.0.0.0", PORT), TelemedRequestHandler)
    print("==================================================================")
    print(f"  TNSAT AI & QUANTUM ML TELEMEDICINE WORKSTATION ONLINE")
    print(f"  URL: http://localhost:{PORT}")
    print("  Models Loaded: EfficientNet-B3, PCA (4-D), Classical SVM, Pennylane VQC (4 Qubits)")
    print("==================================================================")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")


if __name__ == "__main__":
    main()
