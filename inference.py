"""
Diabetic Retinopathy - Inference Script
=========================================
Run this AFTER training. Paste into a new Kaggle cell (in the same notebook,
or a fresh one where you've uploaded best_model.pt as a Kaggle Dataset input).

WHAT TO EDIT BEFORE RUNNING:
1. MODEL_PATH  -> path to your downloaded/uploaded best_model.pt
2. IMAGE_PATH  -> path to the fundus image you want to test

This applies the EXACT same preprocessing used during training (crop black
border -> resize -> Ben Graham method), so predictions are reliable. Using
different preprocessing at inference time than training time is one of the
most common reasons a "working" model gives garbage predictions - don't
skip or alter these steps.
"""

import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F

try:
    import timm
except ImportError:
    os.system("pip install timm -q")
    import timm


# -----------------------------
# CONFIG - edit these two paths
# -----------------------------
MODEL_PATH = "/kaggle/working/best_model_fixed.pt"   # <-- change to your model's path
IMAGE_PATH = "/kaggle/input/competitions/aptos2019-blindness-detection/test_images/0aebb1b2aef1.png" # <-- change to your test image

IMG_SIZE = 380
MODEL_NAME = "efficientnet_b3"
NUM_CLASSES = 5
CLASS_NAMES = ["No DR", "Mild", "Moderate", "Severe", "Proliferative"]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -----------------------------
# PREPROCESSING (must match training exactly)
# -----------------------------
def crop_black_borders(img, tol=7):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    mask = gray > tol
    if mask.sum() == 0:
        return img
    coords = np.argwhere(mask)
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1
    return img[y0:y1, x0:x1]


def ben_graham_preprocess(img, sigma_x=10):
    blurred = cv2.GaussianBlur(img, (0, 0), sigma_x)
    img = cv2.addWeighted(img, 4, blurred, -4, 128)
    return img


def load_and_preprocess(path, img_size=IMG_SIZE):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Could not read image at: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = crop_black_borders(img)
    img = cv2.resize(img, (img_size, img_size))
    img = ben_graham_preprocess(img)
    return img


def to_tensor(img):
    """Normalize + convert to a model-ready tensor (no augmentation - inference only)."""
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = img.astype(np.float32) / 255.0
    img = (img - mean) / std
    img = img.transpose(2, 0, 1)  # HWC -> CHW
    tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0)  # add batch dim
    return tensor


# -----------------------------
# LOAD MODEL
# -----------------------------
def load_dr_model(model_path=MODEL_PATH):
    print(f"Loading model from {model_path} ...")
    model = timm.create_model(MODEL_NAME, pretrained=False, num_classes=NUM_CLASSES)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()
    print("Model loaded successfully.")
    return model


# -----------------------------
# RUN INFERENCE ON ONE IMAGE
# -----------------------------
def predict(image_path, model_path=MODEL_PATH):
    model = load_dr_model(model_path)
    img = load_and_preprocess(image_path)
    tensor = to_tensor(img).to(DEVICE)

    with torch.no_grad():
        outputs = model(tensor)
        probs = F.softmax(outputs, dim=1).cpu().numpy()[0]

    predicted_class = int(np.argmax(probs))
    confidence = float(probs[predicted_class])

    print(f"\nImage: {image_path}")
    print(f"Predicted: {CLASS_NAMES[predicted_class]} (confidence: {confidence*100:.1f}%)")
    print("\nFull class probabilities:")
    for name, p in zip(CLASS_NAMES, probs):
        print(f"  {name:15s}: {p*100:5.1f}%")

    return predicted_class, confidence, probs


if __name__ == "__main__":
    import sys
    img_target = sys.argv[1] if len(sys.argv) > 1 else IMAGE_PATH
    predict(img_target)
