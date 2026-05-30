import os
import io
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import logging

logger = logging.getLogger(__name__)

B0_PATH   = "models/best_model.pth"
B3_PATH   = "models/best_model_b3.pth"
MOCK_MODE = False
DEVICE    = torch.device("cpu")

TARGET_CONDITIONS = ['No Finding', 'Pneumonia', 'Cardiomegaly', 'Effusion', 'Infiltration']
HIGH_THRESHOLD    = 0.6
MEDIUM_THRESHOLD  = 0.3

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

class ModelHolder:
    b0     = None
    b3     = None
    loaded = False
    mock   = False

def _build_b0():
    m = models.efficientnet_b0(weights=None)
    m.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(m.classifier[1].in_features, 5)
    )
    return m

def _build_b3():
    m = models.efficientnet_b3(weights=None)
    m.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(m.classifier[1].in_features, 5)
    )
    return m

def load_models():
    if ModelHolder.loaded:
        return
    b0 = _build_b0()
    b3 = _build_b3()
    if MOCK_MODE:
        ModelHolder.mock = True
        logger.warning("MOCK_MODE is ON — returning dummy predictions.")
    else:
        try:
            b0.load_state_dict(torch.load(B0_PATH, map_location=DEVICE))
            b3.load_state_dict(torch.load(B3_PATH, map_location=DEVICE))
            logger.info("✅ Models loaded successfully")
        except FileNotFoundError as e:
            logger.warning(f"Model file not found: {e} — switching to mock mode")
            ModelHolder.mock = True
    b0.eval()
    b3.eval()
    ModelHolder.b0 = b0
    ModelHolder.b3 = b3
    ModelHolder.loaded = True

def preprocess(image_bytes: bytes) -> torch.Tensor:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return TRANSFORM(img).unsqueeze(0)

def predict_ensemble(tensor: torch.Tensor) -> dict:
    import numpy as np
    with torch.no_grad():
        if ModelHolder.mock:
            torch.manual_seed(42)
            probs_b0 = torch.sigmoid(torch.randn(1, 5)).squeeze().numpy()
            probs_b3 = torch.sigmoid(torch.randn(1, 5)).squeeze().numpy()
        else:
            probs_b0 = torch.sigmoid(ModelHolder.b0(tensor)).squeeze().numpy()
            probs_b3 = torch.sigmoid(ModelHolder.b3(tensor)).squeeze().numpy()

    probs = (probs_b0 + probs_b3) / 2.0
    abnormal = probs[1:]
    max_abn  = float(abnormal.max())
    top_idx  = int(abnormal.argmax()) + 1

    if max_abn >= HIGH_THRESHOLD:
        triage = "HIGH"
    elif max_abn >= MEDIUM_THRESHOLD:
        triage = "MEDIUM"
    else:
        triage = "LOW"

    return {
        "findings":      {c: round(float(p), 4) for c, p in zip(TARGET_CONDITIONS, probs)},
        "triage":        triage,
        "top_condition": TARGET_CONDITIONS[top_idx],
        "confidence":    round(float(probs[top_idx]), 4),
        "b0_findings":   {c: round(float(p), 4) for c, p in zip(TARGET_CONDITIONS, probs_b0)},
        "b3_findings":   {c: round(float(p), 4) for c, p in zip(TARGET_CONDITIONS, probs_b3)},
        "mock_mode":     ModelHolder.mock,
    }

# expose for compatibility
val_transforms = TRANSFORM