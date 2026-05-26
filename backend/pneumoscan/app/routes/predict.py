"""
/predict  —  Core inference endpoint
=====================================
POST an X-ray image → get back structured findings + Grad-CAM heatmap.

Dashboard calls:
    fetch('http://localhost:8000/predict', {
        method: 'POST',
        body: formData   // FormData with 'file' field
    })
"""

import uuid
import random
import torch
import torch.nn.functional as F
from fastapi import APIRouter, UploadFile, File, HTTPException

from app.schemas import PredictionResponse, Finding
from app.model_loader import load_model, preprocess_image, LABELS, MOCK_MODE
from utils.gradcam import GradCAM, mock_gradcam

router = APIRouter()

# Load model once at startup (not per request)
_model = load_model()
_gradcam = GradCAM(_model) if _model is not None else None


# ── Severity thresholds ──
def _severity(conf: float) -> str:
    if conf >= 0.70: return "high"
    if conf >= 0.35: return "medium"
    return "low"


# ── Triage based on top confidence ──
def _triage(conf: float) -> str:
    if conf >= 0.75: return "HIGH"
    if conf >= 0.40: return "MEDIUM"
    return "LOW"


# ── Rough anatomical region mapping ──
REGION_MAP = {
    "Pneumonia":        "Right upper lobe",
    "Tuberculosis":     "Upper lobes bilateral",
    "Cardiomegaly":     "Cardiac silhouette",
    "Pleural Effusion": "Costophrenic angles",
    "Normal":           None,
}


@router.post("/", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    """
    Upload a chest X-ray image (JPEG / PNG / DICOM preview).
    Returns AI findings, confidence scores, triage level, and Grad-CAM heatmap.
    """

    # ── Validate file type ──
    if file.content_type not in ("image/jpeg", "image/png", "image/webp"):
        raise HTTPException(
            status_code=422,
            detail=f"Unsupported file type '{file.content_type}'. Upload JPEG or PNG."
        )

    image_bytes = await file.read()
    scan_id = f"SCN-{uuid.uuid4().hex[:8].upper()}"

    # ────────────────────────────────────────
    # MOCK MODE  — returns realistic dummy data
    # while your model is still training
    # ────────────────────────────────────────
    if MOCK_MODE or _model is None:
        findings = [
            Finding(label="Pneumonia",        confidence=0.913, severity="high",   region="Right upper lobe"),
            Finding(label="Pleural Effusion", confidence=0.347, severity="medium", region="Costophrenic angles"),
            Finding(label="Cardiomegaly",     confidence=0.082, severity="low",    region="Cardiac silhouette"),
            Finding(label="Tuberculosis",     confidence=0.051, severity="low",    region="Upper lobes bilateral"),
            Finding(label="Normal",           confidence=0.024, severity="low",    region=None),
        ]
        return PredictionResponse(
            prediction="Abnormal",
            primary_finding="Pneumonia",
            confidence=0.913,
            triage="HIGH",
            ece=0.032,
            findings=findings,
            gradcam_base64=mock_gradcam(),
            scan_id=scan_id,
            model_version="mock_v0",
            auc_roc=0.924,
            mock_mode=True,
        )

    # ────────────────────────────────────────
    # REAL INFERENCE  — runs once weights loaded
    # ────────────────────────────────────────
    try:
        tensor = preprocess_image(image_bytes)

        with torch.no_grad():
            logits = _model(tensor)
            probs  = F.softmax(logits, dim=1)[0]  # [NUM_CLASSES]

        # Build findings list sorted by confidence
        findings = []
        for i, label in enumerate(LABELS):
            conf = float(probs[i])
            findings.append(Finding(
                label=label,
                confidence=round(conf, 4),
                severity=_severity(conf),
                region=REGION_MAP.get(label),
            ))
        findings.sort(key=lambda f: f.confidence, reverse=True)

        top = findings[0]
        is_normal = top.label == "Normal" and top.confidence > 0.60

        # Grad-CAM on the top (non-normal) class
        cam_class = 0 if not is_normal else int(probs.argmax())
        gradcam_b64 = _gradcam.generate(tensor, cam_class)

        return PredictionResponse(
            prediction="Normal" if is_normal else "Abnormal",
            primary_finding=top.label,
            confidence=round(float(top.confidence), 4),
            triage="LOW" if is_normal else _triage(top.confidence),
            ece=0.032,   # replace with your calibration eval result
            findings=findings,
            gradcam_base64=gradcam_b64,
            scan_id=scan_id,
            model_version="efficientnet_b3_v1",
            auc_roc=0.924,   # replace with your actual val AUC
            mock_mode=False,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")
