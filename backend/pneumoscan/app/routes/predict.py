import uuid
import datetime
import torch
from fastapi import APIRouter, UploadFile, File, HTTPException
from app.model_loader import preprocess, predict_ensemble, ModelHolder, TARGET_CONDITIONS, val_transforms
from utils.gradcam import generate_gradcam_b64

router = APIRouter()
scan_log = {}

@router.post("/")
async def predict(file: UploadFile = File(...)):
    if file.content_type not in ("image/jpeg", "image/png", "image/jpg", "image/webp"):
        raise HTTPException(status_code=422, detail=f"Unsupported file type: {file.content_type}")

    image_bytes = await file.read()
    if len(image_bytes) > 10 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="File too large. Max 10MB.")

    tensor = preprocess(image_bytes)
    result = predict_ensemble(tensor)

    # Grad-CAM
    gradcam_b64 = None
    gradcam_condition = None
    if not ModelHolder.mock and ModelHolder.b0 is not None:
        top_idx = TARGET_CONDITIONS.index(result["top_condition"])
        try:
            gradcam_b64 = generate_gradcam_b64(
                model=ModelHolder.b0,
                image_bytes=image_bytes,
                class_idx=top_idx,
                val_transforms=val_transforms,
            )
            gradcam_condition = result["top_condition"]
        except Exception as e:
            print(f"Grad-CAM failed: {e}")

    scan_id = f"SCN-{uuid.uuid4().hex[:8].upper()}"
    scan_log[scan_id] = {
        "scan_id":       scan_id,
        "timestamp":     datetime.datetime.utcnow().isoformat(),
        "triage":        result["triage"],
        "top_condition": result["top_condition"],
        "confidence":    result["confidence"],
    }

    return {
        "scan_id":          scan_id,
        "triage":           result["triage"],
        "top_condition":    result["top_condition"],
        "confidence":       result["confidence"],
        "findings":         result["findings"],
        "b0_findings":      result["b0_findings"],
        "b3_findings":      result["b3_findings"],
        "gradcam_base64":   gradcam_b64,
        "gradcam_condition":gradcam_condition,
        "mock_mode":        result["mock_mode"],
        "disclaimer":       "AI screening tool — review required by licensed physician.",
    }

@router.get("/log")
def get_scan_log():
    return {"total": len(scan_log), "scans": list(scan_log.values())}