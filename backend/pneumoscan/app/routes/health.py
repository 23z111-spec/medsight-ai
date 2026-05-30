import torch
from fastapi import APIRouter
from app.model_loader import ModelHolder

router = APIRouter()

@router.get("/")
def health_check():
    return {
        "status":     "ok" if ModelHolder.loaded else "not_loaded",
        "b0_loaded":  ModelHolder.b0 is not None,
        "b3_loaded":  ModelHolder.b3 is not None,
        "mock_mode":  ModelHolder.mock,
        "device":     "cuda" if torch.cuda.is_available() else "cpu",
        "model_auc":  0.7749,
        "conditions": ["No Finding", "Pneumonia", "Cardiomegaly", "Effusion", "Infiltration"]
    }