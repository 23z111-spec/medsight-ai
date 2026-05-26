from fastapi import APIRouter
from app.schemas import HealthResponse
from app.model_loader import MOCK_MODE, DEVICE, MODEL_PATH
import os

router = APIRouter()

@router.get("/", response_model=HealthResponse)
def health_check():
    """
    Quick status check — call this from the dashboard on page load
    to confirm the API is up and the model is ready.
    """
    return HealthResponse(
        status="ok",
        model_loaded=not MOCK_MODE and os.path.exists(MODEL_PATH),
        mock_mode=MOCK_MODE,
        device=str(DEVICE),
        model_version="efficientnet_b3_v1",
    )
