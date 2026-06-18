"""
PneumoScan AI — FastAPI Backend
================================
Run with:  uvicorn main:app --reload --port 8000
Docs at:   http://localhost:8000/docs
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from app.routes import predict, health, patients, auth
from app.model_loader import load_models

from app.routes import chat
import os

print("MAIN FILE:", os.path.abspath(__file__))

from app.database import engine, Base
from app import models_db  # noqa: F401 — ensures User table is registered



@asynccontextmanager
async def lifespan(app: FastAPI):
    # Create DB tables if they don't exist
    Base.metadata.create_all(bind=engine)
    # Load ML models at startup
    load_models()
    yield


app = FastAPI(
    title="PneumoScan AI",
    description=(
        "Chest X-Ray screening API — EfficientNet-B0 + B3 Ensemble + Grad-CAM.\n\n"
        "**Clinical Disclaimer:** This tool is intended to assist, not replace, "
        "qualified medical professionals. All results must be reviewed by a licensed "
        "radiologist or physician. Not FDA-approved."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(health.router,   prefix="/health",   tags=["Health"])
app.include_router(predict.router,  prefix="/predict",  tags=["Prediction"])
app.include_router(patients.router, prefix="/patients", tags=["Patients"])
app.include_router(chat.router, prefix="/chat", tags=["Chat"])

@app.get("/")
def root():
    return {"service": "PneumoScan AI", "status": "running", "version": "1.0.0"}

app.include_router(auth.router,     prefix="/auth",      tags=["Authentication"])
app.include_router(health.router,   prefix="/health",    tags=["Health"])
app.include_router(predict.router,  prefix="/predict",   tags=["Prediction"])
app.include_router(patients.router, prefix="/patients",  tags=["Patients"])


@app.get("/")
def root():
    return {
        "service":     "PneumoScan AI",
        "status":      "running",
        "version":     "1.0.0",
        "ensemble":    "EfficientNet-B0 + B3",
        "mean_auc":    0.7968,
        "docs":        "/docs",
        "disclaimer":  "AI screening tool — not for clinical use without physician review."
    }

