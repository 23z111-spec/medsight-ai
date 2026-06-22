"""
PneumoScan AI — FastAPI Backend
================================
Run with:  uvicorn main:app --reload --port 8000
Docs at:   http://localhost:8000/docs
"""
import os
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from app.routes import predict, health, patients, auth, scans, chat
from app.model_loader import load_models
from app.database import engine, Base
from app import models_db  # noqa: F401 — ensures User table is registered

print("MAIN FILE:", os.path.abspath(__file__))

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

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Root endpoint (returns basic project metadata)
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

# Register all routes precisely once
app.include_router(auth.router,     prefix="/auth",      tags=["Authentication"])
app.include_router(health.router,   prefix="/health",    tags=["Health"])
app.include_router(predict.router,  prefix="/predict",   tags=["Prediction"])
app.include_router(patients.router, prefix="/patients",  tags=["Patients"])
app.include_router(scans.router,    prefix="/scans",     tags=["Scans"])
app.include_router(chat.router,     prefix="/chat",      tags=["Chat"])
