"""
PneumoScan AI — FastAPI Backend
================================
Run with:  uvicorn main:app --reload --port 8000
Docs at:   https://medsight-ai-production.up.railway.app/docs
"""
import os
from dotenv import load_dotenv
load_dotenv()

from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from app.routes import predict, health, patients, auth, scans, chat
from app.model_loader import load_models
from app.database import engine, Base
from app import models_db
from app.routes.auth_password_reset import router as password_router
from pathlib import Path

print("MAIN FILE:", os.path.abspath(__file__))
FRONTEND_DIR = Path(__file__).resolve().parent.parent.parent / "frontend"


@asynccontextmanager
async def lifespan(app: FastAPI):
    Base.metadata.create_all(bind=engine)
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
    allow_origins=["https://medsight-ai-production.up.railway.app","http://127.0.0.1:5500",],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return RedirectResponse(url="/login.html")

app.include_router(auth.router,     prefix="/auth",     tags=["Authentication"])
app.include_router(health.router,   prefix="/health",   tags=["Health"])
app.include_router(predict.router,  prefix="/predict",  tags=["Prediction"])
app.include_router(patients.router, prefix="/patients", tags=["Patients"])
app.include_router(scans.router,    prefix="/scans",    tags=["Scans"])
app.include_router(chat.router,     prefix="/chat",     tags=["Chat"])
app.include_router(password_router)

# Must be LAST

import os

_base = os.path.dirname(os.path.abspath(__file__))
_frontend_candidates = [
    os.path.join(_base, "../../frontend"),
    os.path.join(_base, "../frontend"),
    os.path.join(_base, "frontend"),
    "/app/frontend",
]
_frontend_path = next((p for p in _frontend_candidates if os.path.exists(p)), None)

if _frontend_path:
    print(f"Serving frontend from: {_frontend_path}")
    app.mount("/", StaticFiles(directory=_frontend_path, html=True), name="static")
else:
    print("WARNING: frontend folder not found — API-only mode")