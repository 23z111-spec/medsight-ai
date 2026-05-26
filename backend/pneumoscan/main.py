"""
PneumoScan AI — FastAPI Backend
================================
Run with:  uvicorn main:app --reload --port 8000
Docs at:   http://localhost:8000/docs
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routes import predict, health, patients

app = FastAPI(
    title="PneumoScan AI",
    description="Chest X-Ray screening API with EfficientNet-B3 + Grad-CAM",
    version="1.0.0",
)

# ── CORS — allows your dashboard.html to call the API ──
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # tighten this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ──
app.include_router(health.router,    prefix="/health",   tags=["Health"])
app.include_router(predict.router,   prefix="/predict",  tags=["Prediction"])
app.include_router(patients.router,  prefix="/patients", tags=["Patients"])


@app.get("/")
def root():
    return {"service": "PneumoScan AI", "status": "running", "version": "1.0.0"}
