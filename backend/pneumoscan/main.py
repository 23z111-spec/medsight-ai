"""
PneumoScan AI — FastAPI Backend
Run with:  uvicorn main:app --reload --port 8000
Docs at:   http://localhost:8000/docs
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from app.routes import predict, health, patients
from app.model_loader import load_models

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_models()
    yield

app = FastAPI(
    title="PneumoScan AI",
    description="Chest X-Ray screening API with EfficientNet-B0 + B3 Ensemble + Grad-CAM",
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

@app.get("/")
def root():
    return {"service": "PneumoScan AI", "status": "running", "version": "1.0.0"}