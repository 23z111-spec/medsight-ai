"""
Pydantic schemas
================
Defines the shape of every API request and response.
The dashboard reads these JSON structures directly.
"""

from pydantic import BaseModel
from typing import Optional


class Finding(BaseModel):
    label: str
    confidence: float       # 0.0 – 1.0
    severity: str           # "high" | "medium" | "low"
    region: Optional[str]   # e.g. "Right upper lobe"


class PredictionResponse(BaseModel):
    # ── Core result ──
    prediction: str                 # "Abnormal" | "Normal"
    primary_finding: str            # top detected condition
    confidence: float               # model confidence 0–1
    triage: str                     # "HIGH" | "MEDIUM" | "LOW"
    ece: float                      # expected calibration error

    # ── All findings ──
    findings: list[Finding]

    # ── Explainability ──
    gradcam_base64: str             # heatmap PNG as base64

    # ── Scan metadata ──
    scan_id: str
    model_version: str
    auc_roc: float
    mock_mode: bool                 # True when no real weights loaded


class PatientScan(BaseModel):
    scan_id: str
    date: str
    prediction: str
    primary_finding: str
    confidence: float
    triage: str


class PatientRecord(BaseModel):
    patient_id: str
    name: str
    age: int
    gender: str
    ward: str
    referred_by: str
    scans: list[PatientScan]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    mock_mode: bool
    device: str
    model_version: str
