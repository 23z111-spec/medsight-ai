"""
/patients  —  Patient records & scan history
=============================================
In production, wire these to your SQLite/PostgreSQL DB.
For now, returns mock data that matches the dashboard sidebar.
"""

from fastapi import APIRouter, HTTPException
from app.schemas import PatientRecord, PatientScan

router = APIRouter()

# ── Mock DB — replace with real DB queries later ──
MOCK_PATIENTS = {
    "PT-00421": PatientRecord(
        patient_id="PT-00421",
        name="Rajan Mehta",
        age=54,
        gender="Male",
        ward="Pulmonology",
        referred_by="Dr. S. Anand",
        scans=[
            PatientScan(
                scan_id="SCN-A1B2C3D4",
                date="2026-05-20",
                prediction="Abnormal",
                primary_finding="Pneumonia",
                confidence=0.913,
                triage="HIGH",
            ),
            PatientScan(
                scan_id="SCN-E5F6G7H8",
                date="2026-03-02",
                prediction="Abnormal",
                primary_finding="Pleural Effusion",
                confidence=0.612,
                triage="MEDIUM",
            ),
            PatientScan(
                scan_id="SCN-I9J0K1L2",
                date="2025-10-14",
                prediction="Normal",
                primary_finding="Normal",
                confidence=0.891,
                triage="LOW",
            ),
        ],
    )
}


@router.get("/{patient_id}", response_model=PatientRecord)
def get_patient(patient_id: str):
    """Fetch patient record and full scan history by patient ID."""
    patient = MOCK_PATIENTS.get(patient_id)
    if not patient:
        raise HTTPException(status_code=404, detail=f"Patient '{patient_id}' not found.")
    return patient


@router.get("/", response_model=list[PatientRecord])
def list_patients():
    """List all patients (paginate this in production)."""
    return list(MOCK_PATIENTS.values())
