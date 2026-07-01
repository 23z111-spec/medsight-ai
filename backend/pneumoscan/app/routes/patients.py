"""
app/routes/patients.py
GET  /patients/search?name=    — autocomplete search for existing patients
POST /patients                 — create a new patient
GET  /patients/{id}            — patient details + full scan history
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import or_
from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime

from app.database import get_db
from app.models_db import Patient, Scan
from app.routes.auth import get_current_user
from app.models_db import User

router = APIRouter()


# ── Schemas ───────────────────────────────────────────────────────
class PatientCreateRequest(BaseModel):
    name: str
    age: Optional[int] = None
    gender: Optional[str] = None



class PatientSummary(BaseModel):
    id: int
    name: str
    age: Optional[int]
    gender: Optional[str]
    total_scans: int
    last_scan_date: Optional[str] = None
    # ── new fields ──
    department: Optional[str] = None
    top_condition: Optional[str] = None
    confidence: Optional[float] = None
    triage: Optional[str] = None
    doctor_name: Optional[str] = None


class ScanSummary(BaseModel):
    id: int
    department: str
    scan_date: str
    upload_date: str
    triage: Optional[str]
    top_condition: Optional[str]
    confidence: Optional[float]
    doctor_name: Optional[str] = None


class PatientDetailResponse(BaseModel):
    id: int
    name: str
    age: Optional[int]
    gender: Optional[str]
    scans: List[ScanSummary]


@router.get("/search", response_model=List[PatientSummary])
def search_patients(
    name: str = "",
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    query = db.query(Patient)
    if name:
        query = query.filter(Patient.name.ilike(f"%{name}%"))

    patients = query.order_by(Patient.name).limit(20).all()

    results = []
    for p in patients:
        # scans are already loaded via relationship — most recent first
        latest = p.scans[0] if p.scans else None
        results.append(PatientSummary(
            id=p.id,
            name=p.name,
            age=p.age,
            gender=p.gender,
            total_scans=len(p.scans),
            last_scan_date=latest.scan_date.isoformat() if latest else None,
            # ── from latest scan ──
            department=latest.department if latest else None,
            top_condition=latest.top_condition if latest else None,
            confidence=latest.confidence if latest else None,
            triage=latest.triage if latest else None,
            doctor_name=latest.doctor.full_name if (latest and latest.doctor) else None,
        ))
    return results


@router.post("/", response_model=PatientSummary, status_code=201)
def create_patient(
    body: PatientCreateRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Create a new patient record."""
    patient = Patient(name=body.name.strip(), age=body.age, gender=body.gender)
    db.add(patient)
    db.commit()
    db.refresh(patient)

    return PatientSummary(
        id=patient.id,
        name=patient.name,
        age=patient.age,
        gender=patient.gender,
        total_scans=0,
        last_scan_date=None,
    )


@router.get("/{patient_id}", response_model=PatientDetailResponse)
def get_patient(
    patient_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Return patient details and full scan history, most recent first."""
    patient = db.query(Patient).filter(Patient.id == patient_id).first()
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found.")

    scan_summaries = [
        ScanSummary(
            id=s.id,
            department=s.department,
            scan_date=s.scan_date.isoformat(),
            upload_date=s.upload_date.isoformat() if s.upload_date else "",
            triage=s.triage,
            top_condition=s.top_condition,
            confidence=s.confidence,
            doctor_name=s.doctor.full_name if s.doctor else None,
        )
        for s in patient.scans
    ]

    return PatientDetailResponse(
        id=patient.id,
        name=patient.name,
        age=patient.age,
        gender=patient.gender,
        scans=scan_summaries,
    )