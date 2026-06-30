"""
app/models_db.py
SQLAlchemy ORM models (database tables).
Named models_db.py to avoid clashing with model_loader.py's ML models.
"""

from sqlalchemy import Column, Integer, String, DateTime, Date, ForeignKey, Text, Float
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.database import Base


class User(Base):
    __tablename__ = "users"

    id               = Column(Integer, primary_key=True, index=True)
    full_name        = Column(String, nullable=False)
    email            = Column(String, unique=True, index=True, nullable=False)
    hashed_password  = Column(String, nullable=False)
    role             = Column(String, default="doctor")
    created_at       = Column(DateTime(timezone=True), server_default=func.now())

    scans = relationship("Scan", back_populates="doctor")


class Patient(Base):
    __tablename__ = "patients"

    id          = Column(Integer, primary_key=True, index=True)
    name        = Column(String, nullable=False, index=True)
    age         = Column(Integer, nullable=True)
    gender      = Column(String, nullable=True)
    created_at  = Column(DateTime(timezone=True), server_default=func.now())

    scans = relationship("Scan", back_populates="patient", order_by="desc(Scan.created_at)")


class Scan(Base):
    __tablename__ = "scans"

    id                = Column(Integer, primary_key=True, index=True)

    # Relationships
    patient_id        = Column(Integer, ForeignKey("patients.id"), nullable=False)
    doctor_id         = Column(Integer, ForeignKey("users.id"), nullable=False)

    patient           = relationship("Patient", back_populates="scans")
    doctor            = relationship("User", back_populates="scans")

    # Intake details (filled by doctor before upload)
    department        = Column(String, nullable=False)
    scan_date         = Column(Date, nullable=False)          # date the X-ray was physically taken
    upload_date       = Column(DateTime(timezone=True), server_default=func.now())  # when entered into system

    # Image storage
    image_path        = Column(String, nullable=False)        # path to original uploaded X-ray
    gradcam_path       = Column(String, nullable=True)        # path to saved Grad-CAM overlay

    # Model output
    triage             = Column(String, nullable=True)        # HIGH / MEDIUM / LOW
    top_condition      = Column(String, nullable=True)
    confidence         = Column(Float, nullable=True)

    no_finding_prob    = Column(Float, nullable=True)
    pneumonia_prob     = Column(Float, nullable=True)
    cardiomegaly_prob  = Column(Float, nullable=True)
    effusion_prob      = Column(Float, nullable=True)
    infiltration_prob  = Column(Float, nullable=True)

    # Clinician review
    doctor_notes       = Column(Text, nullable=True)
    override           = Column(String, nullable=True)        # confirm / normal / abnormal / refer
    reviewed_at        = Column(DateTime(timezone=True), nullable=True)

    # Patient-reported symptoms recorded at the time of THIS scan.
    # Stored as a JSON-encoded list of objects, e.g.
    # '[{"name": "Cough", "severity": "medium"}, {"name": "Fever", "severity": "high"}]'
    # Decode with json.loads(scan.symptoms) when reading, encode with
    # json.dumps(symptoms_list) when writing. Nullable so older scans
    # created before this column existed simply have no symptom data.
    symptoms           = Column(Text, nullable=True)

    created_at         = Column(DateTime(timezone=True), server_default=func.now())