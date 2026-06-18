"""
app/models_db.py
SQLAlchemy ORM models (database tables).
Named models_db.py to avoid clashing with model_loader.py's ML models.
"""

from sqlalchemy import Column, Integer, String, DateTime
from sqlalchemy.sql import func
from app.database import Base


class User(Base):
    __tablename__ = "users"

    id            = Column(Integer, primary_key=True, index=True)
    full_name     = Column(String, nullable=False)
    email         = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    role          = Column(String, default="doctor")
    created_at    = Column(DateTime(timezone=True), server_default=func.now())
