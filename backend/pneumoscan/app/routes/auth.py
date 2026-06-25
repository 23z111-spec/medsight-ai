"""
app/routes/auth.py
POST /auth/register        — create a new doctor account
POST /auth/login           — authenticate, return JWT token
GET  /auth/me              — return current logged-in user (requires token)
POST /auth/forgot-password — generate reset token, email the link
POST /auth/reset-password  — validate token, set new password
"""

import os
import secrets
from datetime import datetime, timezone, timedelta

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from pydantic import BaseModel, EmailStr

from app.database import get_db
from app.models_db import User
from app.auth_utils import hash_password, verify_password, create_access_token, decode_access_token
from app.mail_utils import send_reset_email

router = APIRouter()

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/token")

FRONTEND_URL        = os.getenv("FRONTEND_URL", "http://127.0.0.1:5500/frontend")
RESET_TOKEN_EXPIRY  = 30  # minutes


# ── Schemas ───────────────────────────────────────────────────────
class RegisterRequest(BaseModel):
    full_name: str
    email: EmailStr
    password: str


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    full_name: str
    email: str
    role: str


class UserResponse(BaseModel):
    id: int
    full_name: str
    email: str
    role: str


class ForgotPasswordRequest(BaseModel):
    email: EmailStr


class ResetPasswordRequest(BaseModel):
    token: str
    new_password: str


class MessageResponse(BaseModel):
    message: str


# ── Dependency: get current user from token ───────────────────────
def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)) -> User:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    payload = decode_access_token(token)
    if payload is None:
        raise credentials_exception

    email = payload.get("sub")
    if email is None:
        raise credentials_exception

    user = db.query(User).filter(User.email == email).first()
    if user is None:
        raise credentials_exception
    return user


# ── Routes ────────────────────────────────────────────────────────
@router.post("/register", response_model=TokenResponse, status_code=status.HTTP_201_CREATED)
def register(body: RegisterRequest, db: Session = Depends(get_db)):
    existing = db.query(User).filter(User.email == body.email).first()
    if existing:
        raise HTTPException(status_code=400, detail="An account with this email already exists.")

    if len(body.password) < 6:
        raise HTTPException(status_code=400, detail="Password must be at least 6 characters.")

    user = User(
        full_name=body.full_name,
        email=body.email,
        hashed_password=hash_password(body.password),
        role="doctor",
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    token = create_access_token({"sub": user.email})
    return TokenResponse(
        access_token=token,
        full_name=user.full_name,
        email=user.email,
        role=user.role,
    )


@router.post("/token", response_model=TokenResponse)
def login_for_swagger(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    """OAuth2 token endpoint used only by the Swagger 'Authorize' button."""
    user = db.query(User).filter(User.email == form_data.username).first()
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password.",
        )
    token = create_access_token({"sub": user.email})
    return TokenResponse(
        access_token=token,
        full_name=user.full_name,
        email=user.email,
        role=user.role,
    )


@router.post("/login", response_model=TokenResponse)
def login(body: LoginRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == body.email).first()
    if not user or not verify_password(body.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password.",
        )

    token = create_access_token({"sub": user.email})
    return TokenResponse(
        access_token=token,
        full_name=user.full_name,
        email=user.email,
        role=user.role,
    )


@router.get("/me", response_model=UserResponse)
def read_current_user(current_user: User = Depends(get_current_user)):
    return UserResponse(
        id=current_user.id,
        full_name=current_user.full_name,
        email=current_user.email,
        role=current_user.role,
    )


# ── Forgot Password ───────────────────────────────────────────────
@router.post("/forgot-password", response_model=MessageResponse)
def forgot_password(body: ForgotPasswordRequest, db: Session = Depends(get_db)):
    """
    Always returns 200 with the same message regardless of whether the email
    exists — this prevents user enumeration attacks.
    """
    user = db.query(User).filter(User.email == body.email).first()

    if user:
        # Generate a cryptographically secure random token
        token  = secrets.token_urlsafe(32)
        expiry = datetime.utcnow() + timedelta(minutes=RESET_TOKEN_EXPIRY)

        user.reset_token        = token
        user.reset_token_expiry = expiry
        db.commit()

        reset_link = f"{FRONTEND_URL}/reset-password.html?token={token}"

        try:
            send_reset_email(
                to_email=user.email,
                reset_link=reset_link,
                full_name=user.full_name,
            )
        except Exception as e:
            # Roll back the token so the user can try again
            user.reset_token        = None
            user.reset_token_expiry = None
            db.commit()
            print(f"[forgot-password] Mail send failed: {e}")
            raise HTTPException(
                status_code=500,
                detail="Could not send reset email. Check server mail configuration.",
            )

    return MessageResponse(
        message="If an account with that email exists, a password reset link has been sent."
    )


# ── Reset Password ────────────────────────────────────────────────
@router.post("/reset-password", response_model=MessageResponse)
def reset_password(body: ResetPasswordRequest, db: Session = Depends(get_db)):
    """
    Validate the reset token and update the user's password.
    Token is single-use — cleared after successful reset.
    """
    if len(body.new_password) < 6:
        raise HTTPException(status_code=400, detail="Password must be at least 6 characters.")

    user = db.query(User).filter(User.reset_token == body.token).first()

    if not user:
        raise HTTPException(status_code=400, detail="Invalid or expired reset link.")

    # Check expiry
    
    now = datetime.utcnow()
    if user.reset_token_expiry is None or user.reset_token_expiry < now:
        # Clean up the expired token
        user.reset_token        = None
        user.reset_token_expiry = None
        db.commit()
        raise HTTPException(status_code=400, detail="Reset link has expired. Please request a new one.")

    # All good — update the password and clear the token
    user.hashed_password    = hash_password(body.new_password)
    user.reset_token        = None
    user.reset_token_expiry = None
    db.commit()

    return MessageResponse(message="Password updated successfully. You can now log in.")