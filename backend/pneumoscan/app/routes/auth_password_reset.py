"""
Password Reset — MedSight AI
Fixed version: uses port 465 SSL with a 10s timeout so it fails fast
instead of hanging. Falls back gracefully with a clear error log.
"""

import os, smtplib, ssl, secrets
from datetime import datetime, timedelta
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from app.database import get_db
from app.models_db import User
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, EmailStr
from jose import jwt, JWTError
from app.auth_utils import hash_password
from dotenv import load_dotenv

load_dotenv()

router = APIRouter(prefix="/auth", tags=["auth"])

# ── Config ────────────────────────────────────────────────────
GMAIL_USER         = os.getenv("GMAIL_USER")
GMAIL_APP_PASSWORD = os.getenv("GMAIL_APP_PASSWORD")
RESET_SECRET       = os.getenv("RESET_SECRET", secrets.token_hex(32))
FRONTEND_URL       = os.getenv("FRONTEND_URL", "http://localhost:5500")
ALGORITHM          = "HS256"
RESET_EXPIRE_MINS  = 30


# ── Schemas ───────────────────────────────────────────────────
class ForgotPasswordRequest(BaseModel):
    email: EmailStr

class ResetPasswordRequest(BaseModel):
    token: str
    new_password: str


# ── Token helpers ─────────────────────────────────────────────
def create_reset_token(email: str) -> str:
    expire = datetime.utcnow() + timedelta(minutes=RESET_EXPIRE_MINS)
    return jwt.encode(
        {"sub": email, "exp": expire, "type": "password_reset"},
        RESET_SECRET,
        algorithm=ALGORITHM,
    )

def verify_reset_token(token: str) -> str:
    try:
        payload = jwt.decode(token, RESET_SECRET, algorithms=[ALGORITHM])
        if payload.get("type") != "password_reset":
            raise ValueError("Wrong token type")
        email = payload.get("sub")
        if not email:
            raise ValueError("No email in token")
        return email
    except JWTError:
        raise HTTPException(status_code=400, detail="Reset link has expired or is invalid.")


# ── Email sender ──────────────────────────────────────────────
def send_reset_email(to_email: str, reset_link: str):
    if not GMAIL_USER or not GMAIL_APP_PASSWORD:
        raise Exception("GMAIL_USER or GMAIL_APP_PASSWORD not set in environment variables.")

    msg = MIMEMultipart("alternative")
    msg["Subject"] = "MedSight AI — Password Reset"
    msg["From"]    = f"MedSight AI <{GMAIL_USER}>"
    msg["To"]      = to_email

    html = f"""
    <div style="font-family:DM Sans,sans-serif;max-width:520px;margin:0 auto;background:#0f1f34;color:#e2efff;border-radius:12px;overflow:hidden;">
      <div style="background:#061528;padding:28px 32px;border-bottom:1px solid rgba(96,165,250,.2);">
        <span style="font-family:monospace;font-size:13px;color:#60a5fa;letter-spacing:1px;">● MEDSIGHT AI</span>
      </div>
      <div style="padding:32px;">
        <h2 style="margin:0 0 12px;font-size:20px;font-weight:600;">Password Reset Request</h2>
        <p style="color:#94a3b8;font-size:14px;line-height:1.7;margin:0 0 24px;">
          We received a request to reset the password for your MedSight account
          (<strong style="color:#e2efff;">{to_email}</strong>).<br><br>
          Click the button below to set a new password. This link expires in
          <strong style="color:#60a5fa;">30 minutes</strong>.
        </p>
        <a href="{reset_link}"
           style="display:inline-block;background:#3b82f6;color:#fff;text-decoration:none;
                  padding:12px 28px;border-radius:8px;font-size:14px;font-weight:500;">
          Reset my password
        </a>
        <p style="color:#475569;font-size:12px;margin-top:24px;line-height:1.6;">
          If you didn't request this, you can safely ignore this email —
          your password won't change.<br><br>
          Or copy this link manually:<br>
          <span style="color:#60a5fa;word-break:break-all;">{reset_link}</span>
        </p>
      </div>
      <div style="padding:16px 32px;background:#061528;font-size:11px;color:#475569;border-top:1px solid rgba(96,165,250,.1);">
        For authorized medical personnel only · MedSight AI
      </div>
    </div>
    """

    msg.attach(MIMEText(html, "html"))

    # Use explicit SSL context with a 10-second timeout
    # so the request fails fast instead of hanging indefinitely
    context = ssl.create_default_context()
    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context, timeout=10) as server:
            server.login(GMAIL_USER, GMAIL_APP_PASSWORD)
            server.sendmail(GMAIL_USER, to_email, msg.as_string())
            print(f"[EMAIL OK] Reset email sent to {to_email}")
    except smtplib.SMTPAuthenticationError:
        raise Exception("Gmail authentication failed. Check GMAIL_USER and GMAIL_APP_PASSWORD.")
    except TimeoutError:
        raise Exception("SMTP connection timed out. Railway may be blocking port 465.")
    except OSError as e:
        raise Exception(f"Network error reaching Gmail SMTP: {e}")


# ── Endpoints ─────────────────────────────────────────────────

@router.post("/forgot-password")
async def forgot_password(body: ForgotPasswordRequest, db=Depends(get_db)):
    email = body.email
    user  = db.query(User).filter(User.email == email).first()

    if not user:
        return {"message": "If that email exists, a reset link has been sent."}

    token      = create_reset_token(email)
    reset_link = f"{FRONTEND_URL}/reset-password.html?token={token}"

    try:
        send_reset_email(email, reset_link)
    except Exception as e:
        print(f"[EMAIL ERROR] {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to send email: {e}"
        )

    return {"message": "If that email exists, a reset link has been sent."}


@router.post("/reset-password")
async def reset_password(body: ResetPasswordRequest, db=Depends(get_db)):
    email = verify_reset_token(body.token)

    if len(body.new_password) < 8:
        raise HTTPException(status_code=400, detail="Password must be at least 8 characters.")

    user = db.query(User).filter(User.email == email).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found.")

    user.hashed_password = hash_password(body.new_password)
    db.commit()

    print(f"[RESET OK] Password updated for {email}")
    return {"message": "Password updated successfully."}