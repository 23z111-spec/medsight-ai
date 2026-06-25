"""
app/mail_utils.py
Simple Gmail SMTP mailer — no third-party mail library required.
Uses only Python stdlib: smtplib + email.mime
"""

import smtplib
import os
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText


MAIL_USERNAME = os.getenv("MAIL_USERNAME", "")
MAIL_PASSWORD = os.getenv("MAIL_PASSWORD", "")
MAIL_FROM     = os.getenv("MAIL_FROM", MAIL_USERNAME)
MAIL_SERVER   = os.getenv("MAIL_SERVER", "smtp.gmail.com")
MAIL_PORT     = int(os.getenv("MAIL_PORT", "587"))


def send_reset_email(to_email: str, reset_link: str, full_name: str) -> None:
    """
    Send a password-reset email to `to_email`.
    Raises smtplib.SMTPException on failure — caller should catch and log.
    """
    subject = "MedSight AI — Password Reset Request"

    html_body = f"""
    <html>
    <body style="font-family: Inter, Arial, sans-serif; background:#f4f7fb; padding:30px;">
      <div style="max-width:480px; margin:0 auto; background:#ffffff;
                  border-radius:12px; border:1px solid #e2e8f0; padding:32px;">

        <div style="display:flex; align-items:center; gap:10px; margin-bottom:24px;">
          <div style="width:36px; height:36px; border-radius:8px;
                      background:linear-gradient(135deg,#3B82F6,#1D4ED8);
                      display:flex; align-items:center; justify-content:center;
                      font-size:18px; color:#fff;">🧠</div>
          <span style="font-family:monospace; font-weight:700;
                       font-size:14px; color:#3B82F6; letter-spacing:.5px;">
            MEDSIGHT · AI
          </span>
        </div>

        <h2 style="color:#1e293b; font-size:20px; margin:0 0 8px;">
          Password Reset Request
        </h2>
        <p style="color:#64748b; font-size:14px; line-height:1.6; margin:0 0 24px;">
          Hi {full_name}, we received a request to reset your password.
          Click the button below — this link expires in <strong>30 minutes</strong>.
        </p>

        <a href="{reset_link}"
           style="display:inline-block; background:linear-gradient(135deg,#3B82F6,#1D4ED8);
                  color:#ffffff; text-decoration:none; padding:12px 28px;
                  border-radius:8px; font-weight:600; font-size:14px;
                  margin-bottom:24px;">
          Reset My Password
        </a>

        <p style="color:#94a3b8; font-size:12px; line-height:1.6; margin:0;">
          If you didn't request this, you can safely ignore this email —
          your password will not change.<br><br>
          Or copy this link into your browser:<br>
          <span style="color:#3B82F6; word-break:break-all;">{reset_link}</span>
        </p>

        <hr style="border:none; border-top:1px solid #e2e8f0; margin:24px 0 16px;">
        <p style="color:#cbd5e1; font-size:11px; margin:0;">
          MedSight AI · AI-assisted chest X-ray screening · Not for diagnostic use
        </p>
      </div>
    </body>
    </html>
    """

    plain_body = (
        f"Hi {full_name},\n\n"
        f"Reset your MedSight AI password using the link below (expires in 30 minutes):\n\n"
        f"{reset_link}\n\n"
        f"If you didn't request this, ignore this email.\n\n"
        f"— MedSight AI"
    )

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = f"MedSight AI <{MAIL_USERNAME}>"
    msg["To"]      = to_email
    msg.attach(MIMEText(plain_body, "plain"))
    msg.attach(MIMEText(html_body, "html"))

    with smtplib.SMTP(MAIL_SERVER, MAIL_PORT) as server:
        server.ehlo()
        server.starttls()
        server.login(MAIL_USERNAME, MAIL_PASSWORD)
        server.sendmail(MAIL_FROM, to_email, msg.as_string())