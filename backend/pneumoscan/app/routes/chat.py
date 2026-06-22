
"""
app/routes/chat.py

POST /chat/
AI chat using Groq, with scan context awareness.
"""

import os

import httpx
from fastapi import APIRouter, HTTPException

router = APIRouter()

# ──────────────────────────────────────────────────────────────
# Groq Configuration
# ──────────────────────────────────────────────────────────────

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

MODEL_NAME = "llama-3.3-70b-versatile"

SYSTEM_PROMPT = """
You are MedSight AI Chat, a clinical assistant helping doctors understand
chest X-ray screening results produced by an AI model
(EfficientNet-B0+B3 ensemble trained on NIH ChestXray14).

Your role:
- Explain AI findings clearly and concisely to doctors
- Define medical terms when relevant
- Help interpret confidence scores and triage levels
- Suggest clinical next steps based on findings
- Answer questions about the conditions:
  - Pneumonia
  - Cardiomegaly
  - Effusion
  - Infiltration

Always:
- Be concise (2–4 sentences unless more detail is requested)
- Remind that this is an AI screening tool and findings require physician confirmation
- Never make definitive diagnoses; only explain what the model found and why

CRITICAL:
This is a doctor-facing tool. Use appropriate medical terminology
while keeping explanations clear.
"""


# ──────────────────────────────────────────────────────────────
# Helper Functions
# ──────────────────────────────────────────────────────────────

def build_scan_context(scan_result: dict) -> str:
    """
    Build a context string from the current scan result.
    """
    if not scan_result:
        return ""

    condition = scan_result.get("top_condition", "Unknown")
    confidence = scan_result.get("confidence", 0)
    triage = scan_result.get("triage", "LOW")
    findings = scan_result.get("findings", {})

    lines = [
        "Current scan findings:",
        f"- Primary condition: {condition} ({confidence * 100:.1f}% confidence)",
        f"- Triage level: {triage}",
        "- All condition probabilities:"
    ]

    for cond, prob in sorted(findings.items(), key=lambda x: -x[1]):
        lines.append(f"  • {cond}: {prob * 100:.1f}%")

    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────
# Chat Endpoint
# ──────────────────────────────────────────────────────────────

@router.post("/")
async def handle_chat_message(payload: dict):
    """
    Handle chat requests and forward them to Groq.
    """

    user_message = payload.get("message") or payload.get("question", "")
    scan_result = payload.get("scan_result")
    chat_history = payload.get("chat_history", [])

    if not user_message:
        raise HTTPException(
            status_code=400,
            detail="No message provided."
        )

    if not GROQ_API_KEY:
        return {
            "status": "error",
            "reply": (
                "⚠️ No Groq API key configured. "
                "Set the GROQ_API_KEY environment variable on the server."
            ),
        }

    # Build conversation
    messages = []

    # Inject scan context
    scan_context = build_scan_context(scan_result)

    if scan_context:
        messages.append(
            {
                "role": "system",
                "content": (
                    "Current chest X-ray scan context:\n\n"
                    f"{scan_context}\n\n"
                    "Use this information when answering the doctor's questions."
                ),
            }
        )

    # Add recent history (last 6 turns)
    for turn in chat_history[-6:]:
        role = turn.get("role", "user")
        content = turn.get("content", "")

        if role in {"user", "assistant"} and content:
            messages.append(
                {
                    "role": role,
                    "content": content,
                }
            )

    # Current user message
    messages.append(
        {
            "role": "user",
            "content": user_message,
        }
    )

    # Call Groq API
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                GROQ_API_URL,
                headers={
                    "Authorization": f"Bearer {GROQ_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": MODEL_NAME,
                    "messages": [
                        {
                            "role": "system",
                            "content": SYSTEM_PROMPT,
                        },
                        *messages,
                    ],
                    "temperature": 0.3,
                    "max_tokens": 500,
                },
            )

            response.raise_for_status()

            data = response.json()

            reply = (
                data.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
            )

            return {
                "status": "success",
                "reply": reply,
            }

    except httpx.HTTPStatusError as e:
        raise HTTPException(
            status_code=502,
            detail=f"Groq API error: {e.response.text}",
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Chat error: {str(e)}",
        )