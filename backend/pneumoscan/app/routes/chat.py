"""
PneumoScan AI — /chat route (Groq API)
Streams responses from Groq's hosted Llama3 model.

Setup:
    pip install httpx python-dotenv
    Add GROQ_API_KEY to backend/pneumoscan/.env

Add to main.py:
    from app.routes import chat
    app.include_router(chat.router, prefix="/chat", tags=["Chat"])
"""

import os
import httpx
import json
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from pathlib import Path

env_path = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(env_path)



router = APIRouter()

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL   = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
GROQ_URL     = "https://api.groq.com/openai/v1/chat/completions"
print("Groq model:", GROQ_MODEL)
print("CHAT MODULE KEY:", GROQ_API_KEY)

if GROQ_API_KEY:
    print("Groq API key loaded successfully")
else:
    print("WARNING: GROQ_API_KEY not found")

# ── Request schema ─────────────────────────────────────────────────
class ChatRequest(BaseModel):
    question:     str
    scan_result:  dict
    chat_history: list[dict] = Field(default_factory=list)# [{"role": "user"|"assistant", "content": "..."}]

# ── System prompt ──────────────────────────────────────────────────
def build_system_prompt(scan: dict) -> str:
    findings = scan.get("findings", {})
    findings_text = "\n".join(
        f"  - {c}: {p*100:.1f}%"
        for c, p in sorted(findings.items(), key=lambda x: -x[1])
    )
    top     = scan.get("top_condition", "Unknown")
    conf    = scan.get("confidence", 0) * 100
    triage  = scan.get("triage", "UNKNOWN")
    scan_id = scan.get("scan_id", "N/A")
    b0      = scan.get("b0_findings", {})
    b3      = scan.get("b3_findings", {})

    return f"""You are a clinical AI assistant.

Use the scan result provided.
Explain findings clearly and professionally.

Always:
- Mention the top predicted condition.
- Mention confidence percentage.
- Mention triage level.
- Explain that this is a screening result, not a diagnosis.
- Recommend physician review when appropriate and You are PneumoScan AI, a medical assistant inside a chest X-ray AI screening platform.
You are talking to a patient or their family member who just received an AI scan result.

SCAN RESULT (ID: {scan_id}):
- Primary finding : {top} ({conf:.1f}% confidence)
- Triage priority : {triage}
- All findings:
{findings_text}
- EfficientNet-B0 confidence for {top}: {b0.get(top, 0)*100:.1f}%
- EfficientNet-B3 confidence for {top}: {b3.get(top, 0)*100:.1f}%

YOUR RULES:
1. Always answer based on the scan data above. Never make up findings.
2. Use plain, simple language. Explain medical terms immediately if you use them.
3. If the user asks about any condition in the findings (even not the top one), answer about that condition specifically.
4. If triage is HIGH, clearly recommend seeing a doctor today.
5. If triage is MEDIUM, recommend seeing a doctor within 1-2 days.
6. If triage is LOW, recommend a routine check-up.
7. Keep answers concise (3-5 sentences). Use bullet points only if the user asks for a list.
8. Never claim to be a doctor or give a definitive diagnosis.
9. End every medical advice response with a one-line reminder to consult a physician.
10. If asked about causes, lifestyle, smoking, family history — answer using general medical knowledge about the detected condition."""


# ── Streaming chat endpoint ────────────────────────────────────────
@router.post("/")
async def chat(req: ChatRequest):
    print("INSIDE CHAT ROUTE KEY =", repr(GROQ_API_KEY))
    if not GROQ_API_KEY:
        async def no_key():
            yield f"data: {json.dumps({'token': 'GROQ_API_KEY is not set in your .env file.', 'done': True})}\n\n"
        return StreamingResponse(no_key(), media_type="text/event-stream")

    # Build messages for Groq (OpenAI-compatible format)
    messages = [{"role": "system", "content": build_system_prompt(req.scan_result)}]
    for msg in req.chat_history[-6:]:
        messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": req.question})

    payload = {
        "model":       GROQ_MODEL,
        "messages":    messages,
        "temperature": 0.3,
        "max_tokens":  400,
        "stream":      True,
    }
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type":  "application/json",
    }

    async def stream_groq():
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                async with client.stream("POST", GROQ_URL, json=payload, headers=headers) as resp:
                    if resp.status_code == 401:
                        yield f"data: {json.dumps({'token': 'Invalid GROQ_API_KEY. Check your .env file.', 'done': True})}\n\n"
                        return
                    if resp.status_code != 200:
                        body = await resp.aread()

                        print("\n" + "="*80)
                        print("GROQ STATUS:", resp.status_code)
                        print("GROQ BODY:")
                        print(body.decode("utf-8"))
                        print("="*80 + "\n")

                        yield f"data: {json.dumps({'token': body.decode('utf-8'), 'done': True})}\n\n"
                        
                        return

                    async for line in resp.aiter_lines():
                        if not line.startswith("data: "):
                            continue
                        data = line[6:]
                        if data.strip() == "[DONE]":
                            yield f"data: {json.dumps({'token': '', 'done': True})}\n\n"
                            break
                        try:
                            chunk  = json.loads(data)
                            delta  = chunk["choices"][0]["delta"]
                            token  = delta.get("content", "")
                            finish = chunk["choices"][0].get("finish_reason")
                            if token:
                                yield f"data: {json.dumps({'token': token, 'done': False})}\n\n"
                            if finish == "stop":
                                yield f"data: {json.dumps({'token': '', 'done': True})}\n\n"
                                break
                        except (json.JSONDecodeError, KeyError):
                            continue

        except httpx.ConnectError:
            fallback = get_fallback_response(req.question, req.scan_result)
            yield f"data: {json.dumps({'token': fallback, 'done': True})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'token': f'Unexpected error: {str(e)}', 'done': True})}\n\n"

    return StreamingResponse(stream_groq(), media_type="text/event-stream")


# ── Rich fallback (when Groq is unreachable) ───────────────────────
def get_fallback_response(question: str, scan: dict) -> str:
    q        = question.lower()
    top      = scan.get("top_condition", "the detected condition")
    conf     = scan.get("confidence", 0) * 100
    triage   = scan.get("triage", "UNKNOWN")
    findings = scan.get("findings", {})

    CONDITION_PLAIN = {
        "Pneumonia": {
            "simple":    "The lungs show signs of infection. Pneumonia happens when germs cause the air sacs to fill with fluid, making breathing harder.",
            "symptoms":  "Common signs are fever, cough with mucus, shortness of breath, chest pain when breathing, and fatigue.",
            "treatment": "Bacterial pneumonia is treated with antibiotics. Viral pneumonia needs rest and fluids. Severe cases need hospital care and oxygen.",
            "causes":    "Pneumonia is caused by bacteria, viruses, or fungi. Risk factors include smoking, older age, and weakened immunity.",
            "doctor":    "Yes — pneumonia can worsen quickly. Please see a doctor today, especially if breathing is difficult.",
        },
        "Cardiomegaly": {
            "simple":    "The heart appears larger than normal — a sign it is working harder than it should, often due to high blood pressure or a heart condition.",
            "symptoms":  "You might feel short of breath, tired easily, notice leg swelling, or feel your heart pounding.",
            "treatment": "Treatment includes medications for blood pressure or heart failure, lifestyle changes, and sometimes procedures.",
            "causes":    "Causes include high blood pressure, heart valve problems, cardiomyopathy, or long-term heavy alcohol use.",
            "doctor":    "Yes — an enlarged heart needs assessment by a cardiologist as soon as possible.",
        },
        "Effusion": {
            "simple":    "Fluid has collected in the space around the lungs. This presses on the lungs and makes breathing harder.",
            "symptoms":  "Sharp chest pain when breathing in, a dry cough, and shortness of breath — especially when lying flat.",
            "treatment": "Small amounts may clear on their own. Larger amounts may need draining. The underlying cause is also treated.",
            "causes":    "Can be caused by pneumonia, heart failure, kidney disease, cancer, or tuberculosis.",
            "doctor":    "Yes — pleural effusion always needs medical evaluation. Please see a doctor soon.",
        },
        "Infiltration": {
            "simple":    "There are patches in the lungs where fluid or cells have collected in the air spaces — a sign of infection or inflammation.",
            "symptoms":  "Cough, fever, breathlessness, and lower oxygen levels are common.",
            "treatment": "Depends on the cause — antibiotics for infection, diuretics for fluid, anti-inflammatory drugs for other causes.",
            "causes":    "Commonly caused by pneumonia, fluid overload, aspiration, or inflammatory conditions.",
            "doctor":    "Yes — lung infiltrates need medical evaluation to find and treat the cause promptly.",
        },
        "No Finding": {
            "simple":    "The chest X-ray looks normal. No signs of infection, fluid, enlarged heart, or major abnormalities were found.",
            "symptoms":  "No specific symptoms are associated with a normal result.",
            "treatment": "No treatment needed based on this scan alone.",
            "causes":    "No abnormality was detected.",
            "doctor":    "This is reassuring, but if you have ongoing symptoms, still see your doctor — a scan is one piece of the picture.",
        },
    }

    info = CONDITION_PLAIN.get(top, {})
    triage_advice = {
        "HIGH":   "Please see a doctor today — this is a high-priority finding.",
        "MEDIUM": "Please see a doctor within 1–2 days.",
        "LOW":    "A routine check-up is recommended.",
    }.get(triage, "Please consult a doctor.")

    # If user asks about a specific non-top condition
    for cond, prob in findings.items():
        if cond.lower() in q and cond != top:
            alt = CONDITION_PLAIN.get(cond, {})
            return (
                f"You asked about {cond} specifically. "
                f"{alt.get('simple', '')} "
                f"The model detected it at {prob*100:.1f}% probability. "
                f"Please consult your doctor for a full assessment."
            )

    if any(w in q for w in ["simple","explain","layman","plain","understand","what is","what does","mean","tell me"]):
        return f"{info.get('simple', f'The scan detected {top}.')} Confidence: {conf:.1f}%. {triage_advice}"

    if any(w in q for w in ["symptom","sign","feel","experience"]):
        return f"{info.get('symptoms', 'Please consult your doctor about symptoms.')} Always see a doctor if symptoms concern you."

    if any(w in q for w in ["treat","medicine","cure","drug","therapy","management"]):
        return f"{info.get('treatment', 'Treatment depends on clinical assessment.')} Your doctor will advise the right plan. Please consult a physician."

    if any(w in q for w in ["cause","why","reason","smoking","habit","lifestyle","risk","factor","grandfather","grandmother","parent","family","relative","he","she","they"]):
        return f"{info.get('causes', 'Causes vary and require clinical investigation.')} {triage_advice} Please consult a doctor."

    if any(w in q for w in ["doctor","hospital","should i","go","visit","asap","urgent","serious","dangerous","worried","concern","bad","how bad"]):
        return f"{info.get('doctor', triage_advice)} {triage_advice} This AI tool cannot replace a clinical examination."

    if any(w in q for w in ["summarize","summary","report","overview","brief","all"]):
        all_f = ", ".join(f"{c}: {p*100:.1f}%" for c, p in sorted(findings.items(), key=lambda x: -x[1]))
        return f"Summary: Primary finding is {top} ({conf:.1f}% confidence), triage {triage}. All findings — {all_f}. {triage_advice}"

    if any(w in q for w in ["ask","what should i ask","questions for"]):
        return (
            f"Good questions to ask your doctor about {top}: "
            f"1) How certain are you about this finding? "
            f"2) Do I need further tests like a CT scan or blood work? "
            f"3) What are my treatment options? "
            f"4) How soon should I start treatment? "
            f"5) What lifestyle changes would help? Please consult a physician."
        )

    # Generic
    return (
        f"The scan detected {top} with {conf:.1f}% confidence (triage: {triage}). "
        f"{info.get('simple', '')} {triage_advice} "
        f"You can ask me about symptoms, causes, treatment, or whether to see a doctor."
    )