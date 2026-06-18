"""
PneumoScan AI — /chat route
Accepts a user question + scan result, sends to local Ollama LLM,
streams back a plain-language medical response.

Add to main.py:
    from app.routes import chat
    app.include_router(chat.router, prefix="/chat", tags=["Chat"])

Requirements:
    pip install httpx
    # Install Ollama: https://ollama.com  then run:  ollama pull llama3
"""

import httpx
import json
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

router = APIRouter()

OLLAMA_URL  = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3"   # change to "mistral" or "phi3" if preferred

# ── Request schema ────────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    question: str
    scan_result: dict
    chat_history: list[dict] = []   # [{"role": "user"|"assistant", "content": "..."}]

# ── System prompt builder ─────────────────────────────────────────────────────
def build_system_prompt(scan: dict) -> str:
    findings = scan.get("findings", {})
    findings_text = "\n".join(
        f"  - {condition}: {prob*100:.1f}%"
        for condition, prob in sorted(findings.items(), key=lambda x: -x[1])
    )
    b0 = scan.get("b0_findings", {})
    b3 = scan.get("b3_findings", {})
    top = scan.get("top_condition", "Unknown")
    conf = scan.get("confidence", 0) * 100
    triage = scan.get("triage", "UNKNOWN")
    scan_id = scan.get("scan_id", "N/A")

    return f"""You are PneumoScan AI, a medical assistant integrated into a chest X-ray screening platform.
You are speaking with a patient or their family member who has just received an AI-generated chest X-ray screening result.

SCAN RESULT (ID: {scan_id}):
- Top finding: {top} ({conf:.1f}% confidence)
- Triage priority: {triage}
- All findings:
{findings_text}
- EfficientNet-B0 top finding confidence: {b0.get(top, 0)*100:.1f}%
- EfficientNet-B3 top finding confidence: {b3.get(top, 0)*100:.1f}%

YOUR ROLE:
1. Answer questions about this specific scan result using the data above.
2. Use plain, simple language a patient or family member can understand. Avoid jargon; if you must use a medical term, explain it immediately.
3. Be honest about uncertainty — you are an AI screening tool, not a definitive diagnosis.
4. If the triage is HIGH, gently but clearly recommend seeing a doctor urgently.
5. If the user asks about a condition mentioned in the findings (even if not the top one), answer about that specific condition.
6. Never make up data. Only reference findings from the scan result provided above.
7. Keep responses concise (3–5 sentences unless detail is needed). Do not use bullet lists unless asked.
8. Always end responses that involve medical advice with a brief reminder to consult a doctor.

IMPORTANT: You are NOT a replacement for a radiologist or physician."""


# ── Streaming chat endpoint ───────────────────────────────────────────────────
@router.post("/")
async def chat(req: ChatRequest):
    system_prompt = build_system_prompt(req.scan_result)

    # Build conversation history for context
    history_text = ""
    for msg in req.chat_history[-6:]:   # last 3 exchanges (6 messages)
        role = "User" if msg["role"] == "user" else "Assistant"
        history_text += f"{role}: {msg['content']}\n"

    full_prompt = f"{history_text}User: {req.question}\nAssistant:"

    payload = {
        "model": OLLAMA_MODEL,
        "system": system_prompt,
        "prompt": full_prompt,
        "stream": True,
        "options": {
            "temperature": 0.3,       # low temp = consistent medical answers
            "num_predict": 300,       # max tokens per response
        }
    }

    async def stream_ollama():
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                async with client.stream("POST", OLLAMA_URL, json=payload) as resp:
                    if resp.status_code != 200:
                        yield f"data: {json.dumps({'error': 'LLM unavailable'})}\n\n"
                        return
                    async for line in resp.aiter_lines():
                        if line:
                            try:
                                chunk = json.loads(line)
                                token = chunk.get("response", "")
                                done  = chunk.get("done", False)
                                yield f"data: {json.dumps({'token': token, 'done': done})}\n\n"
                                if done:
                                    break
                            except json.JSONDecodeError:
                                continue
        except httpx.ConnectError:
            # Ollama not running — send a clear fallback message
            yield f"data: {json.dumps({'token': get_fallback_response(req.question, req.scan_result), 'done': True})}\n\n"

    return StreamingResponse(stream_ollama(), media_type="text/event-stream")


# ── Fallback when Ollama is not installed ─────────────────────────────────────
def get_fallback_response(question: str, scan: dict) -> str:
    """Rich rule-based fallback when Ollama is not available."""
    q = question.lower()
    top   = scan.get("top_condition", "the detected condition")
    conf  = scan.get("confidence", 0) * 100
    triage = scan.get("triage", "UNKNOWN")
    findings = scan.get("findings", {})

    CONDITION_PLAIN = {
        "Pneumonia": {
            "simple": "The lungs show signs of infection. Pneumonia happens when germs (bacteria or viruses) cause the air sacs in the lungs to fill with fluid or pus, making it harder to breathe.",
            "symptoms": "The most common signs are fever, a cough that brings up mucus, shortness of breath, chest pain when breathing, and feeling very tired.",
            "treatment": "Bacterial pneumonia is treated with antibiotics. Viral pneumonia is managed with rest, fluids, and sometimes antivirals. Severe cases need hospital care and oxygen support.",
            "doctor": "Yes — pneumonia can worsen quickly. Please see a doctor today, especially if you have difficulty breathing or a high fever.",
        },
        "Cardiomegaly": {
            "simple": "The heart appears larger than normal. This is not a disease itself but a sign that the heart is working harder than it should — often due to high blood pressure or a heart condition.",
            "symptoms": "You might feel short of breath, tired easily, notice swelling in your legs or ankles, or feel your heart racing or pounding.",
            "treatment": "Treatment depends on the cause. It usually includes medications (for blood pressure or heart failure), lifestyle changes like reducing salt and exercise, and sometimes procedures.",
            "doctor": "Yes — an enlarged heart needs to be assessed by a cardiologist. Please make an appointment as soon as possible.",
        },
        "Effusion": {
            "simple": "There is fluid collecting in the space between the lungs and the chest wall. Think of it like water building up in a sealed bag around the lungs — it can press on the lungs and make breathing harder.",
            "symptoms": "Common signs are sharp chest pain (especially when breathing in), a dry cough, and feeling short of breath, particularly when lying flat.",
            "treatment": "Small amounts of fluid may clear on their own. Larger amounts may need a procedure to drain the fluid. The underlying cause (infection, heart issue, or other) is also treated.",
            "doctor": "Yes — pleural effusion should always be evaluated by a doctor. Please seek medical attention soon.",
        },
        "Infiltration": {
            "simple": "There are patches in the lungs where fluid, cells, or other material has collected in the air spaces. This can be a sign of infection, inflammation, or fluid from the bloodstream leaking into the lungs.",
            "symptoms": "Symptoms often include cough, fever, breathlessness, and lower oxygen levels in the blood.",
            "treatment": "Treatment depends on the cause — antibiotics for infection, diuretics for fluid overload, or anti-inflammatory medication for other causes.",
            "doctor": "Yes — lung infiltrates need medical evaluation to find the cause. Please see a doctor promptly.",
        },
        "No Finding": {
            "simple": "The chest X-ray looks normal. No signs of infection, fluid, enlarged heart, or other major abnormalities were detected.",
            "symptoms": "No specific symptoms are associated with a normal scan result.",
            "treatment": "No treatment is needed based on this scan alone.",
            "doctor": "This is a reassuring result, but if you have ongoing symptoms, still see your doctor — a scan is one piece of the picture.",
        },
    }

    info = CONDITION_PLAIN.get(top, {})
    triage_urgency = {
        "HIGH": "This is a HIGH priority finding. Please see a doctor today.",
        "MEDIUM": "This is a MEDIUM priority finding. Please see a doctor within 1–2 days.",
        "LOW": "This is a LOW priority finding. A routine check-up is advisable.",
    }.get(triage, "Please consult a doctor.")

    # Detect if the user is asking about a specific condition in findings
    for cond in findings:
        if cond.lower() in q and cond != top:
            alt_info = CONDITION_PLAIN.get(cond, {})
            return (
                f"You asked about {cond} specifically. "
                f"{alt_info.get('simple', '')} "
                f"The model detected it at {findings[cond]*100:.1f}% probability. "
                f"Please consult your doctor for a complete assessment."
            )

    if any(w in q for w in ["simple", "explain", "layman", "plain", "understand", "what is", "what does", "mean"]):
        return f"{info.get('simple', f'The scan detected {top}.')} Confidence: {conf:.1f}%. {triage_urgency}"

    if any(w in q for w in ["symptom", "sign", "feel", "experience"]):
        return f"{info.get('symptoms', 'Please consult your doctor about symptoms.')} Always consult a doctor if symptoms concern you."

    if any(w in q for w in ["treat", "medicine", "cure", "drug", "therapy", "management"]):
        return f"{info.get('treatment', 'Treatment depends on clinical assessment.')} Your doctor will advise the right plan for your situation."

    if any(w in q for w in ["doctor", "hospital", "should i", "go", "visit", "asap", "urgent", "serious", "dangerous", "worried", "grandfather", "grandmother", "parent", "family", "relative"]):
        return f"{info.get('doctor', triage_urgency)} {triage_urgency} This AI tool cannot replace a physician's judgement — a clinical exam is essential."

    if any(w in q for w in ["cause", "why", "reason", "smoking", "habit", "lifestyle", "risk factor"]):
        causes = {
            "Pneumonia": "Pneumonia can be caused by bacteria, viruses, or fungi. Risk factors include smoking, older age, weakened immunity, and recent illness.",
            "Cardiomegaly": "An enlarged heart can be caused by high blood pressure, heart valve problems, cardiomyopathy, or long-term heavy alcohol use.",
            "Effusion": "Pleural effusion can be caused by pneumonia, heart failure, kidney disease, cancer, or tuberculosis.",
            "Infiltration": "Lung infiltrates are commonly caused by pneumonia, pulmonary oedema, aspiration, or inflammatory conditions.",
            "No Finding": "No abnormality was detected in this scan.",
        }
        return f"{causes.get(top, 'Causes vary and require clinical investigation.')} Please discuss your specific risk factors with your doctor."

    if any(w in q for w in ["summarize", "summary", "report", "overview", "brief"]):
        all_findings = ", ".join(f"{c}: {p*100:.1f}%" for c, p in sorted(findings.items(), key=lambda x: -x[1]))
        return (
            f"Scan summary: The primary finding is {top} ({conf:.1f}% confidence), "
            f"with triage level {triage}. All findings — {all_findings}. "
            f"{triage_urgency} Please have this reviewed by a radiologist."
        )

    # Generic fallback
    return (
        f"The scan detected {top} with {conf:.1f}% confidence (triage: {triage}). "
        f"{info.get('simple', '')} {triage_urgency} "
        f"Feel free to ask about symptoms, treatment, causes, or whether to see a doctor."
    )
