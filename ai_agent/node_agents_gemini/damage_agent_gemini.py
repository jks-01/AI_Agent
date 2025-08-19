# ai_agent/node_agents_gemini/damage_agent_gemini.py
# - Compare REF(new) vs USED in one call (Gemini 2.5 Flash/Lite)
# - Output ONLY grade (A~E)

import os, io, json, base64, time
from PIL import Image
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

# ----------------- helpers -----------------
def _img_to_b64jpeg(path_or_file, max_side: int = 1024) -> str:
    try:
        img = Image.open(path_or_file).convert("RGB")
    except Exception:
        img = Image.new("RGB", (256, 256), "white")
    w, h = img.size
    s = min(1.0, max_side / max(w, h))
    if s < 1.0:
        img = img.resize((int(w*s), int(h*s)), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode("utf-8")

# ----------------- prompt -----------------
_PROMPT = """
You are a precise toy DAMAGE judge.
Compare two images: REF (new toy) vs USED (secondhand toy).
Decide ONLY the USED toy's damage grade (A-E):

- A = 없음 (no damage)
- B = 미세한 파손 (very minor damage)
- C = 경미한 파손 (minor damage)
- D = 부품 누락 (part missing)
- E = 심각한 파손 (severe damage)

Return STRICT JSON ONLY:
{"grade":"A" | "B" | "C" | "D" | "E"}

No explanations. JSON only.
""".strip()

# ----------------- DamageAgent class -----------------
class DamageAgent:
    def __init__(self, model: str = "gemini-2.5-flash-lite"):
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        self.client = genai.GenerativeModel(model)
        self.model = model

    def analyze(self, ref_image_bytes: bytes, used_image_bytes: bytes) -> str:
        """Compare REF vs USED images -> return grade (A~E, string only)"""
        ref_b64 = base64.b64encode(ref_image_bytes).decode("utf-8")
        used_b64 = base64.b64encode(used_image_bytes).decode("utf-8")

        resp = self.client.generate_content(
            contents=[{
                "role": "user",
                "parts": [
                    {"text": _PROMPT},
                    {"inline_data": {"mime_type": "image/jpeg", "data": ref_b64}},
                    {"inline_data": {"mime_type": "image/jpeg", "data": used_b64}},
                ]
            }],
            generation_config={"response_mime_type": "application/json"}
        )

        try:
            out = json.loads(resp.text.strip())
            grade = out.get("grade", "C")
        except Exception:
            grade = "C"  # fallback

        return grade
