# material_agent_gemini.py  (Google Gemini, single-image; EN output, "plastic" 등만)
import os, io, base64
from typing import Optional
from PIL import Image
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

def _img_bytes_to_b64jpeg(img_bytes: bytes, max_side: int = 1024) -> str:
    """bytes → RGB → resize → JPEG → base64 string"""
    try:
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except Exception:
        img = Image.new("RGB", (256, 256), color="white")
    w, h = img.size
    scale = min(1.0, max_side / max(w, h))
    if scale < 1.0:
        img = img.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=92)
    return base64.b64encode(buf.getvalue()).decode("utf-8")

class MaterialAgent:
    """
    결과는 영어로:
      "plastic" | "metal" | "wood" | "fabric" | "silicone" | "rubber" | "paper_cardboard" | "electronic" | "mixed" | "unknown"
    """
    def __init__(self):
        self.model = "gemini-2.5-flash-lite"
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("환경변수 GEMINI_API_KEY (또는 GOOGLE_API_KEY)가 필요합니다.")
        genai.configure(api_key=api_key)

        # ✅ 출력은 반드시 소재명 하나만
        self.prompt = """
You are a MATERIAL classifier for toys.
One toy image will be provided.

Answer ONLY with the dominant material in English:
plastic | metal | wood | fabric | silicone | rubber | paper_cardboard | electronic | mixed | unknown

- Do NOT include JSON, confidence, notes, or extra words.
- Output must be exactly one word from the list above.
""".strip()

        self._model = genai.GenerativeModel(self.model)

    def analyze(self, image_bytes: bytes) -> str:
        if image_bytes is None:
            raise RuntimeError("한 장의 이미지가 필요합니다.")

        b64 = _img_bytes_to_b64jpeg(image_bytes)
        img_part = {"mime_type": "image/jpeg", "data": base64.b64decode(b64)}

        try:
            resp = self._model.generate_content(
                [self.prompt, img_part],
                generation_config={"temperature": 0.0},
                safety_settings=None,
            )

            text = (getattr(resp, "text", None) or "").strip().lower()

            # 혹시 "material: plastic" 같은 출력이 오면 정리
            for m in ["plastic","metal","wood","fabric","silicone","rubber","paper_cardboard","electronic","mixed","unknown"]:
                if m in text:
                    return m

            return "unknown"

        except Exception as e:
            print(f"MaterialAgent(Gemini) 에러: {e}")
            return "unknown"
