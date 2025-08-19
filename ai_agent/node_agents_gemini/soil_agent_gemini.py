# soil_agent_gemini.py  (single-image absolute grading; E-grade enabled + parse fix; Google Gemini 2.5 Flash)
import os, io, json, base64
from typing import Tuple, Dict, Optional, List

from PIL import Image
from dotenv import load_dotenv

# Google Generative AI SDK
# pip install google-generativeai
import google.generativeai as genai

load_dotenv()

# 라벨 → 절대점수: 4(깨끗) → 0(매우 더러움)
_SOIL_LABEL_TO_ABS = {
    "깨끗": 4,
    "보통": 3,
    "약간 더러움": 2,
    "더러움": 1,
    "매우 더러움": 0,  # E 가능
}

# '더러움'인데 심각 오염 신호가 있으면 0으로 강등
_SEVERE_DIRT_KWS = [
    "끈적", "sticky", "곰팡", "mold", "기름때", "heavy stain", "찌든때",
    "기름", "oil stain", "진흙이 두껍게", "mud caked"
]

def _img_bytes_to_b64jpeg(img_bytes: bytes, max_side: int = 1024) -> str:
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

def _label_to_abs(s: Optional[str]) -> int:
    t = (s or "").strip()
    for k, v in _SOIL_LABEL_TO_ABS.items():
        if k in t:
            return v
    return 2  # 알 수 없음 → 중간값

def _score_to_grade_single(score: int) -> str:
    s = max(0, min(4, int(score)))
    return {4: "A", 3: "B", 2: "C", 1: "D", 0: "E"}[s]

class SoilAgent:
    """
    입력: 단일 이미지 bytes
    반환(최소):
      {
        "overall": {"summary": "<ko one line>"},
        "absolute_score_avg": <float 0..4>,
        "absolute_grade": "A"|"B"|"C"|"D"|"E"
      }
    """
    def __init__(self):
        self.model = "gemini-2.5-flash-lite"

        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("환경변수 GEMINI_API_KEY (또는 GOOGLE_API_KEY)가 필요합니다.")
        genai.configure(api_key=api_key)

        self.prompt = """
You are a **precise toy CLEANLINESS (soil/dirt) judge** for a single image.

Return **STRICT JSON ONLY** (no extra text):
{
  "overall": {"summary":"(Korean, one line)"},
  "soil":"깨끗|보통|약간 더러움|더러움|매우 더러움",
  "notes":"(Korean, short)"
}
Guidelines:
- '깨끗' = 거의 오염 없음. '보통' = 약간의 사용흔적.
- '약간 더러움' = 눈에 띄는 얼룩/먼지/마모. '더러움' = 넓은 면적의 얼룩/때.
- '매우 더러움' = 끈적임/곰팡이/기름때/두꺼운 오염 등 심각 상태.
- 얼굴/데칼/밝은 플라스틱/패브릭은 민감 영역으로 가중.
- 크롭/가림이 있어도 보이는 면을 근거로 판단. **'판단 불가' 금지**.
- JSON only. No extra text.
""".strip()

        self._model = genai.GenerativeModel(self.model)

    def analyze(self, image_bytes: Optional[bytes]) -> Tuple[dict, dict]:
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
        except Exception as e:
            raise RuntimeError(f"Gemini API 호출 실패: {e}")

        # usage (SDK 버전별 필드 차이를 방어적으로 처리)
        meta = getattr(resp, "usage_metadata", None)
        in_tok  = getattr(meta, "prompt_token_count", None)
        out_tok = getattr(meta, "candidates_token_count", None)
        total   = getattr(meta, "total_token_count", None)
        if in_tok is None:
            in_tok = getattr(meta, "input_tokens", 0) if meta else 0
        if out_tok is None:
            out_tok = getattr(meta, "output_tokens", 0) if meta else 0
        if total is None:
            total = (in_tok or 0) + (out_tok or 0)

        usage = {
            "input_tokens": in_tok or 0,
            "output_tokens": out_tok or 0,
            "total_tokens": total or 0,
        }

        # 응답 텍스트 → JSON 파싱
        raw = (getattr(resp, "text", None) or "").strip()
        if raw.startswith("```"):
            raw = raw.strip("`")
            if raw.lstrip().lower().startswith("json"):
                raw = raw.split("\n", 1)[1] if "\n" in raw else ""

        try:
            base = json.loads(raw)
        except Exception:
            base = {
                "overall": {"summary": "자동 판정: 파싱 실패로 보통으로 가정"},
                "soil": "보통",
                "notes": "파싱 실패 기본값"
            }

        summary    = (base.get("overall") or {}).get("summary") or ""
        soil_label = base.get("soil") or "보통"
        notes      = (base.get("notes") or "").lower()

        soil_abs = _label_to_abs(soil_label)  # 0..4

        # '더러움'(1점)인데 심각 키워드가 있으면 0으로 강등
        if soil_abs == 1 and any(k in notes for k in _SEVERE_DIRT_KWS):
            soil_abs = 0

        avg   = float(soil_abs)
        grade = _score_to_grade_single(soil_abs)

        result = {
            "overall": base.get("overall", {"summary": summary}),
            "absolute_score_avg": avg,
            "absolute_grade": grade,
        }
        return result, usage
