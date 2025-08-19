# type_agent_gemini.py  (Google Gemini, single-image; returns ENGLISH categories via gemini-2.5-flash)
import os, io, json, base64
from dotenv import load_dotenv
from PIL import Image

# Google Generative AI SDK
# pip install google-generativeai
import google.generativeai as genai

load_dotenv()

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

class TypeAgent:
    def __init__(self):
        # 모델명 고정
        self.model = "gemini-2.5-flash-lite"

        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("환경변수 GEMINI_API_KEY (또는 GOOGLE_API_KEY)가 필요합니다.")
        genai.configure(api_key=api_key)

        self.system_prompt = "Return STRICT JSON only. No extra text."
        self.user_prompt = """
You are a toy category expert. Look at ONE product image and output STRICT JSON only.

Return EXACTLY:
{
  "type": "robot|building_blocks|dolls|vehicles|educational|action_figures|board_games|musical|sports|others",
  "battery": "battery|non-battery|unknown",
  "size": "small|medium|large|unknown"
}

Rules & mapping:
- If text suggests 'figure', '모형', '피규어', '프라모델' → map to "action_figures".
- Vehicles (car/tank/plane/train/boat) → "vehicles".
- Blocks/LEGO-like/bricks → "building_blocks".
- Dolls/plush/인형류 → "dolls".
- Obvious learning aids (alphabet/shape sorter) → "educational".
- Musical instruments/toys → "musical".
- Board/box games → "board_games".
- Sports/balls/rackets → "sports".
- Anything else → "others".
- Battery means electronic power required (lights/sound/motors) → "battery". If clearly manual → "non-battery". Else "unknown".
- Size is visual/relative: handheld figure/mini car → small; typical 20–30cm figure/vehicle → medium; very large/ride-on/base > ~30cm → large.
- STRICT JSON only. No extra text.
""".strip()

        # 모델 핸들
        self._model = genai.GenerativeModel(self.model)

    def analyze(self, image_bytes: bytes):
        """
        단일 이미지로 장난감 유형/배터리/크기 추정.
        반환: (STRICT JSON string, token_info)
        """
        try:
            b64 = _img_bytes_to_b64jpeg(image_bytes)
            img_part = {"mime_type": "image/jpeg", "data": base64.b64decode(b64)}

            resp = self._model.generate_content(
                [self.system_prompt + "\n\n" + self.user_prompt, img_part],
                generation_config={"temperature": 0.0},
                safety_settings=None,  # 필요시 정책 세팅
            )

            raw = (getattr(resp, "text", None) or "").strip()

            # ```json 래핑 제거
            if raw.startswith("```"):
                raw = raw.strip("`")
                if raw.lstrip().lower().startswith("json"):
                    raw = raw.split("\n", 1)[1] if "\n" in raw else ""

            # 최소 검증: 비어있으면 기본값
            result = raw.strip() or '{"type":"others","battery":"unknown","size":"unknown"}'

            # usage 메타 (SDK 버전에 따라 필드명이 다를 수 있어 방어적 처리)
            meta = getattr(resp, "usage_metadata", None)
            total = getattr(meta, "total_token_count", None)
            if total is None:
                total = (getattr(meta, "prompt_token_count", 0) or 0) + (getattr(meta, "candidates_token_count", 0) or 0)

            token_info = {"total_tokens": total or 0}

            return result, token_info

        except Exception as e:
            print(f"TypeAgent error: {e}")
            return '{"type":"others","battery":"unknown","size":"unknown"}', {"total_tokens": 0}
