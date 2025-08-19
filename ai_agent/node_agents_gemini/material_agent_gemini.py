# material_agent_gemini.py  (Google Gemini, single-image; EN output, "mixed" only when clear)
import os, io, json, base64
from typing import Tuple, Dict, Optional, List

from PIL import Image
from dotenv import load_dotenv

# Google Generative AI SDK
# pip install google-generativeai
import google.generativeai as genai

load_dotenv()

def _img_bytes_to_b64jpeg(img_bytes: bytes, max_side: int = 1024) -> str:
    """bytes → RGB → resize → JPEG → base64"""
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
      material: "plastic|metal|wood|fabric|silicone|rubber|paper_cardboard|electronic|mixed|unknown"
      components: ["plastic","metal"]  # material == "mixed"일 때만
      secondary_hint: 약한 2순위 후보 또는 null
      confidence: 0.0~1.0
      material_detail: 짧은 이유
      notes: 선택
    """
    def __init__(self):
        # 모델명 고정
        self.model = "gemini-2.5-flash-lite"

        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("환경변수 GEMINI_API_KEY (또는 GOOGLE_API_KEY)가 필요합니다.")
        genai.configure(api_key=api_key)

        # 단일 이미지 기준 프롬프트
        self.prompt = """
You are a MATERIAL classifier for toys. ONE product image is provided.

Return STRICT JSON ONLY (no markdown, no prose):
{
  "material": "plastic|metal|wood|fabric|silicone|rubber|paper_cardboard|electronic|mixed|unknown",
  "components": ["plastic","metal"],           // only when material=="mixed", 2–3 items max
  "secondary_hint": "plastic|metal|wood|fabric|silicone|rubber|paper_cardboard|electronic|null",
  "confidence": 0.0,                           // 0.0–1.0 overall confidence
  "material_detail": "...",                    // short reason (<= 120 chars)
  "notes": "..."                               // optional short notes
}

Guidelines:
- OUTPUT IN ENGLISH.
- Be CONSERVATIVE with "mixed": choose "mixed" ONLY when you clearly see at least two distinct materials across different parts/surfaces
  (not just reflections, paint, decals, or lighting artifacts). Require solid visual cues (texture, seams, shine, deformation).
- If you suspect a second material but evidence is weak, DO NOT output "mixed". Instead:
  - choose the single DOMINANT material for "material"
  - and set "secondary_hint" to the weak candidate (otherwise null).
- Prefer "plastic" for typical injection-molded toy bodies unless strong evidence contradicts.
- "fabric" includes cloth/textiles; "paper_cardboard" includes paper, carton; "electronic" is for visible PCBs/solder/ICs dominating the object.
- Keep responses short. STRICT JSON only.
""".strip()

        # 모델 핸들
        self._model = genai.GenerativeModel(self.model)

    def analyze(self, image_bytes: bytes) -> Tuple[str, Dict]:
        if image_bytes is None:
            raise RuntimeError("한 장의 이미지가 필요합니다.")

        b64 = _img_bytes_to_b64jpeg(image_bytes)
        img_part = {"mime_type": "image/jpeg", "data": base64.b64decode(b64)}

        try:
            resp = self._model.generate_content(
                [self.prompt, img_part],
                generation_config={"temperature": 0.0},
                safety_settings=None,  # 필요시 정책 세팅
            )

            # usage 메타 (SDK 버전에 따라 필드명이 다를 수 있어 방어적 처리)
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
            usage = {"total_tokens": total or 0}

            # 텍스트 꺼내기
            raw = (getattr(resp, "text", None) or "").strip()

            # ```json 래핑 제거
            if raw.startswith("```"):
                raw = raw.strip("`")
                if raw.lstrip().lower().startswith("json"):
                    raw = raw.split("\n", 1)[1] if "\n" in raw else ""

            # 파싱 & 필드 보정
            try:
                parsed = json.loads(raw) if raw else {}
            except Exception:
                parsed = {}

            material = (parsed.get("material") or "unknown").strip().lower()
            components = parsed.get("components") if isinstance(parsed.get("components"), list) else []
            if material != "mixed":
                components = []
            parsed["material"] = material
            parsed["components"] = components
            parsed["secondary_hint"] = parsed.get("secondary_hint", None)
            parsed["confidence"] = float(parsed.get("confidence", 0.0) or 0.0)
            parsed["material_detail"] = parsed.get("material_detail", "") or ""
            parsed["notes"] = parsed.get("notes", "") or ""

            out_json = json.dumps(parsed, ensure_ascii=False)
            print(f"MaterialAgent(Gemini) 응답: {out_json} | tokens={usage['total_tokens']}")
            return out_json, usage

        except Exception as e:
            print(f"MaterialAgent(Gemini) 에러: {e}")
            fallback = {
                "material": "unknown",
                "components": [],
                "secondary_hint": None,
                "confidence": 0.0,
                "material_detail": "fallback",
                "notes": ""
            }
            return json.dumps(fallback, ensure_ascii=False), {"total_tokens": 0}
