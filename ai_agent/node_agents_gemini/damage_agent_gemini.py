# damage_agent_gemini.py (single-image; ultra-lenient; **NO-CLIP**; Google Gemini 2.5 Flash)
# - 구조/리턴은 기존 그대로, 내부 호출만 Google Generative AI로 변경
# - Only core-part loss can yield D/E; otherwise >= C

import os, io, json, base64
from typing import Tuple, Dict, Optional, List

from PIL import Image
from dotenv import load_dotenv

# Google Generative AI SDK
# pip install google-generativeai  (또는 최신 SDK: pip install google-genai)
import google.generativeai as genai

load_dotenv()

ALLOWED_DAMAGE = ["없음","미세한 파손","경미한 파손","부품 누락","심각한 파손"]

# ---------- helpers ----------
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

def _damage_to_abs(label: Optional[str]) -> int:
    t = str(label or "").strip()
    if "없음" in t: return 4
    if "미세" in t: return 3
    if "경미" in t: return 2
    if "부품 누락" in t: return 1
    if "심각" in t: return 0
    return 2

def _normalize_label(label: Optional[str]) -> str:
    t = str(label or "").strip()
    if "판단" in t or "불명" in t or not t:
        return "경미한 파손"
    for k in ALLOWED_DAMAGE:
        if k in t:
            return k
    tl = t.lower()
    if any(w in tl for w in ["missing","detach","누락","분리","빠짐","떨어짐"]): return "부품 누락"
    if any(w in tl for w in ["crack","break","broken","깨짐","금","균열","부러짐"]): return "심각한 파손"
    if any(w in tl for w in ["scratch","scuff","scrape","스크래치","마모","흠집","칠 벗","페인트"]): return "경미한 파손"
    return "경미한 파손"

def _score_to_grade_single(score: int) -> str:
    s = max(0, min(4, int(score)))
    return {4:"A", 3:"B", 2:"C", 1:"D", 0:"E"}[s]

# ---------- prompt ----------
_GENERIC_SUFFIX = """
Return STRICT JSON ONLY (no extra text):
{
  "overall":  {"summary":"(Korean, one line)"},
  "checklist": {
    "arm_left":  "present|missing|detached|unknown",
    "arm_right": "present|missing|detached|unknown",
    "leg_left":  "present|missing|detached|unknown",
    "leg_right": "present|missing|detached|unknown",
    "head":      "present|missing|detached|unknown",
    "wheel_missing": true|false
  },
  "counts": { "arms_visible": 0, "legs_visible": 0, "wheels_visible": 0 },
  "confidence": { "arm_left":0.0, "arm_right":0.0, "leg_left":0.0, "leg_right":0.0, "head":0.0 },
  "damage":"없음|미세한 파손|경미한 파손|부품 누락|심각한 파손",
  "missing_parts": true|false,
  "notes":"(Korean, short)"
}

Guidelines:
- **Accessories do NOT count as missing parts** (weapons, shields, decorative side panels, stickers/decals, backpacks, boosters).
- **Core parts only**:
  - Humanoid/figure: arms, legs, head clearly attached to torso/hip.
  - Vehicle: wheels/tracks intact; hull/panels not split; no large gap exposing interior.
  - Animal-like: legs, head, tail, wings, horns.
- If a core part is absent and the joint/seat is visible with no occluder, mark the side as **missing/detached** and set damage="부품 누락".
- If the body is obviously **split** exposing interior, treat as **심각한 파손**.
- When unsure but the joint region is fully visible with no limb mass, prefer **missing** over unknown.
""".strip()

class DamageAgent:
    """
    입력: 단일 이미지 bytes
    반환: (result_dict, usage_dict)
      result_dict = {
        "overall": {"summary": "..."},
        "absolute_score_avg": float,
        "absolute_grade": "A"|"B"|"C"|"D"|"E"
      }
    """
    def __init__(self):
        # 모델명 고정
        self.model = "gemini-2.5-flash-lite"

        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("환경변수 GEMINI_API_KEY (또는 GOOGLE_API_KEY)가 없습니다.")
        genai.configure(api_key=api_key)

        self.prompt = """
You are a **precise but LENIENT** toy DAMAGE judge for a single image.

Policy:
- Only when a **core body part** is truly absent or the **body is clearly split apart** may the grade drop to D/E.
- Missing accessories (weapons, shields, decorative panels, stickers, backpacks, boosters) must NOT reduce the grade.
- Otherwise, never grade below **C**. Minor scuffs/scratches = **B~C**.

Output JSON only.
""".strip()

        # 한 번만 모델 핸들 생성
        self._model = genai.GenerativeModel(self.model)

    def _build_prompt(self) -> str:
        return self.prompt + "\n\n" + _GENERIC_SUFFIX

    def analyze(self, image_bytes: Optional[bytes]) -> Tuple[dict, dict]:
        if image_bytes is None:
            raise RuntimeError("한 장의 이미지가 필요합니다.")

        image_b64 = _img_bytes_to_b64jpeg(image_bytes)
        img_part = {
            "mime_type": "image/jpeg",
            "data": base64.b64decode(image_b64),
        }

        try:
            resp = self._model.generate_content(
                [self._build_prompt(), img_part],
                generation_config={"temperature": 0.0},
                safety_settings=None,  # 필요시 정책 세팅
            )
        except Exception as e:
            raise RuntimeError(f"Gemini API 호출 실패: {e}")

        # usage 메타 파싱 (SDK 버전에 따라 필드명이 다를 수 있으므로 방어적으로 처리)
        meta = getattr(resp, "usage_metadata", None)
        in_tok  = getattr(meta, "prompt_token_count", None)
        out_tok = getattr(meta, "candidates_token_count", None)
        total   = getattr(meta, "total_token_count", None)

        # 대체 필드명 (신규 SDK 호환)
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

        # 응답 텍스트
        raw = (getattr(resp, "text", None) or "").strip()
        if raw.startswith("```"):
            raw = raw.strip("`")
            if raw.lstrip().lower().startswith("json"):
                raw = raw.split("\n", 1)[1] if "\n" in raw else ""

        # JSON 파싱 & fallback
        try:
            base = json.loads(raw)
        except Exception:
            base = {
                "overall": {"summary": "자동 판정: 파싱 실패로 경미한 파손으로 가정"},
                "checklist": {
                    "arm_left":"present","arm_right":"present",
                    "leg_left":"present","leg_right":"present",
                    "head":"present","wheel_missing":False
                },
                "damage":"경미한 파손",
                "missing_parts": False,
                "notes":"fallback"
            }

        checklist = base.get("checklist") or {}
        summary   = (base.get("overall") or {}).get("summary") or ""
        damage    = _normalize_label(base.get("damage"))

        # Only core-part loss allows D/E
        def _is_missing(x: Optional[str]) -> bool:
            s = str(x or "").lower()
            return ("missing" in s) or ("detached" in s)

        limb_missing  = any(_is_missing(checklist.get(k)) for k in ["arm_left","arm_right","leg_left","leg_right","head"])
        wheel_missing = bool(checklist.get("wheel_missing", False))
        label_severe  = ("심각" in damage)

        severe = label_severe or limb_missing or wheel_missing

        base_score = _damage_to_abs(damage)  # 0..4
        if severe:
            per_image_score = min(base_score, 1 if not label_severe else 0)  # allow D/E
        else:
            per_image_score = max(base_score, 2)  # never below C

        grade = _score_to_grade_single(per_image_score)
        result = {
            "overall": base.get("overall", {"summary": summary}),
            "absolute_score_avg": float(per_image_score),
            "absolute_grade": grade,
        }
        return result, usage
