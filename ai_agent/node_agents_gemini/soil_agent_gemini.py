# ai_agent/soil_agent.py
import os
import base64
import re
from typing import Optional
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

ALLOWED_SOIL = ["A", "B", "C", "D", "E"]

def _b64(path: Optional[str]) -> Optional[str]:
    """이미지를 base64 문자열로 변환"""
    if not path:
        return None
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def _extract_soil_letter(text: str) -> str:
    """Gemini 응답에서 soil 등급(A~E)만 추출"""
    # JSON 패턴 우선
    m = re.search(r'"soil"\s*:\s*"([A-E])"', text)
    if m:
        return m.group(1)

    # 혹시 한글로 들어오면 대응
    mapping = {
        "깨끗": "A",
        "보통": "B",
        "약간 더러움": "C",
        "더러움": "D",
        "매우 더러움": "E"
    }
    for k, v in mapping.items():
        if k in text:
            return v

    # 안전장치: 못 찾으면 A로
    return "A"

class SoilAgent:
    def __init__(self, model="gemini-2.5-flash-lite"):
        self.model = genai.GenerativeModel(model)
        self.prompt = """
You are a precise toy CLEANLINESS judge comparing two images:
- First = REFERENCE (new toy)
- Second = USED (possibly dirty toy)

Return STRICT JSON ONLY (no extra text):
{
  "soil": "A" | "B" | "C" | "D" | "E"
}

Guidelines:
- Grade the USED image relative to REFERENCE.
- Definitions:
  A = Clean (거의 오염 없음)
  B = Slightly used (약간 사용흔적)
  C = Noticeable dirt (얼룩 보임)
  D = Dirty (넓은 얼룩)
  E = Very dirty (심각한 오염)
- Output MUST be a single JSON object with one key "soil".
- Value MUST be a single capital letter A–E. No Korean text, no explanation.
""".strip()

    def analyze(self, ref_bytes: bytes, used_bytes: bytes) -> str:
        """SupervisorAgent에서 호출 (bytes 입력)"""
        if not (ref_bytes and used_bytes):
            raise ValueError("이미지 두 장 모두 필요합니다.")

        ref_b64 = base64.b64encode(ref_bytes).decode("utf-8")
        used_b64 = base64.b64encode(used_bytes).decode("utf-8")

        resp = self.model.generate_content(
            [
                {"role": "user", "parts": [
                    {"text": self.prompt},
                    {"inline_data": {"mime_type": "image/jpeg", "data": ref_b64}},
                    {"inline_data": {"mime_type": "image/jpeg", "data": used_b64}}
                ]}
            ],
            generation_config={"temperature": 0.0, "max_output_tokens": 20}
        )

        raw = resp.text.strip()
        return _extract_soil_letter(raw)

    def run(self, ref_path: str, used_path: str) -> str:
        """파일 경로 직접 입력받아 실행"""
        ref_b64 = _b64(ref_path)
        used_b64 = _b64(used_path)
        if not (ref_b64 and used_b64):
            raise ValueError("이미지 두 장 모두 필요합니다.")

        resp = self.model.generate_content(
            [
                {"role": "user", "parts": [
                    {"text": self.prompt},
                    {"inline_data": {"mime_type": "image/jpeg", "data": ref_b64}},
                    {"inline_data": {"mime_type": "image/jpeg", "data": used_b64}}
                ]}
            ],
            generation_config={"temperature": 0.0, "max_output_tokens": 20}
        )

        raw = resp.text.strip()
        return _extract_soil_letter(raw)

if __name__ == "__main__":
    agent = SoilAgent()
    ref  = "test/ref.jpg"
    used = "test/used.jpg"
    out = agent.run(ref, used)
    print(out)   # 항상 "A"~"E" 중 하나 출력
