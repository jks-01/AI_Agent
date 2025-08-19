# ai_agent/image_input.py  (터미널 전용, Streamlit 의존성 없음)
import io
from PIL import Image

def optimize_image_size(img_bytes: bytes, max_side: int = 1024, quality: int = 92) -> bytes:
    """바이트 이미지 → RGB → 리사이즈 → JPEG 바이트로 최적화"""
    try:
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except Exception:
        img = Image.new("RGB", (256, 256), "white")
    w, h = img.size
    scale = min(1.0, max_side / max(w, h))
    if scale < 1.0:
        img = img.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()
