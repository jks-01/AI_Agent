# ai_agent_output.py  (single input — absolute grades, no KR/EN mapping)
# Final JSON: toy_name, retail_price, purchase_price, soil, damage, toy_type, material, elapsed_ms

import os
import re
import json
import base64
import argparse
import tempfile
import atexit
import time  # 추가
from typing import List, Optional

# (1) Similarity pipeline (returns 4 fields)
from predict import run_similarity_brief   # IMAGE_PATH import 제거

# (2) Condition agent (absolute evaluation; ref not used)
from ai_agent.supervisor_agent import SupervisorAgent


# -----------------------
# utils
# -----------------------
def _read_bytes_or_none(p: Optional[str]) -> Optional[bytes]:
    if not p or not os.path.isfile(p):
        return None
    with open(p, "rb") as f:
        return f.read()

def _to_int_or_none(x) -> Optional[int]:
    if x is None:
        return None
    try:
        return int(float(str(x).replace(",", "").strip()))
    except Exception:
        return None


# -----------------------
# Absolute evaluation (single image)
# -----------------------
def run_condition_agents_absolute(image_path: str) -> dict:
    """
    Use only 1 user image. SupervisorAgent returns ENGLISH keys with absolute A~E grades:
      - damage_absolute_grade
      - soil_absolute_grade
      - toy_type
      - material
    """
    img_b = _read_bytes_or_none(image_path)

    if img_b is None:
        # No input at all: conservative defaults
        return {
            "toy_type": "others",
            "material": "unknown",
            "soil_absolute_grade": None,
            "damage_absolute_grade": None,
        }

    sup = SupervisorAgent()
    # 단일 입력만 전달 (SupervisorAgent.process(image_bytes) 형태여야 함)
    result = sup.process(img_b)

    return {
        "toy_type": result.get("toy_type", "others"),
        "material": result.get("material", "unknown"),
        "soil_absolute_grade": result.get("soil_absolute_grade"),
        "damage_absolute_grade": result.get("damage_absolute_grade"),
    }


# -----------------------
# Grade (A~E) → price multiplier
# -----------------------
def _grade_to_multiplier(grade: Optional[str]) -> float:
    """
    A→1.00, B→0.95, C→0.90, D→0.80, E→0.65
    """
    g = (grade or "").strip().upper()
    table = {"A": 1.00, "B": 0.95, "C": 0.90, "D": 0.80, "E": 0.65}
    return table.get(g, 0.90)  # unknown → treat as C


def _round_to_unit(x: float, unit: int = 100) -> int:
    return int(round(x / unit) * unit)


def estimate_purchase_price_by_grade(
    baseline_price: int,
    *,
    damage_grade: Optional[str],   # 'A'..'E' or None
    soil_grade: Optional[str],     # 'A'..'E' or None
    base_bias: float = 0.7
) -> int:
    dmg_mult  = _grade_to_multiplier(damage_grade)
    soil_mult = _grade_to_multiplier(soil_grade)
    raw = baseline_price * float(base_bias) * dmg_mult * soil_mult
    return int(_round_to_unit(raw, 100))


# -----------------------
# data URL support (optional, single only)
# -----------------------
_DATAURL_RE = re.compile(r"^data:(?P<mime>[\w\-/+\.]+)?;base64,(?P<b64>[A-Za-z0-9+/=]+)$")
_TEMP_FILES: List[str] = []

def dataurl_to_bytes(dataurl: Optional[str]) -> Optional[bytes]:
    if not dataurl:
        return None
    m = _DATAURL_RE.match(dataurl.strip())
    if not m:
        return None
    try:
        return base64.b64decode(m.group("b64"))
    except Exception:
        return None

def bytes_to_tempfile(b: bytes, suffix: str = ".jpg") -> str:
    f = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    f.write(b)
    f.flush()
    f.close()
    _TEMP_FILES.append(f.name)
    return f.name

def _cleanup_tempfiles():
    for p in _TEMP_FILES:
        try:
            os.remove(p)
        except Exception:
            pass
atexit.register(_cleanup_tempfiles)


# -----------------------
# Full pipeline (absolute grades end-to-end)
# -----------------------
def run_full_pipeline(
    image_path: str,
    *,
    base_bias: float = 0.7,
) -> dict:
    """
    1) predict.run_similarity_brief(image_path) → 4 fields (toy name and prices)
    2) absolute grades from single user image via SupervisorAgent
    3) purchase price from baseline used_avg and absolute grades
    4) final JSON (7 keys)
    """
    brief = run_similarity_brief(image_path=image_path)  # 입력 이미지 기반 Top-1
    toy_name = brief.get("similar_toy_name")
    retail   = _to_int_or_none(brief.get("similar_retail_price"))   # MSRP
    used_avg = _to_int_or_none(brief.get("similar_used_price"))     # baseline

    attrs = run_condition_agents_absolute(image_path=image_path)

    soil_grade = attrs.get("soil_absolute_grade")      # 'A'..'E'
    dmg_grade  = attrs.get("damage_absolute_grade")    # 'A'..'E'

    purchase_price = None
    if used_avg is not None:
        purchase_price = estimate_purchase_price_by_grade(
            baseline_price=used_avg,
            damage_grade=dmg_grade,
            soil_grade=soil_grade,
            base_bias=base_bias,
        )

    return {
        "toy_name": toy_name,
        "retail_price": retail,
        "purchase_price": int(purchase_price) if purchase_price is not None else None,
        "soil": soil_grade,        # 'A'..'E'
        "damage": dmg_grade,       # 'A'..'E'
        "toy_type": attrs.get("toy_type"),
        "material": attrs.get("material"),
    }


# -----------------------
# CLI
# -----------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Similarity(4-field) + absolute grading pipeline (single image, JSON)")
    # single file path (required OR dataurl)
    p.add_argument("--image", type=str, default=None, help="image path")             
    p.add_argument("--image-dataurl", type=str, default=None, help="image data URL")
    p.add_argument("--base-bias", type=float, default=0.7, help="purchase price baseline bias (default 0.7)")
    return p


def main():
    args = build_parser().parse_args()

    if not args.image and not args.image_dataurl:
        raise SystemExit("Error: provide --image <path> or --image-dataurl <dataurl>")

    t0 = time.perf_counter() 

    image_path = args.image
    if args.image_dataurl:
        b = dataurl_to_bytes(args.image_dataurl)
        if b:
            image_path = bytes_to_tempfile(b, ".jpg")

    out = run_full_pipeline(
        image_path=image_path,
        base_bias=args.base_bias,
    )

    # JSON: stdout
    print(json.dumps(out, ensure_ascii=False))

    # 경과시간: stderr, 초 단위로 표기
    elapsed_s = time.perf_counter() - t0
    os.write(2, f"[elapsed] {elapsed_s:.3f} s\n".encode())


if __name__ == "__main__":
    main()