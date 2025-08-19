# ai_agent_output.py
import os
import time
import json
import argparse
from typing import Optional, Dict, Any
from ai_similarity.similarity import search_by_image_name
from ai_agent.supervisor_agent import SupervisorAgent
# -----------------------
# 유틸
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
def _round_to_unit(x: float, unit: int = 100) -> int:
    return int(round(x / unit) * unit)
# -----------------------
# A~E 등급 → 보정계수
# -----------------------
def _grade_to_multiplier(grade: Optional[str]) -> float:
    """
    A→1.00, B→0.95, C→0.90, D→0.80, E→0.65
    알 수 없으면 C(0.90)로 처리
    """
    g = (grade or "").strip().upper()
    table = {"A": 1.00, "B": 0.95, "C": 0.90, "D": 0.80, "E": 0.65}
    return table.get(g, 0.90)
def _compute_purchase_price(
    baseline_used_price: Optional[int],
    *,
    damage_grade: Optional[str],
    soil_grade: Optional[str],
    base_bias: float = 0.7,
    round_unit: int = 100,
) -> Optional[int]:
    """
    구매가 = baseline_used_price x base_bias x damage_mult x soil_mult
    """
    if baseline_used_price is None:
        return None
    dmg_m = _grade_to_multiplier(damage_grade)
    soil_m = _grade_to_multiplier(soil_grade)
    raw = float(baseline_used_price) * float(base_bias) * dmg_m * soil_m
    return _round_to_unit(raw, round_unit)
# -----------------------
# 메인 파이프라인
# -----------------------
def run_full_pipeline(used_path: str, *, base_bias: float = 0.7) -> Dict[str, Any]:
    """
    1) predict.run_similarity_brief 로 ref 후보(Top-1) 찾기
    2) used/ref 이미지를 supervisor_agent 로 전달 → soil, damage, type, material 분석
    3) price info + condition 결과 합쳐서 최종 JSON 반환
    """
    # --- (1) Top-1 ref 찾기 ---
    sim_info = search_by_image_name(used_path, return_results=True)
    print(sim_info)
    # 기대 키: similar_image_path, similar_toy_name, similar_retail_price, similar_used_price
    ref_path = sim_info.get("similar_image_path")
    # --- (2) 이미지 로드 ---
    used_b = _read_bytes_or_none(os.path.join("test", used_path))
    ref_b  = _read_bytes_or_none(ref_path)
    #print(f'ref_path: {ref_path},')
    # --- (3) 상태 분석 ---
    sup = SupervisorAgent()
    # 시그니처가 process(ref_b, used_b) 라면 아래처럼 호출
    condition_result = sup.process(ref_b, used_b)
    # 기대 결과 예시: {"soil":"B","damage":"C","type":"robot","material":"plastic"}
    # --- (3.5) 구매가 계산 ---
    retail_price = _to_int_or_none(sim_info.get("similar_retail_price"))
    baseline_used = _to_int_or_none(sim_info.get("similar_used_price"))  # 기준 중고가
    soil_grade = condition_result.get("soil")
    damage_grade = condition_result.get("damage")
    purchase_price = _compute_purchase_price(
        baseline_used_price=baseline_used,
        damage_grade=damage_grade,
        soil_grade=soil_grade,
        base_bias=base_bias,
        round_unit=100,   # 100원 단위 반올림
    )
    # --- (4) 최종 JSON ---
    out = {
        "toy_name": sim_info.get("similar_toy_name"),
        "retail_price": retail_price,
        "purchase_price": purchase_price,
        "soil": soil_grade if soil_grade is not None else "unknown",
        "damage": damage_grade if damage_grade is not None else "unknown",
        "toy_type": condition_result.get("type", "unknown"),
        "material": condition_result.get("material", "unknown"),
    }
    return out
# -----------------------
# CLI 실행
# -----------------------
def main():
    t0 = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument("--used", type=str, required=True, help="중고 이미지 경로")
    parser.add_argument("--base-bias", type=float, default=0.7, help="기본 배수 (default=0.7)")
    args = parser.parse_args()
    result = run_full_pipeline(args.used, base_bias=args.base_bias)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    total_time = time.time() - t0
    print(f"🎯 총 소요 시간: {total_time:.3f}초")
if __name__ == "__main__":
    main()