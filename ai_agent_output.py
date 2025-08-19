# ai_agent_output.py
import os
import json
import argparse
from typing import Optional

from predict import run_similarity_brief
from ai_agent.supervisor_agent import SupervisorAgent


# -----------------------
# 유틸
# -----------------------
def _read_bytes_or_none(p: Optional[str]) -> Optional[bytes]:
    if not p:
        return None
    with open(p, "rb") as f:
        return f.read()


# -----------------------
# 메인 파이프라인
# -----------------------
def run_full_pipeline(used_path: str):
    """
    1) predict.run_similarity_brief 로 ref 후보(Top-1) 찾기
    2) used/ref 이미지를 supervisor_agent 로 전달 → soil, damage, type, material 분석
    3) price info + condition 결과 합쳐서 최종 JSON 반환
    """
    # --- (1) Top-1 ref 찾기 ---
    sim_info = run_similarity_brief(used_path)
    ref_path = sim_info["similar_image_path"]

    # --- (2) 이미지 로드 ---
    used_b = _read_bytes_or_none(used_path)
    ref_b = _read_bytes_or_none(ref_path)

    # --- (3) 상태 분석 ---
    sup = SupervisorAgent()
    condition_result = sup.process(ref_b, used_b)
    # 기대 결과: {"soil": "B", "damage": "C", "toy_type": "robot", "material": "plastic"}

    # --- (4) 최종 JSON 구조 맞추기 ---
    out = {
        "toy_name": sim_info["similar_toy_name"],
        "retail_price": sim_info["similar_retail_price"],
        "purchase_price": sim_info["similar_used_price"],   # used price = 구매가
        "soil": condition_result.get("soil", "unknown"),
        "damage": condition_result.get("damage", "unknown"),
        "toy_type": condition_result.get("type", "unknown"),
        "material": condition_result.get("material", "unknown")
    }
    return out


# -----------------------
# CLI 실행
# -----------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--used", type=str, required=True, help="중고 이미지 경로")
    args = parser.parse_args()

    result = run_full_pipeline(args.used)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
