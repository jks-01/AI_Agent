# ai_agent/supervisor_agent.py

from ai_agent.node_agents_gemini.damage_agent_gemini import DamageAgent
from ai_agent.node_agents_gemini.soil_agent_gemini import SoilAgent
from ai_agent.node_agents_gemini.type_agent_gemini import TypeAgent
from ai_agent.node_agents_gemini.material_agent_gemini import MaterialAgent

class SupervisorAgent:
    def __init__(self):
        self.damage = DamageAgent()
        self.soil = SoilAgent()
        self.type = TypeAgent()
        self.material = MaterialAgent()

    def process(self, ref_b: bytes, used_b: bytes) -> dict:
        """
        ref_b: 기준 이미지 (새 상품, bytes)
        used_b: 중고 이미지 (bytes)
        """
        # 각각의 Agent 호출
        damage_result = self.damage.analyze(ref_b, used_b)   # {"grade": "A", "elapsed": ...}
        soil_result   = self.soil.analyze(ref_b, used_b)     # {"soil": "깨끗", "elapsed": ...}
        type_result   = self.type.analyze(used_b)            # {"type": "robot"}
        material_result = self.material.analyze(used_b)      # {"material": "plastic"}

        return {
            "damage": damage_result,
            "soil": soil_result,
            "type": type_result,
            "material": material_result
        }
