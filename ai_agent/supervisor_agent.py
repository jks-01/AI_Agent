# supervisor_agent.py  (single-image product version: absolute grades, english keys, per-task fallback)
import json
import concurrent.futures

# from ai_agent.node_agents_claude.type_agent import TypeAgent
# from ai_agent.node_agents_claude.material_agent import MaterialAgent
# from ai_agent.node_agents_claude.damage_agent import DamageAgent
# from ai_agent.node_agents_claude.soil_agent import SoilAgent
# from ai_agent.node_agents_openai.type_agent_gpt_4o_mini import TypeAgent
# from ai_agent.node_agents_openai.material_agent_gpt_4o_mini import MaterialAgent
# from ai_agent.node_agents_openai.damage_agent_gpt_4o_mini import DamageAgent
# from ai_agent.node_agents_openai.soil_agent_gpt_4o_mini import SoilAgent
from ai_agent.node_agents_gemini.type_agent_gemini import TypeAgent
from ai_agent.node_agents_gemini.material_agent_gemini import MaterialAgent
from ai_agent.node_agents_gemini.damage_agent_gemini import DamageAgent
from ai_agent.node_agents_gemini.soil_agent_gemini import SoilAgent
from ai_agent.image_input import optimize_image_size


class SupervisorAgent:
    def __init__(self):
        self.type_agent = TypeAgent()
        self.material_agent = MaterialAgent()
        self.damage_agent = DamageAgent()
        self.soil_agent = SoilAgent()

    def _run_parallel_with_fallback(self, optimized_bytes):
        """
        Run 4 node agents in parallel with a 30s budget.
        - Collect finished results.
        - Cancel stragglers and re-run only those sequentially (fallback).
        """
        results = {}

        ex = concurrent.futures.ThreadPoolExecutor(max_workers=4)
        try:
            futs = {
                "type":     ex.submit(self.type_agent.analyze, optimized_bytes),
                "material": ex.submit(self.material_agent.analyze, optimized_bytes),
                "damage":   ex.submit(self.damage_agent.analyze, optimized_bytes),
                "soil":     ex.submit(self.soil_agent.analyze, optimized_bytes),
            }

            done, not_done = concurrent.futures.wait(
                futs.values(), timeout=30, return_when=concurrent.futures.FIRST_EXCEPTION
            )

            # Collect finished futures
            for name, fut in futs.items():
                if fut in done and fut.done():
                    try:
                        results[name] = fut.result()  # (response, tokens)
                    except Exception:
                        results[name] = None
                else:
                    # cancel stragglers (may fail if already running)
                    try:
                        fut.cancel()
                    except Exception:
                        pass
        finally:
            # do not wait for unfinished tasks; we already cancelled
            ex.shutdown(wait=False)

        # Per-task fallback (only the missing ones)
        if not results.get("type"):
            results["type"] = self.type_agent.analyze(optimized_bytes)
        if not results.get("material"):
            results["material"] = self.material_agent.analyze(optimized_bytes)
        if not results.get("damage"):
            results["damage"] = self.damage_agent.analyze(optimized_bytes)
        if not results.get("soil"):
            results["soil"] = self.soil_agent.analyze(optimized_bytes)

        return results

    def process(self, image_bytes):
        """
        Absolute evaluation only (A~E) with a single user image.
        Returns ENGLISH keys and A~E grades directly. No donation logic.
        """
        # 0) optimize single input
        optimized = optimize_image_size(image_bytes)

        # 1) run node agents (parallel with per-task fallback)
        results = self._run_parallel_with_fallback(optimized)

        (type_response, type_tokens)         = results["type"]
        (material_response, material_tokens) = results["material"]
        (damage_response, damage_tokens)     = results["damage"]
        (soil_response, soil_tokens)         = results["soil"]

        # 2) safe parse
        def _parse_json_maybe(resp, default_obj):
            if isinstance(resp, dict):
                return resp
            try:
                return json.loads(resp)
            except Exception:
                return default_obj

        type_result = _parse_json_maybe(
            type_response,
            {"type": "others", "battery": "unknown", "size": "unknown"},
        )

        material_result = _parse_json_maybe(
            material_response,
            {
                "material": "unknown",
                "components": [],
                "secondary_hint": None,
                "confidence": 0.0,
                "material_detail": "",
                "notes": "",
            },
        )

        damage_result = _parse_json_maybe(
            damage_response,
            {"overall": {"summary": "unknown"}, "absolute_grade": None},
        )

        soil_result = _parse_json_maybe(
            soil_response,
            {"overall": {"summary": "unknown"}, "absolute_grade": None},
        )

        # 3) aggregate (ENGLISH keys, absolute only)
        toy_type = type_result.get("type", "others")
        battery = type_result.get("battery", "unknown")
        size = type_result.get("size", "unknown")

        # material: english; if mixed and components exist, display "mixed(a,b)"
        material_en = (material_result.get("material") or "unknown").strip().lower()
        components_en = material_result.get("components") or []
        if material_en == "mixed" and components_en:
            material_display = f"mixed({','.join(components_en)})"
        else:
            material_display = material_en

        # damage absolute grade & summaries
        abs_damage_grade = (
            damage_result.get("absolute_grade")
            or damage_result.get("grade")
        )
        damage_score_avg = (
            damage_result.get("absolute_score_avg")
            or damage_result.get("query_damage_score")
        )
        damage_summary = (damage_result.get("overall") or {}).get("summary")

        # soil absolute grade & summaries
        abs_soil_grade = (
            soil_result.get("absolute_grade")
            or soil_result.get("grade")
        )
        soil_level = soil_result.get("soil_level")
        soil_summary = (soil_result.get("overall") or {}).get("summary")

        # 4) token usage (robust to None/non-dict)
        def _tok(x):
            return x.get("total_tokens", 0) if isinstance(x, dict) else 0

        token_usage = {
            "type_agent": _tok(type_tokens),
            "material_agent": _tok(material_tokens),
            "damage_agent": _tok(damage_tokens),
            "soil_agent": _tok(soil_tokens),
            "total": _tok(type_tokens) + _tok(material_tokens) + _tok(damage_tokens) + _tok(soil_tokens),
        }

        # 5) final return — ENGLISH keys only, no donation logic
        return {
            "toy_type": toy_type,                      # e.g., "robot", "vehicles", ...
            "battery": battery,                        # "battery" / "non-battery" / "unknown"
            "material": material_display,              # e.g., "plastic" or "mixed(plastic,metal)"
            "size": size,

            "damage_absolute_grade": abs_damage_grade, # 'A'..'E'
            "damage_score_avg": damage_score_avg,      # optional 0..4
            "damage_summary": damage_summary,          # optional text

            "soil_absolute_grade": abs_soil_grade,     # 'A'..'E'
            "soil_level": soil_level,                  # optional
            "soil_summary": soil_summary,              # optional text

            "token_usage": token_usage,
        }
