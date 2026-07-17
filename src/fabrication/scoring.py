"""Common scoring helpers for prototype fabrication plans."""

from __future__ import annotations

from typing import Dict

from fabrication.context import FabricationContext
from fabrication.contracts import FabricationPlan

DEFAULT_WEIGHTS = {
    "fidelity": 0.25,
    "material_efficiency": 0.20,
    "assembly_simplicity": 0.20,
    "strength_proxy": 0.15,
    "part_count": 0.10,
    "risk": 0.10,
}


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def score_from_components(components: Dict[str, float]) -> Dict[str, float]:
    scores = {key: clamp01(value) for key, value in components.items()}
    overall = 0.0
    for key, weight in DEFAULT_WEIGHTS.items():
        overall += weight * scores.get(key, 0.0)
    scores["overall"] = round(clamp01(overall), 4)
    return scores


def add_basic_score(plan: FabricationPlan, context: FabricationContext) -> None:
    """Fill missing score components with conservative generic estimates."""

    part_count = len(plan.parts)
    total_part_volume = sum(float(part.volume_mm3) for part in plan.parts)
    mesh_volume = max(float(context.mesh_volume_mm3), 1.0)

    components = {
        "fidelity": plan.scores.get("fidelity", 0.5),
        "material_efficiency": plan.scores.get(
            "material_efficiency",
            1.0 / (1.0 + max(0.0, total_part_volume / mesh_volume - 1.0)),
        ),
        "assembly_simplicity": plan.scores.get(
            "assembly_simplicity", 1.0 / (1.0 + part_count / 24.0)
        ),
        "strength_proxy": plan.scores.get("strength_proxy", 0.5),
        "part_count": plan.scores.get("part_count", 1.0 / (1.0 + part_count / 36.0)),
        "risk": plan.scores.get("risk", 0.65 if not plan.warnings else 0.45),
    }
    plan.scores = score_from_components(components)
