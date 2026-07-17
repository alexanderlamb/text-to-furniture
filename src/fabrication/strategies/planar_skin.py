"""Strategy wrapper around the existing OpenSCAD Step 1 panel pipeline."""

from __future__ import annotations

from pathlib import Path
import json

from openscad_step1 import Step1Config, run_step1_pipeline
from openscad_step1.audit import AuditTrail

from fabrication.context import FabricationContext
from fabrication.contracts import FabricationPlan, Operation, Part
from fabrication.scoring import add_basic_score


class PlanarSkinStrategy:
    """Current planar-panel pipeline exposed as a tournament strategy."""

    strategy_id = "planar_skin"

    def generate(
        self, context: FabricationContext, artifacts_dir: Path | None = None
    ) -> FabricationPlan:
        strategy_artifacts = None
        if artifacts_dir is not None:
            strategy_artifacts = Path(artifacts_dir) / self.strategy_id
            strategy_artifacts.mkdir(parents=True, exist_ok=True)

        cfg = context.config
        step1_config = Step1Config(
            mesh_path=str(context.source_mesh_path),
            design_name=f"{cfg.design_name}_{self.strategy_id}",
            material_key=cfg.material_key,
            preferred_thickness_mm=cfg.preferred_thickness_mm,
            auto_scale=cfg.auto_scale,
            target_height_mm=cfg.target_height_mm,
            part_budget_max=cfg.part_budget_max,
            min_feature_mm=cfg.min_feature_mm,
        )
        audit = None
        if strategy_artifacts is not None:
            audit = AuditTrail(
                run_id=f"{cfg.design_name}_{self.strategy_id}",
                artifacts_dir=strategy_artifacts,
            )
        result = run_step1_pipeline(
            config=step1_config,
            run_id=f"{cfg.design_name}_{self.strategy_id}",
            artifacts_dir=strategy_artifacts or Path("."),
            audit=audit,
        )

        parts = [
            Part(
                part_id=panel.panel_id,
                strategy_id=self.strategy_id,
                kind="flat_panel",
                material_thickness_mm=float(panel.thickness_mm),
                area_mm2=float(panel.area_mm2),
                volume_mm3=float(panel.area_mm2 * panel.thickness_mm),
                aabb_min=_panel_aabb(panel)[0],
                aabb_max=_panel_aabb(panel)[1],
                metadata={
                    "family_id": panel.family_id,
                    "source_candidate_ids": list(panel.source_candidate_ids),
                    "panel_role": panel.metadata.get("panel_role", "unknown"),
                    "source_face_count": panel.source_face_count,
                },
            )
            for panel in result.panels
        ]
        warnings = [
            f"{violation.severity}:{violation.code}:{violation.message}"
            for violation in result.violations
        ]
        plan = FabricationPlan(
            strategy_id=self.strategy_id,
            status=result.status,
            parts=parts,
            operations=[
                Operation(
                    operation_id="planar_skin_openscad_emit",
                    strategy_id=self.strategy_id,
                    kind="openscad_panelization",
                    part_ids=[part.part_id for part in parts],
                    metadata={"panel_count": len(parts)},
                )
            ],
            scores={
                "fidelity": 0.75,
                "strength_proxy": 0.65,
                "risk": 0.80 if result.status == "ok" else 0.50,
            },
            warnings=warnings,
            artifacts={},
            debug={
                "panel_families": len(result.panel_families),
                "selected_families": len(result.selected_families),
                "trim_pairs": len(result.trim_decisions),
                "source_strategy": "openscad_step1_clean_slate",
            },
        )
        if strategy_artifacts is not None:
            (strategy_artifacts / "model_step1.scad").write_text(
                result.openscad_code, encoding="utf-8"
            )
            with (strategy_artifacts / "design_step1_openscad.json").open(
                "w", encoding="utf-8"
            ) as f:
                json.dump(result.design_payload, f, indent=2)
            plan.artifacts = {
                "openscad_code": str(strategy_artifacts / "model_step1.scad"),
                "design_json": str(strategy_artifacts / "design_step1_openscad.json"),
                "decision_log": str(result.decision_log_path),
            }
        add_basic_score(plan, context)
        return plan


def _panel_aabb(panel) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    xs = [point[0] for point in panel.outline_2d]
    ys = [point[1] for point in panel.outline_2d]
    if not xs or not ys:
        origin = tuple(float(v) for v in panel.origin_3d)
        return origin, origin

    local = [
        (min(xs), min(ys), 0.0),
        (max(xs), min(ys), 0.0),
        (max(xs), max(ys), 0.0),
        (min(xs), max(ys), 0.0),
        (min(xs), min(ys), panel.thickness_mm),
        (max(xs), min(ys), panel.thickness_mm),
        (max(xs), max(ys), panel.thickness_mm),
        (min(xs), max(ys), panel.thickness_mm),
    ]
    origin = panel.origin_3d
    u = panel.basis_u
    v = panel.basis_v
    n = panel.basis_n
    world = [
        (
            origin[0] + u[0] * x + v[0] * y + n[0] * z,
            origin[1] + u[1] * x + v[1] * y + n[1] * z,
            origin[2] + u[2] * x + v[2] * y + n[2] * z,
        )
        for x, y, z in local
    ]
    mins = tuple(float(min(point[i] for point in world)) for i in range(3))
    maxs = tuple(float(max(point[i] for point in world)) for i in range(3))
    return mins, maxs
