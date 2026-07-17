from __future__ import annotations

from pathlib import Path

import pytest

from fabrication.context import build_fabrication_context
from fabrication.contracts import FabricationConfig
from fabrication.strategies.contour_stack import ContourStackStrategy
from fabrication.tournament import available_strategies


def _context(mesh_path: str, part_budget: int = 16):
    return build_fabrication_context(
        FabricationConfig(
            mesh_path=mesh_path,
            auto_scale=False,
            part_budget_max=part_budget,
        )
    )


def _plan_signature(plan):
    return [
        (
            part.part_id,
            round(part.area_mm2, 6),
            tuple(round(v, 6) for v in part.aabb_min),
            tuple(round(v, 6) for v in part.aabb_max),
            part.metadata["axis_name"],
            part.metadata["layer_index"],
            part.metadata["outline_count"],
        )
        for part in plan.parts
    ]


def test_contour_stack_is_registered_when_module_exists():
    assert "contour_stack" in available_strategies()


def test_contour_stack_generates_deterministic_layer_parts(box_mesh_file: str):
    context = _context(box_mesh_file, part_budget=12)
    strategy = ContourStackStrategy()

    plan_a = strategy.generate(context)
    plan_b = strategy.generate(context)

    assert plan_a.status == "ok"
    assert plan_a.strategy_id == "contour_stack"
    assert 1 <= len(plan_a.parts) <= context.config.part_budget_max
    assert _plan_signature(plan_a) == _plan_signature(plan_b)
    assert plan_a.debug["selected_axis"] == plan_b.debug["selected_axis"]

    first = plan_a.parts[0]
    assert first.kind == "contour_layer"
    assert first.material_thickness_mm == pytest.approx(context.material_thickness_mm)
    assert first.area_mm2 > 0.0
    assert first.volume_mm3 == pytest.approx(
        first.area_mm2 * context.material_thickness_mm
    )
    assert first.metadata["outline_count"] >= 1
    assert first.metadata["outline_summaries"][0]["area_mm2"] > 0.0
    assert len(first.aabb_min) == 3
    assert len(first.aabb_max) == 3
    assert all(lo <= hi for lo, hi in zip(first.aabb_min, first.aabb_max))


def test_contour_stack_respects_budget_with_coarser_spacing(box_mesh_file: str):
    context = _context(box_mesh_file, part_budget=5)
    plan = ContourStackStrategy().generate(context)

    assert plan.status == "ok"
    assert len(plan.parts) <= 5
    assert (
        plan.debug["selected_axis"]["effective_spacing_mm"]
        > context.material_thickness_mm
    )
    assert any("part_budget_max=5" in warning for warning in plan.warnings)
    assert len(plan.joints) == max(0, len(plan.parts) - 1)
    assert {operation.kind for operation in plan.operations} == {
        "mesh_sectioning",
        "profile_cutting",
        "stack_assembly",
    }


@pytest.mark.parametrize(
    "mesh_name",
    [
        "01_box.stl",
        "03_l_bracket.stl",
        "06_step_stool.stl",
        "11_v_bracket.stl",
        "13_trapezoidal_tray.stl",
    ],
)
def test_contour_stack_runs_on_representative_benchmarks(mesh_name: str):
    mesh_path = Path(__file__).parent.parent / "benchmarks" / "meshes" / mesh_name
    context = _context(str(mesh_path), part_budget=10)

    plan = ContourStackStrategy().generate(context)

    assert plan.status == "ok"
    assert 1 <= len(plan.parts) <= 10
    assert plan.overall_score > 0.0
    assert plan.debug["non_empty_slice_count"] == len(plan.parts)
    assert plan.debug["selected_axis"]["axis_name"] in {
        candidate["axis_name"] for candidate in plan.debug["axis_candidates"]
    }
