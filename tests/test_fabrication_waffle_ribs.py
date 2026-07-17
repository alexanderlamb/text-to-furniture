from __future__ import annotations

from pathlib import Path

import pytest

from fabrication.context import build_fabrication_context
from fabrication.contracts import FabricationConfig
from fabrication.strategies.waffle_ribs import WaffleRibsStrategy


def _context(mesh_path: str, part_budget: int = 12):
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
            part.kind,
            part.metadata["rib_set"],
            part.metadata["rib_index"],
            round(part.metadata["station_offset_mm"], 6),
            tuple(round(value, 6) for value in part.aabb_min),
            tuple(round(value, 6) for value in part.aabb_max),
            round(part.area_mm2, 6),
            part.metadata["slot_count_estimate"],
        )
        for part in plan.parts
    ]


def test_waffle_ribs_generates_deterministic_perpendicular_ribs(
    box_mesh_file: str,
):
    context = _context(box_mesh_file, part_budget=10)
    strategy = WaffleRibsStrategy()

    plan_a = strategy.generate(context)
    plan_b = strategy.generate(context)

    assert plan_a.status == "ok"
    assert plan_a.strategy_id == "waffle_ribs"
    assert 2 <= len(plan_a.parts) <= context.config.part_budget_max
    assert _plan_signature(plan_a) == _plan_signature(plan_b)
    assert plan_a.debug["allocation"] == plan_b.debug["allocation"]

    rib_sets = {part.metadata["rib_set"] for part in plan_a.parts}
    assert rib_sets == {"x", "y"}
    assert {part.kind for part in plan_a.parts} == {"waffle_rib"}
    assert all(
        part.material_thickness_mm == pytest.approx(context.material_thickness_mm)
        for part in plan_a.parts
    )
    assert all(part.area_mm2 > 0.0 for part in plan_a.parts)
    assert all(part.volume_mm3 > 0.0 for part in plan_a.parts)
    assert all(
        low <= high
        for part in plan_a.parts
        for low, high in zip(part.aabb_min, part.aabb_max)
    )


def test_waffle_ribs_emit_slot_and_assembly_metadata(box_mesh_file: str):
    context = _context(box_mesh_file, part_budget=8)
    plan = WaffleRibsStrategy().generate(context)

    assert plan.status == "ok"
    assert plan.joints
    assert {joint.kind for joint in plan.joints} == {"half_lap_slot"}

    operation_by_kind = {operation.kind: operation for operation in plan.operations}
    assert set(operation_by_kind) == {
        "coarse_cross_section_banding",
        "rib_profile_cutting",
        "half_lap_slot_cutting",
        "waffle_grid_assembly",
    }
    slot_operation = operation_by_kind["half_lap_slot_cutting"]
    assembly_operation = operation_by_kind["waffle_grid_assembly"]

    assert slot_operation.metadata["slot_count"] == len(plan.joints)
    assert slot_operation.metadata["slot_width_mm"] == pytest.approx(
        context.material_thickness_mm
    )
    assert slot_operation.metadata["slot_layout_preview"]
    assert assembly_operation.metadata["joint_count"] == len(plan.joints)
    assert all(part.metadata["slot_count_estimate"] > 0 for part in plan.parts)
    assert plan.debug["slot_count"] == len(plan.joints)
    assert plan.debug["material_estimate"]["total_volume_mm3"] == pytest.approx(
        sum(part.volume_mm3 for part in plan.parts)
    )


def test_waffle_ribs_respects_part_budget_with_two_sets(box_mesh_file: str):
    context = _context(box_mesh_file, part_budget=4)
    plan = WaffleRibsStrategy().generate(context)

    assert plan.status == "ok"
    assert len(plan.parts) <= 4
    assert {part.metadata["rib_set"] for part in plan.parts} == {"x", "y"}
    assert plan.debug["allocation"]["selected_counts"]["x_running_ribs"] >= 1
    assert plan.debug["allocation"]["selected_counts"]["y_running_ribs"] >= 1


@pytest.mark.parametrize(
    "mesh_name",
    [
        "01_box.stl",
        "02_tall_cabinet.stl",
        "03_l_bracket.stl",
        "04_t_beam.stl",
        "05_u_channel.stl",
        "06_step_stool.stl",
        "07_h_beam.stl",
        "08_shelf_unit.stl",
        "09_desk.stl",
        "10_table_with_stretchers.stl",
        "11_v_bracket.stl",
        "12_angled_flange.stl",
        "13_trapezoidal_tray.stl",
    ],
)
def test_waffle_ribs_runs_on_benchmark_meshes(mesh_name: str):
    mesh_path = Path("benchmarks/meshes") / mesh_name
    context = build_fabrication_context(
        FabricationConfig(
            mesh_path=str(mesh_path),
            design_name=f"waffle_{mesh_path.stem}",
            auto_scale=False,
            part_budget_max=14,
        )
    )

    plan = WaffleRibsStrategy().generate(context)

    assert plan.status in {"ok", "warning"}
    assert 2 <= len(plan.parts) <= 14
    assert {part.metadata["rib_set"] for part in plan.parts} == {"x", "y"}
    assert all(part.kind == "waffle_rib" for part in plan.parts)
    assert plan.operations[2].kind == "half_lap_slot_cutting"
    assert plan.scores["overall"] > 0.0
