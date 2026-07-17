from __future__ import annotations

from pathlib import Path

import pytest
import trimesh

from fabrication.context import build_fabrication_context
from fabrication.contracts import FabricationConfig
from fabrication.strategies.voxel_blocks import VoxelBlocksStrategy
from fabrication.tournament import available_strategies


def _export_mesh(mesh: trimesh.Trimesh, tmp_path: Path, name: str) -> str:
    path = tmp_path / f"{name}.stl"
    mesh.export(path)
    return str(path)


def _box_context(tmp_path: Path):
    mesh = trimesh.creation.box(extents=[80.0, 50.0, 40.0])
    mesh.apply_translation([0.0, 0.0, 20.0])
    return build_fabrication_context(
        FabricationConfig(
            mesh_path=_export_mesh(mesh, tmp_path, "box"),
            design_name="voxel_box_test",
            auto_scale=False,
            preferred_thickness_mm=12.7,
            voxel_pitch_multiplier=1.0,
            max_voxels_per_axis=12,
            part_budget_max=24,
        )
    )


def test_voxel_blocks_generates_deterministic_block_parts(tmp_path: Path):
    context = _box_context(tmp_path)
    strategy = VoxelBlocksStrategy()

    plan_a = strategy.generate(context)
    plan_b = strategy.generate(context)

    assert plan_a.status in {"ok", "warning"}
    assert plan_a.parts
    assert plan_a.scores["overall"] > 0.0
    assert plan_a.debug["occupancy"]["occupied_voxel_count"] >= len(plan_a.parts)

    signature_a = [
        (
            part.part_id,
            part.kind,
            tuple(round(v, 6) for v in part.aabb_min),
            tuple(round(v, 6) for v in part.aabb_max),
            round(part.volume_mm3, 6),
            part.metadata["voxel_count"],
        )
        for part in plan_a.parts
    ]
    signature_b = [
        (
            part.part_id,
            part.kind,
            tuple(round(v, 6) for v in part.aabb_min),
            tuple(round(v, 6) for v in part.aabb_max),
            round(part.volume_mm3, 6),
            part.metadata["voxel_count"],
        )
        for part in plan_b.parts
    ]
    assert signature_a == signature_b


@pytest.mark.parametrize("mesh_name", ["01_box.stl", "03_l_bracket.stl"])
def test_voxel_blocks_handles_benchmark_meshes(mesh_name: str):
    mesh_path = Path("benchmarks/meshes") / mesh_name
    context = build_fabrication_context(
        FabricationConfig(
            mesh_path=str(mesh_path),
            design_name=f"voxel_{mesh_path.stem}",
            auto_scale=False,
            voxel_pitch_multiplier=4.0,
            max_voxels_per_axis=18,
            part_budget_max=128,
        )
    )

    plan = VoxelBlocksStrategy().generate(context)

    assert plan.status in {"ok", "warning"}
    assert len(plan.parts) > 0
    assert plan.operations[0].kind == "voxelize_and_greedy_merge_blocks"
    assert plan.debug["occupancy"]["pitch_mm"] > 0.0
    assert plan.debug["material_estimate"]["total_volume_mm3"] == pytest.approx(
        sum(part.volume_mm3 for part in plan.parts)
    )


def test_voxel_blocks_parts_include_aabb_volume_and_material_estimates(tmp_path: Path):
    context = _box_context(tmp_path)
    plan = VoxelBlocksStrategy().generate(context)

    part = plan.parts[0]
    assert part.kind == "voxel_block"
    assert all(high > low for low, high in zip(part.aabb_min, part.aabb_max))
    assert part.volume_mm3 > 0.0
    assert part.area_mm2 > 0.0
    assert part.metadata["material_volume_mm3"] == pytest.approx(part.volume_mm3)
    assert part.metadata["estimated_mass_kg"] > 0.0
    assert plan.debug["material_estimate"]["estimated_mass_kg"] > 0.0


def test_voxel_blocks_strategy_is_registered():
    registry = available_strategies()

    assert "voxel_blocks" in registry
    assert isinstance(registry["voxel_blocks"], VoxelBlocksStrategy)
