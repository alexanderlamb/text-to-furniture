from __future__ import annotations

import json
from pathlib import Path

from fabrication.context import build_fabrication_context
from fabrication.contracts import (
    FabricationConfig,
    FabricationPlan,
    HybridFabricationPlan,
    HybridRegion,
    Part,
)
from fabrication.hybrid import (
    compose_hybrid_plan,
    run_hybrid_composition,
    write_hybrid_artifacts,
)


def test_hybrid_composition_combines_strategy_regions(
    box_mesh_file: str, tmp_path: Path
):
    config = FabricationConfig(
        mesh_path=box_mesh_file,
        design_name="hybrid_box",
        auto_scale=False,
        strategies=("planar_skin", "contour_stack", "voxel_blocks"),
        part_budget_max=12,
        max_hybrid_regions=3,
    )

    result = run_hybrid_composition(config, artifacts_dir=tmp_path, max_regions=3)
    plan = result["hybrid_plan"]

    assert isinstance(plan, HybridFabricationPlan)
    assert plan.status in {"ok", "warning"}
    assert len(plan.regions) == 3
    assert len(plan.assignments) >= 2
    assert len(plan.parts) >= len(plan.assignments)
    assert plan.scores["overall"] > 0.0
    assert plan.debug["strategy_mix"]
    assert len(set(plan.debug["strategy_mix"])) >= 2


def test_hybrid_artifacts_include_plan_and_source_tournament(
    box_mesh_file: str, tmp_path: Path
):
    config = FabricationConfig(
        mesh_path=box_mesh_file,
        design_name="hybrid_artifacts",
        auto_scale=False,
        strategies=("planar_skin", "contour_stack", "voxel_blocks"),
        part_budget_max=12,
        max_hybrid_regions=3,
    )

    result = run_hybrid_composition(config, artifacts_dir=tmp_path / "work")
    write_hybrid_artifacts(result, tmp_path / "hybrid_fabrication")

    plan_path = tmp_path / "hybrid_fabrication" / "hybrid_plan.json"
    summary_path = tmp_path / "hybrid_fabrication" / "hybrid_summary.json"
    source_ranking = (
        tmp_path / "hybrid_fabrication" / "source_strategies" / "ranking.json"
    )

    assert plan_path.exists()
    assert summary_path.exists()
    assert source_ranking.exists()

    payload = json.loads(plan_path.read_text(encoding="utf-8"))
    assert payload["region_count"] == 3
    assert payload["assignment_count"] >= 2
    assert payload["part_count"] > 0


def test_hybrid_composition_penalizes_reused_source_parts(box_mesh_file: str):
    config = FabricationConfig(
        mesh_path=box_mesh_file,
        design_name="hybrid_reuse_guard",
        auto_scale=False,
        strategies=("waffle_ribs",),
        max_hybrid_regions=2,
    )
    context = build_fabrication_context(config)
    regions = [
        HybridRegion(
            region_id="region_a",
            kind="shell_band",
            aabb_min=(0.0, 0.0, 0.0),
            aabb_max=(10.0, 10.0, 10.0),
            metadata={"band_index": 1, "band_count": 4},
        ),
        HybridRegion(
            region_id="region_b",
            kind="shell_band",
            aabb_min=(10.0, 0.0, 0.0),
            aabb_max=(20.0, 10.0, 10.0),
            metadata={"band_index": 2, "band_count": 4},
        ),
    ]
    source_plan = FabricationPlan(
        strategy_id="waffle_ribs",
        status="ok",
        scores={"overall": 1.0},
        parts=[
            Part(
                part_id="a_local_1",
                strategy_id="waffle_ribs",
                kind="rib",
                aabb_min=(0.0, 0.0, 0.0),
                aabb_max=(10.0, 10.0, 10.0),
                volume_mm3=1000.0,
            ),
            Part(
                part_id="a_local_2",
                strategy_id="waffle_ribs",
                kind="rib",
                aabb_min=(0.0, 0.0, 0.0),
                aabb_max=(10.0, 5.0, 10.0),
                volume_mm3=500.0,
            ),
            Part(
                part_id="b_local_1",
                strategy_id="waffle_ribs",
                kind="rib",
                aabb_min=(10.0, 0.0, 0.0),
                aabb_max=(20.0, 10.0, 10.0),
                volume_mm3=1000.0,
            ),
            Part(
                part_id="b_local_2",
                strategy_id="waffle_ribs",
                kind="rib",
                aabb_min=(10.0, 0.0, 0.0),
                aabb_max=(20.0, 5.0, 10.0),
                volume_mm3=500.0,
            ),
            Part(
                part_id="shared_long_rib",
                strategy_id="waffle_ribs",
                kind="rib",
                aabb_min=(0.0, 0.0, 0.0),
                aabb_max=(20.0, 10.0, 10.0),
                volume_mm3=2000.0,
            ),
        ],
    )

    plan = compose_hybrid_plan(
        context=context,
        regions=regions,
        source_plans={"waffle_ribs": source_plan},
    )

    source_ids = [str(part.metadata["source_part_id"]) for part in plan.parts]
    assert source_ids.count("shared_long_rib") == 1
    assert plan.debug["source_part_reuse_count"] == 0


def test_hybrid_composition_shares_spanning_source_part_once(box_mesh_file: str):
    config = FabricationConfig(
        mesh_path=box_mesh_file,
        design_name="hybrid_shared_part",
        auto_scale=False,
        strategies=("voxel_blocks",),
        max_hybrid_regions=2,
    )
    context = build_fabrication_context(config)
    regions = [
        HybridRegion(
            region_id="region_a",
            kind="blocky_band",
            aabb_min=(0.0, 0.0, 0.0),
            aabb_max=(10.0, 10.0, 10.0),
        ),
        HybridRegion(
            region_id="region_b",
            kind="blocky_band",
            aabb_min=(10.0, 0.0, 0.0),
            aabb_max=(20.0, 10.0, 10.0),
        ),
    ]
    source_plan = FabricationPlan(
        strategy_id="voxel_blocks",
        status="ok",
        scores={"overall": 1.0},
        parts=[
            Part(
                part_id="shared_block",
                strategy_id="voxel_blocks",
                kind="voxel_cluster",
                aabb_min=(0.0, 0.0, 0.0),
                aabb_max=(20.0, 10.0, 10.0),
                volume_mm3=2000.0,
            )
        ],
    )

    plan = compose_hybrid_plan(
        context=context,
        regions=regions,
        source_plans={"voxel_blocks": source_plan},
    )

    assert len(plan.parts) == 1
    assert (
        len(
            {
                part_id
                for assignment in plan.assignments
                for part_id in assignment.part_ids
            }
        )
        == 1
    )
    assert plan.debug["source_part_reuse_count"] == 0
    assert plan.debug["source_part_shared_count"] == 1
    assert plan.parts[0].metadata["hybrid_region_ids"] == ["region_a", "region_b"]


def test_hybrid_composition_requires_structural_evidence_for_rib_role(
    box_mesh_file: str,
):
    config = FabricationConfig(
        mesh_path=box_mesh_file,
        design_name="hybrid_rib_gate",
        auto_scale=False,
        strategies=("contour_stack", "waffle_ribs"),
        max_hybrid_regions=1,
    )
    context = build_fabrication_context(config)
    source_plans = {
        "contour_stack": FabricationPlan(
            strategy_id="contour_stack",
            status="ok",
            scores={"overall": 1.0},
            parts=[
                Part(
                    part_id="contour_layer",
                    strategy_id="contour_stack",
                    kind="layer",
                    aabb_min=(0.0, 0.0, 0.0),
                    aabb_max=(10.0, 10.0, 10.0),
                    volume_mm3=1000.0,
                )
            ],
        ),
        "waffle_ribs": FabricationPlan(
            strategy_id="waffle_ribs",
            status="ok",
            scores={"overall": 1.0},
            parts=[
                Part(
                    part_id="waffle_rib",
                    strategy_id="waffle_ribs",
                    kind="rib",
                    aabb_min=(0.0, 0.0, 0.0),
                    aabb_max=(10.0, 10.0, 10.0),
                    volume_mm3=1000.0,
                )
            ],
        ),
    }

    sparse_plan = compose_hybrid_plan(
        context=context,
        regions=[
            _shell_region_with_occupancy(
                "sparse_interior",
                fill_ratio=0.0,
                sample_inside_count=0,
                vertex_count=0,
                face_centroid_count=5,
            )
        ],
        source_plans=source_plans,
    )
    structural_plan = compose_hybrid_plan(
        context=context,
        regions=[
            _shell_region_with_occupancy(
                "structural_interior",
                fill_ratio=0.0,
                sample_inside_count=0,
                vertex_count=8,
                face_centroid_count=18,
            )
        ],
        source_plans=source_plans,
    )

    assert sparse_plan.assignments[0].strategy_id == "contour_stack"
    assert "assignment_role:layer_band" in sparse_plan.assignments[0].reason_codes
    assert structural_plan.assignments[0].strategy_id == "waffle_ribs"
    assert "assignment_role:rib_band" in structural_plan.assignments[0].reason_codes


def _shell_region_with_occupancy(
    region_id: str,
    *,
    fill_ratio: float,
    sample_inside_count: int,
    vertex_count: int,
    face_centroid_count: int,
) -> HybridRegion:
    return HybridRegion(
        region_id=region_id,
        kind="shell_band",
        aabb_min=(0.0, 0.0, 0.0),
        aabb_max=(10.0, 10.0, 10.0),
        metadata={
            "band_index": 1,
            "band_count": 4,
            "occupancy_proxy": {
                "fill_ratio": fill_ratio,
                "sample_inside_count": sample_inside_count,
                "vertex_count": vertex_count,
                "face_centroid_count": face_centroid_count,
            },
        },
    )
