from __future__ import annotations

from pathlib import Path

from fabrication import FabricationConfig, run_tournament, write_tournament_artifacts
from fabrication.context import build_fabrication_context
from fabrication.tournament import available_strategies


def test_fabrication_context_resolves_mesh_and_material(box_mesh_file: str):
    context = build_fabrication_context(
        FabricationConfig(mesh_path=box_mesh_file, auto_scale=False)
    )

    assert context.mesh_hash_sha256
    assert context.mesh_bounds_mm == (200.0, 140.0, 120.0)
    assert context.material_thickness_mm > 0.0
    assert context.mesh_volume_mm3 > 0.0


def test_tournament_runs_planar_skin_strategy(box_mesh_file: str, tmp_path: Path):
    config = FabricationConfig(
        mesh_path=box_mesh_file,
        design_name="core_tournament",
        auto_scale=False,
        strategies=("planar_skin",),
        part_budget_max=12,
    )
    result = run_tournament(config, artifacts_dir=tmp_path)
    write_tournament_artifacts(result, tmp_path / "fabrication_tournament")

    assert "planar_skin" in available_strategies()
    assert result["ranking"][0]["strategy_id"] == "planar_skin"
    assert result["ranking"][0]["part_count"] > 0
    assert (tmp_path / "fabrication_tournament" / "ranking.json").exists()
    assert (tmp_path / "fabrication_tournament" / "planar_skin" / "plan.json").exists()
