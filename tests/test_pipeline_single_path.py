from __future__ import annotations

import json
from pathlib import Path

from pipeline import PipelineConfig, run_pipeline_from_mesh
from step3_first_principles import Step3Input, build_default_capability_profile


def test_pipeline_creates_run_folder_structure(box_mesh_file: str, tmp_path: Path):
    step3 = Step3Input(
        mesh_path=box_mesh_file,
        design_name="pipeline_box",
        auto_scale=False,
        part_budget_max=5,
        material_preferences=["plywood_baltic_birch"],
        scs_capabilities=build_default_capability_profile(
            material_key="plywood_baltic_birch",
            allow_controlled_bending=False,
        ),
    )
    config = PipelineConfig(runs_dir=str(tmp_path), step3_input=step3)

    result = run_pipeline_from_mesh(
        box_mesh_file, design_name="pipeline_box", config=config
    )

    run_dir = Path(result.run_dir)
    assert run_dir.exists()
    assert (run_dir / "input").exists()
    assert (run_dir / "artifacts").exists()
    assert (run_dir / "manifest.json").exists()
    assert (run_dir / "metrics.json").exists()
    assert (run_dir / "summary.md").exists()
    snapshots_dir = run_dir / "artifacts" / "snapshots"
    assert snapshots_dir.exists()
    capsules = sorted(snapshots_dir.glob("spatial_capsule_phase_*.json"))
    assert capsules, "Expected spatial capsule files alongside phase snapshots"

    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["strategy"] == "first_principles_step3"
    assert manifest["run_id"] == result.run_id
    artifacts = manifest.get("artifacts", {})
    assert len(artifacts.get("phase_spatial_capsules", [])) >= 1
