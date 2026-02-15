from __future__ import annotations

from pathlib import Path

from pipeline import PipelineConfig, run_pipeline_from_mesh
from step3_first_principles import Step3Input, build_default_capability_profile


def test_pipeline_exports_svg_and_dxf(box_mesh_file: str, tmp_path: Path):
    step3 = Step3Input(
        mesh_path=box_mesh_file,
        design_name="exports_box",
        auto_scale=False,
        part_budget_max=5,
        material_preferences=["mild_steel"],
        scs_capabilities=build_default_capability_profile(
            material_key="mild_steel",
            allow_controlled_bending=True,
        ),
    )
    config = PipelineConfig(
        runs_dir=str(tmp_path),
        step3_input=step3,
        export_svg=True,
        export_dxf=True,
    )

    result = run_pipeline_from_mesh(
        box_mesh_file, design_name="exports_box", config=config
    )

    assert result.design_json_path.endswith("design_first_principles.json")
    assert Path(result.design_json_path).exists()

    assert result.svg_paths
    for path in result.svg_paths:
        assert Path(path).exists()

    assert result.dxf_paths
    for path in result.dxf_paths:
        assert Path(path).exists()
