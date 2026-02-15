from __future__ import annotations

from pathlib import Path

import numpy as np

from overlay_viewer import (
    build_overlay_scene,
    normalized_to_original,
    original_to_normalized,
    triangulate_part_polygon,
)
from pipeline import PipelineConfig, run_pipeline_from_mesh
from step3_first_principles import Step3Input, build_default_capability_profile


def _make_overlay_run(mesh_path: str, runs_dir: Path):
    material_key = "plywood_baltic_birch"
    step3_input = Step3Input(
        mesh_path=mesh_path,
        design_name="overlay_case",
        fidelity_weight=0.85,
        part_budget_max=6,
        material_preferences=[material_key],
        scs_capabilities=build_default_capability_profile(
            material_key=material_key,
            allow_controlled_bending=True,
        ),
        target_height_mm=330.0,
        auto_scale=True,
    )
    config = PipelineConfig(
        runs_dir=str(runs_dir),
        export_svg=False,
        export_nested_svg=False,
        export_dxf=False,
        export_nested_dxf=False,
        step3_input=step3_input,
    )
    return run_pipeline_from_mesh(
        mesh_path=mesh_path,
        design_name="overlay_case",
        config=config,
    )


def test_overlay_scene_builds_for_original_and_normalized_spaces(
    box_mesh_file: str, tmp_path: Path
):
    result = _make_overlay_run(box_mesh_file, tmp_path / "runs")
    scene_norm = build_overlay_scene(
        run_dir=result.run_dir,
        space="normalized",
        max_mesh_faces=3000,
    )
    scene_orig = build_overlay_scene(
        run_dir=result.run_dir,
        space="original",
        max_mesh_faces=3000,
    )

    assert scene_norm.space == "normalized"
    assert scene_orig.space == "original"
    assert scene_norm.mesh_faces.shape[0] > 0
    assert scene_orig.mesh_faces.shape[0] > 0
    assert len(scene_norm.parts) > 0
    assert len(scene_orig.parts) > 0


def test_overlay_space_roundtrip_is_consistent(box_mesh_file: str, tmp_path: Path):
    result = _make_overlay_run(box_mesh_file, tmp_path / "runs")
    scene_norm = build_overlay_scene(
        run_dir=result.run_dir,
        space="normalized",
        max_mesh_faces=3000,
    )
    first_part = scene_norm.parts[0]
    point_norm = first_part.centroid[None, :]
    point_orig = normalized_to_original(point_norm, scene_norm.normalization)
    point_back = original_to_normalized(point_orig, scene_norm.normalization)
    assert np.allclose(point_back, point_norm, atol=1e-6)


def test_overlay_normalized_mesh_uses_min_z_anchor(box_mesh_file: str, tmp_path: Path):
    result = _make_overlay_run(box_mesh_file, tmp_path / "runs")
    scene_norm = build_overlay_scene(
        run_dir=result.run_dir,
        space="normalized",
        max_mesh_faces=3000,
    )
    z_vals = scene_norm.mesh_vertices[:, 2]
    assert np.isclose(float(z_vals.min()), 0.0, atol=1e-6)
    assert np.isclose(float(z_vals.max()), 330.0, atol=1e-6)


def test_triangulate_part_polygon_handles_concave_outline():
    concave_outline = [
        [0.0, 0.0],
        [8.0, 0.0],
        [8.0, 3.0],
        [4.0, 3.0],
        [4.0, 7.0],
        [8.0, 7.0],
        [8.0, 10.0],
        [0.0, 10.0],
        [0.0, 0.0],
    ]
    verts, faces, loops = triangulate_part_polygon(concave_outline, cutouts_2d=[])
    assert verts.shape[0] > 0
    assert faces.shape[0] > 0
    assert len(loops) > 0
