from __future__ import annotations

import json
from pathlib import Path

import pytest

from fabrication.visual_report import render_visual_report, write_visual_report


def test_write_visual_report_renders_summary_mix_flags_and_links(tmp_path: Path):
    eval_dir = tmp_path / "artifacts" / "hybrid_evaluation"
    row_a_dir = eval_dir / "01_box"
    row_b_dir = eval_dir / "02_error"
    evaluation = {
        "schema_version": "fabrication.hybrid_evaluation.v0",
        "status": "warning",
        "elapsed_s": 12.3456,
        "strategies": ["planar_skin", "contour_stack", "voxel_blocks"],
        "counts": {
            "meshes": 2,
            "errors": 1,
            "mixed_strategy_meshes": 1,
            "meshes_with_boundary_joints": 1,
        },
        "flag_counts": {
            "exception": 1,
            "single_strategy_assignment": 1,
            "warnings_present": 1,
        },
        "strategy_region_use": {
            "contour_stack": 1,
            "planar_skin": 2,
            "voxel_blocks": 1,
        },
        "strategy_part_use": {
            "contour_stack": 8,
            "planar_skin": 4,
            "voxel_blocks": 2,
        },
        "strategy_volume_use_mm3": {
            "contour_stack": 2_500_000.0,
            "planar_skin": 1_200_000.0,
            "voxel_blocks": 900_000.0,
        },
        "rows": [
            {
                "mesh": "01_box & bracket.stl",
                "status": "ok",
                "overall_score": 0.87654,
                "regions": 3,
                "assignments": 3,
                "parts": 12,
                "joints": 4,
                "strategy_diversity": 2,
                "strategy_mix": {"planar_skin": 2, "contour_stack": 1},
                "strategy_part_burden": {"planar_skin": 4, "contour_stack": 8},
                "strategy_volume_burden_mm3": {
                    "planar_skin": 1_200_000.0,
                    "contour_stack": 2_500_000.0,
                },
                "dominant_part_strategy": "contour_stack",
                "source_winner": "planar_skin",
                "warning_count": 1,
                "warnings": ["thin wall <script>alert(1)</script>"],
                "flags": ["warnings_present"],
                "elapsed_s": 1.234,
                "hybrid_plan": str(row_a_dir / "hybrid_plan.json"),
                "source_ranking": str(row_a_dir / "source_strategies" / "ranking.json"),
            },
            {
                "mesh": "02_error.stl",
                "status": "error",
                "overall_score": 0.0,
                "regions": 0,
                "assignments": 0,
                "parts": 0,
                "joints": 0,
                "strategy_diversity": 0,
                "strategy_mix": {},
                "source_winner": "",
                "warning_count": 1,
                "warnings": ["RuntimeError: failed"],
                "flags": ["exception", "single_strategy_assignment"],
                "elapsed_s": 0.456,
                "hybrid_plan": "",
                "source_ranking": "",
            },
        ],
    }

    report_path = eval_dir / "report.html"
    returned_path = write_visual_report(report_path, evaluation)

    assert returned_path == report_path
    html = report_path.read_text(encoding="utf-8")
    assert html.startswith("<!doctype html>")
    assert "Hybrid Benchmark Visual Report" in html
    assert "fabrication.hybrid_evaluation.v0" in html
    assert "Meshes" in html
    assert "Mixed strategy meshes" in html
    assert "Strategy Mix" in html
    assert "planar_skin" in html
    assert "#2f80ed" in html
    assert "contour_stack" in html
    assert '<svg class="mix-bar"' in html
    assert "Strategy Burden" in html
    assert "Selected Parts" in html
    assert "2.50M mm3" in html
    assert "single_strategy_assignment" in html
    assert "exception" in html
    assert "01_box &amp; bracket.stl" in html
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html
    assert "<script>alert(1)</script>" not in html
    assert 'href="01_box/hybrid_plan.json"' in html
    assert 'href="01_box/source_strategies/ranking.json"' in html
    assert "hybrid_plan unavailable" in html
    assert "source_ranking unavailable" in html


def test_render_visual_report_accepts_rows_and_infers_counts(tmp_path: Path):
    rows = [
        {
            "mesh": "shelf.stl",
            "status": "ok",
            "overall_score": "0.75",
            "regions": 2,
            "parts": 8,
            "joints": 1,
            "strategy_mix": "{'planar_skin': 1, 'voxel_blocks': 1}",
            "flags": "[]",
            "hybrid_plan": str(tmp_path / "shelf" / "hybrid_plan.json"),
            "source_ranking": str(
                tmp_path / "shelf" / "source_strategies" / "ranking.json"
            ),
        }
    ]

    html = render_visual_report(rows, base_dir=tmp_path)

    assert "shelf.stl" in html
    assert "0.750" in html
    assert "voxel_blocks" in html
    assert "No quality flags were recorded." in html
    assert 'href="shelf/hybrid_plan.json"' in html
    assert 'href="shelf/source_strategies/ranking.json"' in html


def test_render_visual_report_includes_region_preview_from_hybrid_plan(tmp_path: Path):
    plan_path = tmp_path / "mesh_a" / "hybrid_plan.json"
    plan_path.parent.mkdir(parents=True)
    plan_path.write_text(
        json.dumps(
            {
                "regions": [
                    {
                        "region_id": "region_a",
                        "aabb_min": [0.0, 0.0, 0.0],
                        "aabb_max": [10.0, 20.0, 5.0],
                    },
                    {
                        "region_id": "region_b",
                        "aabb_min": [10.0, 0.0, 0.0],
                        "aabb_max": [20.0, 20.0, 5.0],
                    },
                ],
                "assignments": [
                    {"region_id": "region_a", "strategy_id": "planar_skin"},
                    {"region_id": "region_b", "strategy_id": "waffle_ribs"},
                ],
            }
        ),
        encoding="utf-8",
    )

    html = render_visual_report(
        [
            {
                "mesh": "mesh_a.stl",
                "status": "ok",
                "hybrid_plan": str(plan_path),
                "strategy_mix": {"planar_skin": 1, "waffle_ribs": 1},
            }
        ],
        base_dir=tmp_path,
    )

    assert '<svg class="region-preview"' in html
    assert "region_a: planar_skin" in html
    assert "region_b: waffle_ribs" in html
    assert "#2f80ed" in html


def test_render_visual_report_is_deterministic(tmp_path: Path):
    evaluation = {
        "rows": [
            {
                "mesh": "mesh.stl",
                "status": "warning",
                "strategy_mix": {"unknown_strategy": 2, "planar_skin": 1},
                "flags": ["single_strategy_assignment"],
            }
        ]
    }

    first = render_visual_report(evaluation, base_dir=tmp_path)
    second = render_visual_report(evaluation, base_dir=tmp_path)

    assert first == second
    assert "unknown_strategy" in first
    assert "hsl(" in first


def test_render_visual_report_rejects_invalid_rows():
    with pytest.raises(TypeError, match="iterable of mappings"):
        render_visual_report("not rows")
