from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
APP_DIR = REPO_ROOT / "app"
APP_PATH = APP_DIR / "streamlit_app.py"
DATA_PATH = APP_DIR / "data.py"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_data_module():
    return _load_module("data", DATA_PATH)


def _load_app_module():
    if str(APP_DIR) not in sys.path:
        sys.path.insert(0, str(APP_DIR))
    return _load_module("streamlit_app", APP_PATH)


def test_data_list_runs_filters_step1_strategy(tmp_path: Path):
    data = _load_data_module()

    run_step1 = tmp_path / "20260201_000000_step1"
    run_legacy = tmp_path / "20260201_000001_legacy"
    run_hybrid = tmp_path / "20260201_000002_hybrid"
    run_step1.mkdir(parents=True, exist_ok=True)
    run_legacy.mkdir(parents=True, exist_ok=True)
    run_hybrid.mkdir(parents=True, exist_ok=True)

    (run_step1 / "manifest.json").write_text(
        json.dumps(
            {
                "run_id": run_step1.name,
                "strategy": "openscad_step1_clean_slate",
                "design_name": "step1",
            }
        ),
        encoding="utf-8",
    )
    (run_step1 / "metrics.json").write_text(
        json.dumps({"status": "ok"}), encoding="utf-8"
    )

    (run_legacy / "manifest.json").write_text(
        json.dumps(
            {
                "run_id": run_legacy.name,
                "strategy": "legacy_strategy",
                "design_name": "legacy",
            }
        ),
        encoding="utf-8",
    )
    (run_legacy / "metrics.json").write_text(
        json.dumps({"status": "error"}), encoding="utf-8"
    )

    (run_hybrid / "manifest.json").write_text(
        json.dumps(
            {
                "run_id": run_hybrid.name,
                "strategy": "hybrid_benchmark_evaluation",
                "design_name": "hybrid",
            }
        ),
        encoding="utf-8",
    )
    (run_hybrid / "metrics.json").write_text(
        json.dumps({"status": "ok"}), encoding="utf-8"
    )

    runs = data.list_runs(str(tmp_path))
    assert len(runs) == 3
    by_id = {row["run_id"]: row for row in runs}
    assert by_id[run_step1.name]["is_step1"] is True
    assert by_id[run_legacy.name]["is_step1"] is False
    assert by_id[run_hybrid.name]["is_hybrid_eval"] is True


def test_streamlit_module_has_main_and_relation_counter():
    module = _load_app_module()
    assert hasattr(module, "main")

    counts = module._relation_counts(
        {
            "relations": [
                {"class": "touching"},
                {"class": "touching"},
                {"class": "disjoint"},
            ]
        }
    )
    assert counts["touching"] == 2
    assert counts["disjoint"] == 1


def test_streamlit_hybrid_scene_items_use_region_assignments():
    module = _load_app_module()
    plan = {
        "regions": [
            {
                "region_id": "region_a",
                "kind": "shell_band",
                "aabb_min": [0, 0, 0],
                "aabb_max": [10, 10, 10],
                "volume_mm3": 1000,
            }
        ],
        "assignments": [{"region_id": "region_a", "strategy_id": "waffle_ribs"}],
        "parts": [
            {
                "part_id": "part_a",
                "strategy_id": "hybrid",
                "kind": "rib",
                "aabb_min": [0, 0, 0],
                "aabb_max": [10, 5, 2],
                "volume_mm3": 100,
                "metadata": {
                    "source_strategy_id": "waffle_ribs",
                    "hybrid_region_ids": ["region_a"],
                },
            }
        ],
    }

    region_items = module._hybrid_scene_items(plan, "Regions")
    part_items = module._hybrid_scene_items(plan, "Parts")

    assert region_items[0]["strategy"] == "waffle_ribs"
    assert "region_a" in region_items[0]["hover"]
    assert part_items[0]["strategy"] == "waffle_ribs"
    assert "part_a" in part_items[0]["hover"]


def test_streamlit_source_strategy_plan_path(tmp_path: Path):
    module = _load_app_module()
    hybrid_plan = tmp_path / "mesh_case" / "hybrid_plan.json"
    hybrid_plan.parent.mkdir(parents=True)

    row = {"hybrid_plan": str(hybrid_plan)}

    assert module._source_strategy_plan_path(row, "voxel_blocks") == (
        hybrid_plan.parent / "source_strategies" / "voxel_blocks" / "plan.json"
    )


def test_streamlit_strategy_board_summary_counts_hybrid_selected_parts():
    module = _load_app_module()
    source_plans = {
        "planar_skin": {
            "status": "ok",
            "part_count": 12,
            "joint_count": 2,
            "scores": {"overall": 0.7},
        },
        "contour_stack": {
            "status": "warning",
            "part_count": 8,
            "joint_count": 7,
            "scores": {"overall": 0.5},
        },
        "waffle_ribs": {
            "status": "ok",
            "part_count": 4,
            "joint_count": 12,
            "scores": {"overall": 0.6},
        },
        "voxel_blocks": {
            "status": "ok",
            "part_count": 2,
            "joint_count": 1,
            "scores": {"overall": 0.9},
        },
    }
    hybrid_plan = {
        "assignments": [
            {"region_id": "region_a", "strategy_id": "planar_skin"},
            {"region_id": "region_b", "strategy_id": "voxel_blocks"},
        ],
        "parts": [
            {
                "part_id": "part_a",
                "strategy_id": "hybrid",
                "volume_mm3": 1000,
                "metadata": {"source_strategy_id": "planar_skin"},
            },
            {
                "part_id": "part_b",
                "strategy_id": "voxel_blocks",
                "volume_mm3": 2000,
                "metadata": {},
            },
        ],
    }

    rows = module._strategy_board_summary_rows(source_plans, hybrid_plan)
    by_strategy = {row["strategy"]: row for row in rows}

    assert by_strategy["Planar Skin"]["hybrid_regions"] == 1
    assert by_strategy["Planar Skin"]["hybrid_selected_parts"] == 1
    assert by_strategy["Planar Skin"]["hybrid_selected_cm3"] == 1.0
    assert by_strategy["Voxel Blocks"]["source_parts"] == 2
    assert by_strategy["Voxel Blocks"]["hybrid_selected_parts"] == 1
