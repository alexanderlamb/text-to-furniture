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
    run_step1.mkdir(parents=True, exist_ok=True)
    run_legacy.mkdir(parents=True, exist_ok=True)

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

    runs = data.list_runs(str(tmp_path))
    assert len(runs) == 2
    by_id = {row["run_id"]: row for row in runs}
    assert by_id[run_step1.name]["is_step1"] is True
    assert by_id[run_legacy.name]["is_step1"] is False


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
