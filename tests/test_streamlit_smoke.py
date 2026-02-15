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
    # Ensure app/ is on path so `from data import ...` works inside streamlit_app.
    if str(APP_DIR) not in sys.path:
        sys.path.insert(0, str(APP_DIR))
    return _load_module("streamlit_app", APP_PATH)


# ---- data.py tests ----


def test_data_read_json(tmp_path: Path):
    data = _load_data_module()
    p = tmp_path / "test.json"
    p.write_text(json.dumps({"a": 1}), encoding="utf-8")
    assert data.read_json(p) == {"a": 1}
    assert data.read_json(tmp_path / "missing.json") == {}


def test_data_status_badge():
    data = _load_data_module()
    assert data.status_badge("success") == "SUCCESS"
    assert data.status_badge("partial") == "PARTIAL"
    assert data.status_badge("fail") == "FAIL"
    assert data.status_badge("") == "UNKNOWN"


def test_data_safe_float():
    data = _load_data_module()
    assert data.safe_float(3.14) == 3.14
    assert data.safe_float("2.5") == 2.5
    assert data.safe_float(None) is None
    assert data.safe_float("bad") is None


def test_data_list_runs(tmp_path: Path):
    data = _load_data_module()
    run_dir = tmp_path / "20260101_000000_demo"
    run_dir.mkdir()
    (run_dir / "manifest.json").write_text(
        json.dumps({"run_id": run_dir.name, "status": "success"}),
        encoding="utf-8",
    )
    runs = data.list_runs(str(tmp_path))
    assert len(runs) == 1
    assert runs[0]["run_id"] == run_dir.name
    assert runs[0]["status"] == "success"


def test_data_list_suite_runs(tmp_path: Path):
    data = _load_data_module()
    suite_dir = tmp_path / "suites" / "20260101_000000_suite"
    suite_dir.mkdir(parents=True, exist_ok=True)
    (suite_dir / "manifest.json").write_text(
        json.dumps({"suite_name": "smoke_suite", "created_utc": "2026-01-01T00:00:00Z"}),
        encoding="utf-8",
    )
    (suite_dir / "results.json").write_text(
        json.dumps(
            {
                "rows": [
                    {"case_id": "a", "status": "success", "overall_score": 0.8},
                    {"case_id": "b", "status": "partial", "overall_score": 0.2},
                ]
            }
        ),
        encoding="utf-8",
    )
    suites = data.list_suite_runs(str(tmp_path / "suites"))
    assert suites
    assert suites[0]["suite_name"] == "smoke_suite"
    assert suites[0]["case_count"] == 2


def test_data_artifact_files(tmp_path: Path):
    data = _load_data_module()
    art_dir = tmp_path / "artifacts"
    art_dir.mkdir()
    (art_dir / "test.svg").write_text("<svg/>", encoding="utf-8")
    files = data.artifact_files(str(tmp_path))
    assert len(files) == 1
    assert files[0].name == "test.svg"


def test_data_infer_failure_modes():
    data = _load_data_module()
    modes = data.infer_failure_modes("fail", {"no_candidates_generated": 1}, {})
    assert any("No candidate" in m for m in modes)

    modes2 = data.infer_failure_modes("partial", {}, {})
    assert any("No dominant" in m for m in modes2)


def test_data_suite_rows_by_case():
    data = _load_data_module()
    rows = [{"case_id": "a", "score": 1}, {"case_id": "b", "score": 2}]
    by_case = data.suite_rows_by_case(rows)
    assert set(by_case.keys()) == {"a", "b"}


# ---- streamlit_app.py tests ----


def test_streamlit_module_has_main():
    module = _load_app_module()
    assert hasattr(module, "main")
