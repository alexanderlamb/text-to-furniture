from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import trimesh

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_mesh_suite_runner_smoke(tmp_path: Path):
    mesh_path = tmp_path / "suite_box.stl"
    mesh = trimesh.creation.box(extents=[180.0, 120.0, 90.0])
    mesh.apply_translation([0.0, 0.0, 45.0])
    mesh.export(mesh_path)

    suite_path = tmp_path / "mesh_suite.json"
    suite_path.write_text(
        json.dumps(
            {
                "suite_name": "smoke_suite",
                "description": "single-case smoke",
                "cases": [
                    {"id": "suite_box", "mesh_path": str(mesh_path)},
                ],
            }
        ),
        encoding="utf-8",
    )

    runs_dir = tmp_path / "runs"
    suites_dir = tmp_path / "suites"
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "run_mesh_suite.py"),
        "--suite-file",
        str(suite_path),
        "--runs-dir",
        str(runs_dir),
        "--suites-dir",
        str(suites_dir),
        "--no-auto-scale",
        "--no-bending",
        "--part-budget",
        "4",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr

    latest = suites_dir / "latest"
    assert latest.exists() or latest.is_symlink()

    if latest.is_symlink():
        suite_out = (suites_dir / latest.readlink()).resolve()
    else:
        suite_out = latest

    results_path = suite_out / "results.json"
    assert results_path.exists()
    payload = json.loads(results_path.read_text(encoding="utf-8"))
    rows = payload.get("rows", [])
    assert len(rows) == 1
    assert rows[0]["case_id"] == "suite_box"
    assert rows[0]["status"] in {"success", "partial", "fail"}
    assert "dominant_failure_mode" in rows[0]
    assert "violation_code_counts" in rows[0]
