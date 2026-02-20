from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_openscad_step1_cli_runs_and_emits_artifacts(
    box_mesh_file: str, tmp_path: Path
):
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "generate_openscad_step1.py"),
        "--mesh",
        box_mesh_file,
        "--name",
        "step1_box",
        "--runs-dir",
        str(tmp_path),
        "--no-auto-scale",
        "--part-budget",
        "10",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    assert "Run ID:" in proc.stdout

    run_dirs = sorted(
        [path for path in tmp_path.iterdir() if path.is_dir() and path.name != "latest"]
    )
    assert len(run_dirs) == 1
    run_dir = run_dirs[0]

    artifacts = run_dir / "artifacts"
    design_path = artifacts / "design_step1_openscad.json"
    scad_path = artifacts / "model_step1.scad"
    capsule_path = artifacts / "spatial_capsule_step1.json"
    checkpoints_dir = artifacts / "checkpoints"
    decision_log_path = artifacts / "decision_log.jsonl"
    decision_chain_path = artifacts / "decision_hash_chain.json"

    assert design_path.exists()
    assert scad_path.exists()
    assert capsule_path.exists()
    assert checkpoints_dir.exists()
    assert decision_log_path.exists()
    assert decision_chain_path.exists()

    design = json.loads(design_path.read_text(encoding="utf-8"))
    assert design["schema_version"] == "openscad_step1.design.v1"
    assert len(design["panels"]) > 0

    scad_text = scad_path.read_text(encoding="utf-8")
    assert "module panel_2d_0" in scad_text
    assert "assembled_panels();" in scad_text

    checkpoint_files = sorted(checkpoints_dir.glob("phase_*.json"))
    assert len(checkpoint_files) >= 5
