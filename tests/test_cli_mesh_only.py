from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_cli_mesh_only_runs(box_mesh_file: str, tmp_path: Path):
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "generate_furniture.py"),
        "--mesh",
        box_mesh_file,
        "--name",
        "cli_box",
        "--runs-dir",
        str(tmp_path),
        "--no-auto-scale",
        "--step3-no-bending",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    assert "Run ID:" in proc.stdout


def test_cli_rejects_removed_legacy_flags(box_mesh_file: str, tmp_path: Path):
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "generate_furniture.py"),
        "--mesh",
        box_mesh_file,
        "--runs-dir",
        str(tmp_path),
        "--manufacturing-v2",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    assert proc.returncode != 0
