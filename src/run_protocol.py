"""Run-folder protocol for fast iteration and feedback."""

from __future__ import annotations

import json
import os
import re
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict


@dataclass
class RunPaths:
    run_id: str
    run_dir: Path
    input_dir: Path
    artifacts_dir: Path
    logs_path: Path
    manifest_path: Path
    metrics_path: Path
    summary_path: Path


def slugify(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    value = re.sub(r"-+", "-", value)
    return value.strip("-") or "run"


def create_run_id(design_name: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"{stamp}_{slugify(design_name)}"


def prepare_run_dir(runs_root: str, design_name: str) -> RunPaths:
    runs_path = Path(runs_root)
    runs_path.mkdir(parents=True, exist_ok=True)

    run_id = create_run_id(design_name)
    run_dir = runs_path / run_id
    input_dir = run_dir / "input"
    artifacts_dir = run_dir / "artifacts"

    input_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    return RunPaths(
        run_id=run_id,
        run_dir=run_dir,
        input_dir=input_dir,
        artifacts_dir=artifacts_dir,
        logs_path=run_dir / "logs.txt",
        manifest_path=run_dir / "manifest.json",
        metrics_path=run_dir / "metrics.json",
        summary_path=run_dir / "summary.md",
    )


def copy_input_mesh(mesh_path: str, input_dir: Path) -> Path:
    src = Path(mesh_path)
    dst = input_dir / src.name
    if src.resolve() != dst.resolve():
        shutil.copy2(src, dst)
    return dst


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write(content)


def update_latest_pointer(runs_root: str, run_dir: Path) -> None:
    runs_path = Path(runs_root)
    latest = runs_path / "latest"

    if latest.exists() or latest.is_symlink():
        if latest.is_symlink() or latest.is_file():
            latest.unlink()
        else:
            shutil.rmtree(latest)

    try:
        target = os.path.relpath(run_dir, runs_path)
        latest.symlink_to(target)
    except OSError:
        # Fallback for filesystems where symlink is not available.
        latest.mkdir(parents=True, exist_ok=True)
        with (latest / "latest_run.txt").open("w", encoding="utf-8") as f:
            f.write(run_dir.name)
