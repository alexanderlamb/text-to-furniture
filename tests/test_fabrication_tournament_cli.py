from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_fabrication_tournament_cli_runs_and_emits_artifacts(
    box_mesh_file: str, tmp_path: Path
):
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "generate_fabrication_tournament.py"),
        "--mesh",
        box_mesh_file,
        "--name",
        "fabrication_box",
        "--runs-dir",
        str(tmp_path),
        "--no-auto-scale",
        "--part-budget",
        "12",
        "--strategy",
        "planar_skin",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    assert "Run ID:" in proc.stdout
    assert "Winner: planar_skin" in proc.stdout

    run_dirs = sorted(
        [path for path in tmp_path.iterdir() if path.is_dir() and path.name != "latest"]
    )
    assert len(run_dirs) == 1
    run_dir = run_dirs[0]

    tournament_dir = run_dir / "artifacts" / "fabrication_tournament"
    ranking_path = tournament_dir / "ranking.json"
    plan_path = tournament_dir / "planar_skin" / "plan.json"

    assert ranking_path.exists()
    assert plan_path.exists()

    ranking = json.loads(ranking_path.read_text(encoding="utf-8"))
    assert ranking["schema_version"] == "fabrication.tournament.v0"
    assert ranking["requested_strategies"] == ["planar_skin"]
    assert ranking["ranking"][0]["strategy_id"] == "planar_skin"
    assert ranking["ranking"][0]["part_count"] > 0

    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    assert plan["strategy_id"] == "planar_skin"
    assert plan["part_count"] > 0
    assert plan["operation_count"] > 0

    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["strategy"] == "fabrication_tournament"
    assert metrics["requested_strategies"] == ["planar_skin"]
    assert metrics["artifacts"]["ranking"] == str(ranking_path)

    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["artifacts"]["fabrication_tournament_dir"] == str(tournament_dir)
    assert manifest["artifacts"]["ranking"] == str(ranking_path)
