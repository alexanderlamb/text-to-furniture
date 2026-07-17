from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_hybrid_fabrication_cli_runs_and_emits_artifacts(
    box_mesh_file: str, tmp_path: Path
):
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "generate_hybrid_fabrication.py"),
        "--mesh",
        box_mesh_file,
        "--name",
        "hybrid_box",
        "--runs-dir",
        str(tmp_path),
        "--no-auto-scale",
        "--material",
        "plywood_baltic_birch",
        "--thickness-mm",
        "12.0",
        "--part-budget",
        "12",
        "--max-regions",
        "2",
        "--strategies",
        "planar_skin,contour_stack",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    assert "Run ID:" in proc.stdout
    assert "Status: OK" in proc.stdout
    assert "Hybrid plan:" in proc.stdout
    assert "Source ranking:" in proc.stdout

    run_dirs = sorted(
        [path for path in tmp_path.iterdir() if path.is_dir() and path.name != "latest"]
    )
    assert len(run_dirs) == 1
    run_dir = run_dirs[0]

    hybrid_dir = run_dir / "artifacts" / "hybrid_fabrication"
    hybrid_plan_path = hybrid_dir / "hybrid_plan.json"
    hybrid_summary_path = hybrid_dir / "hybrid_summary.json"
    source_ranking_path = hybrid_dir / "source_strategies" / "ranking.json"
    planar_plan_path = hybrid_dir / "source_strategies" / "planar_skin" / "plan.json"

    assert hybrid_plan_path.exists()
    assert hybrid_summary_path.exists()
    assert source_ranking_path.exists()
    assert planar_plan_path.exists()

    hybrid_plan = json.loads(hybrid_plan_path.read_text(encoding="utf-8"))
    assert hybrid_plan["status"] == "ok"
    assert hybrid_plan["region_count"] == 2
    assert hybrid_plan["assignment_count"] == 2
    assert hybrid_plan["part_count"] > 0
    assert hybrid_plan["operation_count"] > 0
    assert set(hybrid_plan["source_strategy_scores"]) == {
        "planar_skin",
        "contour_stack",
    }

    source_ranking = json.loads(source_ranking_path.read_text(encoding="utf-8"))
    assert source_ranking["schema_version"] == "fabrication.tournament.v0"
    assert source_ranking["requested_strategies"] == [
        "planar_skin",
        "contour_stack",
    ]
    assert len(source_ranking["ranking"]) == 2

    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["strategy"] == "hybrid_fabrication"
    assert metrics["requested_strategies"] == ["planar_skin", "contour_stack"]
    assert metrics["counts"]["regions"] == 2
    assert metrics["artifacts"]["hybrid_plan"] == str(hybrid_plan_path)
    assert metrics["artifacts"]["source_ranking"] == str(source_ranking_path)

    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["artifacts"]["hybrid_fabrication_dir"] == str(hybrid_dir)
    assert manifest["artifacts"]["hybrid_plan"] == str(hybrid_plan_path)
    assert manifest["artifacts"]["source_ranking"] == str(source_ranking_path)
    assert manifest["config"]["max_hybrid_regions"] == 2
