from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_hybrid_evaluation_cli_runs_one_mesh(box_mesh_file: str, tmp_path: Path):
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "evaluate_hybrid_benchmarks.py"),
        "--mesh",
        box_mesh_file,
        "--name",
        "hybrid_eval_box",
        "--runs-dir",
        str(tmp_path),
        "--no-auto-scale",
        "--part-budget",
        "12",
        "--max-regions",
        "3",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)

    assert proc.returncode == 0, proc.stderr
    assert "Evaluation JSON:" in proc.stdout
    assert "Report HTML:" in proc.stdout
    assert "tmp" in proc.stdout

    run_dirs = sorted(
        [path for path in tmp_path.iterdir() if path.is_dir() and path.name != "latest"]
    )
    assert len(run_dirs) == 1
    run_dir = run_dirs[0]

    eval_dir = run_dir / "artifacts" / "hybrid_evaluation"
    eval_json_path = eval_dir / "evaluation.json"
    eval_csv_path = eval_dir / "evaluation.csv"
    report_html_path = eval_dir / "report.html"

    assert eval_json_path.exists()
    assert eval_csv_path.exists()
    assert report_html_path.exists()

    evaluation = json.loads(eval_json_path.read_text(encoding="utf-8"))
    assert evaluation["schema_version"] == "fabrication.hybrid_evaluation.v0"
    assert evaluation["counts"]["meshes"] == 1
    assert evaluation["counts"]["errors"] == 0
    assert evaluation["rows"][0]["status"] in {"ok", "warning"}
    assert evaluation["rows"][0]["strategy_diversity"] >= 2
    assert evaluation["rows"][0]["joints"] > 0
    assert Path(evaluation["rows"][0]["hybrid_plan"]).exists()
    assert evaluation["strategy_part_use"]
    assert evaluation["strategy_volume_use_mm3"]
    assert evaluation["rows"][0]["strategy_part_burden"]
    assert evaluation["rows"][0]["dominant_part_strategy"]

    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["strategy"] == "hybrid_benchmark_evaluation"
    assert metrics["artifacts"]["evaluation_json"] == str(eval_json_path)
    assert metrics["artifacts"]["report_html"] == str(report_html_path)
