#!/usr/bin/env python3
"""Analyze the latest suite run and print a structured summary.

Usage:
    venv/bin/python3 scripts/analyze_suite.py              # latest suite
    venv/bin/python3 scripts/analyze_suite.py --suite-dir runs/suites/20260215_...
    venv/bin/python3 scripts/analyze_suite.py --compare     # compare latest vs previous

Output is designed to be pasted to a coding agent for actionable analysis.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SUITES_DIR = ROOT / "runs" / "suites"


def _resolve_latest(suites_dir: Path) -> Optional[Path]:
    latest = suites_dir / "latest"
    if latest.is_symlink():
        target = (suites_dir / os.readlink(latest)).resolve()
        if target.is_dir():
            return target
    # Fallback: newest directory
    dirs = sorted(
        [d for d in suites_dir.iterdir() if d.is_dir() and d.name != "latest"],
        key=lambda d: d.name,
        reverse=True,
    )
    return dirs[0] if dirs else None


def _load_results(suite_dir: Path) -> Dict[str, Any]:
    path = suite_dir / "results.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _fmt(val: Any, fmt: str = ".3f") -> str:
    if val is None:
        return "n/a"
    try:
        return format(float(val), fmt)
    except (TypeError, ValueError):
        return str(val)


def print_summary(suite_dir: Path) -> None:
    results = _load_results(suite_dir)
    rows: List[Dict[str, Any]] = results.get("rows", [])
    if not rows:
        print(f"No results in {suite_dir}")
        return

    suite_id = results.get("suite_run_id", suite_dir.name)
    statuses = {}
    for r in rows:
        s = str(r.get("status", "unknown")).lower()
        statuses[s] = statuses.get(s, 0) + 1

    scores = [r["overall_score"] for r in rows if isinstance(r.get("overall_score"), (int, float))]
    hausdorffs = [r["hausdorff_mm"] for r in rows if isinstance(r.get("hausdorff_mm"), (int, float))]

    print(f"=== Suite: {suite_id} ===")
    print(f"Cases: {len(rows)}")
    print(f"Status breakdown: {', '.join(f'{k}={v}' for k, v in sorted(statuses.items()))}")
    if scores:
        print(f"Scores: mean={sum(scores)/len(scores):.3f}, min={min(scores):.3f}, max={max(scores):.3f}")
    if hausdorffs:
        print(f"Hausdorff (mm): mean={sum(hausdorffs)/len(hausdorffs):.1f}, max={max(hausdorffs):.1f}")
    print()

    # Per-case table
    print(f"{'case_id':<30} {'status':<10} {'score':>7} {'parts':>5} {'errs':>5} {'failure_mode'}")
    print("-" * 90)
    for r in rows:
        print(
            f"{str(r.get('case_id','')):<30} "
            f"{str(r.get('status','')):.<10} "
            f"{_fmt(r.get('overall_score')):>7} "
            f"{str(r.get('part_count', '')):>5} "
            f"{str(r.get('errors', '')):>5} "
            f"{r.get('dominant_failure_mode', '')}"
        )
    print()

    # Failure mode aggregation
    modes: Dict[str, int] = {}
    for r in rows:
        m = str(r.get("dominant_failure_mode", "")).strip()
        if m and m != "ok":
            modes[m] = modes.get(m, 0) + 1
    if modes:
        print("Failure modes:")
        for mode, count in sorted(modes.items(), key=lambda x: -x[1]):
            print(f"  {mode}: {count} cases")
        print()

    # Violation code aggregation across all cases
    all_codes: Dict[str, int] = {}
    for r in rows:
        for code, count in (r.get("violation_code_counts") or {}).items():
            all_codes[code] = all_codes.get(code, 0) + count
    if all_codes:
        print("Violation codes (total across all cases):")
        for code, count in sorted(all_codes.items(), key=lambda x: -x[1]):
            print(f"  {code}: {count}")
        print()

    # Cases that need attention (non-success, sorted by score ascending)
    problem_cases = [r for r in rows if str(r.get("status", "")).lower() != "success"]
    if problem_cases:
        problem_cases.sort(key=lambda r: r.get("overall_score") if isinstance(r.get("overall_score"), (int, float)) else -1)
        print("Priority cases (worst first):")
        for r in problem_cases:
            case_id = r.get("case_id", "")
            status = r.get("status", "")
            score = _fmt(r.get("overall_score"))
            mode = r.get("dominant_failure_mode", "")
            codes = ", ".join(r.get("violation_codes", []))
            print(f"  {case_id}: status={status}, score={score}, mode={mode}")
            if codes:
                print(f"    violations: {codes}")
            # Show key debug metrics
            coverage = r.get("coverage_ratio_unique_faces")
            dropped = r.get("intersection_dropped_count", 0)
            if coverage is not None:
                print(f"    coverage={_fmt(coverage)}, intersection_dropped={dropped}")
        print()


def print_comparison(suites_dir: Path) -> None:
    dirs = sorted(
        [d for d in suites_dir.iterdir() if d.is_dir() and d.name != "latest"],
        key=lambda d: d.name,
        reverse=True,
    )
    if len(dirs) < 2:
        print("Need at least 2 suite runs to compare.")
        return

    current_dir, previous_dir = dirs[0], dirs[1]
    current = _load_results(current_dir)
    previous = _load_results(previous_dir)

    cur_rows = {str(r.get("case_id")): r for r in current.get("rows", [])}
    prev_rows = {str(r.get("case_id")): r for r in previous.get("rows", [])}

    print(f"=== Comparison: {current_dir.name} vs {previous_dir.name} ===")
    print()
    print(f"{'case_id':<30} {'prev_score':>10} {'curr_score':>10} {'delta':>8} {'status_change'}")
    print("-" * 80)

    common = sorted(set(cur_rows.keys()) & set(prev_rows.keys()))
    for case_id in common:
        cr, pr = cur_rows[case_id], prev_rows[case_id]
        cs = cr.get("overall_score")
        ps = pr.get("overall_score")
        if isinstance(cs, (int, float)) and isinstance(ps, (int, float)):
            delta = cs - ps
            delta_str = f"{delta:+.3f}"
        else:
            delta_str = "n/a"
        status_change = ""
        if str(cr.get("status")) != str(pr.get("status")):
            status_change = f"{pr.get('status')} → {cr.get('status')}"
        print(f"{case_id:<30} {_fmt(ps):>10} {_fmt(cs):>10} {delta_str:>8} {status_change}")

    new_cases = sorted(set(cur_rows.keys()) - set(prev_rows.keys()))
    if new_cases:
        print(f"\nNew cases: {', '.join(new_cases)}")
    removed = sorted(set(prev_rows.keys()) - set(cur_rows.keys()))
    if removed:
        print(f"Removed cases: {', '.join(removed)}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze suite results")
    parser.add_argument("--suite-dir", default=None, help="Specific suite run directory")
    parser.add_argument("--suites-dir", default=str(DEFAULT_SUITES_DIR))
    parser.add_argument("--compare", action="store_true", help="Compare latest vs previous")
    args = parser.parse_args()

    suites_dir = Path(args.suites_dir)
    if not suites_dir.exists():
        print(f"Suites directory not found: {suites_dir}")
        sys.exit(1)

    if args.suite_dir:
        suite_dir = Path(args.suite_dir)
    else:
        suite_dir = _resolve_latest(suites_dir)
        if not suite_dir:
            print("No suite runs found.")
            sys.exit(1)

    print_summary(suite_dir)
    if args.compare:
        print_comparison(suites_dir)


if __name__ == "__main__":
    main()
