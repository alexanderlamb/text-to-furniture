#!/usr/bin/env python3
"""
Diagnose mesh decomposition quality.

Runs the v2 manufacturing decomposition pipeline on a mesh file, evaluates
quality gates, and writes a structured diagnosis_report.json. Designed for
LLM-driven diagnose-fix-rerun loops.

Usage:
    venv/bin/python3 scripts/diagnose_mesh.py --input model.glb
    venv/bin/python3 scripts/diagnose_mesh.py --input model.glb --simulate --compare output/run1/diagnosis_report.json

Exit codes:
    0 — all quality gates pass
    1 — at least one warning
    2 — at least one failure
"""
import sys
import os
import argparse
import json
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from manufacturing_decomposer_v2 import (
    ManufacturingDecompositionConfigV2,
    decompose_manufacturing_v2,
)
from slab_candidates import Slab3DConfig
from slab_selector import SlabSelectionConfig
from mesh_decomposer import load_mesh, DecompositionConfig
from mesh_cleanup import MeshCleanupConfig
from materials import MATERIALS
from diagnosis import (
    build_diagnosis_report,
    compare_reports,
    report_to_json,
    report_to_markdown,
)


def main():
    parser = argparse.ArgumentParser(
        description="Diagnose mesh decomposition quality"
    )
    parser.add_argument(
        "--input", required=True, type=str, help="Mesh file (GLB/STL/OBJ)"
    )
    parser.add_argument(
        "--output", type=str, default="output/diagnosis",
        help="Output directory (default: output/diagnosis/)",
    )
    parser.add_argument(
        "--height", type=float, default=750.0,
        help="Target height in mm (default: 750)",
    )
    parser.add_argument(
        "--max-slabs", type=int, default=15,
        help="Max slab count (default: 15)",
    )
    parser.add_argument(
        "--coverage", type=float, default=0.80,
        help="Coverage target 0-1 (default: 0.80)",
    )
    parser.add_argument(
        "--selection-objective", type=str, default="volume_fill",
        choices=["surface_coverage", "volume_fill"],
        help="Selection objective for slab picking (default: volume_fill)",
    )
    parser.add_argument(
        "--target-volume-fill", type=float, default=0.55,
        help="Interior volume fill target 0-1 for volume_fill mode",
    )
    parser.add_argument(
        "--min-volume-contribution", type=float, default=0.002,
        help="Minimum per-slab volume contribution for volume_fill mode",
    )
    parser.add_argument(
        "--plane-penalty-weight", type=float, default=0.0005,
        help="Per-slab penalty in volume_fill objective",
    )
    parser.add_argument(
        "--material", type=str, default="plywood_baltic_birch",
        choices=list(MATERIALS.keys()),
        help="Material (default: plywood_baltic_birch)",
    )
    parser.add_argument(
        "--simulate", action="store_true",
        help="Run physics simulation",
    )
    parser.add_argument(
        "--compare", type=str, default=None,
        help="Previous diagnosis_report.json for delta comparison",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="DEBUG logging",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    # Build config
    config = ManufacturingDecompositionConfigV2(
        slab_candidates=Slab3DConfig(
            material_key=args.material,
            target_height_mm=args.height,
        ),
        slab_selection=SlabSelectionConfig(
            coverage_target=args.coverage,
            max_slabs=args.max_slabs,
            objective_mode=args.selection_objective,
            target_volume_fill=args.target_volume_fill,
            min_volume_contribution=args.min_volume_contribution,
            plane_penalty_weight=args.plane_penalty_weight,
        ),
        default_material=args.material,
        target_height_mm=args.height,
        mesh_cleanup=MeshCleanupConfig(enabled=True),
    )

    # Step 1: Load mesh
    decomp_config = DecompositionConfig(
        default_material=args.material,
        target_height_mm=args.height,
        auto_scale=True,
        mesh_cleanup=config.mesh_cleanup,
    )
    mesh = load_mesh(args.input, decomp_config)

    # Step 2: Run decomposition
    mfg_result = decompose_manufacturing_v2(args.input, config)

    # Step 3: Optional simulation
    sim_result = None
    if args.simulate:
        try:
            from simulator import quick_stability_test
            sim_result = quick_stability_test(mfg_result.design)
        except Exception as e:
            logging.getLogger(__name__).warning("Simulation failed: %s", e)

    # Step 4: Build diagnosis report
    report = build_diagnosis_report(
        mesh_path=args.input,
        mesh=mesh,
        mfg_result=mfg_result,
        sim_result=sim_result,
        config=config,
    )

    # Step 5: Compare with previous run
    if args.compare:
        try:
            compare_reports(report, args.compare)
        except Exception as e:
            logging.getLogger(__name__).warning("Comparison failed: %s", e)

    # Step 6: Write output
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    report_path = out_dir / "diagnosis_report.json"
    with open(report_path, "w") as f:
        json.dump(report_to_json(report), f, indent=2)

    # Step 7: Print markdown summary
    print(report_to_markdown(report))
    print(f"Report written to: {report_path}")

    # Step 8: Exit code
    if report.overall_status == "fail":
        return 2
    elif report.overall_status == "warn":
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
