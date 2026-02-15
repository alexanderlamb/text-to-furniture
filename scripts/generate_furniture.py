#!/usr/bin/env python3
"""Mesh-only CLI for the first-principles strategy."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from materials import MATERIALS
from pipeline import PipelineConfig, run_pipeline_from_mesh
from step3_first_principles import Step3Input, build_default_capability_profile


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run first-principles decomposition from a local mesh"
    )
    parser.add_argument(
        "--mesh", required=True, help="Path to .stl/.obj/.glb/.ply mesh"
    )
    parser.add_argument("--name", default="design", help="Design name for this run")
    parser.add_argument("--runs-dir", default="runs", help="Run output root directory")

    parser.add_argument(
        "--material",
        type=str,
        default="plywood_baltic_birch",
        choices=sorted(MATERIALS.keys()),
        help="Material key",
    )
    parser.add_argument(
        "--step3-fidelity-weight",
        type=float,
        default=0.75,
        help="Fidelity weight (0-1)",
    )
    parser.add_argument(
        "--step3-part-budget",
        type=int,
        default=10,
        help="Maximum selected parts",
    )

    bend = parser.add_mutually_exclusive_group()
    bend.add_argument(
        "--step3-bending",
        action="store_true",
        dest="step3_bending",
        help="Enable controlled bending (default)",
    )
    bend.add_argument(
        "--step3-no-bending",
        action="store_false",
        dest="step3_bending",
        help="Disable controlled bending",
    )
    parser.set_defaults(step3_bending=True)

    parser.add_argument("--target-height-mm", type=float, default=750.0)
    parser.add_argument("--no-auto-scale", action="store_true")
    parser.add_argument(
        "--step3-no-planar-stacking",
        action="store_true",
        help="Disable stacking multiple identical planar sheets for thickness fill",
    )
    parser.add_argument(
        "--step3-max-stack-layers",
        type=int,
        default=4,
        help="Maximum laminate layers allowed per planar region",
    )
    parser.add_argument(
        "--step3-stack-roundup-bias",
        type=float,
        default=0.35,
        help=(
            "Bias for rounding sheet layer count toward additional layers "
            "(0.0 aggressive, 1.0 conservative)"
        ),
    )
    parser.add_argument(
        "--step3-stack-extra-layer-gain",
        type=float,
        default=0.65,
        help="Relative selection gain for extra laminate layers (0-1.5)",
    )
    parser.add_argument(
        "--step3-no-thin-side-suppression",
        action="store_true",
        help="Disable suppression of thin side planes when stacked sheets cover thickness",
    )
    parser.add_argument(
        "--step3-thin-side-dim-multiplier",
        type=float,
        default=1.10,
        help="Max thin-side short dimension relative to inferred member thickness",
    )
    parser.add_argument(
        "--step3-thin-side-aspect-limit",
        type=float,
        default=0.25,
        help="Max short/long aspect ratio for classifying a planar candidate as thin-side",
    )
    parser.add_argument(
        "--step3-thin-side-coverage-start",
        type=float,
        default=0.40,
        help="Coverage ratio at which thin-side candidates begin to be penalized",
    )
    parser.add_argument(
        "--step3-thin-side-coverage-drop",
        type=float,
        default=0.62,
        help="Coverage ratio at which thin-side candidates are dropped",
    )
    parser.add_argument(
        "--step3-no-intersection-filter",
        action="store_true",
        help="Disable filtering of overlapping non-stack parts",
    )
    parser.add_argument(
        "--step3-no-joint-intent-crossings",
        action="store_true",
        help="Disallow orthogonal intersection exceptions for joinery intent",
    )
    parser.add_argument(
        "--step3-intersection-clearance-mm",
        type=float,
        default=0.5,
        help="Clearance used when checking part-part intersections",
    )
    parser.add_argument(
        "--step3-joint-contact-tolerance-mm",
        type=float,
        default=2.0,
        help="Maximum gap for forming joints from geometric contact",
    )
    parser.add_argument(
        "--step3-joint-parallel-dot-threshold",
        type=float,
        default=0.95,
        help="Skip explicit joints when |dot(normals)| exceeds this threshold",
    )
    parser.add_argument("--no-svg", action="store_true", help="Skip SVG export")
    parser.add_argument("--no-dxf", action="store_true", help="Skip DXF export")
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    capability = build_default_capability_profile(
        material_key=args.material,
        allow_controlled_bending=args.step3_bending,
    )

    step3_input = Step3Input(
        mesh_path=args.mesh,
        design_name=args.name,
        fidelity_weight=max(0.0, min(1.0, args.step3_fidelity_weight)),
        part_budget_max=max(1, args.step3_part_budget),
        material_preferences=[args.material],
        scs_capabilities=capability,
        target_height_mm=args.target_height_mm,
        auto_scale=not args.no_auto_scale,
        enable_planar_stacking=not args.step3_no_planar_stacking,
        max_stack_layers_per_region=max(1, args.step3_max_stack_layers),
        stack_roundup_bias=max(0.0, min(0.95, args.step3_stack_roundup_bias)),
        stack_extra_layer_gain=max(0.0, min(1.5, args.step3_stack_extra_layer_gain)),
        enable_thin_side_suppression=not args.step3_no_thin_side_suppression,
        thin_side_dim_multiplier=max(1.0, args.step3_thin_side_dim_multiplier),
        thin_side_aspect_limit=max(0.05, min(0.95, args.step3_thin_side_aspect_limit)),
        thin_side_coverage_penalty_start=max(0.0, args.step3_thin_side_coverage_start),
        thin_side_coverage_drop_threshold=max(
            args.step3_thin_side_coverage_start + 1e-3,
            args.step3_thin_side_coverage_drop,
        ),
        enable_intersection_filter=not args.step3_no_intersection_filter,
        allow_joint_intent_intersections=not args.step3_no_joint_intent_crossings,
        intersection_clearance_mm=max(0.0, args.step3_intersection_clearance_mm),
        joint_contact_tolerance_mm=max(0.0, args.step3_joint_contact_tolerance_mm),
        joint_parallel_dot_threshold=max(
            0.0, min(0.999999, args.step3_joint_parallel_dot_threshold)
        ),
    )

    pipeline_config = PipelineConfig(
        runs_dir=args.runs_dir,
        export_svg=not args.no_svg,
        export_dxf=not args.no_dxf,
        step3_input=step3_input,
    )

    result = run_pipeline_from_mesh(
        args.mesh, design_name=args.name, config=pipeline_config
    )
    output = result.step3_output
    assert output is not None

    errors = sum(1 for v in output.violations if v.severity == "error")
    warnings = sum(1 for v in output.violations if v.severity == "warning")

    print(f"Run ID: {result.run_id}")
    print(f"Run dir: {result.run_dir}")
    print(f"Status: {output.status.upper()}")
    print(f"Parts: {output.quality_metrics.part_count}")
    print(f"Joints: {len(output.joints)}")
    print(f"Score: {output.quality_metrics.overall_score:.3f}")
    print(f"Hausdorff: {output.quality_metrics.hausdorff_mm:.2f} mm")
    print(f"Normal error: {output.quality_metrics.normal_error_deg:.2f} deg")
    print(f"Violations: {errors} errors, {warnings} warnings")
    print(f"Design JSON: {result.design_json_path}")
    print(f"Metrics: {result.metrics_path}")
    print(f"Summary: {result.summary_path}")

    if result.svg_paths:
        print(f"SVG files: {len(result.svg_paths)}")
    if result.dxf_paths:
        print(f"DXF files: {len(result.dxf_paths)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
