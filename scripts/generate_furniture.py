#!/usr/bin/env python3
"""
Generate flat-pack furniture from text prompts, images, or existing meshes.

Usage:
    # From text (requires API key)
    python scripts/generate_furniture.py --text "a modern coffee table" --provider tripo
    python scripts/generate_furniture.py --text "wooden bookshelf" --provider meshy

    # From image (requires API key)
    python scripts/generate_furniture.py --image photo.jpg --provider tripo

    # From existing mesh file (no API key needed)
    python scripts/generate_furniture.py --mesh model.glb

API keys are read from TRIPO_API_KEY or MESHY_API_KEY environment variables,
or passed via --api-key.
"""
import sys
import os
import argparse
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mesh_provider import ProviderConfig, ProviderError
from mesh_decomposer import DecompositionConfig
from mesh_cleanup import MeshCleanupConfig
from manufacturing_decomposer_v2 import ManufacturingDecompositionConfigV2
from slab_candidates import Slab3DConfig
from slab_selector import SlabSelectionConfig
from pipeline import run_pipeline, run_pipeline_from_mesh, PipelineConfig
from materials import MATERIALS


def get_provider(name, api_key=None):
    """Create a provider instance by name."""
    if name == "tripo":
        key = api_key or os.environ.get("TRIPO_API_KEY")
        if not key:
            print("Error: Set TRIPO_API_KEY environment variable or pass --api-key")
            sys.exit(1)
        from tripo_provider import TripoProvider
        return TripoProvider(ProviderConfig(api_key=key))
    elif name == "meshy":
        key = api_key or os.environ.get("MESHY_API_KEY")
        if not key:
            print("Error: Set MESHY_API_KEY environment variable or pass --api-key")
            sys.exit(1)
        from meshy_provider import MeshyProvider
        return MeshyProvider(ProviderConfig(api_key=key))
    else:
        print(f"Error: Unknown provider '{name}'. Use 'tripo' or 'meshy'.")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Generate flat-pack furniture from text, images, or meshes"
    )

    # Input mode (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--text", type=str, help="Text prompt describing furniture")
    input_group.add_argument("--image", type=str, help="Path to input image")
    input_group.add_argument(
        "--mesh", type=str, help="Path to existing mesh file (skip generation)"
    )

    # Provider options
    parser.add_argument(
        "--provider", type=str, default="tripo", choices=["tripo", "meshy"],
        help="Cloud provider for mesh generation (default: tripo)",
    )
    parser.add_argument("--api-key", type=str, default=None, help="API key override")

    # Decomposition options
    parser.add_argument(
        "--material", type=str, default="plywood_baltic_birch",
        choices=list(MATERIALS.keys()),
        help="Material for flat-pack components (default: plywood_baltic_birch)",
    )
    parser.add_argument(
        "--height", type=float, default=750.0, help="Target height in mm (default: 750)"
    )
    parser.add_argument(
        "--max-slabs", type=int, default=15, help="Max flat-pack components (default: 15)"
    )
    parser.add_argument(
        "--coverage", type=float, default=0.80,
        help="Voxel coverage target 0-1 (default: 0.80)",
    )
    parser.add_argument(
        "--selection-objective", type=str, default="volume_fill",
        choices=["surface_coverage", "volume_fill"],
        help="Selection objective for manufacturing v2 (default: volume_fill)",
    )
    parser.add_argument(
        "--target-volume-fill", type=float, default=0.55,
        help="Interior volume fill target 0-1 for volume_fill mode (default: 0.55)",
    )
    parser.add_argument(
        "--min-volume-contribution", type=float, default=0.002,
        help="Minimum per-slab volume contribution in volume_fill mode (default: 0.002)",
    )
    parser.add_argument(
        "--plane-penalty-weight", type=float, default=0.0005,
        help="Per-slab penalty in volume_fill objective (default: 0.0005)",
    )
    cleanup_group = parser.add_mutually_exclusive_group()
    cleanup_group.add_argument(
        "--mesh-cleanup",
        dest="mesh_cleanup",
        action="store_true",
        help="Enable mesh cleanup (flatten near-planar faces, align near-parallel faces)",
    )
    cleanup_group.add_argument(
        "--no-mesh-cleanup",
        dest="mesh_cleanup",
        action="store_false",
        help="Disable mesh cleanup",
    )
    parser.set_defaults(mesh_cleanup=None)
    parser.add_argument(
        "--cleanup-planar-angle-deg", type=float, default=8.0,
        help="Max face-angle deviation for planar grouping (default: 8.0)",
    )
    parser.add_argument(
        "--cleanup-planar-distance-mm", type=float, default=1.5,
        help="Max distance from plane for planar grouping (default: 1.5 mm)",
    )
    parser.add_argument(
        "--cleanup-parallel-angle-deg", type=float, default=5.0,
        help="Max normal-angle gap to enforce parallelism (default: 5.0)",
    )
    parser.add_argument(
        "--cleanup-simplify-mm", type=float, default=2.0,
        help="Planar-region quantization tolerance for simplification (default: 2.0 mm)",
    )
    parser.add_argument(
        "--cleanup-min-region-area-mm2", type=float, default=400.0,
        help="Minimum planar-region area to process (default: 400 mm^2)",
    )
    parser.add_argument(
        "--cleanup-iterations", type=int, default=2,
        help="Maximum cleanup iterations (default: 2)",
    )

    # Pipeline options
    parser.add_argument("--no-simulate", action="store_true", help="Skip physics simulation")
    parser.add_argument("--no-svg", action="store_true", help="Skip SVG export")
    parser.add_argument("--no-optimize", action="store_true", help="Skip decomposition optimization")
    parser.add_argument("--output", type=str, default="output", help="Output directory")
    parser.add_argument("--name", type=str, default="furniture", help="Design name")
    parser.add_argument(
        "--manufacturing-v2", action="store_true",
        help="Use v2 3D-first manufacturing decomposition",
    )
    parser.add_argument(
        "--export-dxf", action="store_true",
        help="Export DXF cut files (manufacturing-aware mode)",
    )
    parser.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    cleanup_enabled = args.mesh_cleanup
    if cleanup_enabled is None:
        cleanup_enabled = args.manufacturing_v2

    cleanup_config = MeshCleanupConfig(
        enabled=cleanup_enabled,
        planar_angle_threshold_deg=args.cleanup_planar_angle_deg,
        planar_distance_threshold_mm=args.cleanup_planar_distance_mm,
        parallel_angle_threshold_deg=args.cleanup_parallel_angle_deg,
        boundary_simplify_tolerance_mm=args.cleanup_simplify_mm,
        min_region_area_mm2=args.cleanup_min_region_area_mm2,
        max_iterations=args.cleanup_iterations,
    )

    mfg_cleanup_config = MeshCleanupConfig(
        enabled=cleanup_enabled,
        planar_angle_threshold_deg=args.cleanup_planar_angle_deg,
        planar_distance_threshold_mm=args.cleanup_planar_distance_mm,
        parallel_angle_threshold_deg=args.cleanup_parallel_angle_deg,
        boundary_simplify_tolerance_mm=args.cleanup_simplify_mm,
        min_region_area_mm2=args.cleanup_min_region_area_mm2,
        max_iterations=args.cleanup_iterations,
    )

    pipeline_config = PipelineConfig(
        decomposition=DecompositionConfig(
            default_material=args.material,
            target_height_mm=args.height,
            max_slabs=args.max_slabs,
            coverage_target=args.coverage,
            selection_objective_mode=args.selection_objective,
            target_volume_fill=args.target_volume_fill,
            min_volume_contribution=args.min_volume_contribution,
            plane_penalty_weight=args.plane_penalty_weight,
            mesh_cleanup=cleanup_config,
        ),
        run_simulation=not args.no_simulate,
        export_svg=not args.no_svg,
        optimize_decomposition=not args.no_optimize,
        output_dir=args.output,
        use_manufacturing_v2=args.manufacturing_v2,
        manufacturing_v2_config=ManufacturingDecompositionConfigV2(
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
            mesh_cleanup=mfg_cleanup_config,
        ),
        export_dxf=args.export_dxf or args.manufacturing_v2,
    )

    try:
        if args.mesh:
            result = run_pipeline_from_mesh(args.mesh, args.name, pipeline_config)
        else:
            provider = get_provider(args.provider, args.api_key)
            result = run_pipeline(
                provider=provider,
                prompt=args.text or "",
                image_path=args.image or "",
                design_name=args.name,
                config=pipeline_config,
            )
    except ProviderError as e:
        print(f"Error: {e}")
        return 1

    # Print results
    print(f"\nDesign: {result.design.name}")
    print(f"Mesh: {result.mesh_path}")
    print(f"Components: {len(result.design.components)}")
    print(f"Joints: {len(result.design.assembly.joints)}")

    for comp in result.design.components:
        w = float(comp.profile[2][0]) if len(comp.profile) >= 3 else 0
        h = float(comp.profile[2][1]) if len(comp.profile) >= 3 else 0
        print(
            f"  {comp.name}: {w:.0f} x {h:.0f} x {comp.thickness:.1f} mm "
            f"({comp.type.value}, {comp.material})"
        )

    if result.simulation:
        status = "STABLE" if result.simulation.stable else "UNSTABLE"
        print(f"\nPhysics: {status}")
        print(f"  Position change: {result.simulation.position_change:.4f} m")
        print(f"  Rotation change: {result.simulation.rotation_change:.4f} rad")

    if result.manufacturing_result:
        mfg = result.manufacturing_result
        print(f"\nManufacturing score: {mfg.score.overall_score:.2f}")
        print(f"  Hausdorff: {mfg.score.hausdorff_mm:.1f} mm")
        print(f"  DFM errors: {mfg.score.dfm_violations_error}")
        print(f"  DFM warnings: {mfg.score.dfm_violations_warning}")
        print(f"  Structural plausibility: {mfg.score.structural_plausibility:.2f}")
        print(f"  Assembly steps: {len(mfg.assembly_steps)}")

    if result.svg_paths:
        print(f"\nSVG files: {len(result.svg_paths)} exported to {args.output}/")

    if result.dxf_paths:
        print(f"DXF files: {len(result.dxf_paths)} exported to {args.output}/")

    if result.design_json_path:
        print(f"Design JSON: {result.design_json_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
