#!/usr/bin/env python3
"""
Decompose a 3D mesh into flat-pack furniture components.

Takes an AI-generated or hand-modelled mesh and produces a FurnitureDesign
with rectangular components suitable for SendCutSend manufacturing.

Usage:
    python scripts/decompose_mesh.py --input model.stl
    python scripts/decompose_mesh.py --input model.glb --material plywood_baltic_birch --height 750
    python scripts/decompose_mesh.py --input model.obj --max-slabs 12 --output output_dir/ --simulate --export-svg
"""
import sys
import os
import json
import argparse
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mesh_decomposer import decompose, DecompositionConfig
from materials import MATERIALS


def main():
    parser = argparse.ArgumentParser(
        description="Decompose a 3D mesh into flat-pack furniture components.",
    )
    parser.add_argument(
        "--input", required=True,
        help="Path to input mesh file (STL, OBJ, GLB, PLY)",
    )
    parser.add_argument(
        "--output", default=None,
        help="Output directory (default: <input_dir>/<input_stem>_decomposed/)",
    )
    parser.add_argument(
        "--material", default="plywood_baltic_birch",
        choices=list(MATERIALS.keys()),
        help="SendCutSend material key (default: plywood_baltic_birch)",
    )
    parser.add_argument(
        "--height", type=float, default=750.0,
        help="Target height in mm (default: 750)",
    )
    parser.add_argument(
        "--max-slabs", type=int, default=15,
        help="Maximum number of slabs (default: 15)",
    )
    parser.add_argument(
        "--coverage-target", type=float, default=0.80,
        help="Voxel coverage target 0-1 (default: 0.80)",
    )
    parser.add_argument(
        "--no-optimize", action="store_true",
        help="Skip the local optimization pass",
    )
    parser.add_argument(
        "--simulate", action="store_true",
        help="Run physics simulation on the result",
    )
    parser.add_argument(
        "--export-svg", action="store_true",
        help="Export SVG cut files for each component",
    )
    parser.add_argument(
        "--visualize", action="store_true",
        help="Show matplotlib figure with all pipeline stages",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    # Logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    # Resolve paths
    input_path = os.path.abspath(args.input)
    if not os.path.isfile(input_path):
        parser.error(f"Input file not found: {input_path}")

    if args.output:
        output_dir = os.path.abspath(args.output)
    else:
        stem = Path(input_path).stem
        output_dir = os.path.join(os.path.dirname(input_path), f"{stem}_decomposed")
    os.makedirs(output_dir, exist_ok=True)

    # Build config
    config = DecompositionConfig(
        default_material=args.material,
        target_height_mm=args.height,
        max_slabs=args.max_slabs,
        coverage_target=args.coverage_target,
    )

    # Run decomposition (with visualization if requested)
    if args.visualize:
        from visualize_decomposition import run_pipeline_with_visualization
        design = run_pipeline_with_visualization(
            input_path, config, output_dir, optimize=not args.no_optimize,
        )
    else:
        print(f"Decomposing {input_path} ...")
        design = decompose(
            filepath=input_path,
            config=config,
            optimize=not args.no_optimize,
        )

    # Summary
    is_valid, errors = design.validate()
    print(f"\nResult: {len(design.components)} components, "
          f"{len(design.assembly.joints)} joints")
    if errors:
        print(f"Validation warnings: {errors}")

    for comp in design.components:
        dims = comp.get_dimensions()
        print(f"  {comp.name}: {dims[0]:.0f} x {dims[1]:.0f} x {comp.thickness:.1f} mm "
              f"({comp.type.value})")

    # Save design as JSON
    design_json = {
        "name": design.name,
        "components": [
            {
                "name": c.name,
                "type": c.type.value,
                "profile": c.profile,
                "thickness": c.thickness,
                "position": c.position.tolist(),
                "rotation": c.rotation.tolist(),
                "material": c.material,
                "dimensions": list(c.get_dimensions()),
            }
            for c in design.components
        ],
        "joints": [
            {
                "component_a": j.component_a,
                "component_b": j.component_b,
                "joint_type": j.joint_type.value,
                "position_a": list(j.position_a),
                "position_b": list(j.position_b),
            }
            for j in design.assembly.joints
        ],
    }
    json_path = os.path.join(output_dir, "design.json")
    with open(json_path, "w") as f:
        json.dump(design_json, f, indent=2)
    print(f"\nDesign saved to {json_path}")

    # Optional: export SVGs
    if args.export_svg:
        from svg_exporter import design_to_svg
        svg_dir = os.path.join(output_dir, "svg")
        paths = design_to_svg(design, svg_dir)
        print(f"Exported {len(paths)} SVG cut files to {svg_dir}")

    # Optional: physics simulation
    if args.simulate:
        from simulator import FurnitureSimulator
        print("\nRunning physics simulation ...")
        sim = FurnitureSimulator(gui=False)
        try:
            result = sim.simulate(design, duration=3.0)
            status = "STABLE" if result.stable else "UNSTABLE"
            print(f"  Result: {status}")
            print(f"  Position change: {result.position_change:.4f} m")
            print(f"  Rotation change: {float(result.rotation_change):.2f} rad")
            print(f"  Height drop: {result.height_drop:.4f} m")

            sim_json = result.to_dict()
            sim_path = os.path.join(output_dir, "simulation.json")
            with open(sim_path, "w") as f:
                json.dump(sim_json, f, indent=2)
            print(f"  Simulation results saved to {sim_path}")
        finally:
            sim.close()

    print("\nDone.")


if __name__ == "__main__":
    main()
