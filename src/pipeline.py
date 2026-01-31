"""
End-to-end pipeline demonstration.

Shows the complete flow:
1. Generate random furniture genome
2. Convert to FurnitureDesign
3. Run physics simulation
4. Export SVG cut files (if stable)

This is the foundation for the genetic algorithm - each design
goes through this pipeline to determine fitness.
"""
import os
import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass

from genome import (
    FurnitureGenome, create_table_genome, create_shelf_genome, 
    create_random_genome
)
from simulator import FurnitureSimulator, SimulationResult
from svg_exporter import design_to_svg, design_to_nested_svg


@dataclass
class PipelineResult:
    """Result from processing a single design through the pipeline."""
    genome: FurnitureGenome
    design_name: str
    simulation: SimulationResult
    svg_paths: List[str]
    nested_svg: Optional[str]
    
    @property
    def is_valid(self) -> bool:
        """Design is valid if it's physically stable."""
        return self.simulation.stable


def process_design(
    genome: FurnitureGenome,
    design_name: str,
    output_dir: str,
    simulator: FurnitureSimulator,
    export_svg: bool = True,
    sim_duration: float = 2.0,
) -> PipelineResult:
    """
    Process a single furniture genome through the full pipeline.
    
    Args:
        genome: Furniture genome to process
        design_name: Name for this design
        output_dir: Directory for output files
        simulator: Pre-initialized simulator instance
        export_svg: Whether to export SVG files
        sim_duration: Physics simulation duration
    
    Returns:
        PipelineResult with all outputs
    """
    # Convert genome to design
    design = genome.to_furniture_design(design_name)
    
    # Run physics simulation
    sim_result = simulator.simulate(design, duration=sim_duration)
    
    # Export SVG if requested and design is stable
    svg_paths = []
    nested_svg = None
    
    if export_svg and sim_result.stable:
        design_dir = os.path.join(output_dir, design_name)
        svg_paths = design_to_svg(design, design_dir)
        nested_svg = os.path.join(design_dir, f"{design_name}_nested.svg")
        design_to_nested_svg(design, nested_svg)
    
    return PipelineResult(
        genome=genome,
        design_name=design_name,
        simulation=sim_result,
        svg_paths=svg_paths,
        nested_svg=nested_svg,
    )


def batch_evaluate(
    genomes: List[FurnitureGenome],
    output_dir: str,
    export_svg: bool = False,
    sim_duration: float = 2.0,
) -> List[PipelineResult]:
    """
    Evaluate a batch of genomes through the pipeline.
    
    This is efficient because it reuses a single simulator instance.
    
    Args:
        genomes: List of genomes to evaluate
        output_dir: Directory for output files
        export_svg: Whether to export SVG files for stable designs
        sim_duration: Physics simulation duration per design
    
    Returns:
        List of PipelineResult objects
    """
    simulator = FurnitureSimulator(gui=False)
    results = []
    
    try:
        for i, genome in enumerate(genomes):
            result = process_design(
                genome,
                design_name=f"design_{i:04d}",
                output_dir=output_dir,
                simulator=simulator,
                export_svg=export_svg,
                sim_duration=sim_duration,
            )
            results.append(result)
    finally:
        simulator.close()
    
    return results


def generate_and_evaluate(
    n_designs: int,
    furniture_type: str = None,
    output_dir: str = "output",
    export_svg: bool = True,
) -> Tuple[List[PipelineResult], dict]:
    """
    Generate random designs and evaluate them.
    
    This demonstrates the basic loop that the GA will use:
    1. Generate random genomes
    2. Evaluate fitness via physics sim
    3. Keep the stable ones
    
    Args:
        n_designs: Number of designs to generate
        furniture_type: Type of furniture ("table", "shelf", or None for random)
        output_dir: Directory for output files
        export_svg: Export SVG for stable designs
    
    Returns:
        (results, stats_dict)
    """
    print(f"Generating {n_designs} random {furniture_type or 'mixed'} designs...")
    
    # Generate genomes
    genomes = [create_random_genome(furniture_type) for _ in range(n_designs)]
    
    # Evaluate
    print("Running physics simulations...")
    results = batch_evaluate(
        genomes, 
        output_dir, 
        export_svg=export_svg,
        sim_duration=2.0
    )
    
    # Compute stats
    stable_count = sum(1 for r in results if r.is_valid)
    unstable_count = n_designs - stable_count
    
    stats = {
        "total": n_designs,
        "stable": stable_count,
        "unstable": unstable_count,
        "stability_rate": stable_count / n_designs,
    }
    
    return results, stats


if __name__ == "__main__":
    import time
    
    print("="*60)
    print("TEXT-TO-FURNITURE: End-to-End Pipeline Demo")
    print("="*60)
    
    output_dir = "/Users/alexanderlamb/projects/text-to-furniture/examples/pipeline_output"
    os.makedirs(output_dir, exist_ok=True)
    
    # Demo 1: Single design through full pipeline
    print("\n--- Demo 1: Single Design Pipeline ---")
    
    genome = create_table_genome()
    design = genome.to_furniture_design("demo_table")
    
    print(f"Generated table with {len(design.components)} components")
    
    simulator = FurnitureSimulator(gui=False)
    sim_result = simulator.simulate(design, duration=2.0)
    simulator.close()
    
    print(f"Physics simulation: {'STABLE ✓' if sim_result.stable else 'UNSTABLE ✗'}")
    print(f"  - Rotation: {np.degrees(sim_result.rotation_change):.1f}°")
    print(f"  - Position change: {sim_result.position_change:.4f}m")
    
    if sim_result.stable:
        svg_paths = design_to_svg(design, os.path.join(output_dir, "demo_table"))
        print(f"Exported {len(svg_paths)} SVG cut files")
    
    # Demo 2: Batch evaluation - Tables
    print("\n--- Demo 2: Batch Evaluation (20 Tables) ---")
    start = time.time()
    
    results, stats = generate_and_evaluate(
        n_designs=20,
        furniture_type="table",
        output_dir=output_dir,
        export_svg=True,
    )
    
    elapsed = time.time() - start
    print(f"\nResults:")
    print(f"  - Stable: {stats['stable']}/{stats['total']} ({stats['stability_rate']*100:.0f}%)")
    print(f"  - Time: {elapsed:.1f}s ({elapsed/20:.2f}s per design)")
    
    # Show some details
    print("\nSample results:")
    for r in results[:5]:
        status = "✓" if r.is_valid else "✗"
        rot = np.degrees(r.simulation.rotation_change)
        print(f"  {r.design_name}: {status} (rot={rot:.1f}°)")
    
    # Demo 3: Batch evaluation - Shelves
    print("\n--- Demo 3: Batch Evaluation (20 Shelves) ---")
    start = time.time()
    
    results, stats = generate_and_evaluate(
        n_designs=20,
        furniture_type="shelf",
        output_dir=output_dir,
        export_svg=True,
    )
    
    elapsed = time.time() - start
    print(f"\nResults:")
    print(f"  - Stable: {stats['stable']}/{stats['total']} ({stats['stability_rate']*100:.0f}%)")
    print(f"  - Time: {elapsed:.1f}s ({elapsed/20:.2f}s per design)")
    
    # Demo 4: Mixed batch
    print("\n--- Demo 4: Mixed Batch (50 Designs) ---")
    start = time.time()
    
    results, stats = generate_and_evaluate(
        n_designs=50,
        furniture_type=None,  # Random mix
        output_dir=output_dir,
        export_svg=False,  # Skip SVG for speed
    )
    
    elapsed = time.time() - start
    print(f"\nResults:")
    print(f"  - Stable: {stats['stable']}/{stats['total']} ({stats['stability_rate']*100:.0f}%)")
    print(f"  - Time: {elapsed:.1f}s ({elapsed/50:.2f}s per design)")
    print(f"  - Throughput: {50/elapsed:.1f} designs/second")
    
    print("\n" + "="*60)
    print("Pipeline demo complete!")
    print(f"Output saved to: {output_dir}")
    print("="*60)
