#!/usr/bin/env python3
"""
Generate diverse furniture training data using genetic algorithm.

This script:
1. Evolves a population of freeform furniture designs
2. Filters for stable, load-bearing designs
3. Exports to multiple formats for ML training

Output formats:
- JSON: Full design data (geometry, materials, fitness scores)
- URDF: Physics simulation files
- PNG: Rendered images from multiple angles
"""
import sys
import os
import json
import argparse
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from evolution import EvolutionConfig, EvolutionEngine, Individual
from furniture import FurnitureDesign
from urdf_generator import furniture_to_urdf
import numpy as np


def export_design(
    individual: Individual,
    design_id: int,
    output_dir: Path,
    export_urdf: bool = True,
    export_json: bool = True,
) -> dict:
    """Export a single design to various formats."""
    design = individual.genome.to_furniture_design()
    design_dir = output_dir / f"design_{design_id:04d}"
    design_dir.mkdir(parents=True, exist_ok=True)
    
    export_data = {
        "design_id": design_id,
        "generation": individual.genome.generation,
        "num_components": len(individual.genome.get_active_components()),
    }
    
    # Export JSON (design data + fitness)
    if export_json:
        json_path = design_dir / "design.json"
        
        # Build component data with sequential indices
        active_components = individual.genome.get_active_components()
        old_to_new_idx = {}  # Map original indices to sequential 0,1,2...
        
        components = []
        for new_idx, (old_idx, gene) in enumerate(active_components):
            old_to_new_idx[old_idx] = new_idx
            comp_data = {
                "index": new_idx,
                "type": gene.component_type.value,
                "dimensions": {
                    "width": gene.width,
                    "height": gene.height,
                    "thickness": gene.thickness,
                },
                "position": {
                    "x": gene.x,
                    "y": gene.y,
                    "z": gene.z,
                },
                "rotation": {
                    "rx": gene.rx,
                    "ry": gene.ry,
                    "rz": gene.rz,
                },
                "material": gene.modifiers.get("material", "unknown"),
            }
            components.append(comp_data)
        
        # Build connection data
        connections = []
        for conn in individual.genome.get_active_connections():
            # Only include if both components are in our active set
            if conn.component_a_idx in old_to_new_idx and conn.component_b_idx in old_to_new_idx:
                conn_data = {
                    "from": old_to_new_idx[conn.component_a_idx],
                    "to": old_to_new_idx[conn.component_b_idx],
                    "joint_type": conn.joint_type.value,
                    "anchor_from": list(conn.anchor_a),
                    "anchor_to": list(conn.anchor_b),
                }
                connections.append(conn_data)
        
        # Fitness scores
        fitness_data = individual.fitness.to_dict() if individual.fitness else {}
        
        design_data = {
            "design_id": design_id,
            "generation": individual.genome.generation,
            "components": components,
            "connections": connections,
            "fitness": fitness_data,
            "novelty_score": individual.novelty_score,
            "fingerprint": individual.fingerprint.tolist() if individual.fingerprint is not None else [],
        }
        
        with open(json_path, 'w') as f:
            json.dump(design_data, f, indent=2)
        
        export_data["json_path"] = str(json_path)
    
    # Export URDF
    if export_urdf:
        urdf_path = design_dir / "model.urdf"
        urdf_content = furniture_to_urdf(design)
        
        with open(urdf_path, 'w') as f:
            f.write(urdf_content)
        
        export_data["urdf_path"] = str(urdf_path)
    
    return export_data


def main():
    parser = argparse.ArgumentParser(description="Generate furniture training data")
    parser.add_argument("--population", type=int, default=50,
                        help="Population size per generation")
    parser.add_argument("--generations", type=int, default=100,
                        help="Number of generations to evolve")
    parser.add_argument("--archive-size", type=int, default=1000,
                        help="Maximum designs to keep in archive")
    parser.add_argument("--output", type=str, default="training_data",
                        help="Output directory for training data")
    parser.add_argument("--min-fitness", type=float, default=0.4,
                        help="Minimum fitness score to export")
    parser.add_argument("--no-urdf", action="store_true",
                        help="Skip URDF export")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    if args.seed is not None:
        np.random.seed(args.seed)
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("FURNITURE TRAINING DATA GENERATOR")
    print("=" * 60)
    print(f"Population:     {args.population}")
    print(f"Generations:    {args.generations}")
    print(f"Archive size:   {args.archive_size}")
    print(f"Output dir:     {output_dir.absolute()}")
    print(f"Min fitness:    {args.min_fitness}")
    print("=" * 60)
    
    # Configure evolution
    config = EvolutionConfig(
        population_size=args.population,
        generations=args.generations,
        archive_size=args.archive_size,
        mutation_rate=0.3,
        crossover_rate=0.3,
        elitism=5,
        novelty_k=15,
        min_archive_distance=0.12,  # Slightly lower for more diversity
        simulation_duration=2.0,
        load_test=True,
        output_dir=str(output_dir / "evolution"),
        save_interval=10,
    )
    
    # Run evolution
    start_time = time.time()
    
    engine = EvolutionEngine(config)
    archive = engine.run()
    
    evolution_time = time.time() - start_time
    print(f"\nEvolution completed in {evolution_time:.1f}s")
    
    # Filter and export designs
    print("\n" + "=" * 60)
    print("EXPORTING TRAINING DATA")
    print("=" * 60)
    
    # Filter by fitness
    qualified = [
        ind for ind in archive 
        if ind.fitness and ind.fitness.total >= args.min_fitness
    ]
    
    print(f"Qualified designs: {len(qualified)}/{len(archive)} (fitness >= {args.min_fitness})")
    
    # Export each design
    designs_dir = output_dir / "designs"
    export_manifest = []
    
    for i, ind in enumerate(qualified):
        export_data = export_design(
            ind, 
            design_id=i,
            output_dir=designs_dir,
            export_urdf=not args.no_urdf,
        )
        export_manifest.append(export_data)
        
        if (i + 1) % 50 == 0:
            print(f"  Exported {i + 1}/{len(qualified)} designs...")
    
    # Save manifest
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, 'w') as f:
        json.dump({
            "total_designs": len(qualified),
            "evolution_config": {
                "population_size": args.population,
                "generations": args.generations,
                "archive_size": args.archive_size,
            },
            "min_fitness": args.min_fitness,
            "designs": export_manifest,
        }, f, indent=2)
    
    # Save evolution stats
    stats_path = output_dir / "evolution_stats.json"
    with open(stats_path, 'w') as f:
        json.dump(engine.stats_history, f, indent=2)
    
    # Summary stats
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total designs exported:  {len(qualified)}")
    print(f"Output directory:        {output_dir.absolute()}")
    print(f"Manifest:                {manifest_path}")
    
    if qualified:
        fitnesses = [ind.fitness.total for ind in qualified]
        symmetries = [ind.fitness.mirror_symmetry_x for ind in qualified]
        
        print(f"\nFitness range:           {min(fitnesses):.3f} - {max(fitnesses):.3f}")
        print(f"Mean fitness:            {np.mean(fitnesses):.3f}")
        print(f"Mean X-symmetry:         {np.mean(symmetries):.3f}")
    
    print(f"\nTotal time:              {time.time() - start_time:.1f}s")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
