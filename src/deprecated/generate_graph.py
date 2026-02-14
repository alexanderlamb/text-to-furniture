#!/usr/bin/env python3
"""
Generate furniture from trained graph model.

The graph model outputs:
- Component properties (dimensions, material)
- Connection graph (parent, face, offset)

This script converts that to 3D positions and simulates physics.
"""
import sys
import os
import json
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
import numpy as np

from model_graph import GraphFurnitureTransformer, GraphModelConfig, FACE_VOCAB
from furniture import FurnitureDesign, Component, ComponentType
from simulator import FurnitureSimulator
from fitness import evaluate_design


def compute_position_from_attachment(
    parent_dims: dict,
    parent_pos: np.ndarray,
    child_dims: dict,
    face: str,
    offset: list,
) -> np.ndarray:
    """
    Compute child position based on attachment to parent.
    
    Args:
        parent_dims: {"width", "height", "thickness"}
        parent_pos: (x, y, z) of parent
        child_dims: {"width", "height", "thickness"}  
        face: Which face of parent to attach to
        offset: [u, v] position on that face (0-1)
    
    Returns:
        (x, y, z) position for child
    """
    pw, ph, pt = parent_dims["width"], parent_dims["height"], parent_dims["thickness"]
    cw, ch, ct = child_dims["width"], child_dims["height"], child_dims["thickness"]
    u, v = offset
    
    px, py, pz = parent_pos
    
    if face == "z+":  # Top of parent
        x = px + u * pw - cw / 2
        y = py + v * ph - ch / 2
        z = pz + pt
    elif face == "z-":  # Bottom of parent
        x = px + u * pw - cw / 2
        y = py + v * ph - ch / 2
        z = pz - ct
    elif face == "x+":  # Right of parent
        x = px + pw
        y = py + u * ph - ch / 2
        z = pz + v * pt - ct / 2
    elif face == "x-":  # Left of parent
        x = px - cw
        y = py + u * ph - ch / 2
        z = pz + v * pt - ct / 2
    elif face == "y+":  # Back of parent
        x = px + u * pw - cw / 2
        y = py + ph
        z = pz + v * pt - ct / 2
    elif face == "y-":  # Front of parent
        x = px + u * pw - cw / 2
        y = py - ch
        z = pz + v * pt - ct / 2
    else:
        x, y, z = px, py, pz + pt
    
    return np.array([x, y, max(0, z)])


def graph_to_design(components: list, connections: list, name: str = "generated") -> FurnitureDesign:
    """
    Convert graph model output to FurnitureDesign with computed positions.
    """
    design = FurnitureDesign(name=name)
    
    if not components:
        return design
    
    # Track positions for each component
    positions = [None] * len(components)
    
    # First component starts at origin
    positions[0] = np.array([0.0, 0.0, 0.0])
    
    # Build position for each component based on its connection
    # Process in order (connections are from parent to child)
    for conn in connections:
        parent_idx = conn["from"]
        child_idx = conn["to"]
        face = conn["face"]
        offset = conn["offset"]
        
        if parent_idx < len(components) and child_idx < len(components):
            if positions[parent_idx] is not None:
                positions[child_idx] = compute_position_from_attachment(
                    components[parent_idx]["dimensions"],
                    positions[parent_idx],
                    components[child_idx]["dimensions"],
                    face,
                    offset,
                )
    
    # Fill any missing positions (fallback to origin)
    for i in range(len(positions)):
        if positions[i] is None:
            positions[i] = np.array([0.0, 0.0, float(i) * 100])
    
    # Create furniture components
    for i, (comp, pos) in enumerate(zip(components, positions)):
        dims = comp["dimensions"]
        rot = comp["rotation"]
        
        profile = [
            (0.0, 0.0),
            (dims["width"], 0.0),
            (dims["width"], dims["height"]),
            (0.0, dims["height"]),
        ]
        
        component = Component(
            name=f"part_{i}",
            type=ComponentType.PANEL,
            profile=profile,
            thickness=dims["thickness"],
            material=comp.get("material", "plywood"),
            position=pos,
            rotation=np.array([rot["rx"], rot["ry"], rot["rz"]]),
        )
        
        design.add_component(component)
    
    return design


def main():
    parser = argparse.ArgumentParser(description="Generate from graph model")
    parser.add_argument("--model", type=str, required=True, help="Model checkpoint")
    parser.add_argument("--num", type=int, default=10, help="Number to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--max-components", type=int, default=10, help="Max components")
    parser.add_argument("--visualize", action="store_true", help="Show in PyBullet")
    parser.add_argument("--output", type=str, default=None, help="Output directory")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    
    args = parser.parse_args()
    
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
    
    # Load model
    print(f"Loading model from {args.model}...")
    checkpoint = torch.load(args.model, map_location="cpu", weights_only=False)
    
    config = checkpoint.get("config")
    if config is None:
        config = GraphModelConfig(d_model=256, n_layers=6, n_graph_layers=3)
    
    model = GraphFurnitureTransformer(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    print(f"Model loaded (epoch {checkpoint.get('epoch', '?')})")
    
    # Setup simulator
    simulator = FurnitureSimulator(gui=args.visualize)
    
    # Generate
    print(f"\nGenerating {args.num} designs (temp={args.temperature})...")
    print("=" * 60)
    
    results = []
    stable_count = 0
    
    for i in range(args.num):
        # Generate graph
        components, connections = model.generate(
            max_components=args.max_components,
            temperature=args.temperature,
        )
        
        if not components:
            print(f"Design {i+1}: Empty")
            continue
        
        # Convert to design
        design = graph_to_design(components, connections, name=f"gen_{i}")
        
        # Simulate
        result = simulator.simulate(design, duration=2.0, load_test=True)
        
        # Evaluate
        fitness = evaluate_design(design, result.to_dict())
        
        status = "STABLE" if result.stable else "UNSTABLE"
        if result.stable:
            stable_count += 1
        
        print(f"Design {i+1}: {len(components)} parts, {len(connections)} connections | "
              f"{status} | Fitness: {fitness.total:.3f}")
        
        results.append({
            "components": components,
            "connections": connections,
            "stable": result.stable,
            "fitness": fitness.total,
        })
        
        if args.visualize and result.stable:
            input("  Press Enter for next...")
    
    print("=" * 60)
    print(f"Generated {len(results)} designs, {stable_count} stable ({100*stable_count/max(len(results),1):.0f}%)")
    
    if stable_count > 0:
        stable_fitness = [r["fitness"] for r in results if r["stable"]]
        print(f"Stable fitness: {min(stable_fitness):.3f} - {max(stable_fitness):.3f} (mean {np.mean(stable_fitness):.3f})")
    
    # Save
    if args.output:
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        with open(output_dir / "results.json", 'w') as f:
            json.dump(results, f, indent=2, default=lambda x: bool(x) if isinstance(x, np.bool_) else x)
        
        print(f"Saved to {output_dir}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
