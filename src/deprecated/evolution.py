"""
Genetic Algorithm for evolving freeform furniture designs.

Key features:
- Multi-objective optimization (stability, aesthetics, efficiency)
- Diversity preservation through novelty search and niching
- Island model for parallel exploration of different design spaces

This generates diverse training data, not just "the best" design.
"""
import numpy as np
from typing import List, Tuple, Dict, Optional, Set
from dataclasses import dataclass, field
import copy
import time
import json
import os

from genome import FurnitureGenome, ComponentGene, ConnectionGene, JointType
from freeform_genome import (
    create_freeform_genome, 
    random_material, 
    random_thickness,
    MATERIALS,
    MIN_OVERLAP_MM,
    AttachmentFace,
    gene_to_bbox,
    compute_attachment_position,
)
from furniture import ComponentType, FurnitureDesign
from simulator import FurnitureSimulator, SimulationResult
from fitness import FitnessScores, evaluate_design


# =============================================================================
# Design Fingerprinting for Diversity
# =============================================================================

def compute_design_fingerprint(genome: FurnitureGenome) -> np.ndarray:
    """
    Compute a fingerprint vector that captures structural properties.
    
    Used for diversity comparison - similar fingerprints = similar designs.
    """
    components = [g for g in genome.components if g.active]
    
    if not components:
        return np.zeros(20)
    
    # Feature vector components:
    features = []
    
    # 1. Component count (normalized)
    features.append(len(components) / 15.0)
    
    # 2. Bounding box aspect ratios
    if components:
        positions = np.array([[g.x, g.y, g.z] for g in components])
        dims = np.array([[g.width, g.height, g.thickness] for g in components])
        
        mins = positions.min(axis=0)
        maxs = (positions + dims).max(axis=0)
        bbox = maxs - mins
        
        # Aspect ratios (clamped)
        if bbox[1] > 0:
            features.append(np.clip(bbox[0] / bbox[1], 0, 3) / 3)
        else:
            features.append(0.5)
        if bbox[2] > 0:
            features.append(np.clip(bbox[0] / bbox[2], 0, 3) / 3)
            features.append(np.clip(bbox[1] / bbox[2], 0, 3) / 3)
        else:
            features.append(0.5)
            features.append(0.5)
    else:
        features.extend([0.5, 0.5, 0.5])
    
    # 3. Height distribution (histogram of component centers)
    if components and bbox[2] > 0:
        heights = [(g.z + g.thickness/2 - mins[2]) / bbox[2] for g in components]
        hist, _ = np.histogram(heights, bins=4, range=(0, 1))
        features.extend(hist / max(len(components), 1))
    else:
        features.extend([0.25, 0.25, 0.25, 0.25])
    
    # 4. Rotation statistics
    rotations = np.array([[g.rx, g.ry, g.rz] for g in components])
    features.append(np.mean(np.abs(rotations[:, 0])) / 0.5)  # Avg X rotation
    features.append(np.mean(np.abs(rotations[:, 1])) / 0.5)  # Avg Y rotation
    features.append(np.mean(np.abs(rotations[:, 2])) / 0.5)  # Avg Z rotation
    features.append(np.sum(np.any(np.abs(rotations) > 0.01, axis=1)) / len(components))  # Fraction rotated
    
    # 5. Material diversity
    materials = set()
    for g in components:
        mat = g.modifiers.get("material", "unknown")
        materials.add(mat)
    features.append(len(materials) / 5.0)  # Normalized by typical max
    
    # 6. Size variation
    volumes = [g.width * g.height * g.thickness for g in components]
    if len(volumes) > 1:
        features.append(np.std(volumes) / (np.mean(volumes) + 1))
    else:
        features.append(0)
    
    # 7. Connection density
    active_connections = [c for c in genome.connections if c.active]
    max_connections = len(components) * (len(components) - 1) / 2
    if max_connections > 0:
        features.append(len(active_connections) / max_connections)
    else:
        features.append(0)
    
    # Pad to fixed length
    while len(features) < 20:
        features.append(0)
    
    return np.array(features[:20])


def fingerprint_distance(fp1: np.ndarray, fp2: np.ndarray) -> float:
    """Euclidean distance between fingerprints."""
    return np.linalg.norm(fp1 - fp2)


# =============================================================================
# Mutation Operators
# =============================================================================

def mutate_dimensions(gene: ComponentGene, strength: float = 0.2) -> ComponentGene:
    """Mutate component dimensions."""
    gene = copy.deepcopy(gene)
    
    # Random scaling
    scale = 1.0 + np.random.uniform(-strength, strength)
    gene.width *= scale
    gene.height *= scale
    
    # Clamp to reasonable sizes
    gene.width = np.clip(gene.width, MIN_OVERLAP_MM * 2, 1000)
    gene.height = np.clip(gene.height, MIN_OVERLAP_MM * 2, 1000)
    
    return gene


def mutate_position(gene: ComponentGene, strength: float = 50.0) -> ComponentGene:
    """Mutate component position."""
    gene = copy.deepcopy(gene)
    
    gene.x += np.random.uniform(-strength, strength)
    gene.y += np.random.uniform(-strength, strength)
    gene.z += np.random.uniform(-strength, strength)
    
    # Keep above ground
    gene.z = max(0, gene.z)
    
    return gene


def mutate_rotation(gene: ComponentGene, strength: float = 0.2) -> ComponentGene:
    """Mutate component rotation."""
    gene = copy.deepcopy(gene)
    
    gene.rx += np.random.uniform(-strength, strength)
    gene.ry += np.random.uniform(-strength, strength)
    gene.rz += np.random.uniform(-strength, strength)
    
    # Clamp to reasonable range
    gene.rx = np.clip(gene.rx, -0.5, 0.5)
    gene.ry = np.clip(gene.ry, -0.5, 0.5)
    gene.rz = np.clip(gene.rz, -0.5, 0.5)
    
    return gene


def mutate_material(gene: ComponentGene) -> ComponentGene:
    """Change component material."""
    gene = copy.deepcopy(gene)
    
    mat_name, material = random_material()
    gene.thickness = random_thickness(material)
    gene.modifiers["material"] = mat_name
    
    return gene


def add_component(genome: FurnitureGenome) -> FurnitureGenome:
    """Add a new component attached to an existing one."""
    genome = genome.copy()
    
    active_components = genome.get_active_components()
    if not active_components:
        return genome
    
    # Pick random parent
    parent_idx, parent_gene = active_components[np.random.randint(len(active_components))]
    parent_bbox = gene_to_bbox(parent_gene)
    
    # Generate new component
    mat_name, material = random_material()
    thickness = random_thickness(material)
    width = np.random.uniform(100, 600)
    height = np.random.uniform(100, 600)
    
    # Random face and position
    face = np.random.choice(list(AttachmentFace))
    if face == AttachmentFace.Z_NEG and parent_bbox.z_min <= 0:
        face = AttachmentFace.Z_POS
    
    offset_u = np.random.uniform(0.3, 0.7)
    offset_v = np.random.uniform(0.3, 0.7)
    
    # Random rotation
    angle = (0.0, 0.0, 0.0)
    if np.random.random() < 0.3:
        angle = (
            np.random.uniform(-0.3, 0.3),
            np.random.uniform(-0.3, 0.3),
            np.random.uniform(-0.3, 0.3),
        )
    
    (x, y, z), (rx, ry, rz) = compute_attachment_position(
        parent_bbox, face, width, height, thickness, offset_u, offset_v, angle
    )
    
    new_gene = ComponentGene(
        component_type=ComponentType.PANEL,
        width=width,
        height=height,
        thickness=thickness,
        x=max(0, x) if z <= 0 else x,
        y=y,
        z=max(0, z),
        rx=rx, ry=ry, rz=rz,
    )
    new_gene.modifiers["material"] = mat_name
    
    new_idx = genome.add_component(new_gene)
    genome.add_connection(ConnectionGene(
        component_a_idx=parent_idx,
        component_b_idx=new_idx,
        joint_type=JointType.BUTT,
    ))
    
    return genome


def remove_component(genome: FurnitureGenome) -> FurnitureGenome:
    """Remove a random component (mark as inactive)."""
    genome = genome.copy()
    
    active_components = genome.get_active_components()
    if len(active_components) <= 3:  # Keep minimum
        return genome
    
    # Pick random component to remove (not the first one)
    idx, _ = active_components[np.random.randint(1, len(active_components))]
    genome.components[idx].active = False
    
    return genome


def mutate_genome(genome: FurnitureGenome, mutation_rate: float = 0.3) -> FurnitureGenome:
    """Apply random mutations to a genome."""
    genome = genome.copy()
    
    # Structural mutations (add/remove components)
    if np.random.random() < mutation_rate * 0.3:
        if np.random.random() < 0.6:
            genome = add_component(genome)
        else:
            genome = remove_component(genome)
    
    # Component-level mutations
    for i, gene in enumerate(genome.components):
        if not gene.active:
            continue
        
        if np.random.random() < mutation_rate:
            mutation_type = np.random.choice(['dims', 'pos', 'rot', 'mat'], p=[0.3, 0.3, 0.25, 0.15])
            
            if mutation_type == 'dims':
                genome.components[i] = mutate_dimensions(gene)
            elif mutation_type == 'pos':
                genome.components[i] = mutate_position(gene)
            elif mutation_type == 'rot':
                genome.components[i] = mutate_rotation(gene)
            else:
                genome.components[i] = mutate_material(gene)
    
    genome.generation += 1
    return genome


# =============================================================================
# Crossover Operators
# =============================================================================

def crossover_genomes(parent1: FurnitureGenome, parent2: FurnitureGenome) -> FurnitureGenome:
    """Create offspring by combining two parent genomes."""
    # Start with copy of parent1
    child = parent1.copy()
    
    # Take some components from parent2
    p2_components = parent2.get_active_components()
    
    if p2_components:
        # Add 1-3 components from parent2
        num_to_add = min(len(p2_components), np.random.randint(1, 4))
        indices = np.random.choice(len(p2_components), num_to_add, replace=False)
        
        for idx in indices:
            _, gene = p2_components[idx]
            gene_copy = copy.deepcopy(gene)
            
            # Adjust position to connect to child
            child_components = child.get_active_components()
            if child_components:
                # Attach to random existing component
                parent_idx, parent_gene = child_components[np.random.randint(len(child_components))]
                parent_bbox = gene_to_bbox(parent_gene)
                
                # Reposition the gene near the parent
                gene_copy.x = parent_gene.x + np.random.uniform(-100, 100)
                gene_copy.y = parent_gene.y + np.random.uniform(-100, 100)
                gene_copy.z = max(0, parent_gene.z + np.random.uniform(-50, 200))
                
                new_idx = child.add_component(gene_copy)
                child.add_connection(ConnectionGene(
                    component_a_idx=parent_idx,
                    component_b_idx=new_idx,
                    joint_type=JointType.BUTT,
                ))
    
    child.generation = max(parent1.generation, parent2.generation) + 1
    return child


# =============================================================================
# Selection with Diversity
# =============================================================================

@dataclass
class Individual:
    """An individual in the population."""
    genome: FurnitureGenome
    fitness: FitnessScores = None
    fingerprint: np.ndarray = None
    novelty_score: float = 0.0
    simulation_result: SimulationResult = None
    
    @property
    def combined_score(self) -> float:
        """Combined fitness + novelty score."""
        if self.fitness is None:
            return 0.0
        # Weight novelty to encourage diversity
        return self.fitness.total * 0.7 + self.novelty_score * 0.3


def compute_novelty(individual: Individual, archive: List[Individual], k: int = 15) -> float:
    """
    Compute novelty score based on distance to k-nearest neighbors.
    
    Higher score = more different from existing designs.
    """
    if individual.fingerprint is None:
        return 0.0
    
    if not archive:
        return 1.0
    
    # Compute distances to all archive members
    distances = []
    for other in archive:
        if other.fingerprint is not None:
            dist = fingerprint_distance(individual.fingerprint, other.fingerprint)
            distances.append(dist)
    
    if not distances:
        return 1.0
    
    # Average distance to k nearest neighbors
    distances.sort()
    k_nearest = distances[:min(k, len(distances))]
    
    novelty = np.mean(k_nearest)
    
    # Normalize to 0-1 range (empirically tuned)
    return min(novelty / 2.0, 1.0)


def tournament_selection(population: List[Individual], tournament_size: int = 3) -> Individual:
    """Select individual using tournament selection."""
    tournament = np.random.choice(population, size=min(tournament_size, len(population)), replace=False)
    return max(tournament, key=lambda x: x.combined_score)


def select_diverse(population: List[Individual], num_select: int, min_distance: float = 0.15) -> List[Individual]:
    """
    Select individuals while maintaining diversity.
    
    Uses a greedy algorithm that picks best individuals while ensuring
    minimum fingerprint distance between selections.
    """
    if not population:
        return []
    
    # Sort by combined score
    sorted_pop = sorted(population, key=lambda x: x.combined_score, reverse=True)
    
    selected = []
    for ind in sorted_pop:
        if len(selected) >= num_select:
            break
        
        # Check distance to already selected
        if ind.fingerprint is not None:
            too_close = False
            for sel in selected:
                if sel.fingerprint is not None:
                    if fingerprint_distance(ind.fingerprint, sel.fingerprint) < min_distance:
                        too_close = True
                        break
            
            if not too_close:
                selected.append(ind)
        else:
            selected.append(ind)
    
    # If we couldn't get enough diverse individuals, fill with best remaining
    if len(selected) < num_select:
        for ind in sorted_pop:
            if ind not in selected:
                selected.append(ind)
                if len(selected) >= num_select:
                    break
    
    return selected


# =============================================================================
# Evolution Engine
# =============================================================================

@dataclass
class EvolutionConfig:
    """Configuration for evolution run."""
    population_size: int = 50
    generations: int = 100
    mutation_rate: float = 0.3
    crossover_rate: float = 0.3
    elitism: int = 5
    tournament_size: int = 3
    
    # Diversity settings
    archive_size: int = 500  # Max designs to keep in archive
    novelty_k: int = 15  # K for k-nearest novelty
    min_archive_distance: float = 0.15  # Min fingerprint distance for archive
    
    # Simulation settings
    simulation_duration: float = 2.0
    load_test: bool = True
    
    # Output settings
    output_dir: str = "evolved_designs"
    save_interval: int = 10  # Save every N generations


class EvolutionEngine:
    """
    Main evolution engine for generating diverse furniture designs.
    """
    
    def __init__(self, config: EvolutionConfig = None):
        self.config = config or EvolutionConfig()
        self.simulator = FurnitureSimulator(gui=False)
        
        self.population: List[Individual] = []
        self.archive: List[Individual] = []  # Best diverse designs
        self.generation = 0
        
        # Statistics
        self.stats_history = []
    
    def initialize_population(self):
        """Create initial random population."""
        self.population = []
        
        for _ in range(self.config.population_size):
            genome = create_freeform_genome()
            ind = Individual(genome=genome)
            ind.fingerprint = compute_design_fingerprint(genome)
            self.population.append(ind)
        
        print(f"Initialized population with {len(self.population)} individuals")
    
    def evaluate_individual(self, individual: Individual) -> Individual:
        """Evaluate fitness of an individual."""
        design = individual.genome.to_furniture_design()
        
        # Run physics simulation
        result = self.simulator.simulate(
            design,
            duration=self.config.simulation_duration,
            load_test=self.config.load_test,
        )
        individual.simulation_result = result
        
        # Compute fitness scores
        individual.fitness = evaluate_design(design, result.to_dict())
        
        # Compute novelty
        individual.novelty_score = compute_novelty(individual, self.archive, self.config.novelty_k)
        
        return individual
    
    def evaluate_population(self):
        """Evaluate all individuals in population."""
        for i, ind in enumerate(self.population):
            if ind.fitness is None:
                self.population[i] = self.evaluate_individual(ind)
    
    def update_archive(self):
        """Add good, diverse individuals to archive."""
        for ind in self.population:
            if ind.fitness is None:
                continue
            
            # Must be stable to enter archive
            if ind.fitness.stability < 0.5:
                continue
            
            # Check diversity against archive
            if ind.fingerprint is not None:
                is_novel = True
                for archived in self.archive:
                    if archived.fingerprint is not None:
                        if fingerprint_distance(ind.fingerprint, archived.fingerprint) < self.config.min_archive_distance:
                            is_novel = False
                            break
                
                if is_novel:
                    self.archive.append(copy.deepcopy(ind))
        
        # Trim archive if too large (keep most diverse)
        if len(self.archive) > self.config.archive_size:
            # Sort by combined score and keep best
            self.archive.sort(key=lambda x: x.combined_score, reverse=True)
            self.archive = self.archive[:self.config.archive_size]
    
    def create_next_generation(self):
        """Create next generation through selection, crossover, mutation."""
        new_population = []
        
        # Elitism: keep best individuals
        elite = select_diverse(self.population, self.config.elitism)
        new_population.extend([copy.deepcopy(e) for e in elite])
        
        # Fill rest through breeding
        while len(new_population) < self.config.population_size:
            if np.random.random() < self.config.crossover_rate and len(self.population) >= 2:
                # Crossover
                parent1 = tournament_selection(self.population, self.config.tournament_size)
                parent2 = tournament_selection(self.population, self.config.tournament_size)
                child_genome = crossover_genomes(parent1.genome, parent2.genome)
            else:
                # Clone and mutate
                parent = tournament_selection(self.population, self.config.tournament_size)
                child_genome = parent.genome.copy()
            
            # Apply mutation
            child_genome = mutate_genome(child_genome, self.config.mutation_rate)
            
            child = Individual(genome=child_genome)
            child.fingerprint = compute_design_fingerprint(child_genome)
            new_population.append(child)
        
        self.population = new_population
        self.generation += 1
    
    def get_stats(self) -> Dict:
        """Get current generation statistics."""
        if not self.population:
            return {}
        
        fitnesses = [ind.fitness.total for ind in self.population if ind.fitness]
        stabilities = [ind.fitness.stability for ind in self.population if ind.fitness]
        novelties = [ind.novelty_score for ind in self.population]
        
        stable_count = sum(1 for ind in self.population if ind.fitness and ind.fitness.stability > 0.5)
        load_passed = sum(1 for ind in self.population if ind.fitness and ind.fitness.load_test > 0.5)
        
        return {
            "generation": self.generation,
            "population_size": len(self.population),
            "archive_size": len(self.archive),
            "stable_count": stable_count,
            "load_passed": load_passed,
            "fitness_mean": np.mean(fitnesses) if fitnesses else 0,
            "fitness_max": np.max(fitnesses) if fitnesses else 0,
            "fitness_std": np.std(fitnesses) if fitnesses else 0,
            "novelty_mean": np.mean(novelties) if novelties else 0,
            "stability_rate": np.mean(stabilities) if stabilities else 0,
        }
    
    def save_archive(self, path: str = None):
        """Save archive to disk."""
        if path is None:
            path = os.path.join(self.config.output_dir, f"archive_gen{self.generation}.json")
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        archive_data = []
        for ind in self.archive:
            design = ind.genome.to_furniture_design()
            data = {
                "generation": ind.genome.generation,
                "fitness": ind.fitness.to_dict() if ind.fitness else {},
                "novelty": ind.novelty_score,
                "num_components": len(ind.genome.get_active_components()),
                "fingerprint": ind.fingerprint.tolist() if ind.fingerprint is not None else [],
            }
            archive_data.append(data)
        
        with open(path, 'w') as f:
            json.dump({
                "generation": self.generation,
                "archive_size": len(self.archive),
                "designs": archive_data,
            }, f, indent=2)
        
        print(f"Saved archive ({len(self.archive)} designs) to {path}")
    
    def run(self, generations: int = None):
        """Run evolution for specified generations."""
        if generations is None:
            generations = self.config.generations
        
        if not self.population:
            self.initialize_population()
        
        print(f"\nStarting evolution for {generations} generations")
        print(f"Population: {self.config.population_size}, Archive max: {self.config.archive_size}")
        print("=" * 60)
        
        start_time = time.time()
        
        for gen in range(generations):
            gen_start = time.time()
            
            # Evaluate
            self.evaluate_population()
            
            # Update archive
            self.update_archive()
            
            # Get stats
            stats = self.get_stats()
            self.stats_history.append(stats)
            
            gen_time = time.time() - gen_start
            
            # Print progress
            print(f"Gen {self.generation:3d} | "
                  f"Stable: {stats['stable_count']:2d}/{self.config.population_size} | "
                  f"Archive: {stats['archive_size']:3d} | "
                  f"Fitness: {stats['fitness_mean']:.3f} (max {stats['fitness_max']:.3f}) | "
                  f"Novelty: {stats['novelty_mean']:.3f} | "
                  f"Time: {gen_time:.1f}s")
            
            # Save periodically
            if self.generation % self.config.save_interval == 0:
                self.save_archive()
            
            # Create next generation
            self.create_next_generation()
        
        total_time = time.time() - start_time
        print("=" * 60)
        print(f"Evolution complete! {generations} generations in {total_time:.1f}s")
        print(f"Final archive: {len(self.archive)} diverse designs")
        
        # Final save
        self.save_archive()
        
        return self.archive


# =============================================================================
# Convenience functions
# =============================================================================

def quick_evolve(generations: int = 50, population_size: int = 30) -> List[Individual]:
    """Quick evolution run with sensible defaults."""
    config = EvolutionConfig(
        population_size=population_size,
        generations=generations,
        simulation_duration=2.0,
        load_test=True,
    )
    
    engine = EvolutionEngine(config)
    return engine.run()


if __name__ == "__main__":
    # Test run
    print("Testing evolution engine...")
    
    config = EvolutionConfig(
        population_size=20,
        generations=10,
        archive_size=100,
        save_interval=5,
    )
    
    engine = EvolutionEngine(config)
    archive = engine.run()
    
    print(f"\nFinal archive contains {len(archive)} designs")
    
    # Show some stats
    if archive:
        fitnesses = [ind.fitness.total for ind in archive if ind.fitness]
        print(f"Fitness range: {min(fitnesses):.3f} - {max(fitnesses):.3f}")
        
        # Show top 5
        print("\nTop 5 designs:")
        sorted_archive = sorted(archive, key=lambda x: x.combined_score, reverse=True)
        for i, ind in enumerate(sorted_archive[:5]):
            print(f"  {i+1}. Score: {ind.combined_score:.3f}, "
                  f"Components: {len(ind.genome.get_active_components())}, "
                  f"Symmetry: {ind.fitness.mirror_symmetry_x:.2f}")
