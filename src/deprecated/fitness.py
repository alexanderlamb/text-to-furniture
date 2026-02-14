"""
Fitness functions for evaluating furniture designs.

Beyond just "does it stand up", we evaluate:
- Symmetry (mirror and rotational)
- Proportions and visual balance
- Material efficiency
- Structural coherence

These scores guide the GA toward designs that are both functional AND beautiful.
"""
import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass

from furniture import FurnitureDesign, Component
from genome import FurnitureGenome, ComponentGene


@dataclass
class FitnessScores:
    """Collection of fitness scores for a design."""
    # Physical stability (from simulation)
    stability: float = 0.0  # 0-1, did it pass physics test?
    load_test: float = 0.0  # 0-1, did it support the load?
    
    # Symmetry scores (0-1, higher is more symmetric)
    mirror_symmetry_x: float = 0.0  # Left-right mirror symmetry
    mirror_symmetry_y: float = 0.0  # Front-back mirror symmetry
    rotational_symmetry: float = 0.0  # 180° or 90° rotational symmetry
    
    # Aesthetic scores (0-1)
    proportion_score: float = 0.0  # Golden ratio adherence
    balance_score: float = 0.0  # Center of mass alignment
    visual_weight_balance: float = 0.0  # Perceived weight distribution
    
    # Efficiency scores (0-1)
    material_efficiency: float = 0.0  # Less waste = higher
    structural_efficiency: float = 0.0  # Strength-to-weight ratio
    
    # Overall composite score
    @property
    def total(self) -> float:
        """Weighted combination of all scores."""
        weights = {
            'stability': 3.0,  # Must be stable
            'load_test': 2.0,  # Must support load
            'mirror_symmetry_x': 1.0,
            'mirror_symmetry_y': 0.5,
            'rotational_symmetry': 0.8,
            'proportion_score': 0.7,
            'balance_score': 1.0,
            'visual_weight_balance': 0.5,
            'material_efficiency': 0.3,
            'structural_efficiency': 0.4,
        }
        
        score = (
            weights['stability'] * self.stability +
            weights['load_test'] * self.load_test +
            weights['mirror_symmetry_x'] * self.mirror_symmetry_x +
            weights['mirror_symmetry_y'] * self.mirror_symmetry_y +
            weights['rotational_symmetry'] * self.rotational_symmetry +
            weights['proportion_score'] * self.proportion_score +
            weights['balance_score'] * self.balance_score +
            weights['visual_weight_balance'] * self.visual_weight_balance +
            weights['material_efficiency'] * self.material_efficiency +
            weights['structural_efficiency'] * self.structural_efficiency
        )
        
        max_score = sum(weights.values())
        return score / max_score
    
    def to_dict(self) -> Dict[str, float]:
        return {
            'stability': self.stability,
            'load_test': self.load_test,
            'mirror_symmetry_x': self.mirror_symmetry_x,
            'mirror_symmetry_y': self.mirror_symmetry_y,
            'rotational_symmetry': self.rotational_symmetry,
            'proportion_score': self.proportion_score,
            'balance_score': self.balance_score,
            'visual_weight_balance': self.visual_weight_balance,
            'material_efficiency': self.material_efficiency,
            'structural_efficiency': self.structural_efficiency,
            'total': self.total,
        }


# =============================================================================
# Symmetry Analysis
# =============================================================================

def get_component_center(comp: Component) -> np.ndarray:
    """Get 3D center of a component."""
    dims = comp.get_dimensions()
    return np.array([
        comp.position[0] + dims[0] / 2,
        comp.position[1] + dims[1] / 2,
        comp.position[2] + dims[2] / 2,
    ])


def get_design_centroid(design: FurnitureDesign) -> np.ndarray:
    """Get the centroid of all components."""
    if not design.components:
        return np.array([0, 0, 0])
    
    centers = [get_component_center(c) for c in design.components]
    return np.mean(centers, axis=0)


def get_design_bounds(design: FurnitureDesign) -> Tuple[np.ndarray, np.ndarray]:
    """Get min and max bounds of the design."""
    if not design.components:
        return np.array([0, 0, 0]), np.array([0, 0, 0])
    
    mins = np.array([float('inf'), float('inf'), float('inf')])
    maxs = np.array([float('-inf'), float('-inf'), float('-inf')])
    
    for comp in design.components:
        dims = comp.get_dimensions()
        pos = comp.position
        
        mins = np.minimum(mins, pos)
        maxs = np.maximum(maxs, pos + np.array(dims))
    
    return mins, maxs


def compute_mirror_symmetry(design: FurnitureDesign, axis: int) -> float:
    """
    Compute mirror symmetry score along an axis.
    
    Args:
        design: The furniture design
        axis: 0 for X (left-right), 1 for Y (front-back)
    
    Returns:
        Score from 0 (asymmetric) to 1 (perfectly symmetric)
    """
    if len(design.components) < 2:
        return 1.0  # Single component is trivially symmetric
    
    # Get design center
    mins, maxs = get_design_bounds(design)
    center = (mins + maxs) / 2
    mirror_plane = center[axis]
    
    # For each component, find its mirror and check if one exists
    total_match_score = 0.0
    
    for comp in design.components:
        comp_center = get_component_center(comp)
        comp_dims = np.array(comp.get_dimensions())
        
        # Mirror the center across the plane
        mirrored_center = comp_center.copy()
        mirrored_center[axis] = 2 * mirror_plane - comp_center[axis]
        
        # Find best matching component on the other side
        best_match = 0.0
        for other in design.components:
            if other is comp:
                continue
            
            other_center = get_component_center(other)
            other_dims = np.array(other.get_dimensions())
            
            # Distance between mirrored position and other component
            dist = np.linalg.norm(mirrored_center - other_center)
            
            # Dimension similarity (should be same or mirrored)
            dim_diff = np.linalg.norm(comp_dims - other_dims)
            
            # Score based on position match and dimension match
            if dist < 50:  # Within 50mm
                position_score = 1.0 - (dist / 50)
                dim_score = 1.0 - min(dim_diff / 100, 1.0)
                match_score = (position_score + dim_score) / 2
                best_match = max(best_match, match_score)
        
        total_match_score += best_match
    
    return total_match_score / len(design.components)


def compute_rotational_symmetry(design: FurnitureDesign, angle_degrees: float = 180) -> float:
    """
    Compute rotational symmetry score around the Z axis.
    
    Args:
        design: The furniture design
        angle_degrees: Rotation angle to check (180 for 2-fold, 90 for 4-fold)
    
    Returns:
        Score from 0 (asymmetric) to 1 (rotationally symmetric)
    """
    if len(design.components) < 2:
        return 1.0
    
    # Get design center (rotation point)
    mins, maxs = get_design_bounds(design)
    center = (mins + maxs) / 2
    
    angle_rad = np.radians(angle_degrees)
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    
    total_match_score = 0.0
    
    for comp in design.components:
        comp_center = get_component_center(comp)
        comp_dims = np.array(comp.get_dimensions())
        
        # Rotate center around Z axis
        rel_pos = comp_center - center
        rotated_x = rel_pos[0] * cos_a - rel_pos[1] * sin_a
        rotated_y = rel_pos[0] * sin_a + rel_pos[1] * cos_a
        rotated_center = center + np.array([rotated_x, rotated_y, rel_pos[2]])
        
        # Find best matching component at rotated position
        best_match = 0.0
        for other in design.components:
            if other is comp:
                continue
            
            other_center = get_component_center(other)
            other_dims = np.array(other.get_dimensions())
            
            dist = np.linalg.norm(rotated_center - other_center)
            
            # For rotational symmetry, dimensions might be swapped
            dim_diff1 = np.linalg.norm(comp_dims - other_dims)
            dim_diff2 = np.linalg.norm(comp_dims - np.array([other_dims[1], other_dims[0], other_dims[2]]))
            dim_diff = min(dim_diff1, dim_diff2)
            
            if dist < 50:
                position_score = 1.0 - (dist / 50)
                dim_score = 1.0 - min(dim_diff / 100, 1.0)
                match_score = (position_score + dim_score) / 2
                best_match = max(best_match, match_score)
        
        total_match_score += best_match
    
    return total_match_score / len(design.components)


# =============================================================================
# Proportion and Balance
# =============================================================================

GOLDEN_RATIO = 1.618033988749895


def compute_proportion_score(design: FurnitureDesign) -> float:
    """
    Score how well proportions match pleasing ratios (golden ratio, etc.)
    
    Returns:
        Score from 0 to 1
    """
    mins, maxs = get_design_bounds(design)
    dims = maxs - mins
    
    if min(dims) == 0:
        return 0.0
    
    # Check ratios against golden ratio and simple ratios (1:1, 1:2, 2:3)
    pleasing_ratios = [1.0, GOLDEN_RATIO, 2.0, 1.5, 1/GOLDEN_RATIO, 0.5]
    
    # Get aspect ratios
    ratios = []
    if dims[1] > 0:
        ratios.append(dims[0] / dims[1])  # Width to depth
    if dims[2] > 0:
        ratios.append(dims[0] / dims[2])  # Width to height
        ratios.append(dims[1] / dims[2])  # Depth to height
    
    if not ratios:
        return 0.5
    
    # Score each ratio by how close it is to a pleasing ratio
    ratio_scores = []
    for r in ratios:
        best_match = min(abs(r - pr) / max(pr, 0.1) for pr in pleasing_ratios)
        score = max(0, 1.0 - best_match)
        ratio_scores.append(score)
    
    return np.mean(ratio_scores)


def compute_balance_score(design: FurnitureDesign) -> float:
    """
    Score how well balanced the design is (center of mass over base).
    
    A well-balanced design has its center of mass centered over its footprint.
    
    Returns:
        Score from 0 to 1
    """
    if not design.components:
        return 0.0
    
    # Compute weighted center of mass (assuming uniform density)
    total_mass = 0.0
    com = np.array([0.0, 0.0, 0.0])
    
    for comp in design.components:
        dims = comp.get_dimensions()
        volume = dims[0] * dims[1] * dims[2]
        center = get_component_center(comp)
        
        com += center * volume
        total_mass += volume
    
    if total_mass == 0:
        return 0.0
    
    com /= total_mass
    
    # Get base footprint (XY bounds of components touching ground)
    base_mins = np.array([float('inf'), float('inf')])
    base_maxs = np.array([float('-inf'), float('-inf')])
    
    for comp in design.components:
        if comp.position[2] < 10:  # Within 10mm of ground
            dims = comp.get_dimensions()
            base_mins = np.minimum(base_mins, comp.position[:2])
            base_maxs = np.maximum(base_maxs, comp.position[:2] + np.array(dims[:2]))
    
    if base_mins[0] == float('inf'):
        # Nothing touching ground - use full XY extent
        mins, maxs = get_design_bounds(design)
        base_mins = mins[:2]
        base_maxs = maxs[:2]
    
    # Check if COM XY is within base footprint
    base_center = (base_mins + base_maxs) / 2
    base_size = base_maxs - base_mins
    
    if min(base_size) == 0:
        return 0.0
    
    # Distance from COM to base center, normalized by base size
    com_offset = np.abs(com[:2] - base_center) / base_size
    
    # Score: 1.0 if perfectly centered, decreasing as offset increases
    offset_magnitude = np.linalg.norm(com_offset)
    return max(0, 1.0 - offset_magnitude)


def compute_visual_weight_balance(design: FurnitureDesign) -> float:
    """
    Score visual weight balance (larger/heavier-looking components lower).
    
    Good design typically has visual weight concentrated at the bottom.
    
    Returns:
        Score from 0 to 1
    """
    if not design.components:
        return 0.0
    
    mins, maxs = get_design_bounds(design)
    height_range = maxs[2] - mins[2]
    
    if height_range == 0:
        return 1.0
    
    # Compute weighted average of volume by height
    total_volume = 0.0
    weighted_height_sum = 0.0
    
    for comp in design.components:
        dims = comp.get_dimensions()
        volume = dims[0] * dims[1] * dims[2]
        center = get_component_center(comp)
        
        # Normalize height to 0-1
        norm_height = (center[2] - mins[2]) / height_range
        
        weighted_height_sum += volume * norm_height
        total_volume += volume
    
    if total_volume == 0:
        return 0.5
    
    avg_weighted_height = weighted_height_sum / total_volume
    
    # Ideal: heavy things at bottom (avg_weighted_height close to 0)
    # Score of 1.0 if all mass at bottom, 0.0 if all at top
    return 1.0 - avg_weighted_height


# =============================================================================
# Efficiency Metrics
# =============================================================================

def compute_material_efficiency(design: FurnitureDesign) -> float:
    """
    Score material efficiency (how much of bounding box is filled).
    
    Low scores mean lots of empty space (might be good for aesthetics, 
    but wasteful for shipping/material).
    
    Returns:
        Score from 0 to 1
    """
    mins, maxs = get_design_bounds(design)
    bbox_volume = np.prod(maxs - mins)
    
    if bbox_volume == 0:
        return 0.0
    
    # Sum of all component volumes
    total_volume = 0.0
    for comp in design.components:
        dims = comp.get_dimensions()
        total_volume += dims[0] * dims[1] * dims[2]
    
    # Ratio of filled to bounding box
    fill_ratio = total_volume / bbox_volume
    
    # Don't want 100% filled (that's a solid block), aim for ~20-40%
    # Score peaks around 30% fill
    ideal_fill = 0.3
    deviation = abs(fill_ratio - ideal_fill)
    
    return max(0, 1.0 - deviation / ideal_fill)


def compute_structural_efficiency(design: FurnitureDesign) -> float:
    """
    Score structural efficiency (height supported per unit material).
    
    Returns:
        Score from 0 to 1
    """
    mins, maxs = get_design_bounds(design)
    height = maxs[2] - mins[2]
    
    if height == 0:
        return 0.0
    
    # Total material volume
    total_volume = 0.0
    for comp in design.components:
        dims = comp.get_dimensions()
        total_volume += dims[0] * dims[1] * dims[2]
    
    if total_volume == 0:
        return 0.0
    
    # Height per unit volume (in sensible units)
    # Convert to meters and liters for reasonable numbers
    height_m = height / 1000.0
    volume_liters = total_volume / 1e6
    
    efficiency = height_m / volume_liters
    
    # Normalize: typical furniture might have 0.5-2.0 m/L
    normalized = min(efficiency / 2.0, 1.0)
    
    return normalized


# =============================================================================
# Combined Fitness Evaluation
# =============================================================================

def evaluate_design(
    design: FurnitureDesign,
    stability_result: Optional[dict] = None
) -> FitnessScores:
    """
    Evaluate all fitness metrics for a design.
    
    Args:
        design: The furniture design to evaluate
        stability_result: Optional dict with 'stable' and 'load_test_passed' keys
    
    Returns:
        FitnessScores object with all metrics
    """
    scores = FitnessScores()
    
    # Physical scores from simulation
    if stability_result:
        scores.stability = 1.0 if stability_result.get('stable', False) else 0.0
        scores.load_test = 1.0 if stability_result.get('load_test_passed', False) else 0.0
    
    # Symmetry scores
    scores.mirror_symmetry_x = compute_mirror_symmetry(design, axis=0)
    scores.mirror_symmetry_y = compute_mirror_symmetry(design, axis=1)
    scores.rotational_symmetry = compute_rotational_symmetry(design, 180)
    
    # Proportion and balance
    scores.proportion_score = compute_proportion_score(design)
    scores.balance_score = compute_balance_score(design)
    scores.visual_weight_balance = compute_visual_weight_balance(design)
    
    # Efficiency
    scores.material_efficiency = compute_material_efficiency(design)
    scores.structural_efficiency = compute_structural_efficiency(design)
    
    return scores


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    from genome import create_table_genome, create_shelf_genome
    from freeform_genome import create_freeform_genome
    
    print("Testing fitness functions...")
    print("=" * 60)
    
    # Test on a table (should be fairly symmetric)
    print("\n1. Random Table:")
    genome = create_table_genome()
    design = genome.to_furniture_design("test_table")
    scores = evaluate_design(design)
    
    for key, value in scores.to_dict().items():
        print(f"   {key}: {value:.3f}")
    
    # Test on a shelf
    print("\n2. Random Shelf:")
    genome = create_shelf_genome()
    design = genome.to_furniture_design("test_shelf")
    scores = evaluate_design(design)
    
    for key, value in scores.to_dict().items():
        print(f"   {key}: {value:.3f}")
    
    # Test on freeform (likely less symmetric)
    print("\n3. Freeform Design:")
    genome = create_freeform_genome()
    design = genome.to_furniture_design("test_freeform")
    scores = evaluate_design(design)
    
    for key, value in scores.to_dict().items():
        print(f"   {key}: {value:.3f}")
    
    print("\n" + "=" * 60)
    print("Fitness evaluation complete!")
