"""
Furniture genome representation for genetic algorithms.

Instead of hardcoded templates, this represents furniture as a flexible
genome that can be mutated, crossed over, and evolved.

The genome encodes:
- Variable number of components
- Component parameters (size, position, rotation)
- Connection graph between components
- Joint types and positions
"""
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
from enum import Enum
import numpy as np
import copy

from furniture import (
    Component, ComponentType, Joint, JointType,
    AssemblyGraph, FurnitureDesign
)


@dataclass
class ComponentGene:
    """
    Gene representing a single component.
    
    This is a more flexible representation than the Component class,
    designed for mutation and crossover operations.
    """
    # Type of component
    component_type: ComponentType
    
    # Dimensions (interpreted based on component_type)
    width: float  # mm
    height: float  # mm  
    thickness: float  # mm (material thickness)
    
    # Position in 3D space
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    
    # Rotation (in radians)
    rx: float = 0.0
    ry: float = 0.0
    rz: float = 0.0
    
    # Whether this component is active (allows "deleting" without changing list size)
    active: bool = True
    
    # Optional shape modifications
    # e.g., cutouts, rounded corners, tabs
    modifiers: Dict[str, Any] = field(default_factory=dict)
    
    def to_component(self, name: str) -> Component:
        """Convert gene to Component object."""
        # Create rectangular profile
        profile = [
            (0.0, 0.0),
            (self.width, 0.0),
            (self.width, self.height),
            (0.0, self.height),
        ]
        
        return Component(
            name=name,
            type=self.component_type,
            profile=profile,
            thickness=self.thickness,
            position=np.array([self.x, self.y, self.z]),
            rotation=np.array([self.rx, self.ry, self.rz]),
        )


@dataclass 
class ConnectionGene:
    """
    Gene representing a connection between two components.
    """
    # Indices into the component list
    component_a_idx: int
    component_b_idx: int
    
    # Joint type
    joint_type: JointType
    
    # Connection points (relative to component, 0-1 range)
    # e.g., (0.5, 0.5, 0.0) = center of bottom face
    anchor_a: Tuple[float, float, float] = (0.5, 0.5, 0.0)
    anchor_b: Tuple[float, float, float] = (0.5, 0.5, 1.0)
    
    # Whether this connection is active
    active: bool = True


@dataclass
class FurnitureGenome:
    """
    Complete genome for a piece of furniture.
    
    This is the representation that gets mutated and evolved
    by the genetic algorithm.
    """
    # List of component genes (variable length)
    components: List[ComponentGene] = field(default_factory=list)
    
    # List of connection genes (variable length)
    connections: List[ConnectionGene] = field(default_factory=list)
    
    # Global parameters
    base_thickness: float = 18.0  # Default material thickness
    
    # Metadata
    generation: int = 0
    fitness_scores: Dict[str, float] = field(default_factory=dict)
    
    def add_component(self, gene: ComponentGene) -> int:
        """Add a component, return its index."""
        self.components.append(gene)
        return len(self.components) - 1
    
    def add_connection(self, gene: ConnectionGene):
        """Add a connection between components."""
        self.connections.append(gene)
    
    def get_active_components(self) -> List[Tuple[int, ComponentGene]]:
        """Get list of (index, gene) for active components only."""
        return [(i, g) for i, g in enumerate(self.components) if g.active]
    
    def get_active_connections(self) -> List[ConnectionGene]:
        """Get connections where both endpoints are active."""
        active_indices = {i for i, g in enumerate(self.components) if g.active}
        return [
            c for c in self.connections 
            if c.active and c.component_a_idx in active_indices and c.component_b_idx in active_indices
        ]
    
    def to_furniture_design(self, name: str = "evolved_design") -> FurnitureDesign:
        """Convert genome to FurnitureDesign for simulation/export."""
        design = FurnitureDesign(name=name)
        
        # Create components
        active_comps = self.get_active_components()
        idx_to_name = {}
        
        for idx, gene in active_comps:
            comp_name = f"comp_{idx}"
            idx_to_name[idx] = comp_name
            component = gene.to_component(comp_name)
            design.add_component(component)
        
        # Create joints
        for conn in self.get_active_connections():
            if conn.component_a_idx not in idx_to_name or conn.component_b_idx not in idx_to_name:
                continue
            
            gene_a = self.components[conn.component_a_idx]
            gene_b = self.components[conn.component_b_idx]
            
            # Convert relative anchors to absolute positions
            pos_a = (
                conn.anchor_a[0] * gene_a.width,
                conn.anchor_a[1] * gene_a.height,
                conn.anchor_a[2] * gene_a.thickness,
            )
            pos_b = (
                conn.anchor_b[0] * gene_b.width,
                conn.anchor_b[1] * gene_b.height,
                conn.anchor_b[2] * gene_b.thickness,
            )
            
            joint = Joint(
                component_a=idx_to_name[conn.component_a_idx],
                component_b=idx_to_name[conn.component_b_idx],
                joint_type=conn.joint_type,
                position_a=pos_a,
                position_b=pos_b,
            )
            design.assembly.add_joint(joint)
        
        return design
    
    def copy(self) -> 'FurnitureGenome':
        """Create a deep copy of this genome."""
        return copy.deepcopy(self)


# =============================================================================
# Genome Factory Functions - Create starting genomes for different furniture types
# =============================================================================

def create_table_genome(
    height_range: Tuple[float, float] = (650, 850),
    width_range: Tuple[float, float] = (800, 1600),
    depth_range: Tuple[float, float] = (600, 1000),
    num_legs_options: List[int] = [3, 4, 6],
    thickness: float = 18.0,
) -> FurnitureGenome:
    """
    Create a randomized table genome.
    
    This is NOT a fixed template - it generates random valid starting points
    for evolution.
    
    Component dimensions use the convention:
    - width: X dimension
    - height: Y dimension  
    - thickness: Z dimension (vertical height for legs)
    
    NO rotations are used - all components have rx=ry=rz=0.
    """
    genome = FurnitureGenome(base_thickness=thickness)
    
    # Small gap to prevent collision issues
    gap = 1.0  # 1mm
    
    # Random dimensions
    total_height = np.random.uniform(*height_range)
    width = np.random.uniform(*width_range)
    depth = np.random.uniform(*depth_range)
    
    # Leg dimensions
    num_legs = np.random.choice(num_legs_options)
    leg_width = np.random.uniform(40, 80)  # X and Y cross-section
    leg_height = total_height - thickness - gap  # Z dimension (vertical)
    
    # Tabletop - sits on top of legs
    # Position: z = leg_height + gap (on top of legs)
    top_idx = genome.add_component(ComponentGene(
        component_type=ComponentType.PANEL,
        width=width,           # X
        height=depth,          # Y
        thickness=thickness,   # Z
        x=0, 
        y=0, 
        z=leg_height + gap,    # Sits on legs
        rx=0, ry=0, rz=0,
    ))
    
    # Position legs based on count
    if num_legs == 3:
        # Triangle configuration
        inset = 0.1
        leg_positions = [
            (width * 0.5, depth * inset),           # Front center
            (width * inset, depth * (1-inset)),     # Back left
            (width * (1-inset), depth * (1-inset)), # Back right
        ]
    elif num_legs == 4:
        # Corner configuration with random inset
        inset = np.random.uniform(0.08, 0.15)
        leg_positions = [
            (width * inset, depth * inset),
            (width * (1-inset), depth * inset),
            (width * (1-inset), depth * (1-inset)),
            (width * inset, depth * (1-inset)),
        ]
    else:  # 6 legs
        inset = np.random.uniform(0.08, 0.15)
        leg_positions = [
            (width * inset, depth * inset),
            (width * 0.5, depth * inset),
            (width * (1-inset), depth * inset),
            (width * (1-inset), depth * (1-inset)),
            (width * 0.5, depth * (1-inset)),
            (width * inset, depth * (1-inset)),
        ]
    
    # Create legs - vertical boxes starting at z=0
    for i, (lx, ly) in enumerate(leg_positions):
        leg_idx = genome.add_component(ComponentGene(
            component_type=ComponentType.LEG,
            width=leg_width,      # X dimension
            height=leg_width,     # Y dimension (square cross-section)
            thickness=leg_height, # Z dimension (vertical height)
            x=lx - leg_width/2,   # Center leg at position
            y=ly - leg_width/2,
            z=0,                  # Start at ground
            rx=0, ry=0, rz=0,     # No rotation!
        ))
        
        # Connect leg to tabletop
        genome.add_connection(ConnectionGene(
            component_a_idx=top_idx,
            component_b_idx=leg_idx,
            joint_type=JointType.TAB_SLOT,
            anchor_a=(lx/width, ly/depth, 0.0),
            anchor_b=(0.5, 0.5, 1.0),  # Top center of leg
        ))
    
    return genome


def create_shelf_genome(
    height_range: Tuple[float, float] = (800, 2000),
    width_range: Tuple[float, float] = (600, 1200),
    depth_range: Tuple[float, float] = (200, 400),
    num_shelves_range: Tuple[int, int] = (2, 6),
    thickness: float = 18.0,
) -> FurnitureGenome:
    """
    Create a randomized shelf/bookcase genome.
    
    Supports various configurations:
    - 0, 1, or 2 side panels
    - Variable number of shelves
    - Optional back panel
    - Optional top panel
    
    Note: All components use rx=ry=rz=0 to avoid rotation/collision issues.
    The width/height/thickness define the 3D box directly.
    For side panels: width=thickness, height=depth, thickness=height (the tall dimension)
    
    A small gap (1mm) is added between adjacent components to prevent physics
    collision artifacts from interpenetrating geometry.
    """
    genome = FurnitureGenome(base_thickness=thickness)
    
    # Small gap to prevent collision artifacts
    gap = 1.0  # 1mm gap between components
    
    # Random dimensions (these are the ASSEMBLED shelf dimensions)
    total_height = np.random.uniform(*height_range)
    total_width = np.random.uniform(*width_range)
    total_depth = np.random.uniform(*depth_range)
    
    # Decide on configuration - bias toward more stable configs
    num_sides = np.random.choice([1, 2], p=[0.2, 0.8])  # At least 1 side for stability
    num_shelves = np.random.randint(*num_shelves_range)
    has_back = np.random.random() < 0.6  # Back helps stability
    has_top = np.random.random() < 0.8
    has_bottom = True  # Always have a bottom shelf for stability
    
    side_indices = []
    shelf_indices = []
    
    # Create side panels (vertical orientation)
    # Side panel dimensions: thin in X, full depth in Y, full height in Z
    # Side panels rest on the bottom shelf, so they start at z=gap+thickness (above bottom shelf)
    side_z = 0  # Side panels go all the way to ground
    side_height = total_height  # Full height
    
    if num_sides >= 1:
        # Left side panel
        left_idx = genome.add_component(ComponentGene(
            component_type=ComponentType.PANEL,
            width=thickness,
            height=total_depth,
            thickness=side_height,
            x=0, 
            y=0, 
            z=side_z,
            rx=0, ry=0, rz=0,
        ))
        side_indices.append(left_idx)
    
    if num_sides >= 2:
        # Right side panel
        right_idx = genome.add_component(ComponentGene(
            component_type=ComponentType.PANEL,
            width=thickness,
            height=total_depth,
            thickness=side_height,
            x=total_width - thickness, 
            y=0, 
            z=side_z,
            rx=0, ry=0, rz=0,
        ))
        side_indices.append(right_idx)
    
    # Inner width (between side panels, with gaps)
    inner_width = total_width - (thickness * num_sides) - (gap * num_sides)
    shelf_x_offset = (thickness + gap) if num_sides >= 1 else 0
    
    # Bottom shelf (always present for stability)
    if has_bottom:
        bottom_idx = genome.add_component(ComponentGene(
            component_type=ComponentType.SHELF,
            width=inner_width,
            height=total_depth - gap,  # Slight gap from back
            thickness=thickness,
            x=shelf_x_offset,
            y=0,
            z=gap,  # Tiny gap above ground to avoid z-fighting
            rx=0, ry=0, rz=0,
        ))
        shelf_indices.append(bottom_idx)
        for side_idx in side_indices:
            genome.add_connection(ConnectionGene(
                component_a_idx=bottom_idx,
                component_b_idx=side_idx,
                joint_type=JointType.TAB_SLOT,
            ))
    
    # Interior shelves with spacing
    if num_shelves > 0:
        # Ensure minimum spacing between shelves (150mm)
        min_spacing = 150
        available_height = total_height - thickness * 2  # Leave room for top/bottom
        max_shelves = int(available_height / min_spacing)
        num_shelves = min(num_shelves, max_shelves)
        
        if num_shelves > 0:
            # Evenly distribute shelves with some randomness
            base_spacing = available_height / (num_shelves + 1)
            
            for i in range(num_shelves):
                # Add some randomness to position but keep reasonable spacing
                z_pos = thickness + base_spacing * (i + 1)
                z_pos += np.random.uniform(-base_spacing * 0.2, base_spacing * 0.2)
                z_pos = max(thickness + 50, min(z_pos, total_height - thickness - 50))
                
                shelf_idx = genome.add_component(ComponentGene(
                    component_type=ComponentType.SHELF,
                    width=inner_width,
                    height=total_depth - gap,
                    thickness=thickness,
                    x=shelf_x_offset,
                    y=0,
                    z=z_pos,
                    rx=0, ry=0, rz=0,
                ))
                shelf_indices.append(shelf_idx)
                
                for side_idx in side_indices:
                    genome.add_connection(ConnectionGene(
                        component_a_idx=shelf_idx,
                        component_b_idx=side_idx,
                        joint_type=JointType.TAB_SLOT,
                    ))
    
    # Optional back panel (helps with stability significantly)
    if has_back:
        back_thickness = thickness / 2  # Back panels often thinner
        back_idx = genome.add_component(ComponentGene(
            component_type=ComponentType.PANEL,
            width=inner_width,
            height=back_thickness,
            thickness=total_height - gap,  # Slightly shorter to avoid overlap with top
            x=shelf_x_offset,
            y=total_depth - back_thickness,
            z=gap,  # Start above ground
            rx=0, ry=0, rz=0,
        ))
        
        for side_idx in side_indices:
            genome.add_connection(ConnectionGene(
                component_a_idx=back_idx,
                component_b_idx=side_idx,
                joint_type=JointType.BUTT,
            ))
    
    # Optional top panel
    if has_top:
        top_idx = genome.add_component(ComponentGene(
            component_type=ComponentType.PANEL,
            width=total_width,
            height=total_depth,
            thickness=thickness,
            x=0, 
            y=0, 
            z=total_height - thickness + gap,  # Sit on top of side panels with gap
            rx=0, ry=0, rz=0,
        ))
        
        for side_idx in side_indices:
            genome.add_connection(ConnectionGene(
                component_a_idx=top_idx,
                component_b_idx=side_idx,
                joint_type=JointType.TAB_SLOT,
            ))
    
    return genome


def create_random_genome(furniture_type: str = None) -> FurnitureGenome:
    """
    Create a random genome, optionally of a specific type.
    
    This is the main entry point for generating starting populations.
    """
    if furniture_type is None:
        furniture_type = np.random.choice(["table", "shelf"])
    
    if furniture_type == "table":
        return create_table_genome()
    elif furniture_type == "shelf":
        return create_shelf_genome()
    else:
        raise ValueError(f"Unknown furniture type: {furniture_type}")


if __name__ == "__main__":
    # Test genome generation
    print("Generating random tables...")
    for i in range(3):
        genome = create_table_genome()
        design = genome.to_furniture_design(f"table_{i}")
        print(f"  Table {i}: {len(genome.get_active_components())} components, "
              f"{len(genome.get_active_connections())} connections")
    
    print("\nGenerating random shelves...")
    for i in range(3):
        genome = create_shelf_genome()
        design = genome.to_furniture_design(f"shelf_{i}")
        print(f"  Shelf {i}: {len(genome.get_active_components())} components, "
              f"{len(genome.get_active_connections())} connections")
