"""
Core data structures for parametric furniture representation.
"""
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
from enum import Enum
import numpy as np


class JointType(Enum):
    """Types of joints for flat-pack furniture."""
    TAB_SLOT = "tab_slot"
    FINGER = "finger"
    THROUGH_BOLT = "through_bolt"
    BUTT = "butt"


class ComponentType(Enum):
    """Types of furniture components."""
    PANEL = "panel"
    LEG = "leg"
    SHELF = "shelf"
    SUPPORT = "support"
    BRACE = "brace"


@dataclass
class Component:
    """
    Represents a 2D component that will be cut from flat sheet material.
    
    Attributes:
        name: Unique identifier for this component
        type: Component type (panel, leg, etc.)
        profile: 2D shape as list of (x, y) coordinates (closed polygon)
        thickness: Material thickness in mm
        position: 3D position (x, y, z) in assembled furniture
        rotation: 3D rotation (rx, ry, rz) in radians
        material: Material type (plywood, MDF, etc.)
    """
    name: str
    type: ComponentType
    profile: List[Tuple[float, float]]  # 2D coordinates
    thickness: float  # mm
    position: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0]))
    rotation: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0]))
    material: str = "plywood"
    
    def get_dimensions(self) -> Tuple[float, float, float]:
        """Get bounding box dimensions (width, depth, height)."""
        if not self.profile:
            return (0.0, 0.0, 0.0)
        
        xs = [p[0] for p in self.profile]
        ys = [p[1] for p in self.profile]
        
        width = max(xs) - min(xs)
        depth = max(ys) - min(ys)
        
        return (width, depth, self.thickness)
    
    def get_center_2d(self) -> Tuple[float, float]:
        """Get 2D centroid of profile."""
        if not self.profile:
            return (0.0, 0.0)
        
        xs = [p[0] for p in self.profile]
        ys = [p[1] for p in self.profile]
        
        return (sum(xs) / len(xs), sum(ys) / len(ys))


@dataclass
class Joint:
    """
    Represents a connection between two components.
    
    Attributes:
        component_a: Name of first component
        component_b: Name of second component
        joint_type: Type of joint
        position_a: Position on component A (local coordinates)
        position_b: Position on component B (local coordinates)
        parameters: Additional joint-specific parameters (tab size, bolt size, etc.)
    """
    component_a: str
    component_b: str
    joint_type: JointType
    position_a: Tuple[float, float, float]
    position_b: Tuple[float, float, float]
    parameters: Dict = field(default_factory=dict)


@dataclass
class AssemblyGraph:
    """
    Represents the assembly relationships between components.
    
    This is essentially a graph where nodes are components and edges are joints.
    Also includes assembly order information.
    """
    joints: List[Joint] = field(default_factory=list)
    assembly_order: List[Tuple[str, str]] = field(default_factory=list)  # (part_a, part_b) order
    
    def add_joint(self, joint: Joint):
        """Add a joint to the assembly."""
        self.joints.append(joint)
    
    def get_connected_components(self, component_name: str) -> List[str]:
        """Get all components connected to the given component."""
        connected = []
        for joint in self.joints:
            if joint.component_a == component_name:
                connected.append(joint.component_b)
            elif joint.component_b == component_name:
                connected.append(joint.component_a)
        return connected


@dataclass
class FurnitureDesign:
    """
    Complete furniture design with components, assembly, and metadata.
    
    This is the top-level representation of a piece of furniture.
    """
    name: str
    components: List[Component] = field(default_factory=list)
    assembly: AssemblyGraph = field(default_factory=AssemblyGraph)
    metadata: Dict = field(default_factory=dict)
    
    def add_component(self, component: Component):
        """Add a component to the design."""
        self.components.append(component)
    
    def get_component(self, name: str) -> Optional[Component]:
        """Get a component by name."""
        for comp in self.components:
            if comp.name == name:
                return comp
        return None
    
    def get_bounding_box(self) -> Tuple[float, float, float]:
        """Get overall bounding box of assembled furniture (width, depth, height)."""
        if not self.components:
            return (0.0, 0.0, 0.0)
        
        min_x = min_y = min_z = float('inf')
        max_x = max_y = max_z = float('-inf')
        
        for comp in self.components:
            dims = comp.get_dimensions()
            pos = comp.position
            
            # Simple bounding box (doesn't account for rotation)
            min_x = min(min_x, pos[0])
            min_y = min(min_y, pos[1])
            min_z = min(min_z, pos[2])
            
            max_x = max(max_x, pos[0] + dims[0])
            max_y = max(max_y, pos[1] + dims[1])
            max_z = max(max_z, pos[2] + dims[2])
        
        return (max_x - min_x, max_y - min_y, max_z - min_z)
    
    def get_total_material(self) -> float:
        """Get total material area in mm²."""
        total = 0.0
        for comp in self.components:
            dims = comp.get_dimensions()
            total += dims[0] * dims[1]  # width * depth
        return total
    
    def validate(self) -> Tuple[bool, List[str]]:
        """
        Basic validation of the design.
        
        Returns:
            (is_valid, list_of_errors)
        """
        errors = []
        
        if not self.components:
            errors.append("Design has no components")
        
        # Check for duplicate component names
        names = [c.name for c in self.components]
        if len(names) != len(set(names)):
            errors.append("Duplicate component names found")
        
        # Check that all joints reference existing components
        component_names = set(names)
        for joint in self.assembly.joints:
            if joint.component_a not in component_names:
                errors.append(f"Joint references unknown component: {joint.component_a}")
            if joint.component_b not in component_names:
                errors.append(f"Joint references unknown component: {joint.component_b}")
        
        return (len(errors) == 0, errors)
