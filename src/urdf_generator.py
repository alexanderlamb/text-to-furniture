"""
URDF Generator for furniture designs.

Converts FurnitureDesign objects to URDF format for PyBullet physics simulation.
"""
import os
import xml.etree.ElementTree as ET
from xml.dom import minidom
import numpy as np
from typing import Optional
import tempfile

from furniture import FurnitureDesign, Component, ComponentType


# Material properties for physics simulation
MATERIAL_DENSITIES = {
    "plywood": 680,  # kg/m³
    "mdf": 750,
    "particle_board": 700,
    "hardwood": 800,
}


def compute_box_inertia(mass: float, width: float, height: float, depth: float):
    """
    Compute inertia tensor for a box.
    
    Args:
        mass: Mass in kg
        width, height, depth: Dimensions in meters
    
    Returns:
        (ixx, iyy, izz) diagonal inertia values
    """
    ixx = (1/12) * mass * (height**2 + depth**2)
    iyy = (1/12) * mass * (width**2 + depth**2)
    izz = (1/12) * mass * (width**2 + height**2)
    return ixx, iyy, izz


def component_to_link(component: Component, link_name: str) -> ET.Element:
    """
    Convert a furniture component to a URDF link element.
    
    Args:
        component: Furniture component
        link_name: Name for the URDF link
    
    Returns:
        XML Element for the link
    """
    dims = component.get_dimensions()
    
    # Convert mm to meters for URDF
    width_m = dims[0] / 1000.0
    depth_m = dims[1] / 1000.0
    height_m = dims[2] / 1000.0
    
    # Calculate mass from density
    density = MATERIAL_DENSITIES.get(component.material, 700)
    volume = width_m * depth_m * height_m
    mass = density * volume
    
    # Compute inertia
    ixx, iyy, izz = compute_box_inertia(mass, width_m, depth_m, height_m)
    
    # Create link element
    link = ET.Element("link", name=link_name)
    
    # Inertial properties
    inertial = ET.SubElement(link, "inertial")
    ET.SubElement(inertial, "mass", value=f"{mass:.6f}")
    ET.SubElement(inertial, "origin", xyz="0 0 0", rpy="0 0 0")
    ET.SubElement(inertial, "inertia", 
                  ixx=f"{ixx:.9f}", ixy="0", ixz="0",
                  iyy=f"{iyy:.9f}", iyz="0", izz=f"{izz:.9f}")
    
    # Visual properties
    visual = ET.SubElement(link, "visual")
    ET.SubElement(visual, "origin", xyz="0 0 0", rpy="0 0 0")
    vis_geometry = ET.SubElement(visual, "geometry")
    ET.SubElement(vis_geometry, "box", size=f"{width_m:.6f} {depth_m:.6f} {height_m:.6f}")
    
    # Material color based on component type
    material = ET.SubElement(visual, "material", name=f"{link_name}_material")
    if component.type == ComponentType.LEG:
        ET.SubElement(material, "color", rgba="0.6 0.4 0.2 1")  # Brown
    elif component.type == ComponentType.PANEL:
        ET.SubElement(material, "color", rgba="0.8 0.7 0.5 1")  # Light wood
    elif component.type == ComponentType.SHELF:
        ET.SubElement(material, "color", rgba="0.7 0.6 0.4 1")  # Medium wood
    else:
        ET.SubElement(material, "color", rgba="0.5 0.5 0.5 1")  # Gray
    
    # Collision properties
    collision = ET.SubElement(link, "collision")
    ET.SubElement(collision, "origin", xyz="0 0 0", rpy="0 0 0")
    col_geometry = ET.SubElement(collision, "geometry")
    ET.SubElement(col_geometry, "box", size=f"{width_m:.6f} {depth_m:.6f} {height_m:.6f}")
    
    return link


def create_fixed_joint(parent_name: str, child_name: str, 
                       position: np.ndarray, rotation: np.ndarray) -> ET.Element:
    """
    Create a fixed joint connecting two links.
    
    Args:
        parent_name: Name of parent link
        child_name: Name of child link
        position: Position offset (x, y, z) in meters
        rotation: Rotation (roll, pitch, yaw) in radians
    
    Returns:
        XML Element for the joint
    """
    joint = ET.Element("joint", name=f"{parent_name}_to_{child_name}", type="fixed")
    ET.SubElement(joint, "parent", link=parent_name)
    ET.SubElement(joint, "child", link=child_name)
    ET.SubElement(joint, "origin", 
                  xyz=f"{position[0]:.6f} {position[1]:.6f} {position[2]:.6f}",
                  rpy=f"{rotation[0]:.6f} {rotation[1]:.6f} {rotation[2]:.6f}")
    return joint


def furniture_to_urdf(design: FurnitureDesign, 
                      fixed_base: bool = False) -> str:
    """
    Convert a FurnitureDesign to URDF XML string.
    
    For simulation, we create the furniture as a single rigid body by using
    fixed joints between all components. This is simpler than articulated
    joints and works well for stability testing.
    
    Args:
        design: FurnitureDesign object
        fixed_base: If True, fix the furniture to the world (won't fall)
    
    Returns:
        URDF XML string
    """
    if not design.components:
        raise ValueError("Design has no components")
    
    # Create robot element (URDF root)
    robot = ET.Element("robot", name=design.name)
    
    # Add base link (required for URDF)
    base_link = ET.SubElement(robot, "link", name="base_link")
    # Empty base link - just a reference point
    
    # Add all components as links
    for i, component in enumerate(design.components):
        link_name = f"link_{component.name}"
        link = component_to_link(component, link_name)
        robot.append(link)
        
        # Get component center position in world frame
        dims = component.get_dimensions()
        # Position is at corner, we need center for URDF
        center_x = (component.position[0] + dims[0]/2) / 1000.0
        center_y = (component.position[1] + dims[1]/2) / 1000.0
        center_z = (component.position[2] + dims[2]/2) / 1000.0
        
        # Create joint from base to this component
        if i == 0:
            # First component connects to base_link
            parent = "base_link"
        else:
            # All other components also connect to base_link (forming a tree)
            parent = "base_link"
        
        joint = create_fixed_joint(
            parent, 
            link_name,
            np.array([center_x, center_y, center_z]),
            component.rotation
        )
        robot.append(joint)
    
    # Convert to pretty-printed XML string
    rough_string = ET.tostring(robot, encoding='unicode')
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="  ")


def save_urdf(design: FurnitureDesign, filepath: str, fixed_base: bool = False):
    """
    Save furniture design as URDF file.
    
    Args:
        design: FurnitureDesign object
        filepath: Path to save URDF file
        fixed_base: If True, fix furniture to world
    """
    urdf_string = furniture_to_urdf(design, fixed_base)
    
    with open(filepath, 'w') as f:
        f.write(urdf_string)
    
    return filepath


def save_urdf_temp(design: FurnitureDesign, fixed_base: bool = False) -> str:
    """
    Save furniture design to a temporary URDF file.
    
    Useful for quick simulation without managing file cleanup.
    
    Args:
        design: FurnitureDesign object
        fixed_base: If True, fix furniture to world
    
    Returns:
        Path to temporary URDF file
    """
    urdf_string = furniture_to_urdf(design, fixed_base)
    
    # Create temp file that won't be auto-deleted
    fd, filepath = tempfile.mkstemp(suffix='.urdf', prefix=f'{design.name}_')
    with os.fdopen(fd, 'w') as f:
        f.write(urdf_string)
    
    return filepath


if __name__ == "__main__":
    # Test URDF generation
    from templates import create_simple_table
    from genome import create_table_genome, create_shelf_genome
    
    # Test with simple table
    print("Testing URDF generation...")
    
    table = create_simple_table(
        height=750,
        top_width=1200,
        top_depth=800,
    )
    
    urdf_path = save_urdf_temp(table)
    print(f"Saved table URDF to: {urdf_path}")
    
    # Print first 50 lines
    with open(urdf_path, 'r') as f:
        lines = f.readlines()[:50]
        print("\nURDF preview (first 50 lines):")
        print("".join(lines))
    
    # Test with genome-generated furniture
    print("\n" + "="*50)
    print("Testing with genome-generated table...")
    
    genome = create_table_genome()
    design = genome.to_furniture_design("genome_table")
    urdf_path2 = save_urdf_temp(design)
    print(f"Saved genome table URDF to: {urdf_path2}")
    print(f"Components: {len(design.components)}")
