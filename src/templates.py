"""
Furniture template generators.

These create parametric furniture designs from simple parameters.
Used for initial testing and as starting points for genetic algorithms.
"""
import numpy as np
from furniture import (
    Component, ComponentType, Joint, JointType,
    AssemblyGraph, FurnitureDesign
)


def create_simple_table(
    height: float = 750.0,  # mm
    top_width: float = 1200.0,  # mm
    top_depth: float = 800.0,  # mm
    top_thickness: float = 18.0,  # mm
    leg_thickness: float = 60.0,  # mm
    leg_inset: float = 50.0,  # mm from edge
) -> FurnitureDesign:
    """
    Create a simple 4-leg table design.
    
    Args:
        height: Total table height (mm)
        top_width: Tabletop width (mm)
        top_depth: Tabletop depth (mm)
        top_thickness: Tabletop material thickness (mm)
        leg_thickness: Leg cross-section size (square legs) (mm)
        leg_inset: Distance from edge to leg center (mm)
    
    Returns:
        FurnitureDesign object representing the table
    """
    design = FurnitureDesign(name="simple_table")
    
    # Create tabletop component
    # Profile is a rectangle
    tabletop_profile = [
        (0.0, 0.0),
        (top_width, 0.0),
        (top_width, top_depth),
        (0.0, top_depth),
    ]
    
    tabletop = Component(
        name="tabletop",
        type=ComponentType.PANEL,
        profile=tabletop_profile,
        thickness=top_thickness,
        position=np.array([0.0, 0.0, height - top_thickness]),
        rotation=np.array([0.0, 0.0, 0.0]),
    )
    design.add_component(tabletop)
    
    # Create 4 legs
    leg_height = height - top_thickness
    leg_positions = [
        # (x, y) positions for leg centers
        (leg_inset, leg_inset),  # front-left
        (top_width - leg_inset, leg_inset),  # front-right
        (top_width - leg_inset, top_depth - leg_inset),  # back-right
        (leg_inset, top_depth - leg_inset),  # back-left
    ]
    
    for i, (leg_x, leg_y) in enumerate(leg_positions):
        # Leg profile is a square
        leg_profile = [
            (0.0, 0.0),
            (leg_thickness, 0.0),
            (leg_thickness, leg_height),
            (0.0, leg_height),
        ]
        
        leg = Component(
            name=f"leg_{i+1}",
            type=ComponentType.LEG,
            profile=leg_profile,
            thickness=leg_thickness,
            # Position leg so it's centered at (leg_x, leg_y) and on the ground
            position=np.array([
                leg_x - leg_thickness/2,
                leg_y - leg_thickness/2,
                0.0
            ]),
            # Leg stands vertical, so we rotate the 2D profile 90 degrees
            rotation=np.array([np.pi/2, 0.0, 0.0]),
        )
        design.add_component(leg)
    
    # Create joints between tabletop and legs
    # For simplicity, we'll use BUTT joints (legs simply touch underside of top)
    for i in range(4):
        leg_name = f"leg_{i+1}"
        joint = Joint(
            component_a="tabletop",
            component_b=leg_name,
            joint_type=JointType.BUTT,
            # Position on tabletop (bottom surface)
            position_a=leg_positions[i] + (0.0,),  # Add z=0
            # Position on leg (top end)
            position_b=(leg_thickness/2, leg_thickness/2, leg_height),
            parameters={"connection": "top_of_leg_to_underside_of_top"},
        )
        design.assembly.add_joint(joint)
    
    # Add metadata
    design.metadata = {
        "furniture_type": "table",
        "description": f"{top_width}x{top_depth}mm table at {height}mm height with 4 legs",
        "parameters": {
            "height": height,
            "top_width": top_width,
            "top_depth": top_depth,
            "top_thickness": top_thickness,
            "leg_thickness": leg_thickness,
            "leg_inset": leg_inset,
        }
    }
    
    return design


def create_simple_shelf(
    height: float = 1800.0,  # mm
    width: float = 900.0,  # mm
    depth: float = 300.0,  # mm
    num_shelves: int = 4,
    thickness: float = 18.0,  # mm
) -> FurnitureDesign:
    """
    Create a simple bookshelf design with vertical sides and horizontal shelves.
    
    Args:
        height: Total shelf height (mm)
        width: Shelf width (mm)
        depth: Shelf depth (mm)
        num_shelves: Number of horizontal shelves (including top)
        thickness: Material thickness (mm)
    
    Returns:
        FurnitureDesign object representing the bookshelf
    """
    design = FurnitureDesign(name="simple_shelf")
    
    # Create two vertical side panels
    side_profile = [
        (0.0, 0.0),
        (thickness, 0.0),
        (thickness, height),
        (0.0, height),
    ]
    
    left_side = Component(
        name="side_left",
        type=ComponentType.PANEL,
        profile=side_profile,
        thickness=depth,
        position=np.array([0.0, 0.0, 0.0]),
        rotation=np.array([np.pi/2, 0.0, np.pi/2]),  # Vertical panel
    )
    design.add_component(left_side)
    
    right_side = Component(
        name="side_right",
        type=ComponentType.PANEL,
        profile=side_profile,
        thickness=depth,
        position=np.array([width - thickness, 0.0, 0.0]),
        rotation=np.array([np.pi/2, 0.0, np.pi/2]),  # Vertical panel
    )
    design.add_component(right_side)
    
    # Create horizontal shelves
    shelf_profile = [
        (0.0, 0.0),
        (width - 2*thickness, 0.0),  # Shelf fits between sides
        (width - 2*thickness, depth),
        (0.0, depth),
    ]
    
    shelf_spacing = (height - thickness) / num_shelves
    
    for i in range(num_shelves):
        shelf_z = i * shelf_spacing
        
        shelf = Component(
            name=f"shelf_{i+1}",
            type=ComponentType.SHELF,
            profile=shelf_profile,
            thickness=thickness,
            position=np.array([thickness, 0.0, shelf_z]),
            rotation=np.array([0.0, 0.0, 0.0]),  # Horizontal
        )
        design.add_component(shelf)
        
        # Add joints to both sides
        for side in ["side_left", "side_right"]:
            joint = Joint(
                component_a=f"shelf_{i+1}",
                component_b=side,
                joint_type=JointType.BUTT,
                position_a=(0.0, depth/2, 0.0) if side == "side_left" else (width - 2*thickness, depth/2, 0.0),
                position_b=(thickness/2, depth/2, shelf_z),
                parameters={},
            )
            design.assembly.add_joint(joint)
    
    design.metadata = {
        "furniture_type": "bookshelf",
        "description": f"{width}x{depth}x{height}mm shelf with {num_shelves} shelves",
        "parameters": {
            "height": height,
            "width": width,
            "depth": depth,
            "num_shelves": num_shelves,
            "thickness": thickness,
        }
    }
    
    return design


if __name__ == "__main__":
    # Test table generation
    table = create_simple_table()
    is_valid, errors = table.validate()
    
    print(f"Table: {table.name}")
    print(f"Components: {len(table.components)}")
    print(f"Joints: {len(table.assembly.joints)}")
    print(f"Bounding box: {table.get_bounding_box()} mm")
    print(f"Valid: {is_valid}")
    if errors:
        print(f"Errors: {errors}")
    
    print("\n" + "="*50 + "\n")
    
    # Test shelf generation
    shelf = create_simple_shelf()
    is_valid, errors = shelf.validate()
    
    print(f"Shelf: {shelf.name}")
    print(f"Components: {len(shelf.components)}")
    print(f"Joints: {len(shelf.assembly.joints)}")
    print(f"Bounding box: {shelf.get_bounding_box()} mm")
    print(f"Valid: {is_valid}")
    if errors:
        print(f"Errors: {errors}")
