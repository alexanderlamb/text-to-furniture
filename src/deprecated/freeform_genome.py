"""
Freeform furniture genome generator.

Unlike the structured table/shelf generators, this creates random arrangements
of panels with minimal assumptions about what furniture "should" look like.

Constraints:
1. All components use real material thicknesses from SendCutSend
2. Each component must be placed adjacent to (touching) an existing component
3. Physics simulation determines what survives

This allows the GA to discover novel furniture forms.
"""
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
from enum import Enum

from furniture import ComponentType
from genome import FurnitureGenome, ComponentGene, ConnectionGene, JointType


# =============================================================================
# SendCutSend Material Catalog
# Thicknesses in mm (converted from inches)
# =============================================================================

INCH_TO_MM = 25.4

@dataclass
class Material:
    """A material available from SendCutSend."""
    name: str
    thicknesses_inch: List[float]  # Available thicknesses in inches
    density: float  # kg/m³ for physics
    max_size_inch: Tuple[float, float] = (48, 96)  # Max sheet size (w, h)
    
    @property
    def thicknesses_mm(self) -> List[float]:
        return [t * INCH_TO_MM for t in self.thicknesses_inch]
    
    @property
    def max_size_mm(self) -> Tuple[float, float]:
        return (self.max_size_inch[0] * INCH_TO_MM, self.max_size_inch[1] * INCH_TO_MM)


# SendCutSend material catalog (subset of commonly used materials)
MATERIALS = {
    # Wood-based
    "plywood_baltic_birch": Material(
        name="Baltic Birch Plywood",
        thicknesses_inch=[0.125, 0.187, 0.250, 0.375, 0.500, 0.750],
        density=680,
        max_size_inch=(24, 30),
    ),
    "mdf": Material(
        name="MDF",
        thicknesses_inch=[0.125, 0.187, 0.250, 0.500],
        density=750,
        max_size_inch=(24, 48),
    ),
    "hardboard": Material(
        name="Hardboard",
        thicknesses_inch=[0.125],
        density=900,
        max_size_inch=(24, 48),
    ),
    
    # Metals
    "mild_steel": Material(
        name="Mild Steel",
        thicknesses_inch=[0.030, 0.048, 0.060, 0.075, 0.090, 0.120, 0.135, 0.187, 0.250, 0.375, 0.500],
        density=7850,
        max_size_inch=(48, 96),
    ),
    "aluminum_5052": Material(
        name="Aluminum 5052",
        thicknesses_inch=[0.032, 0.040, 0.050, 0.063, 0.080, 0.090, 0.100, 0.125, 0.160, 0.187, 0.250],
        density=2680,
        max_size_inch=(48, 96),
    ),
    "stainless_304": Material(
        name="Stainless Steel 304",
        thicknesses_inch=[0.030, 0.036, 0.048, 0.060, 0.075, 0.090, 0.120, 0.187],
        density=8000,
        max_size_inch=(48, 96),
    ),
    
    # Plastics
    "acrylic_clear": Material(
        name="Acrylic Clear",
        thicknesses_inch=[0.060, 0.118, 0.177, 0.220, 0.354, 0.472],
        density=1180,
        max_size_inch=(24, 48),
    ),
    "acrylic_black": Material(
        name="Acrylic Black",
        thicknesses_inch=[0.118, 0.220],
        density=1180,
        max_size_inch=(24, 48),
    ),
    "hdpe": Material(
        name="HDPE",
        thicknesses_inch=[0.250, 0.375, 0.500],
        density=970,
        max_size_inch=(24, 48),
    ),
    "polycarbonate": Material(
        name="Polycarbonate",
        thicknesses_inch=[0.118, 0.177, 0.220],
        density=1200,
        max_size_inch=(24, 48),
    ),
    
    # Rubber/Foam
    "neoprene": Material(
        name="Neoprene",
        thicknesses_inch=[0.063, 0.125],
        density=1230,
        max_size_inch=(24, 36),
    ),
}

# Default material weights for random selection (favor structural materials)
DEFAULT_MATERIAL_WEIGHTS = {
    "plywood_baltic_birch": 0.35,
    "mdf": 0.20,
    "mild_steel": 0.15,
    "aluminum_5052": 0.10,
    "acrylic_clear": 0.08,
    "hdpe": 0.05,
    "stainless_304": 0.04,
    "polycarbonate": 0.02,
    "hardboard": 0.01,
}


def random_material(weights: Dict[str, float] = None) -> Tuple[str, Material]:
    """Select a random material based on weights."""
    if weights is None:
        weights = DEFAULT_MATERIAL_WEIGHTS
    
    materials = list(weights.keys())
    probs = np.array([weights[m] for m in materials])
    probs = probs / probs.sum()
    
    choice = np.random.choice(materials, p=probs)
    return choice, MATERIALS[choice]


def random_thickness(material: Material) -> float:
    """Select a random available thickness for a material."""
    return np.random.choice(material.thicknesses_mm)


# =============================================================================
# Adjacency / Attachment Logic
# =============================================================================

class AttachmentFace(Enum):
    """Which face of a box to attach to."""
    X_NEG = "x-"  # Left face
    X_POS = "x+"  # Right face
    Y_NEG = "y-"  # Front face
    Y_POS = "y+"  # Back face
    Z_NEG = "z-"  # Bottom face
    Z_POS = "z+"  # Top face


# Minimum overlap for a valid screw connection (mm)
# Two #8 screws need at least ~20mm spacing, plus edge margin
MIN_OVERLAP_MM = 25.0


@dataclass
class BoundingBox:
    """Axis-aligned bounding box."""
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    z_min: float
    z_max: float
    
    @property
    def center(self) -> Tuple[float, float, float]:
        return (
            (self.x_min + self.x_max) / 2,
            (self.y_min + self.y_max) / 2,
            (self.z_min + self.z_max) / 2,
        )
    
    @property
    def size(self) -> Tuple[float, float, float]:
        return (
            self.x_max - self.x_min,
            self.y_max - self.y_min,
            self.z_max - self.z_min,
        )
    
    def get_face_center(self, face: AttachmentFace) -> Tuple[float, float, float]:
        """Get the center point of a face."""
        cx, cy, cz = self.center
        sx, sy, sz = self.size
        
        if face == AttachmentFace.X_NEG:
            return (self.x_min, cy, cz)
        elif face == AttachmentFace.X_POS:
            return (self.x_max, cy, cz)
        elif face == AttachmentFace.Y_NEG:
            return (cx, self.y_min, cz)
        elif face == AttachmentFace.Y_POS:
            return (cx, self.y_max, cz)
        elif face == AttachmentFace.Z_NEG:
            return (cx, cy, self.z_min)
        elif face == AttachmentFace.Z_POS:
            return (cx, cy, self.z_max)
    
    def get_face_size(self, face: AttachmentFace) -> Tuple[float, float]:
        """Get the dimensions of a face (width, height in face coords)."""
        sx, sy, sz = self.size
        
        if face in (AttachmentFace.X_NEG, AttachmentFace.X_POS):
            return (sy, sz)  # Y-Z plane
        elif face in (AttachmentFace.Y_NEG, AttachmentFace.Y_POS):
            return (sx, sz)  # X-Z plane
        else:  # Z faces
            return (sx, sy)  # X-Y plane


def gene_to_bbox(gene: ComponentGene) -> BoundingBox:
    """Convert a component gene to its bounding box."""
    return BoundingBox(
        x_min=gene.x,
        x_max=gene.x + gene.width,
        y_min=gene.y,
        y_max=gene.y + gene.height,
        z_min=gene.z,
        z_max=gene.z + gene.thickness,
    )


def compute_overlap(bbox1: BoundingBox, bbox2: BoundingBox) -> Tuple[float, float, float]:
    """
    Compute overlap dimensions between two bounding boxes.
    
    Returns:
        (overlap_x, overlap_y, overlap_z) - overlap in each dimension.
        If any is <= 0, boxes don't overlap in that dimension.
    """
    overlap_x = min(bbox1.x_max, bbox2.x_max) - max(bbox1.x_min, bbox2.x_min)
    overlap_y = min(bbox1.y_max, bbox2.y_max) - max(bbox1.y_min, bbox2.y_min)
    overlap_z = min(bbox1.z_max, bbox2.z_max) - max(bbox1.z_min, bbox2.z_min)
    return (overlap_x, overlap_y, overlap_z)


def compute_contact_area(bbox1: BoundingBox, bbox2: BoundingBox, face: AttachmentFace) -> float:
    """
    Compute the contact area between two boxes at a given face.
    
    For a valid screw joint, we need sufficient contact area.
    
    Returns:
        Contact area in mm², or 0 if boxes don't touch at that face.
    """
    overlap = compute_overlap(bbox1, bbox2)
    
    # Contact area depends on which face
    if face in (AttachmentFace.X_NEG, AttachmentFace.X_POS):
        # Contact in Y-Z plane
        if overlap[1] > 0 and overlap[2] > 0:
            return overlap[1] * overlap[2]
    elif face in (AttachmentFace.Y_NEG, AttachmentFace.Y_POS):
        # Contact in X-Z plane
        if overlap[0] > 0 and overlap[2] > 0:
            return overlap[0] * overlap[2]
    else:  # Z faces
        # Contact in X-Y plane
        if overlap[0] > 0 and overlap[1] > 0:
            return overlap[0] * overlap[1]
    
    return 0.0


def compute_attachment_position(
    parent_bbox: BoundingBox,
    parent_face: AttachmentFace,
    new_width: float,
    new_height: float, 
    new_thickness: float,
    offset_u: float = 0.5,  # 0-1, position along face width
    offset_v: float = 0.5,  # 0-1, position along face height
    angle: float = 0.0,     # Rotation angle in radians (around face normal)
) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
    """
    Compute position and rotation for a new component attached to a parent's face.
    
    The new component will be placed with one of its faces touching
    the parent's specified face, potentially at an angle.
    
    Args:
        parent_bbox: Bounding box of parent component
        parent_face: Which face of parent to attach to
        new_width, new_height, new_thickness: Dimensions of new component
        offset_u, offset_v: Relative position on the face (0.5, 0.5 = centered)
        angle: Rotation angle around the face normal (radians)
    
    Returns:
        ((x, y, z), (rx, ry, rz)) - position and rotation for the new component
    """
    face_center = parent_bbox.get_face_center(parent_face)
    face_size = parent_bbox.get_face_size(parent_face)
    
    # Calculate offset from face center based on u,v
    du = (offset_u - 0.5) * face_size[0]
    dv = (offset_v - 0.5) * face_size[1]
    
    # Small gap to prevent interpenetration
    gap = 0.5  # 0.5mm
    
    # Default rotation (no angle)
    rx, ry, rz = 0.0, 0.0, 0.0
    
    if parent_face == AttachmentFace.X_NEG:
        x = face_center[0] - new_width - gap
        y = face_center[1] + du - new_height / 2
        z = face_center[2] + dv - new_thickness / 2
    elif parent_face == AttachmentFace.X_POS:
        x = face_center[0] + gap
        y = face_center[1] + du - new_height / 2
        z = face_center[2] + dv - new_thickness / 2
    elif parent_face == AttachmentFace.Y_NEG:
        x = face_center[0] + du - new_width / 2
        y = face_center[1] - new_height - gap
        z = face_center[2] + dv - new_thickness / 2
    elif parent_face == AttachmentFace.Y_POS:
        x = face_center[0] + du - new_width / 2
        y = face_center[1] + gap
        z = face_center[2] + dv - new_thickness / 2
    elif parent_face == AttachmentFace.Z_NEG:
        x = face_center[0] + du - new_width / 2
        y = face_center[1] + dv - new_height / 2
        z = face_center[2] - new_thickness - gap
    else:  # Z_POS
        x = face_center[0] + du - new_width / 2
        y = face_center[1] + dv - new_height / 2
        z = face_center[2] + gap
    
    # Apply rotation - angle is now a tuple (rx, ry, rz) for full 3D rotation
    if isinstance(angle, tuple):
        rx, ry, rz = angle
    else:
        # Legacy single-angle: apply around face normal
        rx, ry, rz = 0.0, 0.0, 0.0
        if parent_face in (AttachmentFace.X_NEG, AttachmentFace.X_POS):
            rx = angle
        elif parent_face in (AttachmentFace.Y_NEG, AttachmentFace.Y_POS):
            ry = angle
        else:
            rz = angle
    
    return ((x, y, z), (rx, ry, rz))


# =============================================================================
# Freeform Genome Generator
# =============================================================================

def create_freeform_genome(
    num_components_range: Tuple[int, int] = (3, 12),
    size_range: Tuple[float, float] = (100, 800),  # mm, for width/height (min 100 for screw margins)
    material_weights: Dict[str, float] = None,
    ground_contact: bool = True,
    allow_angles: bool = True,
    max_angle: float = 0.5,  # Max angle in radians (~30 degrees)
    require_flat_surface: bool = True,  # Must have at least one horizontal surface
    min_surface_size: float = 150.0,  # Minimum flat surface size in mm (each dimension)
) -> FurnitureGenome:
    """
    Create a freeform furniture genome with random connected components.
    
    Unlike structured generators, this has no concept of "table" or "shelf".
    It just creates connected panels that physics will judge.
    
    Constraints:
    - All thicknesses match SendCutSend materials
    - Each component must have sufficient overlap with parent for screw attachment
    - Components can be angled (not just 90°)
    - At least one component must be a flat horizontal surface (for usability)
    
    Args:
        num_components_range: Min/max number of components
        size_range: Min/max size for component width/height (not thickness)
        material_weights: Probability weights for material selection
        ground_contact: If True, ensure at least one component touches z=0
        allow_angles: If True, allow non-90° connections
        max_angle: Maximum rotation angle in radians (when allow_angles=True)
        require_flat_surface: If True, ensure at least one horizontal flat surface
        min_surface_size: Minimum size for the flat surface (mm)
    
    Returns:
        FurnitureGenome with randomly arranged, connected components
    """
    genome = FurnitureGenome()
    num_components = np.random.randint(*num_components_range)
    
    # Track bounding boxes for attachment
    component_bboxes: List[BoundingBox] = []
    
    # Track which component index is the "flat surface" for load testing
    flat_surface_idx: Optional[int] = None
    
    # First component: random panel, positioned to touch ground if required
    mat_name, material = random_material(material_weights)
    thickness = random_thickness(material)
    width = np.random.uniform(*size_range)
    height = np.random.uniform(*size_range)
    
    # Clamp to material max size
    max_w, max_h = material.max_size_mm
    width = min(width, max_w)
    height = min(height, max_h)
    
    # Ensure minimum size for screw attachment
    width = max(width, MIN_OVERLAP_MM * 2)
    height = max(height, MIN_OVERLAP_MM * 2)
    
    # Random orientation: the "thickness" dimension could be X, Y, or Z
    dims = [width, height, thickness]
    np.random.shuffle(dims)
    w, h, t = dims
    
    # Position first component
    if ground_contact:
        z = 0
    else:
        z = np.random.uniform(0, 500)
    
    first_gene = ComponentGene(
        component_type=ComponentType.PANEL,
        width=w,
        height=h,
        thickness=t,
        x=0,
        y=0,
        z=z,
        rx=0, ry=0, rz=0,
    )
    first_gene.modifiers["material"] = mat_name
    genome.add_component(first_gene)
    component_bboxes.append(gene_to_bbox(first_gene))
    
    # Add remaining components, each attached to an existing one
    attempts = 0
    max_attempts = num_components * 10  # Prevent infinite loops
    i = 1
    
    while i < num_components and attempts < max_attempts:
        attempts += 1
        
        # Pick a random existing component to attach to
        parent_idx = np.random.randint(0, len(component_bboxes))
        parent_bbox = component_bboxes[parent_idx]
        
        # Pick a random face to attach to
        face = np.random.choice(list(AttachmentFace))
        
        # For ground contact, avoid attaching below ground
        if ground_contact and face == AttachmentFace.Z_NEG and parent_bbox.z_min <= 0:
            face = np.random.choice([f for f in AttachmentFace if f != AttachmentFace.Z_NEG])
        
        # Generate new component dimensions
        mat_name, material = random_material(material_weights)
        thickness = random_thickness(material)
        width = np.random.uniform(*size_range)
        height = np.random.uniform(*size_range)
        
        # Clamp to material max size
        max_w, max_h = material.max_size_mm
        width = min(width, max_w)
        height = min(height, max_h)
        
        # Ensure minimum size for screw attachment
        width = max(width, MIN_OVERLAP_MM * 2)
        height = max(height, MIN_OVERLAP_MM * 2)
        
        # Shuffle dimensions for variety
        dims = [width, height, thickness]
        np.random.shuffle(dims)
        w, h, t = dims
        
        # Random attachment offset on the face
        # Bias toward center to ensure overlap
        offset_u = np.random.uniform(0.3, 0.7)
        offset_v = np.random.uniform(0.3, 0.7)
        
        # Random angles (if allowed) - can now rotate on multiple axes
        angle = (0.0, 0.0, 0.0)
        if allow_angles and np.random.random() < 0.4:  # 40% chance of angled connection
            # Decide how many axes to rotate on
            num_axes = np.random.choice([1, 2, 3], p=[0.5, 0.35, 0.15])  # Usually 1, sometimes 2-3
            
            rx = np.random.uniform(-max_angle, max_angle) if num_axes >= 1 and np.random.random() < 0.5 else 0.0
            ry = np.random.uniform(-max_angle, max_angle) if num_axes >= 2 or (num_axes == 1 and rx == 0) else 0.0
            rz = np.random.uniform(-max_angle, max_angle) if num_axes >= 3 or (num_axes == 1 and rx == 0 and ry == 0) else 0.0
            
            # Ensure at least one axis has rotation if we wanted angles
            if rx == 0 and ry == 0 and rz == 0:
                axis = np.random.choice([0, 1, 2])
                if axis == 0:
                    rx = np.random.uniform(-max_angle, max_angle)
                elif axis == 1:
                    ry = np.random.uniform(-max_angle, max_angle)
                else:
                    rz = np.random.uniform(-max_angle, max_angle)
            
            angle = (rx, ry, rz)
        
        # Compute position and rotation
        (x, y, z), (rx, ry, rz) = compute_attachment_position(
            parent_bbox, face, w, h, t, offset_u, offset_v, angle
        )
        
        # Ensure not below ground
        if ground_contact and z < 0:
            z = 0
        
        # Create candidate gene
        new_gene = ComponentGene(
            component_type=ComponentType.PANEL,
            width=w,
            height=h,
            thickness=t,
            x=x,
            y=y,
            z=z,
            rx=rx, ry=ry, rz=rz,
        )
        new_gene.modifiers["material"] = mat_name
        
        # Check overlap constraint
        # For angled components, we use a simplified check based on face dimensions
        new_bbox = gene_to_bbox(new_gene)
        face_size = parent_bbox.get_face_size(face)
        
        # Estimate contact area (simplified - assumes alignment)
        # For proper angled joints, we'd need more complex geometry
        min_contact = MIN_OVERLAP_MM * MIN_OVERLAP_MM  # 25mm x 25mm minimum
        
        # Check that both the new component and parent face are large enough
        new_face_size = (w, h) if face in (AttachmentFace.Z_NEG, AttachmentFace.Z_POS) else \
                        (w, t) if face in (AttachmentFace.Y_NEG, AttachmentFace.Y_POS) else \
                        (h, t)
        
        parent_face_area = face_size[0] * face_size[1]
        new_face_area = new_face_size[0] * new_face_size[1]
        
        # Both faces must be large enough, and we need sufficient overlap
        if parent_face_area < min_contact or new_face_area < min_contact:
            # Not enough area for screws, retry
            continue
        
        # Accept this component
        new_idx = genome.add_component(new_gene)
        component_bboxes.append(new_bbox)
        
        # Add connection to parent
        genome.add_connection(ConnectionGene(
            component_a_idx=parent_idx,
            component_b_idx=new_idx,
            joint_type=JointType.BUTT,
        ))
        
        i += 1
    
    # Ensure we have at least one flat horizontal surface
    if require_flat_surface:
        # Check existing components for flat surfaces
        for idx, gene in enumerate(genome.components):
            # A flat horizontal surface has:
            # - No rotation (or negligible rotation)
            # - Thickness is the Z dimension (small compared to width/height)
            # - Sufficient size for placing objects
            is_flat = (abs(gene.rx) < 0.01 and abs(gene.ry) < 0.01)
            is_horizontal = (gene.thickness < gene.width and gene.thickness < gene.height)
            is_large_enough = (gene.width >= min_surface_size and gene.height >= min_surface_size)
            
            if is_flat and is_horizontal and is_large_enough:
                flat_surface_idx = idx
                break
        
        # If no flat surface found, add one
        if flat_surface_idx is None and len(component_bboxes) > 0:
            # Pick a random parent to attach a flat surface to
            parent_idx = np.random.randint(0, len(component_bboxes))
            parent_bbox = component_bboxes[parent_idx]
            
            # Create a flat horizontal panel on top
            mat_name, material = random_material(material_weights)
            thickness = random_thickness(material)
            
            # Ensure it's large enough to be useful
            surface_width = max(np.random.uniform(min_surface_size, size_range[1]), min_surface_size)
            surface_height = max(np.random.uniform(min_surface_size, size_range[1]), min_surface_size)
            
            # Clamp to material max
            max_w, max_h = material.max_size_mm
            surface_width = min(surface_width, max_w)
            surface_height = min(surface_height, max_h)
            
            # Position on top of parent
            (x, y, z), _ = compute_attachment_position(
                parent_bbox, AttachmentFace.Z_POS, 
                surface_width, surface_height, thickness,
                0.5, 0.5, 0.0  # Centered, no angle
            )
            
            flat_gene = ComponentGene(
                component_type=ComponentType.SHELF,  # Mark as shelf for identification
                width=surface_width,
                height=surface_height,
                thickness=thickness,
                x=x,
                y=y,
                z=z,
                rx=0, ry=0, rz=0,  # Perfectly flat
            )
            flat_gene.modifiers["material"] = mat_name
            flat_gene.modifiers["is_flat_surface"] = True
            
            flat_surface_idx = genome.add_component(flat_gene)
            component_bboxes.append(gene_to_bbox(flat_gene))
            
            genome.add_connection(ConnectionGene(
                component_a_idx=parent_idx,
                component_b_idx=flat_surface_idx,
                joint_type=JointType.BUTT,
            ))
    
    # Store flat surface index in genome metadata
    if flat_surface_idx is not None:
        genome.fitness_scores["flat_surface_idx"] = float(flat_surface_idx)
    
    return genome


def create_random_freeform_genome() -> FurnitureGenome:
    """Convenience function with default parameters."""
    return create_freeform_genome()


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    print("Testing freeform genome generator...")
    print(f"Available materials: {len(MATERIALS)}")
    
    for mat_name, mat in MATERIALS.items():
        print(f"  {mat.name}: {len(mat.thicknesses_mm)} thicknesses "
              f"({min(mat.thicknesses_mm):.1f}mm - {max(mat.thicknesses_mm):.1f}mm)")
    
    print("\nGenerating freeform designs...")
    for i in range(5):
        genome = create_freeform_genome()
        design = genome.to_furniture_design(f"freeform_{i}")
        
        materials_used = set()
        for gene in genome.components:
            if "material" in gene.modifiers:
                materials_used.add(gene.modifiers["material"])
        
        print(f"  Design {i}: {len(genome.components)} components, "
              f"{len(genome.connections)} connections, "
              f"materials: {materials_used}")
