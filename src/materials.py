"""
SendCutSend material catalog.

Real material thicknesses, densities, and sheet sizes for CNC-cut flat-pack furniture.
Used by mesh_decomposer.py for slab material assignment.
"""

from dataclasses import dataclass
from typing import List, Tuple

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
        thicknesses_inch=[
            0.030,
            0.048,
            0.060,
            0.075,
            0.090,
            0.120,
            0.135,
            0.187,
            0.250,
            0.375,
            0.500,
        ],
        density=7850,
        max_size_inch=(48, 96),
    ),
    "aluminum_5052": Material(
        name="Aluminum 5052",
        thicknesses_inch=[
            0.032,
            0.040,
            0.050,
            0.063,
            0.080,
            0.090,
            0.100,
            0.125,
            0.160,
            0.187,
            0.250,
        ],
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

# Minimum overlap for a valid screw connection (mm)
# Two #8 screws need at least ~20mm spacing, plus edge margin
MIN_OVERLAP_MM = 25.0
