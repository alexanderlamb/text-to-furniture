"""Built-in fabrication strategy prototypes."""

from fabrication.strategies.contour_stack import ContourStackStrategy
from fabrication.strategies.planar_skin import PlanarSkinStrategy
from fabrication.strategies.waffle_ribs import WaffleRibsStrategy
from fabrication.strategies.voxel_blocks import VoxelBlocksStrategy

__all__ = [
    "ContourStackStrategy",
    "PlanarSkinStrategy",
    "WaffleRibsStrategy",
    "VoxelBlocksStrategy",
]
