"""
Design-for-Manufacturing (DFM) validation rules for SendCutSend CNC routing.

Checks part profiles against manufacturing constraints (minimum slot widths,
internal radii, bridge widths, aspect ratios, sheet sizes). Also provides
dogbone relief insertion for internal right-angle corners.
"""

import logging
import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
from shapely.geometry import Point, Polygon, LineString, MultiPolygon
from shapely.ops import unary_union

from geometry_primitives import PartProfile2D
from materials import MATERIALS, INCH_TO_MM

logger = logging.getLogger(__name__)


@dataclass
class DFMConfig:
    """CNC routing manufacturing constraints."""

    min_slot_width_inch: float = 0.125  # 1/8" end mill
    min_internal_radius_inch: float = 0.063  # 1/16" inside corner radius
    slip_fit_clearance_inch: float = 0.010  # per-side clearance for press fit
    min_bridge_width_inch: float = 0.125  # minimum remaining material
    max_aspect_ratio: float = 20.0  # length/width before warping risk
    max_sheet_width_mm: float = 10000.0  # effectively unlimited
    max_sheet_height_mm: float = 10000.0  # effectively unlimited

    @property
    def min_slot_width_mm(self) -> float:
        return self.min_slot_width_inch * INCH_TO_MM

    @property
    def min_internal_radius_mm(self) -> float:
        return self.min_internal_radius_inch * INCH_TO_MM

    @property
    def slip_fit_clearance_mm(self) -> float:
        return self.slip_fit_clearance_inch * INCH_TO_MM

    @property
    def min_bridge_width_mm(self) -> float:
        return self.min_bridge_width_inch * INCH_TO_MM

    @classmethod
    def from_material(cls, material_key: str) -> "DFMConfig":
        """Create a DFMConfig with sheet size from a material."""
        mat = MATERIALS.get(material_key)
        if mat is None:
            return cls()
        max_w, max_h = mat.max_size_mm
        return cls(max_sheet_width_mm=max_w, max_sheet_height_mm=max_h)


@dataclass
class DFMViolation:
    """A single DFM rule violation."""

    rule_name: str
    severity: str  # "error" or "warning"
    message: str
    location: Optional[Point] = None
    value: float = 0.0
    limit: float = 0.0


def check_part_dfm(
    profile: PartProfile2D,
    config: Optional[DFMConfig] = None,
) -> List[DFMViolation]:
    """Run all DFM checks on a part profile.

    Args:
        profile: The 2D part to validate.
        config: Manufacturing constraints.

    Returns:
        List of violations (empty = passes all checks).
    """
    if config is None:
        config = DFMConfig.from_material(profile.material_key)

    violations: List[DFMViolation] = []

    violations.extend(_check_sheet_size(profile, config))
    violations.extend(_check_aspect_ratio(profile, config))
    violations.extend(_check_slot_widths(profile, config))
    violations.extend(_check_bridge_widths(profile, config))
    violations.extend(_check_internal_radii(profile, config))

    return violations


# ─── Individual checks ───────────────────────────────────────────────────────


def _check_sheet_size(
    profile: PartProfile2D,
    config: DFMConfig,
) -> List[DFMViolation]:
    """Check that part fits within max sheet dimensions."""
    violations = []
    bounds = profile.outline.bounds  # (minx, miny, maxx, maxy)
    width = bounds[2] - bounds[0]
    height = bounds[3] - bounds[1]

    if width > config.max_sheet_width_mm:
        violations.append(
            DFMViolation(
                rule_name="sheet_size_width",
                severity="error",
                message=f"Part width {width:.1f}mm exceeds sheet width {config.max_sheet_width_mm:.1f}mm",
                value=width,
                limit=config.max_sheet_width_mm,
            )
        )
    if height > config.max_sheet_height_mm:
        violations.append(
            DFMViolation(
                rule_name="sheet_size_height",
                severity="error",
                message=f"Part height {height:.1f}mm exceeds sheet height {config.max_sheet_height_mm:.1f}mm",
                value=height,
                limit=config.max_sheet_height_mm,
            )
        )
    return violations


def _check_aspect_ratio(
    profile: PartProfile2D,
    config: DFMConfig,
) -> List[DFMViolation]:
    """Check part aspect ratio."""
    bounds = profile.outline.bounds
    width = bounds[2] - bounds[0]
    height = bounds[3] - bounds[1]

    if width < 1e-6 or height < 1e-6:
        return []

    ratio = max(width, height) / min(width, height)
    if ratio > config.max_aspect_ratio:
        return [
            DFMViolation(
                rule_name="aspect_ratio",
                severity="warning",
                message=f"Aspect ratio {ratio:.1f} exceeds {config.max_aspect_ratio:.1f} (warping risk)",
                value=ratio,
                limit=config.max_aspect_ratio,
            )
        ]
    return []


def _check_slot_widths(
    profile: PartProfile2D,
    config: DFMConfig,
) -> List[DFMViolation]:
    """Check that all cutout slots are at least min_slot_width wide."""
    violations = []
    min_w = config.min_slot_width_mm

    for i, cutout in enumerate(profile.cutouts):
        if cutout.is_empty:
            continue
        # Use the minimum dimension of the cutout's bounding box
        b = cutout.bounds
        slot_w = min(b[2] - b[0], b[3] - b[1])
        if slot_w < min_w:
            centroid = cutout.centroid
            violations.append(
                DFMViolation(
                    rule_name="slot_width",
                    severity="error",
                    message=f"Cutout {i} width {slot_w:.2f}mm < minimum {min_w:.2f}mm",
                    location=centroid,
                    value=slot_w,
                    limit=min_w,
                )
            )
    return violations


def _check_bridge_widths(
    profile: PartProfile2D,
    config: DFMConfig,
) -> List[DFMViolation]:
    """Check minimum material bridge between cutouts and edges."""
    violations = []
    min_bridge = config.min_bridge_width_mm

    net = profile.net_polygon()
    if net.is_empty:
        violations.append(
            DFMViolation(
                rule_name="bridge_width",
                severity="error",
                message="Net polygon is empty (cutouts consume entire part)",
                value=0.0,
                limit=min_bridge,
            )
        )
        return violations

    # Check distance between each cutout and the outline boundary
    outline = profile.outline
    if isinstance(outline, MultiPolygon):
        outline = max(outline.geoms, key=lambda g: g.area)
    for i, cutout in enumerate(profile.cutouts):
        if cutout.is_empty:
            continue
        dist = outline.exterior.distance(cutout)
        if dist < min_bridge:
            violations.append(
                DFMViolation(
                    rule_name="bridge_width",
                    severity="error" if dist < min_bridge * 0.5 else "warning",
                    message=f"Cutout {i} is {dist:.2f}mm from edge (min {min_bridge:.2f}mm)",
                    location=cutout.centroid,
                    value=dist,
                    limit=min_bridge,
                )
            )

    # Check distance between pairs of cutouts
    for i in range(len(profile.cutouts)):
        for j in range(i + 1, len(profile.cutouts)):
            if profile.cutouts[i].is_empty or profile.cutouts[j].is_empty:
                continue
            dist = profile.cutouts[i].distance(profile.cutouts[j])
            if dist < min_bridge:
                violations.append(
                    DFMViolation(
                        rule_name="bridge_width",
                        severity="warning",
                        message=f"Cutouts {i} and {j} are {dist:.2f}mm apart (min {min_bridge:.2f}mm)",
                        value=dist,
                        limit=min_bridge,
                    )
                )

    return violations


def _check_internal_radii(
    profile: PartProfile2D,
    config: DFMConfig,
) -> List[DFMViolation]:
    """Check that internal corners have sufficient radius."""
    violations = []
    min_r = config.min_internal_radius_mm

    # Check cutout corners
    for i, cutout in enumerate(profile.cutouts):
        if cutout.is_empty:
            continue
        sharp_corners = _find_sharp_internal_corners(cutout, min_r)
        for pt, angle in sharp_corners:
            violations.append(
                DFMViolation(
                    rule_name="internal_radius",
                    severity="warning",
                    message=f"Cutout {i} has sharp internal corner ({math.degrees(angle):.0f} deg) - needs dogbone",
                    location=pt,
                    value=0.0,
                    limit=min_r,
                )
            )

    return violations


def _find_sharp_internal_corners(
    polygon: Polygon,
    min_radius: float,
) -> List[Tuple[Point, float]]:
    """Find corners in a polygon that are too sharp for CNC routing."""
    coords = list(polygon.exterior.coords[:-1])
    if len(coords) < 3:
        return []

    sharp = []
    n = len(coords)
    for i in range(n):
        p0 = np.array(coords[(i - 1) % n])
        p1 = np.array(coords[i])
        p2 = np.array(coords[(i + 1) % n])

        v1 = p0 - p1
        v2 = p2 - p1
        len1 = np.linalg.norm(v1)
        len2 = np.linalg.norm(v2)
        if len1 < 1e-8 or len2 < 1e-8:
            continue

        cos_angle = np.clip(np.dot(v1, v2) / (len1 * len2), -1, 1)
        angle = math.acos(cos_angle)

        # Internal corners with angle < 120 degrees need dogbone relief
        if angle < math.radians(120):
            sharp.append((Point(p1[0], p1[1]), angle))

    return sharp


# ─── Dogbone Relief ─────────────────────────────────────────────────────────


def add_dogbone_relief(
    polygon: Polygon,
    radius_mm: float,
    angle_threshold_deg: float = 100.0,
) -> Polygon:
    """Add dogbone relief circles to internal right-angle corners.

    For each corner with angle < angle_threshold_deg, place a circle on the
    bisector direction into the material, then union with the polygon.

    Args:
        polygon: The slot/cutout polygon to add dogbones to.
        radius_mm: Radius of dogbone relief circles (typically = bit radius).
        angle_threshold_deg: Corners sharper than this get dogbones.

    Returns:
        Polygon with dogbone circles added.
    """
    coords = list(polygon.exterior.coords[:-1])
    if len(coords) < 3:
        return polygon

    circles = []
    n = len(coords)
    threshold = math.radians(angle_threshold_deg)

    for i in range(n):
        p0 = np.array(coords[(i - 1) % n])
        p1 = np.array(coords[i])
        p2 = np.array(coords[(i + 1) % n])

        v1 = p0 - p1
        v2 = p2 - p1
        len1 = np.linalg.norm(v1)
        len2 = np.linalg.norm(v2)
        if len1 < 1e-8 or len2 < 1e-8:
            continue

        v1n = v1 / len1
        v2n = v2 / len2
        cos_angle = np.clip(np.dot(v1n, v2n), -1, 1)
        angle = math.acos(cos_angle)

        if angle < threshold:
            # Bisector direction (into the material = away from the angle)
            bisector = v1n + v2n
            bis_len = np.linalg.norm(bisector)
            if bis_len < 1e-8:
                continue
            bisector /= bis_len

            # Place circle center at corner + bisector * radius
            center = p1 + bisector * radius_mm
            circle = Point(float(center[0]), float(center[1])).buffer(
                radius_mm,
                quad_segs=4,
            )
            circles.append(circle)

    if not circles:
        return polygon

    result = polygon
    for circle in circles:
        result = result.union(circle)

    # Ensure we return a Polygon (not MultiPolygon)
    if isinstance(result, MultiPolygon):
        result = max(result.geoms, key=lambda g: g.area)

    return result
