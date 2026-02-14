"""
DXF export for CNC-ready flat-pack parts.

Uses ezdxf to produce DXF files with proper layers:
  - CUT (red, ACI 1): through-cut outlines and cutouts
  - ENGRAVE (blue, ACI 5): engravings, labels, fold lines

Units: millimeters. Format: R2010.
"""
import logging
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import ezdxf
from ezdxf.enums import TextEntityAlignment
from shapely.geometry import Polygon, MultiPolygon

from geometry_primitives import PartProfile2D, PartFeature, FeatureType

logger = logging.getLogger(__name__)


@dataclass
class DXFExportConfig:
    """Configuration for DXF export."""
    cut_layer: str = "CUT"
    engrave_layer: str = "ENGRAVE"
    cut_color: int = 1       # ACI red
    engrave_color: int = 5   # ACI blue
    units: str = "mm"
    add_part_labels: bool = True
    label_height_mm: float = 5.0


def part_to_dxf(
    profile: PartProfile2D,
    filepath: str,
    part_name: str = "part",
    config: Optional[DXFExportConfig] = None,
) -> str:
    """Export a single part to a DXF file.

    Args:
        profile: The 2D part profile.
        filepath: Output DXF file path.
        part_name: Label for the part.
        config: DXF export settings.

    Returns:
        Path to created DXF file.
    """
    if config is None:
        config = DXFExportConfig()

    doc = ezdxf.new("R2010")
    doc.units = ezdxf.units.MM
    msp = doc.modelspace()

    _setup_layers(doc, config)
    _add_part_geometry(msp, profile, config)

    if config.add_part_labels:
        _add_label(msp, profile, part_name, config)

    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    doc.saveas(filepath)
    logger.info("Exported DXF: %s", filepath)
    return filepath


def design_to_dxf(
    parts: List[Tuple[str, PartProfile2D]],
    output_dir: str,
    config: Optional[DXFExportConfig] = None,
) -> List[str]:
    """Export each part to a separate DXF file.

    Args:
        parts: List of (name, profile) tuples.
        output_dir: Directory for output files.
        config: DXF export settings.

    Returns:
        List of created DXF file paths.
    """
    os.makedirs(output_dir, exist_ok=True)
    paths = []

    for name, profile in parts:
        filepath = os.path.join(output_dir, f"{name}.dxf")
        part_to_dxf(profile, filepath, part_name=name, config=config)
        paths.append(filepath)

    return paths


def design_to_nested_dxf(
    parts: List[Tuple[str, PartProfile2D]],
    filepath: str,
    sheet_width_mm: float = 609.6,
    sheet_height_mm: float = 762.0,
    spacing_mm: float = 10.0,
    config: Optional[DXFExportConfig] = None,
) -> str:
    """Export all parts nested onto a single sheet DXF.

    Uses a simple row-based nesting algorithm.

    Args:
        parts: List of (name, profile) tuples.
        filepath: Output DXF file path.
        sheet_width_mm: Sheet width.
        sheet_height_mm: Sheet height.
        spacing_mm: Spacing between parts.
        config: DXF export settings.

    Returns:
        Path to created DXF file.
    """
    if config is None:
        config = DXFExportConfig()

    doc = ezdxf.new("R2010")
    doc.units = ezdxf.units.MM
    msp = doc.modelspace()
    _setup_layers(doc, config)

    # Draw sheet outline
    sheet = Polygon([
        (0, 0), (sheet_width_mm, 0),
        (sheet_width_mm, sheet_height_mm), (0, sheet_height_mm),
    ])
    _add_polygon_to_dxf(msp, sheet, config.engrave_layer)

    # Simple row-based nesting
    current_x = spacing_mm
    current_y = spacing_mm
    row_height = 0.0

    for name, profile in parts:
        bounds = profile.outline.bounds
        w = bounds[2] - bounds[0]
        h = bounds[3] - bounds[1]

        if current_x + w + spacing_mm > sheet_width_mm:
            current_x = spacing_mm
            current_y += row_height + spacing_mm
            row_height = 0.0

        if current_y + h + spacing_mm > sheet_height_mm:
            logger.warning("Part %s doesn't fit on sheet", name)
            continue

        # Translate part to current position
        offset_x = current_x - bounds[0]
        offset_y = current_y - bounds[1]

        from shapely import affinity
        translated = affinity.translate(profile.outline, offset_x, offset_y)
        _add_polygon_to_dxf(msp, translated, config.cut_layer)

        # Translate and add cutouts
        for cutout in profile.cutouts:
            translated_cutout = affinity.translate(cutout, offset_x, offset_y)
            _add_polygon_to_dxf(msp, translated_cutout, config.cut_layer)

        # Add label
        if config.add_part_labels:
            label_x = current_x + w / 2
            label_y = current_y + h / 2
            msp.add_text(
                name,
                height=config.label_height_mm,
                dxfattribs={"layer": config.engrave_layer},
            ).set_placement((label_x, label_y), align=TextEntityAlignment.MIDDLE_CENTER)

        current_x += w + spacing_mm
        row_height = max(row_height, h)

    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    doc.saveas(filepath)
    logger.info("Exported nested DXF: %s", filepath)
    return filepath


# ─── Internal helpers ────────────────────────────────────────────────────────

def _setup_layers(doc, config: DXFExportConfig) -> None:
    """Create CUT and ENGRAVE layers."""
    doc.layers.add(config.cut_layer, color=config.cut_color)
    doc.layers.add(config.engrave_layer, color=config.engrave_color)


def _add_part_geometry(
    msp,
    profile: PartProfile2D,
    config: DXFExportConfig,
) -> None:
    """Add part outline and cutouts to modelspace."""
    # Outer outline
    _add_polygon_to_dxf(msp, profile.outline, config.cut_layer)

    # Cutouts
    for cutout in profile.cutouts:
        _add_polygon_to_dxf(msp, cutout, config.cut_layer)

    # Engrave features
    for feature in profile.features:
        if feature.feature_type == FeatureType.ENGRAVE:
            if hasattr(feature.geometry, 'coords'):
                # LineString
                coords = list(feature.geometry.coords)
                if len(coords) >= 2:
                    msp.add_lwpolyline(
                        coords,
                        dxfattribs={"layer": config.engrave_layer},
                    )


def _add_polygon_to_dxf(msp, polygon: Polygon, layer: str) -> None:
    """Add a Shapely polygon as a closed LWPolyline."""
    if polygon.is_empty:
        return

    if isinstance(polygon, MultiPolygon):
        for geom in polygon.geoms:
            _add_polygon_to_dxf(msp, geom, layer)
        return

    # Exterior ring
    coords = list(polygon.exterior.coords)
    if len(coords) >= 3:
        msp.add_lwpolyline(
            coords,
            close=True,
            dxfattribs={"layer": layer},
        )

    # Interior rings (holes)
    for interior in polygon.interiors:
        coords = list(interior.coords)
        if len(coords) >= 3:
            msp.add_lwpolyline(
                coords,
                close=True,
                dxfattribs={"layer": layer},
            )


def _add_label(
    msp,
    profile: PartProfile2D,
    name: str,
    config: DXFExportConfig,
) -> None:
    """Add a text label at the centroid of the part."""
    centroid = profile.outline.centroid
    msp.add_text(
        name,
        height=config.label_height_mm,
        dxfattribs={"layer": config.engrave_layer},
    ).set_placement(
        (centroid.x, centroid.y),
        align=TextEntityAlignment.MIDDLE_CENTER,
    )

    # Add thickness annotation
    msp.add_text(
        f"t={profile.thickness_mm:.1f}mm",
        height=config.label_height_mm * 0.7,
        dxfattribs={"layer": config.engrave_layer},
    ).set_placement(
        (centroid.x, centroid.y - config.label_height_mm * 2),
        align=TextEntityAlignment.MIDDLE_CENTER,
    )
