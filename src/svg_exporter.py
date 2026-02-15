"""
SVG Exporter for furniture components.

Generates 2D cut files for CNC/laser cutting from FurnitureDesign objects.
"""

import os
import svgwrite
from typing import List, Tuple, Optional
import numpy as np

from furniture import FurnitureDesign, Component, ComponentType

# SVG styling
STYLES = {
    "cut_line": {
        "stroke": "#ff0000",  # Red for cut lines (common convention)
        "stroke_width": "0.5",
        "fill": "none",
    },
    "engrave_line": {
        "stroke": "#0000ff",  # Blue for engraving
        "stroke_width": "0.25",
        "fill": "none",
    },
    "label": {
        "font_size": "12px",
        "font_family": "Arial, sans-serif",
        "fill": "#333333",
    },
    "dimension": {
        "font_size": "10px",
        "font_family": "Arial, sans-serif",
        "fill": "#666666",
    },
}


def component_to_svg(
    component: Component,
    filepath: str,
    add_labels: bool = True,
    add_dimensions: bool = True,
    margin: float = 20.0,  # mm
    scale: float = 1.0,  # 1.0 = 1mm per pixel
) -> str:
    """
    Export a single component to SVG.

    Args:
        component: Furniture component to export
        filepath: Output SVG file path
        add_labels: Add part name/info labels
        add_dimensions: Add dimension annotations
        margin: Margin around the part (mm)
        scale: Scale factor (1.0 = 1:1)

    Returns:
        Path to created SVG file
    """
    # Get component dimensions
    dims = component.get_dimensions()
    width = dims[0] * scale
    height = dims[1] * scale

    # SVG canvas size
    canvas_width = width + 2 * margin
    canvas_height = height + 2 * margin

    # Create SVG document
    dwg = svgwrite.Drawing(
        filepath,
        size=(f"{canvas_width}mm", f"{canvas_height}mm"),
        viewBox=f"0 0 {canvas_width} {canvas_height}",
    )

    # Add definitions for styles
    dwg.defs.add(dwg.style("""
        .cut { stroke: #ff0000; stroke-width: 0.5; fill: none; }
        .engrave { stroke: #0000ff; stroke-width: 0.25; fill: none; }
        .label { font-size: 12px; font-family: Arial, sans-serif; fill: #333; }
        .dim { font-size: 10px; font-family: Arial, sans-serif; fill: #666; }
    """))

    # Draw component outline (cut line)
    # Transform profile to SVG coordinates
    points = []
    for x, y in component.profile:
        svg_x = margin + x * scale
        svg_y = margin + y * scale
        points.append((svg_x, svg_y))

    # Close the polygon
    if points:
        polygon = dwg.polygon(points, class_="cut")
        dwg.add(polygon)

    # Draw cutout features (slots, holes) from Component.features
    for feature in getattr(component, "features", []):
        if not isinstance(feature, dict):
            continue
        feat_type = feature.get("type", "")
        if feat_type in ("slot", "finger_slot", "bolt_hole", "dogbone"):
            # Features with "outline" key contain polygon coords
            outline = feature.get("outline")
            if outline and isinstance(outline, (list, tuple)):
                cutout_points = []
                for fx, fy in outline:
                    cutout_points.append((margin + fx * scale, margin + fy * scale))
                if cutout_points:
                    dwg.add(dwg.polygon(cutout_points, class_="cut"))
        elif feat_type == "engrave":
            coords = feature.get("coords")
            if coords and isinstance(coords, (list, tuple)) and len(coords) >= 2:
                engrave_points = [
                    (margin + x * scale, margin + y * scale) for x, y in coords
                ]
                dwg.add(dwg.polyline(engrave_points, class_="engrave"))

    # Add component label
    if add_labels:
        label_x = margin + width / 2
        label_y = margin + height / 2

        # Part name
        dwg.add(
            dwg.text(
                component.name,
                insert=(label_x, label_y - 10),
                class_="label",
                text_anchor="middle",
            )
        )

        # Part type
        dwg.add(
            dwg.text(
                f"({component.type.value})",
                insert=(label_x, label_y + 5),
                class_="dim",
                text_anchor="middle",
            )
        )

        # Thickness
        dwg.add(
            dwg.text(
                f"t={component.thickness:.1f}mm",
                insert=(label_x, label_y + 18),
                class_="dim",
                text_anchor="middle",
            )
        )

    # Add dimensions
    if add_dimensions:
        # Width dimension (bottom)
        dim_y = canvas_height - margin / 2
        dwg.add(
            dwg.line(
                start=(margin, dim_y - 5),
                end=(margin + width, dim_y - 5),
                class_="engrave",
            )
        )
        dwg.add(
            dwg.text(
                f"{dims[0]:.0f}mm",
                insert=(margin + width / 2, dim_y),
                class_="dim",
                text_anchor="middle",
            )
        )

        # Height dimension (right)
        dim_x = canvas_width - margin / 2
        dwg.add(
            dwg.line(
                start=(dim_x - 5, margin),
                end=(dim_x - 5, margin + height),
                class_="engrave",
            )
        )
        # Rotate text for vertical dimension
        dwg.add(
            dwg.text(
                f"{dims[1]:.0f}mm",
                insert=(dim_x, margin + height / 2),
                class_="dim",
                text_anchor="middle",
                transform=f"rotate(-90, {dim_x}, {margin + height/2})",
            )
        )

    # Save SVG
    dwg.save()
    return filepath


def design_to_svg(
    design: FurnitureDesign,
    output_dir: str,
    add_labels: bool = True,
    add_dimensions: bool = True,
) -> List[str]:
    """
    Export all components of a design to separate SVG files.

    Args:
        design: FurnitureDesign to export
        output_dir: Directory to save SVG files
        add_labels: Add part name/info labels
        add_dimensions: Add dimension annotations

    Returns:
        List of created SVG file paths
    """
    os.makedirs(output_dir, exist_ok=True)

    paths = []
    for component in design.components:
        filename = f"{design.name}_{component.name}.svg"
        filepath = os.path.join(output_dir, filename)

        component_to_svg(
            component,
            filepath,
            add_labels=add_labels,
            add_dimensions=add_dimensions,
        )
        paths.append(filepath)

    return paths


def design_to_nested_svg(
    design: FurnitureDesign,
    filepath: str,
    sheet_width: float = 2440.0,  # mm (standard 8' sheet)
    sheet_height: float = 1220.0,  # mm (standard 4' sheet)
    spacing: float = 10.0,  # mm between parts
    add_labels: bool = True,
) -> str:
    """
    Export all components to a single SVG with simple nesting.

    This is a basic row-based nesting algorithm. For production,
    you'd want a more sophisticated nesting optimizer.

    Args:
        design: FurnitureDesign to export
        filepath: Output SVG file path
        sheet_width: Sheet material width (mm)
        sheet_height: Sheet material height (mm)
        spacing: Spacing between parts (mm)
        add_labels: Add part labels

    Returns:
        Path to created SVG file
    """
    # Create SVG document
    dwg = svgwrite.Drawing(
        filepath,
        size=(f"{sheet_width}mm", f"{sheet_height}mm"),
        viewBox=f"0 0 {sheet_width} {sheet_height}",
    )

    # Add styles
    dwg.defs.add(dwg.style("""
        .cut { stroke: #ff0000; stroke-width: 0.5; fill: none; }
        .sheet { stroke: #cccccc; stroke-width: 1; fill: #f9f9f9; }
        .label { font-size: 8px; font-family: Arial, sans-serif; fill: #333; }
    """))

    # Draw sheet outline
    dwg.add(dwg.rect(insert=(0, 0), size=(sheet_width, sheet_height), class_="sheet"))

    # Simple row-based nesting
    current_x = spacing
    current_y = spacing
    row_height = 0

    for component in design.components:
        dims = component.get_dimensions()
        width = dims[0]
        height = dims[1]

        # Check if part fits in current row
        if current_x + width + spacing > sheet_width:
            # Move to next row
            current_x = spacing
            current_y += row_height + spacing
            row_height = 0

        # Check if part fits on sheet
        if current_y + height + spacing > sheet_height:
            # Sheet is full - in production, would start new sheet
            print(f"Warning: Component {component.name} doesn't fit on sheet")
            continue

        # Draw component
        points = []
        for px, py in component.profile:
            points.append((current_x + px, current_y + py))

        dwg.add(dwg.polygon(points, class_="cut"))

        # Add label
        if add_labels:
            dwg.add(
                dwg.text(
                    component.name,
                    insert=(current_x + width / 2, current_y + height / 2),
                    class_="label",
                    text_anchor="middle",
                )
            )

        # Update position
        current_x += width + spacing
        row_height = max(row_height, height)

    dwg.save()
    return filepath


if __name__ == "__main__":
    import numpy as np
    from furniture import Component, ComponentType, FurnitureDesign

    print("Testing SVG export...")

    table = FurnitureDesign(name="test_table")
    table.add_component(
        Component(
            name="top",
            type=ComponentType.PANEL,
            profile=[(0, 0), (800, 0), (800, 500), (0, 500)],
            thickness=19.0,
            material="plywood",
            position=np.array([0.0, 0.0, 700.0]),
        )
    )
    for i, (x, y) in enumerate([(20, 20), (720, 20), (20, 420), (720, 420)]):
        table.add_component(
            Component(
                name=f"leg_{i}",
                type=ComponentType.LEG,
                profile=[(0, 0), (60, 0), (60, 60), (0, 60)],
                thickness=700.0,
                material="plywood",
                position=np.array([float(x), float(y), 0.0]),
            )
        )

    output_dir = os.path.join(os.path.dirname(__file__), "..", "examples", "svg_output")
    paths = design_to_svg(table, output_dir)
    print(f"\nExported {len(paths)} component SVGs to {output_dir}")
    for p in paths:
        print(f"  - {os.path.basename(p)}")

    nested_path = os.path.join(output_dir, f"{table.name}_nested.svg")
    design_to_nested_svg(table, nested_path)
    print(f"\nExported nested layout to {nested_path}")
