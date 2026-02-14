"""
Reusable UI builder functions for the NiceGUI web interface.

Designed to render inside overlay panels — compact layout, no outer cards.
"""

import os
import time
import json as _json
from dataclasses import replace
from typing import Callable, List, Optional

from nicegui import ui

from materials import MATERIALS
from ui.state import RunSlot
from ui.workers import read_progress

# Role badge colors for structural segments
_ROLE_COLORS = {
    "horizontal_surface": "blue",
    "vertical_panel": "green",
    "leg": "orange",
    "brace": "purple",
    "organic": "grey",
}


def build_parameter_panel(
    slot: RunSlot,
    other_slot: RunSlot,
    slot_label: str,
    other_label: str,
    on_change: Callable,
) -> None:
    """Build decomposition parameter controls for one run slot (compact)."""
    cfg = slot.config
    material_options = {k: v.name for k, v in MATERIALS.items()}

    ui.select(
        material_options,
        value=cfg.default_material,
        label="Material",
        on_change=lambda e: _set(cfg, "default_material", e.value, on_change),
    ).classes("w-full")

    _slider(
        "Height (mm)", cfg.target_height_mm, 100, 2000, 10,
        lambda v: _set(cfg, "target_height_mm", v, on_change),
    )
    _slider(
        "Max Slabs", cfg.max_slabs, 3, 25, 1,
        lambda v: _set(cfg, "max_slabs", int(v), on_change),
    )
    ui.select(
        {
            "volume_fill": "Volume Fill",
            "surface_coverage": "Surface Coverage",
        },
        value=cfg.selection_objective_mode,
        label="Selection Objective",
        on_change=lambda e: _set(
            cfg, "selection_objective_mode", e.value, on_change,
        ),
    ).classes("w-full")
    _slider(
        "Coverage", cfg.coverage_target, 0.50, 1.00, 0.01,
        lambda v: _set(cfg, "coverage_target", round(v, 2), on_change),
        fmt=":.0%",
    )
    _slider(
        "Target Vol Fill", cfg.target_volume_fill, 0.20, 0.95, 0.01,
        lambda v: _set(cfg, "target_volume_fill", round(v, 2), on_change),
        fmt=":.0%",
    )
    _slider(
        "Min Vol Contrib", cfg.min_volume_contribution, 0.0005, 0.02, 0.0005,
        lambda v: _set(cfg, "min_volume_contribution", round(v, 4), on_change),
    )
    _slider(
        "Plane Penalty", cfg.plane_penalty_weight, 0.0, 0.01, 0.0001,
        lambda v: _set(cfg, "plane_penalty_weight", round(v, 4), on_change),
    )
    _slider(
        "Voxel Res (mm)", cfg.voxel_resolution_mm, 2, 15, 0.5,
        lambda v: _set(cfg, "voxel_resolution_mm", v, on_change),
    )
    _slider(
        "Cluster (deg)", cfg.normal_cluster_threshold_deg, 5, 45, 1,
        lambda v: _set(cfg, "normal_cluster_threshold_deg", v, on_change),
    )

    with ui.row().classes("w-full items-center justify-between"):
        ui.switch(
            "Physics Sim",
            value=slot.run_simulation,
            on_change=lambda e: _set_attr(
                slot, "run_simulation", e.value, on_change
            ),
        )
        ui.button(
            f"Copy {other_label}",
            on_click=lambda: _copy_config(other_slot, slot, on_change),
        ).props("flat dense size=sm")


def build_component_table(slot: RunSlot) -> None:
    """Render a table of components with dimensions."""
    result = slot.result
    if result is None:
        return

    design = result.design
    columns = [
        {"name": "name", "label": "Name", "field": "name", "align": "left"},
        {"name": "type", "label": "Type", "field": "type"},
        {"name": "width", "label": "W", "field": "width"},
        {"name": "height", "label": "H", "field": "height"},
        {"name": "thickness", "label": "T", "field": "thickness"},
        {"name": "material", "label": "Mat", "field": "material"},
    ]

    rows = []
    for comp in design.components:
        dims = comp.get_dimensions()
        rows.append({
            "name": comp.name,
            "type": comp.type.value,
            "width": f"{dims[0]:.0f}",
            "height": f"{dims[1]:.0f}",
            "thickness": f"{comp.thickness:.1f}",
            "material": comp.material,
        })

    ui.table(
        columns=columns, rows=rows, row_key="name"
    ).classes("w-full").props("dense flat")


def build_svg_preview(slot: RunSlot) -> None:
    """Show inline SVG preview."""
    result = slot.result
    if result is None:
        return

    if result.nested_svg_path and os.path.isfile(result.nested_svg_path):
        with open(result.nested_svg_path) as f:
            svg_content = f.read()
        with ui.expansion("Nested SVG", icon="view_compact").classes("w-full"):
            ui.html(
                f'<div style="max-width:100%;overflow:auto">{svg_content}</div>'
            )

    if result.svg_paths:
        with ui.expansion("Part SVGs", icon="grid_view").classes("w-full"):
            for svg_path in result.svg_paths:
                if os.path.isfile(svg_path):
                    name = os.path.basename(svg_path)
                    with open(svg_path) as f:
                        svg_content = f.read()
                    with ui.expansion(name).classes("w-full"):
                        ui.html(
                            f'<div style="max-width:100%;overflow:auto">'
                            f"{svg_content}</div>"
                        )


def build_progress_log(slot: RunSlot) -> Optional[ui.timer]:
    """Render a live progress log that polls the slot's progress file.

    Shows elapsed time, current step, and a scrollable log of all messages.
    Returns the timer so the caller can deactivate it when the run completes.
    """
    if not slot.progress_file:
        return None

    start_time = time.time()

    # Container for dynamic content
    with ui.column().classes("w-full gap-1") as container:
        elapsed_label = ui.label("").classes("text-xs text-gray-500")
        step_label = ui.label("Starting...").classes(
            "text-sm font-medium text-blue-700"
        )
        log_area = ui.element("div").classes(
            "w-full max-h-32 overflow-y-auto bg-gray-50 rounded p-1"
        ).style("font-family: monospace; font-size: 11px;")

    seen_count = {"n": 0}

    def _poll():
        entries = read_progress(slot.progress_file)
        if not entries:
            return

        # Elapsed time
        elapsed = time.time() - start_time
        elapsed_label.text = f"Elapsed: {elapsed:.0f}s"

        # Latest step as the headline
        latest = entries[-1]
        level = latest.get("level", "info")
        detail = latest.get("detail", "")
        if level == "error":
            step_label.text = "Error"
            step_label.classes(replace="text-sm font-medium text-red-600")
        else:
            step_label.text = detail[:80] if detail else latest.get("step", "")
            step_label.classes(replace="text-sm font-medium text-blue-700")

        # Only append new entries to the log area
        if len(entries) > seen_count["n"]:
            new_entries = entries[seen_count["n"]:]
            seen_count["n"] = len(entries)
            with log_area:
                for entry in new_entries:
                    lvl = entry.get("level", "info")
                    txt = entry.get("detail", entry.get("step", ""))
                    if lvl == "error":
                        color_cls = "text-red-600"
                    elif lvl == "warning":
                        color_cls = "text-amber-600"
                    else:
                        color_cls = "text-gray-700"
                    # Timestamp relative to start
                    dt = entry.get("t", time.time()) - start_time
                    ui.label(
                        f"[{dt:5.1f}s] {txt}"
                    ).classes(f"text-[10px] leading-tight {color_cls}")

        # Auto-stop when slot is done
        if not slot.running:
            elapsed_label.text = f"Done in {elapsed:.1f}s"
            timer.deactivate()

    timer = ui.timer(0.5, _poll)
    return timer


def build_manufacturing_results(mfg_summary: dict) -> None:
    """Render manufacturing pipeline intermediate results as expandable sections."""
    stages = mfg_summary.get("stages", {})
    scoring = stages.get("scoring")

    # Top-level metrics row
    if scoring:
        ui.separator().classes("my-1")
        ui.label("Manufacturing Results").classes("text-sm font-bold")
        with ui.row().classes("gap-3 flex-wrap"):
            _metric("Score", f"{scoring['overall_score']:.2f}")
            _metric("Parts", str(scoring["part_count"]))
            _metric("DFM Err", str(scoring["dfm_errors"]))
            _metric("DFM Warn", str(scoring["dfm_warnings"]))
            h = scoring["hausdorff_mm"]
            _metric("Hausdorff", f"{h:.1f}" if h < 1e6 else "N/A")

    # Stage 1: Patches
    patches = stages.get("patches", [])
    if patches:
        with ui.expansion(
            f"Patches ({len(patches)})", icon="layers"
        ).classes("w-full"):
            columns = [
                {"name": "idx", "label": "#", "field": "idx"},
                {"name": "area", "label": "Area (mm2)", "field": "area"},
                {"name": "normal", "label": "Normal", "field": "normal"},
                {"name": "faces", "label": "Faces", "field": "faces"},
            ]
            rows = [
                {
                    "idx": str(i),
                    "area": f"{p['area_mm2']:.0f}",
                    "normal": _fmt_vec(p["normal"]),
                    "faces": str(p["n_faces"]),
                }
                for i, p in enumerate(patches)
            ]
            ui.table(
                columns=columns, rows=rows, row_key="idx"
            ).classes("w-full").props("dense flat")

    # Stage 2: Segments
    segments = stages.get("segments", [])
    if segments:
        with ui.expansion(
            f"Segments ({len(segments)})", icon="view_module"
        ).classes("w-full"):
            for i, seg in enumerate(segments):
                with ui.row().classes("w-full items-center gap-2 py-1"):
                    role = seg["role"]
                    color = _ROLE_COLORS.get(role, "grey")
                    ui.badge(role, color=color).props("outline dense")
                    ui.label(
                        f"{seg['n_patches']} patches, "
                        f"{seg['total_area_mm2']:.0f} mm2"
                    ).classes("text-xs")

    # Stage 3: Ladder candidates
    candidates = stages.get("candidates", [])
    if candidates:
        with ui.expansion(
            f"Ladder ({len(candidates)})", icon="stairs"
        ).classes("w-full"):
            columns = [
                {"name": "idx", "label": "#", "field": "idx"},
                {"name": "rung", "label": "Rung", "field": "rung"},
                {"name": "parts", "label": "Parts", "field": "parts"},
                {"name": "score", "label": "Score", "field": "score"},
                {"name": "dfm", "label": "DFM", "field": "dfm"},
            ]
            rows = [
                {
                    "idx": str(i),
                    "rung": str(c["rung"]),
                    "parts": str(c["n_parts"]),
                    "score": f"{c['score']:.2f}",
                    "dfm": f"{c['n_dfm_errors']}E/{c['n_dfm_warnings']}W",
                }
                for i, c in enumerate(candidates)
            ]
            ui.table(
                columns=columns, rows=rows, row_key="idx"
            ).classes("w-full").props("dense flat")

    # Stage 4: Parts 2D profiles
    parts_2d = mfg_summary.get("parts_2d", [])
    if parts_2d:
        with ui.expansion(
            f"2D Profiles ({len(parts_2d)})", icon="crop_square"
        ).classes("w-full"):
            for part in parts_2d:
                with ui.column().classes("w-full mb-2"):
                    ui.label(
                        f"{part['name']} - {part['width_mm']:.0f} x "
                        f"{part['height_mm']:.0f} x {part['thickness_mm']:.1f} mm"
                    ).classes("text-xs font-medium")
                    svg_str = _profile_to_inline_svg(
                        part["outline_coords"],
                        part.get("cutout_coords", []),
                    )
                    ui.html(svg_str)

    # Stage 5: Joints
    joints = stages.get("joints", [])
    if joints:
        with ui.expansion(
            f"Joints ({len(joints)})", icon="link"
        ).classes("w-full"):
            columns = [
                {"name": "a", "label": "Part A", "field": "a", "align": "left"},
                {"name": "b", "label": "Part B", "field": "b", "align": "left"},
                {"name": "type", "label": "Type", "field": "type"},
            ]
            rows = [
                {
                    "a": j["component_a"],
                    "b": j["component_b"],
                    "type": j["joint_type"],
                }
                for j in joints
            ]
            ui.table(
                columns=columns, rows=rows, row_key="a"
            ).classes("w-full").props("dense flat")

    # Stage 6: DFM Violations
    dfm_violations = stages.get("dfm_violations", [])
    if dfm_violations:
        with ui.expansion(
            f"DFM Violations ({len(dfm_violations)})", icon="warning"
        ).classes("w-full"):
            columns = [
                {"name": "rule", "label": "Rule", "field": "rule", "align": "left"},
                {"name": "sev", "label": "Sev", "field": "sev"},
                {"name": "msg", "label": "Message", "field": "msg", "align": "left"},
                {"name": "val", "label": "Value", "field": "val"},
                {"name": "lim", "label": "Limit", "field": "lim"},
            ]
            rows = [
                {
                    "rule": v["rule"],
                    "sev": v["severity"],
                    "msg": v["message"],
                    "val": f"{v['value']:.2f}",
                    "lim": f"{v['limit']:.2f}",
                }
                for v in dfm_violations
            ]
            ui.table(
                columns=columns, rows=rows, row_key="rule"
            ).classes("w-full").props("dense flat")
    elif scoring:
        ui.label("No DFM violations").classes("text-xs text-green-600")

    # Stage 7: Assembly sequence
    assembly_steps = stages.get("assembly_steps", [])
    if assembly_steps:
        with ui.expansion(
            f"Assembly ({len(assembly_steps)} steps)", icon="build"
        ).classes("w-full"):
            for step in assembly_steps:
                with ui.row().classes("w-full items-start gap-2 py-1"):
                    ui.badge(
                        str(step["step"]), color="primary"
                    ).props("dense")
                    with ui.column().classes("gap-0"):
                        action_text = f"{step['action']} {step['component']}"
                        if step.get("attach_to"):
                            action_text += f" -> {step['attach_to']}"
                        ui.label(action_text).classes("text-xs font-medium")
                        if step.get("notes"):
                            ui.label(step["notes"]).classes(
                                "text-[10px] text-gray-500"
                            )


def _pick_metric(fit: dict, decomp_key: str, viewer_key: str):
    """Pick decomp metric if present (even 0.0), else viewer, else None."""
    v = fit.get(decomp_key)
    if v is not None:
        return v
    return fit.get(viewer_key)


def build_debug_report(slot: RunSlot) -> None:
    """Render mesh-vs-parts diagnostics and where the JSON report was saved."""
    report = slot.debug_report
    if not report:
        return

    fit = report.get("fit_metrics", {})
    mesh_source = report.get("mesh_source") or {}
    mesh_decomp = report.get("mesh_decomposition_mm") or {}
    parts_decomp = report.get("parts_decomposition_mm_aabb") or {}
    decomp_debug = report.get("decomposition_debug") or {}
    mfg_debug = report.get("manufacturing_debug") or {}

    with ui.expansion("Diagnostics", icon="analytics").classes("w-full"):
        if slot.debug_report_path:
            ui.label(f"Report: {slot.debug_report_path}").classes(
                "text-[10px] text-gray-500 break-all"
            )

        with ui.row().classes("gap-3 flex-wrap"):
            _metric(
                "Fit H",
                _fmt_ratio(
                    _pick_metric(fit, "decomp_height_ratio_parts_to_mesh",
                                 "viewer_height_ratio_parts_to_mesh")
                ),
            )
            _metric(
                "Fit W",
                _fmt_ratio(
                    _pick_metric(fit, "decomp_width_ratio_parts_to_mesh",
                                 "viewer_width_ratio_parts_to_mesh")
                ),
            )
            _metric(
                "Fit D",
                _fmt_ratio(
                    _pick_metric(fit, "decomp_depth_ratio_parts_to_mesh",
                                 "viewer_depth_ratio_parts_to_mesh")
                ),
            )
            _metric(
                "Parts",
                str(len(slot.result.design.components)) if slot.result else "0",
            )

        if mesh_source:
            ui.label(
                f"Source mesh: {mesh_source.get('vertices', '?')} verts, "
                f"{mesh_source.get('faces', '?')} faces, "
                f"extents { _fmt_extents(mesh_source.get('extents')) }"
            ).classes("text-xs text-gray-700")

        if mesh_decomp.get("extents"):
            ui.label(
                f"Decomp mesh (mm): { _fmt_extents(mesh_decomp.get('extents')) }"
            ).classes("text-xs text-gray-700")

        if parts_decomp.get("extents"):
            ui.label(
                f"Parts AABB (mm): { _fmt_extents(parts_decomp.get('extents')) }"
            ).classes("text-xs text-gray-700")

        if decomp_debug:
            with ui.row().classes("gap-3 flex-wrap"):
                _metric(
                    "Candidates",
                    str(decomp_debug.get("candidate_count",
                        decomp_debug.get("candidate_slab_count", "n/a"))),
                )
                _metric(
                    "Selected",
                    str(decomp_debug.get("selected_slab_count", "n/a")),
                )
                cov = decomp_debug.get("coverage_fraction")
                _metric(
                    "Coverage",
                    f"{cov:.0%}" if cov is not None else "n/a",
                )
                _metric(
                    "Joints",
                    str(decomp_debug.get("joint_count", "n/a")),
                )

            # Source histogram
            histogram = decomp_debug.get("selected_source_histogram")
            if histogram:
                parts = ", ".join(f"{k}: {v}" for k, v in histogram.items())
                ui.label(f"Sources: {parts}").classes("text-xs text-gray-700")

            # Fit ratios from pipeline
            fr = decomp_debug.get("fit_ratios")
            if fr and len(fr) == 3:
                ui.label(
                    f"Fit ratios: X={fr[0]:.3f}, Y={fr[1]:.3f}, Z={fr[2]:.3f}"
                ).classes("text-xs text-gray-700")

            if decomp_debug.get("simplify_mode") is not None:
                before_faces = decomp_debug.get("simplify_before_faces")
                after_faces = decomp_debug.get("simplify_after_faces")
                ui.label(
                    f"Simplify: {decomp_debug.get('simplify_mode')} "
                    f"({before_faces} -> {after_faces} faces)"
                ).classes("text-xs text-gray-700")

        if mfg_debug:
            with ui.row().classes("gap-3 flex-wrap"):
                _metric("Patches", str(mfg_debug.get("patch_count", "n/a")))
                _metric("Segments", str(mfg_debug.get("segment_count", "n/a")))
                _metric("MFG Cand", str(mfg_debug.get("candidate_count", "n/a")))
                _metric("DFM", f"{mfg_debug.get('dfm_errors', 0)}E/{mfg_debug.get('dfm_warnings', 0)}W")

        with ui.expansion("Raw JSON", icon="data_object").classes("w-full"):
            ui.code(
                _json.dumps(report, indent=2),
                language="json",
            ).classes("w-full text-[10px]")


def _profile_to_inline_svg(
    outline_coords: List[List[float]],
    cutout_coords: List[List[List[float]]],
    max_width_px: int = 280,
) -> str:
    """Generate an inline SVG string from 2D profile coordinates."""
    if not outline_coords:
        return ""

    # Compute bounding box
    xs = [c[0] for c in outline_coords]
    ys = [c[1] for c in outline_coords]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    w = max_x - min_x
    h = max_y - min_y

    if w < 0.01 or h < 0.01:
        return ""

    # Scale to fit max_width_px, maintaining aspect ratio
    scale = max_width_px / max(w, h)
    svg_w = w * scale
    svg_h = h * scale
    margin = 4

    def to_svg(x: float, y: float) -> str:
        sx = (x - min_x) * scale + margin
        sy = (y - min_y) * scale + margin
        return f"{sx:.1f},{sy:.1f}"

    # Build outline polygon points
    outline_pts = " ".join(to_svg(c[0], c[1]) for c in outline_coords)

    parts = [
        f'<svg width="{svg_w + 2 * margin:.0f}" '
        f'height="{svg_h + 2 * margin:.0f}" '
        f'xmlns="http://www.w3.org/2000/svg">',
        f'<polygon points="{outline_pts}" '
        f'fill="#dbeafe" stroke="#1e40af" stroke-width="1.5" />',
    ]

    # Draw cutouts
    for cutout in cutout_coords:
        if cutout:
            cutout_pts = " ".join(to_svg(c[0], c[1]) for c in cutout)
            parts.append(
                f'<polygon points="{cutout_pts}" '
                f'fill="white" stroke="#dc2626" stroke-width="1" />'
            )

    parts.append("</svg>")
    return "\n".join(parts)


def _fmt_vec(v: List[float]) -> str:
    """Format a 3D vector for display."""
    return f"({v[0]:.2f}, {v[1]:.2f}, {v[2]:.2f})"


def _fmt_ratio(v) -> str:
    if v is None:
        return "n/a"
    return f"{v:.2f}x"


def _fmt_extents(extents) -> str:
    if not extents or len(extents) != 3:
        return "n/a"
    return f"{extents[0]:.1f}, {extents[1]:.1f}, {extents[2]:.1f}"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _slider(
    label: str,
    value: float,
    min_val: float,
    max_val: float,
    step: float,
    on_change: Callable,
    fmt: str = "",
) -> None:
    def _format(v):
        return f"{v * 100:.0f}%" if fmt == ":.0%" else f"{v}"

    with ui.row().classes("w-full items-center gap-1"):
        ui.label(label).classes("text-xs w-24")
        val_label = ui.label(_format(value)).classes("text-xs w-12 text-right")

        def _on_slide(e, cb=on_change):
            val_label.text = _format(e.value)
            cb(e.value)

        ui.slider(
            min=min_val, max=max_val, step=step, value=value,
            on_change=_on_slide,
        ).classes("flex-grow")


def _metric(label: str, value: str) -> None:
    with ui.column().classes("items-center gap-0"):
        ui.label(value).classes("text-lg font-bold")
        ui.label(label).classes("text-[10px] text-gray-500")


def _set(obj, attr: str, value, callback: Callable) -> None:
    setattr(obj, attr, value)
    callback()


def _set_attr(obj, attr: str, value, callback: Callable) -> None:
    setattr(obj, attr, value)
    callback()


def _copy_config(src_slot: RunSlot, dst_slot: RunSlot, callback: Callable) -> None:
    dst_slot.config = replace(src_slot.config)
    dst_slot.run_simulation = src_slot.run_simulation
    callback()
