"""
Background task wrappers for decomposition and mesh generation.

Uses NiceGUI's run.cpu_bound / run.io_bound to keep the UI responsive
while heavy work runs in a subprocess or thread.
"""

import os
import json as _json
import logging
import tempfile
import time
import traceback
from datetime import datetime, timezone
from concurrent.futures.process import BrokenProcessPool
from typing import Callable

from nicegui import run, ui

from ui.state import AppState, RunSlot

logger = logging.getLogger(__name__)


async def _run_cpu_preferred(callback, *args):
    """Prefer cpu_bound execution, but fall back to io_bound when unavailable."""
    try:
        return await run.cpu_bound(callback, *args)
    except Exception as exc:
        if not _is_process_pool_unavailable_error(exc):
            raise
        logger.warning(
            "cpu_bound unavailable (%s); falling back to io_bound", exc
        )
        return await run.io_bound(callback, *args)


def _is_process_pool_unavailable_error(exc: Exception) -> bool:
    """Return True if an exception indicates process pool execution is unavailable."""
    if isinstance(exc, (PermissionError, BrokenProcessPool)):
        return True
    if isinstance(exc, RuntimeError) and "Process pool not set up" in str(exc):
        return True
    if isinstance(exc, OSError) and "SC_SEM_NSEMS_MAX" in str(exc):
        return True
    return False


def _rotation_matrix_xyz(rx: float, ry: float, rz: float):
    """Build a 3x3 rotation matrix for XYZ Euler angles."""
    import numpy as np

    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)

    rx_m = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]])
    ry_m = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])
    rz_m = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]])
    return rz_m @ ry_m @ rx_m


def _compute_components_aabb(
    components: list[dict],
    scale: float,
    apply_group_rotation: bool,
) -> dict | None:
    """Compute AABB for component boxes in either decomp or viewer space."""
    import math
    import numpy as np

    if not components:
        return None

    group_rotation = _rotation_matrix_xyz(-math.pi / 2.0, 0.0, 0.0)
    all_points = []

    for comp in components:
        w = float(comp["width"]) * scale
        h = float(comp["height"]) * scale
        t = float(comp["thickness"]) * scale
        px, py, pz = [float(v) * scale for v in comp["position"]]
        rx, ry, rz = [float(v) for v in comp["rotation"]]

        local = np.array([
            [-w / 2.0, -h / 2.0, -t / 2.0],
            [-w / 2.0, -h / 2.0,  t / 2.0],
            [-w / 2.0,  h / 2.0, -t / 2.0],
            [-w / 2.0,  h / 2.0,  t / 2.0],
            [ w / 2.0, -h / 2.0, -t / 2.0],
            [ w / 2.0, -h / 2.0,  t / 2.0],
            [ w / 2.0,  h / 2.0, -t / 2.0],
            [ w / 2.0,  h / 2.0,  t / 2.0],
        ])

        rot = _rotation_matrix_xyz(rx, ry, rz)
        points = (local @ rot.T) + np.array([px, py, pz])
        if apply_group_rotation:
            points = points @ group_rotation.T
        all_points.append(points)

    pts = np.vstack(all_points)
    mins = pts.min(axis=0)
    maxs = pts.max(axis=0)
    extents = maxs - mins
    return {
        "min": [float(v) for v in mins],
        "max": [float(v) for v in maxs],
        "extents": [float(v) for v in extents],
    }


def _load_mesh_stats(mesh_path: str | None) -> dict | None:
    """Load mesh stats (counts, bounds, extents) for diagnostics."""
    import numpy as np
    import trimesh

    if not mesh_path:
        return None
    if not os.path.isfile(mesh_path):
        return {"path": mesh_path, "error": "file_not_found"}

    try:
        scene_or_mesh = trimesh.load(mesh_path)
        if isinstance(scene_or_mesh, trimesh.Scene):
            mesh = scene_or_mesh.to_mesh()
        else:
            mesh = scene_or_mesh
        bounds = mesh.bounds
        extents = bounds[1] - bounds[0]
        return {
            "path": mesh_path,
            "vertices": int(len(mesh.vertices)),
            "faces": int(len(mesh.faces)),
            "bounds": [[float(v) for v in bounds[0]], [float(v) for v in bounds[1]]],
            "extents": [float(v) for v in extents],
            "watertight": bool(mesh.is_watertight),
            "surface_area": float(getattr(mesh, "area", 0.0)),
            "volume": float(getattr(mesh, "volume", 0.0)),
        }
    except Exception as exc:
        return {"path": mesh_path, "error": str(exc)}


def _resolve_viewer_mesh_path(state: AppState) -> str | None:
    """Resolve the normalized mesh file path currently displayed in the viewer."""
    if not state.mesh_serving_url or not state.mesh_path:
        return None
    name = state.mesh_serving_url.rsplit("/", 1)[-1]
    uploads_dir = os.path.dirname(state.mesh_path)
    candidate = os.path.join(uploads_dir, name)
    if os.path.isfile(candidate):
        return candidate
    return None


def _safe_ratio(numerator: float, denominator: float) -> float | None:
    if abs(denominator) < 1e-9:
        return None
    return float(numerator / denominator)


def _pick_metric(fit: dict, decomp_key: str, viewer_key: str):
    """Pick decomp metric if present (even 0.0), else viewer, else None."""
    v = fit.get(decomp_key)
    if v is not None:
        return v
    return fit.get(viewer_key)


def _format_fit_metrics(report: dict) -> str:
    fit = report.get("fit_metrics", {})
    h = _pick_metric(fit, "decomp_height_ratio_parts_to_mesh", "viewer_height_ratio_parts_to_mesh")
    w = _pick_metric(fit, "decomp_width_ratio_parts_to_mesh", "viewer_width_ratio_parts_to_mesh")
    d = _pick_metric(fit, "decomp_depth_ratio_parts_to_mesh", "viewer_depth_ratio_parts_to_mesh")

    def _fmt(v):
        return "n/a" if v is None else f"{v:.2f}x"

    return f"Mesh/parts fit ratios H={_fmt(h)}, W={_fmt(w)}, D={_fmt(d)}"


def _to_extents(bounds_2x3: list[list[float]] | None) -> list[float] | None:
    if not bounds_2x3:
        return None
    return [
        float(bounds_2x3[1][0] - bounds_2x3[0][0]),
        float(bounds_2x3[1][1] - bounds_2x3[0][1]),
        float(bounds_2x3[1][2] - bounds_2x3[0][2]),
    ]


def _build_debug_report(
    state: AppState,
    slot: RunSlot,
    slot_idx: int,
    summary: dict,
    output_dir: str,
) -> tuple[dict, str]:
    """Build and persist a JSON report to compare mesh vs generated parts."""
    components = summary.get("components", [])
    source_mesh = _load_mesh_stats(state.mesh_path)
    viewer_mesh_path = _resolve_viewer_mesh_path(state)
    viewer_mesh = _load_mesh_stats(viewer_mesh_path)

    mode = "manufacturing"
    decomp_debug = summary.get("debug", {})
    decomp_bounds = (
        decomp_debug.get("mesh_bounds_mm")
        or summary.get("decomp_bounds")
    )
    decomp_extents = (
        decomp_debug.get("mesh_extents_mm")
        or _to_extents(decomp_bounds)
    )

    parts_decomp_aabb = _compute_components_aabb(
        components, scale=1.0, apply_group_rotation=False,
    )

    viewer_scale = 0.001
    if (
        mode == "manufacturing"
        and viewer_mesh
        and "extents" in viewer_mesh
        and decomp_extents
    ):
        viewer_scale = _safe_ratio(
            float(viewer_mesh["extents"][1]),
            float(decomp_extents[2]),
        ) or 0.001

    parts_viewer_aabb = _compute_components_aabb(
        components, scale=viewer_scale, apply_group_rotation=True,
    )

    fit_metrics = {}
    if decomp_extents and parts_decomp_aabb:
        p_ext = parts_decomp_aabb["extents"]
        fit_metrics["decomp_height_ratio_parts_to_mesh"] = _safe_ratio(
            p_ext[2], decomp_extents[2]
        )
        fit_metrics["decomp_width_ratio_parts_to_mesh"] = _safe_ratio(
            p_ext[0], decomp_extents[0]
        )
        fit_metrics["decomp_depth_ratio_parts_to_mesh"] = _safe_ratio(
            p_ext[1], decomp_extents[1]
        )

    if viewer_mesh and "extents" in viewer_mesh and parts_viewer_aabb:
        v_ext = viewer_mesh["extents"]
        p_ext = parts_viewer_aabb["extents"]
        fit_metrics["viewer_height_ratio_parts_to_mesh"] = _safe_ratio(
            p_ext[1], v_ext[1]
        )
        fit_metrics["viewer_width_ratio_parts_to_mesh"] = _safe_ratio(
            p_ext[0], v_ext[0]
        )
        fit_metrics["viewer_depth_ratio_parts_to_mesh"] = _safe_ratio(
            p_ext[2], v_ext[2]
        )

    manufacturing = None
    if mode == "manufacturing":
        stages = summary.get("stages", {})
        dfm = stages.get("dfm_violations", [])
        decomp_debug = summary.get("debug", {})
        manufacturing = {
            "patch_count": len(stages.get("patches", [])),
            "segment_count": len(stages.get("segments", [])),
            "candidate_count": int(
                decomp_debug.get("candidate_count", len(stages.get("candidates", [])))
            ),
            "assembly_steps": len(stages.get("assembly_steps", [])),
            "dfm_errors": sum(1 for v in dfm if v.get("severity") == "error"),
            "dfm_warnings": sum(1 for v in dfm if v.get("severity") == "warning"),
        }

    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "slot": "A" if slot_idx == 0 else "B",
        "mode": mode,
        "mesh_source": source_mesh,
        "mesh_viewer": viewer_mesh,
        "mesh_decomposition_mm": {
            "bounds": decomp_bounds,
            "extents": decomp_extents,
        },
        "parts_decomposition_mm_aabb": parts_decomp_aabb,
        "parts_viewer_aabb": parts_viewer_aabb,
        "fit_metrics": fit_metrics,
        "decomposition_debug": decomp_debug,
        "manufacturing_debug": manufacturing,
        "run_config": {
            "material": slot.config.default_material,
            "target_height_mm": float(slot.config.target_height_mm),
            "max_slabs": int(slot.config.max_slabs),
            "coverage_target": float(slot.config.coverage_target),
            "selection_objective_mode": slot.config.selection_objective_mode,
            "target_volume_fill": float(slot.config.target_volume_fill),
            "min_volume_contribution": float(slot.config.min_volume_contribution),
            "plane_penalty_weight": float(slot.config.plane_penalty_weight),
            "voxel_resolution_mm": float(slot.config.voxel_resolution_mm),
            "optimize_iterations": int(slot.config.optimize_iterations),
            "run_simulation": bool(slot.run_simulation),
        },
    }

    debug_dir = os.path.join(output_dir, "decomposition", "debug")
    os.makedirs(debug_dir, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(debug_dir, f"run_{report['slot']}_{stamp}.json")
    with open(path, "w") as f:
        _json.dump(report, f, indent=2)

    return report, path


# ---------------------------------------------------------------------------
# Progress file helpers (used by subprocess workers)
# ---------------------------------------------------------------------------

def _write_progress(progress_file: str, step: str, detail: str = "",
                    level: str = "info") -> None:
    """Append a progress line to the shared progress file.

    Each line is a JSON object with timestamp, step name, detail, and level.
    Safe to call from a subprocess — uses append mode with newline flush.
    """
    entry = {
        "t": time.time(),
        "step": step,
        "detail": detail,
        "level": level,
    }
    with open(progress_file, "a") as f:
        f.write(_json.dumps(entry) + "\n")
        f.flush()


class _ProgressLogHandler(logging.Handler):
    """Logging handler that writes to a progress file.

    Attaches to pipeline loggers in the subprocess so all their
    logger.info/warning/error calls show up in the UI progress panel.
    """

    def __init__(self, progress_file: str):
        super().__init__()
        self.progress_file = progress_file

    def emit(self, record):
        try:
            _write_progress(
                self.progress_file,
                step=record.name,
                detail=record.getMessage(),
                level=record.levelname.lower(),
            )
        except Exception:
            pass  # never crash the pipeline for progress logging


_PIPELINE_LOGGER_NAMES = [
    "pipeline", "mesh_decomposer", "manufacturing_decomposer_v2",
    "plane_extraction",
    "slab_candidates", "slab_selector", "slab_joints",
    "scoring", "assembly_sequence", "joint_synthesizer", "dfm_rules",
    "svg_exporter", "dxf_exporter", "simulator",
]


def _attach_progress_logging(progress_file: str) -> _ProgressLogHandler:
    """Attach progress handler to all pipeline-related loggers."""
    handler = _ProgressLogHandler(progress_file)
    handler.setLevel(logging.DEBUG)
    for name in _PIPELINE_LOGGER_NAMES:
        lgr = logging.getLogger(name)
        lgr.addHandler(handler)
        # Subprocess loggers default to WARNING — lower to INFO so
        # pipeline info/warning messages reach our handler.
        if lgr.level == logging.NOTSET or lgr.level > logging.INFO:
            lgr.setLevel(logging.INFO)
    return handler


def _detach_progress_logging(handler: _ProgressLogHandler) -> None:
    """Remove the progress handler from all loggers."""
    for name in _PIPELINE_LOGGER_NAMES:
        logging.getLogger(name).removeHandler(handler)


def create_progress_file() -> str:
    """Create a temporary progress file. Returns the path."""
    fd, path = tempfile.mkstemp(prefix="ttf_progress_", suffix=".jsonl")
    os.close(fd)
    return path


def read_progress(progress_file: str) -> list[dict]:
    """Read all progress entries from a progress file."""
    entries = []
    if not progress_file or not os.path.isfile(progress_file):
        return entries
    try:
        with open(progress_file) as f:
            for line in f:
                line = line.strip()
                if line:
                    entries.append(_json.loads(line))
    except (OSError, _json.JSONDecodeError):
        pass
    return entries


# ---------------------------------------------------------------------------
# Decomposition (CPU-bound, runs in subprocess via run.cpu_bound)
# ---------------------------------------------------------------------------

def _do_v2_pipeline(mesh_path: str, output_dir: str, material: str,
                    target_height_mm: float, max_slabs: int,
                    coverage_target: float, voxel_resolution_mm: float,
                    selection_objective_mode: str, target_volume_fill: float,
                    min_volume_contribution: float, plane_penalty_weight: float,
                    progress_file: str = "") -> dict:
    """Top-level picklable function for v2 manufacturing decomposition.

    Returns a serialisable dict with all intermediate pipeline stage data.
    """
    import sys
    from pathlib import Path

    src_dir = str(Path(__file__).resolve().parent.parent)
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)

    # Set up progress logging if file provided
    handler = None
    if progress_file:
        handler = _attach_progress_logging(progress_file)
        _write_progress(progress_file, "pipeline",
                        "Starting manufacturing v2 pipeline")

    from manufacturing_decomposer_v2 import (
        ManufacturingDecompositionConfigV2,
        decompose_manufacturing_v2,
    )

    mfg_config = ManufacturingDecompositionConfigV2(
        default_material=material,
        target_height_mm=target_height_mm,
    )
    mfg_config.slab_selection.max_slabs = max_slabs
    mfg_config.slab_selection.coverage_target = coverage_target
    mfg_config.slab_selection.voxel_resolution_mm = voxel_resolution_mm
    mfg_config.slab_selection.objective_mode = selection_objective_mode
    mfg_config.slab_selection.target_volume_fill = target_volume_fill
    mfg_config.slab_selection.min_volume_contribution = min_volume_contribution
    mfg_config.slab_selection.plane_penalty_weight = plane_penalty_weight

    if progress_file:
        _write_progress(
            progress_file, "selection_config",
            f"coverage={coverage_target:.0%}, max_slabs={max_slabs}, "
            f"objective={selection_objective_mode}, "
            f"target_volume_fill={target_volume_fill:.0%}, "
            f"min_volume_contribution={min_volume_contribution:.4f}, "
            f"plane_penalty={plane_penalty_weight:.4f}, "
            f"voxel_res={voxel_resolution_mm}mm, "
            f"min_contribution={mfg_config.slab_selection.min_coverage_contribution}, "
            f"dfm_check={mfg_config.slab_selection.dfm_check}",
        )

    try:
        mfg = decompose_manufacturing_v2(mesh_path, mfg_config)
    except Exception:
        if progress_file:
            _write_progress(progress_file, "error",
                            traceback.format_exc(), level="error")
        raise
    finally:
        if handler:
            _detach_progress_logging(handler)

    if progress_file:
        n_parts = len(mfg.design.components)
        n_joints = len(mfg.design.assembly.joints)
        score = mfg.score.overall_score
        _write_progress(
            progress_file, "selection_result",
            f"Selected {n_parts} parts, {n_joints} joints, score={score:.2f}",
        )
        _write_progress(progress_file, "pipeline", "Pipeline complete")

    # Load the decomposition mesh to get its bounds (after auto-scale).
    # This lets the renderer map part positions to the viewer coordinate system.
    from mesh_decomposer import DecompositionConfig as _DC, load_mesh as _load
    _decomp_mesh = _load(mesh_path, _DC(
        default_material=material,
        target_height_mm=target_height_mm,
    ))
    decomp_bounds = _decomp_mesh.bounds.tolist()

    result_design = mfg.design

    # Build picklable components
    components = []
    for c in result_design.components:
        dims = c.get_dimensions()
        components.append({
            "name": c.name,
            "type": c.type.value,
            "width": dims[0],
            "height": dims[1],
            "thickness": c.thickness,
            "material": c.material,
            "position": [float(x) for x in c.position],
            "rotation": [float(x) for x in c.rotation],
        })

    # Serialize patches (v2 has no segmentation, so patches come out empty)
    patches_data = []
    if mfg.segmentation:
        for seg in mfg.segmentation.segments:
            for patch in seg.patches:
                patches_data.append({
                    "area_mm2": float(patch.area_mm2),
                    "normal": [float(x) for x in patch.plane_normal],
                    "centroid_3d": [float(x) for x in patch.centroid_3d],
                    "n_faces": len(patch.face_indices),
                })

    # Serialize segments
    segments_data = []
    if mfg.segmentation:
        for seg in mfg.segmentation.segments:
            bbox_min = [float(x) for x in seg.bounding_box_3d[0]]
            bbox_max = [float(x) for x in seg.bounding_box_3d[1]]
            segments_data.append({
                "role": seg.role.value,
                "n_patches": len(seg.patches),
                "total_area_mm2": float(seg.total_area_mm2),
                "bbox_min": bbox_min,
                "bbox_max": bbox_max,
                "principal_normal": [float(x) for x in seg.principal_normal],
            })

    # Serialize candidates (v2 returns empty candidates list)
    candidates_data = []
    for cand in mfg.candidates:
        n_dfm_errors = sum(1 for v in cand.dfm_violations if v.severity == "error")
        n_dfm_warnings = sum(1 for v in cand.dfm_violations if v.severity == "warning")
        candidates_data.append({
            "rung": cand.rung,
            "n_parts": len(cand.parts),
            "score": float(cand.score),
            "n_dfm_errors": n_dfm_errors,
            "n_dfm_warnings": n_dfm_warnings,
        })

    # Serialize joints
    joints_data = []
    for j in result_design.assembly.joints:
        joints_data.append({
            "component_a": j.component_a,
            "component_b": j.component_b,
            "joint_type": j.joint_type.value,
        })

    # Serialize DFM violations
    dfm_data = []
    for v in mfg.dfm_violations:
        dfm_data.append({
            "rule": v.rule_name,
            "severity": v.severity,
            "message": v.message,
            "value": float(v.value),
            "limit": float(v.limit),
        })

    # Serialize scoring
    s = mfg.score
    scoring_data = {
        "hausdorff_mm": float(s.hausdorff_mm),
        "mean_distance_mm": float(s.mean_distance_mm),
        "part_count": s.part_count,
        "dfm_errors": s.dfm_violations_error,
        "dfm_warnings": s.dfm_violations_warning,
        "structural_plausibility": float(s.structural_plausibility),
        "overall_score": float(s.overall_score),
    }

    # Serialize assembly steps
    assembly_data = []
    for step in mfg.assembly_steps:
        assembly_data.append({
            "step": step.step_number,
            "component": step.component_name,
            "action": step.action,
            "attach_to": step.attach_to,
            "notes": step.notes,
        })

    # Serialize 2D part profiles
    from shapely.geometry import MultiPolygon as _MP
    parts_2d = []
    for name, profile in mfg.parts.items():
        outline = profile.outline
        if isinstance(outline, _MP):
            outline = max(outline.geoms, key=lambda g: g.area)
        outline_coords = [list(c) for c in outline.exterior.coords]
        cutout_coords = [
            [list(c) for c in cutout.exterior.coords]
            for cutout in profile.cutouts
        ]
        bounds = outline.bounds
        parts_2d.append({
            "name": name,
            "outline_coords": outline_coords,
            "cutout_coords": cutout_coords,
            "material": profile.material_key,
            "thickness_mm": float(profile.thickness_mm),
            "width_mm": float(bounds[2] - bounds[0]),
            "height_mm": float(bounds[3] - bounds[1]),
        })

    # Extract debug telemetry from pipeline result
    debug = mfg.design.metadata.get("decomposition_debug", {})

    return {
        "mode": "manufacturing",
        "stages": {
            "patches": patches_data,
            "segments": segments_data,
            "candidates": candidates_data,
            "joints": joints_data,
            "dfm_violations": dfm_data,
            "scoring": scoring_data,
            "assembly_steps": assembly_data,
        },
        "parts_2d": parts_2d,
        "design_name": result_design.name,
        "components": components,
        "joints_count": len(result_design.assembly.joints),
        "coverage": None,
        "simulation": None,
        "svg_paths": [],
        "nested_svg_path": None,
        "mesh_path": mesh_path,
        "decomp_bounds": decomp_bounds,
        "debug": debug,
    }


async def run_decomposition(
    state: AppState,
    slot_idx: int,
    output_dir: str,
    notify: Callable,
) -> None:
    """Run decomposition in a subprocess, update slot when done."""
    slot = state.slots[slot_idx]
    slot.running = True
    slot.error = None
    slot.result = None
    slot.mfg_summary = None
    slot.debug_report = None
    slot.debug_report_path = None
    slot.progress_file = create_progress_file()
    try:
        notify()
    except Exception:
        pass  # Client disconnected

    try:
        summary = await _run_cpu_preferred(
            _do_v2_pipeline,
            state.mesh_path,
            output_dir,
            slot.config.default_material,
            slot.config.target_height_mm,
            slot.config.max_slabs,
            slot.config.coverage_target,
            slot.config.voxel_resolution_mm,
            slot.config.selection_objective_mode,
            slot.config.target_volume_fill,
            slot.config.min_volume_contribution,
            slot.config.plane_penalty_weight,
            slot.progress_file,
        )
        slot.result = _summary_to_result(summary)
        slot.mfg_summary = summary
        try:
            report, report_path = _build_debug_report(
                state, slot, slot_idx, summary, output_dir,
            )
            slot.debug_report = report
            slot.debug_report_path = report_path
            if slot.progress_file:
                _write_progress(
                    slot.progress_file,
                    "diagnostics",
                    f"Debug report saved: {report_path}",
                )
                _write_progress(
                    slot.progress_file,
                    "diagnostics",
                    _format_fit_metrics(report),
                )
        except Exception as dbg_exc:
            logger.exception("Failed to build debug report for slot %d", slot_idx)
            # Surface the error: write to progress file and store a minimal
            # report so the Diagnostics section renders instead of vanishing.
            if slot.progress_file:
                _write_progress(
                    slot.progress_file, "diagnostics",
                    f"Debug report build failed: {dbg_exc}",
                    level="error",
                )
            slot.debug_report = {
                "error": str(dbg_exc),
                "decomposition_debug": summary.get("debug", {}),
            }
    except Exception as exc:
        logger.exception("Decomposition failed for slot %d", slot_idx)
        slot.error = str(exc)
    finally:
        slot.running = False
        try:
            notify()
        except Exception:
            pass  # Client disconnected during run


def _summary_to_result(summary: dict):
    """Convert the picklable summary dict back into a PipelineResult.

    We reconstruct just enough for the UI to render results, tables,
    and SVG previews.
    """
    import numpy as np
    from furniture import (
        Component, ComponentType, AssemblyGraph, FurnitureDesign,
    )
    from simulator import SimulationResult
    from pipeline import PipelineResult

    design = FurnitureDesign(name=summary["design_name"])
    for cd in summary["components"]:
        comp = Component(
            name=cd["name"],
            type=ComponentType(cd["type"]),
            profile=[
                (0.0, 0.0),
                (cd["width"], 0.0),
                (cd["width"], cd["height"]),
                (0.0, cd["height"]),
            ],
            thickness=cd["thickness"],
            position=np.array(cd["position"]),
            rotation=np.array(cd["rotation"]),
            material=cd["material"],
        )
        design.add_component(comp)

    design.metadata["coverage"] = summary.get("coverage")
    if summary.get("debug"):
        design.metadata["decomposition_debug"] = summary.get("debug")

    sim_result = None
    if summary.get("simulation"):
        sd = summary["simulation"]
        sim_result = SimulationResult(
            stable=sd["stable"],
            initial_position=np.zeros(3),
            final_position=np.zeros(3),
            position_change=sd["position_change"],
            initial_rotation=np.array([0, 0, 0, 1.0]),
            final_rotation=np.array([0, 0, 0, 1.0]),
            rotation_change=sd["rotation_change"],
            initial_height=0.0,
            final_height=0.0,
            height_drop=sd["height_drop"],
            simulation_time=sd["simulation_time"],
            wall_time=sd["wall_time"],
            fell_over=sd["fell_over"],
            touched_ground=sd["touched_ground"],
            load_test_passed=sd.get("load_test_passed", True),
            load_final_height=sd.get("load_final_height", 0.0),
        )

    return PipelineResult(
        design=design,
        mesh_path=summary["mesh_path"],
        simulation=sim_result,
        svg_paths=summary.get("svg_paths", []),
        nested_svg_path=summary.get("nested_svg_path"),
    )


# ---------------------------------------------------------------------------
# Mesh Generation (IO-bound, runs in thread via run.io_bound)
# ---------------------------------------------------------------------------

def _do_generate_mesh(
    provider_name: str,
    api_key: str,
    prompt: str,
    output_path: str,
    progress_file: str = "",
) -> str:
    """Top-level function for run.io_bound — generates a mesh via cloud API.

    Returns the path to the saved mesh file.
    """
    import sys
    from pathlib import Path

    src_dir = str(Path(__file__).resolve().parent.parent)
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)

    if progress_file:
        _write_progress(progress_file, "generate",
                        f"Connecting to {provider_name} API...")

    from mesh_provider import ProviderConfig, MeshFormat

    config = ProviderConfig(api_key=api_key)

    if provider_name == "tripo":
        from tripo_provider import TripoProvider
        provider = TripoProvider(config)
        if progress_file:
            _write_progress(progress_file, "generate",
                            "Sending prompt to Tripo3D...")
    elif provider_name == "meshy":
        from meshy_provider import MeshyProvider
        provider = MeshyProvider(config)
        if progress_file:
            _write_progress(progress_file, "generate",
                            "Sending prompt to Meshy...")
    else:
        raise ValueError(f"Unknown provider: {provider_name}")

    try:
        if progress_file:
            _write_progress(progress_file, "generate",
                            "Generating mesh (this may take 1-3 minutes)...")
        result = provider.text_to_mesh(prompt, output_path)
        if progress_file:
            _write_progress(progress_file, "generate",
                            f"Mesh saved: {os.path.basename(result.mesh_path)}")
        return result.mesh_path
    except Exception:
        if progress_file:
            _write_progress(progress_file, "error",
                            traceback.format_exc(), level="error")
        raise


async def run_mesh_generation(
    state: AppState,
    provider_name: str,
    api_key: str,
    prompt: str,
    upload_dir: str,
    notify: Callable,
    progress_file: str = "",
) -> None:
    """Generate a mesh via cloud API, update state when done."""
    output_path = os.path.join(upload_dir, "generated_mesh.glb")

    try:
        mesh_path = await run.io_bound(
            _do_generate_mesh,
            provider_name,
            api_key,
            prompt,
            output_path,
            progress_file,
        )
        state.mesh_path = mesh_path
        state.mesh_filename = os.path.basename(mesh_path)
        state.mesh_serving_url = f"/meshes/{os.path.basename(mesh_path)}"
    except Exception as exc:
        logger.exception("Mesh generation failed")
        ui.notify(f"Mesh generation failed: {exc}", type="negative")
    finally:
        notify()
