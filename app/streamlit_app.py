"""Minimal visual review tool — two tabs for inspecting pipeline outputs."""

from __future__ import annotations

import math
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import streamlit as st
from scipy.spatial.transform import Rotation
from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import triangulate

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from data import (
    artifact_files,
    infer_failure_modes,
    list_runs,
    list_suite_runs,
    read_json,
    read_suite_progress,
    safe_float,
    status_badge,
    suite_rows_by_case,
)
from overlay_viewer import OverlaySceneData, build_overlay_scene

try:
    import plotly.graph_objects as go
except ModuleNotFoundError:
    go = None

DEFAULT_MAX_MESH_FACES = 8000
PREVIEW_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}


# ---------------------------------------------------------------------------
# Helpers: table selection
# ---------------------------------------------------------------------------


def _selected_rows(event: Any) -> List[int]:
    if event is None:
        return []
    if isinstance(event, dict):
        sel = event.get("selection", {})
        rows = sel.get("rows", [])
        return rows if isinstance(rows, list) else []
    sel = getattr(event, "selection", None)
    if sel is None:
        return []
    if isinstance(sel, dict):
        rows = sel.get("rows", [])
        return rows if isinstance(rows, list) else []
    rows = getattr(sel, "rows", None)
    return rows if isinstance(rows, list) else []


# ---------------------------------------------------------------------------
# 3D Overlay rendering (Plotly primary, matplotlib fallback)
# ---------------------------------------------------------------------------


@st.cache_data(show_spinner=False)
def _cached_scene(run_dir: str, space: str, max_faces: int) -> OverlaySceneData:
    return build_overlay_scene(run_dir, space=space, max_mesh_faces=max_faces)


def _hex_to_rgba(hex_color: str, alpha: float) -> tuple:
    c = hex_color.lstrip("#")
    if len(c) != 6:
        return (0.2, 0.6, 0.9, alpha)
    return (int(c[0:2], 16) / 255, int(c[2:4], 16) / 255, int(c[4:6], 16) / 255, alpha)


class _OverlapPatch:
    __slots__ = ("part_id", "fill_vertices", "fill_faces", "edge_loops")

    def __init__(
        self,
        part_id: str,
        fill_vertices: np.ndarray,
        fill_faces: np.ndarray,
        edge_loops: List[np.ndarray],
    ) -> None:
        self.part_id = part_id
        self.fill_vertices = fill_vertices
        self.fill_faces = fill_faces
        self.edge_loops = edge_loops


@st.cache_data(show_spinner=False)
def _part_payload_map(payload_path: str) -> Dict[str, Dict[str, Any]]:
    payload = read_json(Path(payload_path))
    parts = payload.get("parts", [])
    out: Dict[str, Dict[str, Any]] = {}
    if not isinstance(parts, list):
        return out
    for idx, part in enumerate(parts):
        if not isinstance(part, dict):
            continue
        part_id = str(part.get("part_id", f"part_{idx:02d}")).strip()
        if not part_id:
            continue
        out[part_id] = part
    return out


def _rotation_matrix_xyz(rotation_xyz: List[float]) -> np.ndarray:
    angles = np.asarray(rotation_xyz, dtype=float)
    return Rotation.from_euler("xyz", angles).as_matrix()


def _outline_polygon(outline_2d: Any) -> Optional[Polygon]:
    if not isinstance(outline_2d, list) or len(outline_2d) < 3:
        return None
    shell = np.asarray(outline_2d, dtype=float)
    if shell.shape[0] < 3:
        return None
    if np.allclose(shell[0], shell[-1]):
        shell = shell[:-1]
    if shell.shape[0] < 3:
        return None
    poly = Polygon(shell.tolist())
    if poly.is_empty:
        return None
    poly = poly if poly.is_valid else poly.buffer(0)
    if poly.is_empty:
        return None
    if isinstance(poly, MultiPolygon):
        poly = max(poly.geoms, key=lambda g: g.area)
    if not isinstance(poly, Polygon) or poly.area <= 1e-6:
        return None
    return poly


def _geometry_to_polygons(geom: Any) -> List[Polygon]:
    if geom is None or geom.is_empty:
        return []
    clean = geom if geom.is_valid else geom.buffer(0)
    if clean.is_empty:
        return []
    if isinstance(clean, Polygon):
        return [clean] if clean.area > 1e-6 else []
    if isinstance(clean, MultiPolygon):
        return [g for g in clean.geoms if g.area > 1e-6]
    if hasattr(clean, "geoms"):
        polys = [g for g in clean.geoms if isinstance(g, Polygon) and g.area > 1e-6]
        return polys
    return []


def _clip_polygon_halfplane(
    poly: Polygon, a: float, b: float, c_rhs: float
) -> List[Polygon]:
    ab_norm = math.hypot(a, b)
    if ab_norm < 1e-9:
        return []

    line_normal = np.array([a, b], dtype=float) / ab_norm
    line_dir = np.array([-line_normal[1], line_normal[0]], dtype=float)

    if abs(a) > abs(b):
        line_pt = np.array([c_rhs / a, 0.0], dtype=float)
    else:
        line_pt = np.array([0.0, c_rhs / b], dtype=float)

    bounds = poly.bounds
    extent = max(bounds[2] - bounds[0], bounds[3] - bounds[1], 100.0) * 4.0
    p1 = line_pt - extent * line_dir
    p2 = line_pt + extent * line_dir
    p3 = p2 + extent * line_normal
    p4 = p1 + extent * line_normal
    half_plane = Polygon([tuple(p1), tuple(p2), tuple(p3), tuple(p4)])

    try:
        clipped = poly.intersection(half_plane)
    except Exception:
        return []
    return _geometry_to_polygons(clipped)


def _clip_polygon_strip(
    poly: Polygon, a: float, b: float, c0: float, lo: float, hi: float
) -> List[Polygon]:
    if hi <= lo:
        return []

    # lo <= a*u + b*v + c0 <= hi
    # 1) a*u + b*v >= lo - c0
    # 2) -a*u - b*v >= c0 - hi
    first = _clip_polygon_halfplane(poly, a, b, lo - c0)
    if not first:
        return []
    out: List[Polygon] = []
    for p in first:
        out.extend(_clip_polygon_halfplane(p, -a, -b, c0 - hi))
    return out


def _triangulate_polygon(
    poly: Polygon,
) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
    vertices: List[List[float]] = []
    faces: List[List[int]] = []
    for tri in triangulate(poly):
        if tri.is_empty or tri.area <= 1e-9:
            continue
        if not poly.covers(tri.representative_point()):
            continue
        coords = np.asarray(list(tri.exterior.coords)[:3], dtype=float)
        base = len(vertices)
        vertices.extend(coords.tolist())
        faces.append([base, base + 1, base + 2])
    if not vertices or not faces:
        return np.zeros((0, 2), dtype=float), np.zeros((0, 3), dtype=int), []
    loops = [np.asarray(poly.exterior.coords[:-1], dtype=float)]
    loops.extend(np.asarray(ring.coords[:-1], dtype=float) for ring in poly.interiors)
    return np.asarray(vertices, dtype=float), np.asarray(faces, dtype=int), loops


def _extrude_patch(
    verts_2d: np.ndarray,
    faces_2d: np.ndarray,
    edge_loops_2d: List[np.ndarray],
    depth_mm: float,
) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
    n_fill = len(verts_2d)
    if n_fill == 0 or len(faces_2d) == 0:
        return np.zeros((0, 3), dtype=float), np.zeros((0, 3), dtype=int), []

    outer_fill = np.column_stack([verts_2d, np.zeros(n_fill)])
    inner_fill = np.column_stack([verts_2d, np.full(n_fill, -depth_mm)])
    all_verts: List[np.ndarray] = [outer_fill, inner_fill]
    all_faces: List[np.ndarray] = [faces_2d.copy(), faces_2d[:, ::-1] + n_fill]

    base_idx = 2 * n_fill
    loops_out: List[np.ndarray] = []
    for loop in edge_loops_2d:
        n_loop = len(loop)
        if n_loop < 2:
            continue
        outer_edge = np.column_stack([loop, np.zeros(n_loop)])
        inner_edge = np.column_stack([loop, np.full(n_loop, -depth_mm)])
        all_verts.extend([outer_edge, inner_edge])
        loops_out.extend([outer_edge, inner_edge])

        side_tris: List[List[int]] = []
        for k in range(n_loop):
            k_next = (k + 1) % n_loop
            o0 = base_idx + k
            o1 = base_idx + k_next
            i0 = base_idx + n_loop + k
            i1 = base_idx + n_loop + k_next
            side_tris.append([o0, o1, i0])
            side_tris.append([i0, o1, i1])
        all_faces.append(np.asarray(side_tris, dtype=int))
        base_idx += 2 * n_loop

    return (
        np.vstack(all_verts) if all_verts else np.zeros((0, 3), dtype=float),
        np.vstack(all_faces) if all_faces else np.zeros((0, 3), dtype=int),
        loops_out,
    )


def _transform_points(
    local_points: np.ndarray, rot: np.ndarray, origin: np.ndarray
) -> np.ndarray:
    if len(local_points) == 0:
        return np.zeros((0, 3), dtype=float)
    return (rot @ local_points.T).T + origin


def _to_scene_space(points: np.ndarray, scene: OverlaySceneData) -> np.ndarray:
    if scene.space == "normalized":
        return points
    scale = scene.normalization.scale
    if abs(scale) < 1e-12:
        scale = 1.0
    return points / scale + scene.normalization.anchor_raw


def _build_directional_overlap_patches(
    src_part_id: str,
    src: Dict[str, Any],
    dst: Dict[str, Any],
    overlap_depth_mm: float,
    scene: OverlaySceneData,
) -> List[_OverlapPatch]:
    src_poly = _outline_polygon(src.get("outline_2d"))
    if src_poly is None:
        return []

    src_rot = _rotation_matrix_xyz(src.get("rotation_3d") or [0.0, 0.0, 0.0])
    dst_rot = _rotation_matrix_xyz(dst.get("rotation_3d") or [0.0, 0.0, 0.0])
    src_origin = np.asarray(
        src.get("origin_3d") or src.get("position_3d") or [0.0, 0.0, 0.0], dtype=float
    )
    dst_origin = np.asarray(
        dst.get("origin_3d") or dst.get("position_3d") or [0.0, 0.0, 0.0], dtype=float
    )
    src_thickness = float(src.get("thickness_mm", 0.0) or 0.0)
    dst_thickness = float(dst.get("thickness_mm", 0.0) or 0.0)
    if src_thickness <= 1e-6 or dst_thickness <= 1e-6:
        return []

    u_src = src_rot[:, 0]
    v_src = src_rot[:, 1]
    n_dst = dst_rot[:, 2]

    a = float(np.dot(n_dst, u_src))
    b = float(np.dot(n_dst, v_src))
    if math.hypot(a, b) < 1e-9:
        return []

    d_surface = float(np.dot(n_dst, dst_origin))
    c0 = float(np.dot(n_dst, src_origin) - d_surface)
    tol = 0.5
    strip_polys = _clip_polygon_strip(
        src_poly, a, b, c0, lo=-dst_thickness + tol, hi=-tol
    )
    if not strip_polys:
        return []

    depth_mm = min(max(0.0, overlap_depth_mm), src_thickness)
    if depth_mm <= 0.01:
        return []

    out: List[_OverlapPatch] = []
    for poly in strip_polys:
        verts_2d, faces_2d, loops_2d = _triangulate_polygon(poly)
        if len(verts_2d) == 0 or len(faces_2d) == 0:
            continue
        local_verts, local_faces, local_loops = _extrude_patch(
            verts_2d, faces_2d, loops_2d, depth_mm
        )
        world_verts = _transform_points(local_verts, src_rot, src_origin)
        world_loops = [
            _transform_points(loop, src_rot, src_origin) for loop in local_loops
        ]
        world_verts = _to_scene_space(world_verts, scene)
        world_loops = [_to_scene_space(loop, scene) for loop in world_loops]
        out.append(
            _OverlapPatch(
                part_id=src_part_id,
                fill_vertices=world_verts,
                fill_faces=local_faces,
                edge_loops=world_loops,
            )
        )
    return out


def _build_overlap_patches(
    part_payloads: Dict[str, Dict[str, Any]],
    overlap_details: Any,
    scene: OverlaySceneData,
) -> List[_OverlapPatch]:
    if not isinstance(overlap_details, list):
        return []

    patches: List[_OverlapPatch] = []
    for item in overlap_details:
        if not isinstance(item, dict):
            continue
        part_a = str(item.get("part_a", "")).strip()
        part_b = str(item.get("part_b", "")).strip()
        overlap_mm = float(item.get("overlap_mm", 0.0) or 0.0)
        if not part_a or not part_b or overlap_mm <= 0.01:
            continue
        pa = part_payloads.get(part_a)
        pb = part_payloads.get(part_b)
        if not pa or not pb:
            continue
        patches.extend(
            _build_directional_overlap_patches(part_a, pa, pb, overlap_mm, scene)
        )
        patches.extend(
            _build_directional_overlap_patches(part_b, pb, pa, overlap_mm, scene)
        )
    return patches


def _plotly_figure(
    scene: OverlaySceneData,
    selected: Set[str],
    mesh_opacity: float,
    part_opacity: float,
    show_labels: bool,
    overlap_patches: Optional[List[_OverlapPatch]] = None,
) -> Any:
    fig = go.Figure()
    overlap_patches = overlap_patches or []
    v, f = scene.mesh_vertices, scene.mesh_faces
    if len(v) and len(f):
        fig.add_trace(
            go.Mesh3d(
                x=v[:, 0],
                y=v[:, 1],
                z=v[:, 2],
                i=f[:, 0],
                j=f[:, 1],
                k=f[:, 2],
                color="#A8A8A8",
                opacity=mesh_opacity,
                name="source_mesh",
                hoverinfo="skip",
                showscale=False,
            )
        )
    for part in scene.parts:
        if part.part_id not in selected:
            continue
        pv, pf = part.fill_vertices, part.fill_faces
        if len(pv) and len(pf):
            fig.add_trace(
                go.Mesh3d(
                    x=pv[:, 0],
                    y=pv[:, 1],
                    z=pv[:, 2],
                    i=pf[:, 0],
                    j=pf[:, 1],
                    k=pf[:, 2],
                    color=part.color,
                    opacity=part_opacity,
                    name=part.part_id,
                    legendgroup=part.part_id,
                    flatshading=True,
                    showscale=False,
                )
            )
        for loop in part.edge_loops:
            if not len(loop):
                continue
            closed = loop if (loop[0] == loop[-1]).all() else np.vstack([loop, loop[0]])
            fig.add_trace(
                go.Scatter3d(
                    x=closed[:, 0],
                    y=closed[:, 1],
                    z=closed[:, 2],
                    mode="lines",
                    line={"color": part.color, "width": 5},
                    name=f"{part.part_id}_outline",
                    legendgroup=part.part_id,
                    showlegend=False,
                    hoverinfo="skip",
                )
            )
        if show_labels:
            fig.add_trace(
                go.Scatter3d(
                    x=[part.centroid[0]],
                    y=[part.centroid[1]],
                    z=[part.centroid[2]],
                    mode="text",
                    text=[part.part_id],
                    textposition="middle center",
                    name=f"{part.part_id}_label",
                    legendgroup=part.part_id,
                    showlegend=False,
                    hoverinfo="skip",
                )
            )
    for idx, patch in enumerate(overlap_patches):
        if patch.part_id not in selected:
            continue
        if len(patch.fill_vertices) and len(patch.fill_faces):
            ov = patch.fill_vertices
            of = patch.fill_faces
            fig.add_trace(
                go.Mesh3d(
                    x=ov[:, 0],
                    y=ov[:, 1],
                    z=ov[:, 2],
                    i=of[:, 0],
                    j=of[:, 1],
                    k=of[:, 2],
                    color="#FF0000",
                    opacity=0.80,
                    name=f"overlap_{idx}",
                    showlegend=False,
                    hoverinfo="skip",
                    flatshading=True,
                )
            )
        for loop in patch.edge_loops:
            if not len(loop):
                continue
            closed = loop if (loop[0] == loop[-1]).all() else np.vstack([loop, loop[0]])
            fig.add_trace(
                go.Scatter3d(
                    x=closed[:, 0],
                    y=closed[:, 1],
                    z=closed[:, 2],
                    mode="lines",
                    line={"color": "#FF0000", "width": 8},
                    name=f"overlap_edge_{idx}",
                    showlegend=False,
                    hoverinfo="skip",
                )
            )
    fig.update_layout(
        scene={"aspectmode": "data", "uirevision": "keep"},
        margin={"l": 10, "r": 10, "t": 30, "b": 10},
        legend={"orientation": "h"},
        uirevision="keep",
    )
    return fig


def _matplotlib_figure(
    scene: OverlaySceneData,
    selected: Set[str],
    mesh_opacity: float,
    part_opacity: float,
    show_labels: bool,
    overlap_patches: Optional[List[_OverlapPatch]] = None,
) -> Any:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    overlap_patches = overlap_patches or []
    fig = plt.figure(figsize=(9.5, 7.0))
    ax = fig.add_subplot(111, projection="3d")
    bounds: List[np.ndarray] = []
    v, f = scene.mesh_vertices, scene.mesh_faces
    if len(v) and len(f):
        ax.add_collection3d(
            Poly3DCollection(
                v[f],
                facecolors=(0.65, 0.65, 0.65, mesh_opacity),
                edgecolors=(0.65, 0.65, 0.65, mesh_opacity * 0.2),
                linewidths=0.05,
            )
        )
        bounds.append(v)
    for part in scene.parts:
        if part.part_id not in selected:
            continue
        if len(part.fill_vertices) and len(part.fill_faces):
            ax.add_collection3d(
                Poly3DCollection(
                    part.fill_vertices[part.fill_faces],
                    facecolors=_hex_to_rgba(part.color, part_opacity),
                    edgecolors=_hex_to_rgba(part.color, min(1.0, part_opacity + 0.15)),
                    linewidths=0.1,
                )
            )
            bounds.append(part.fill_vertices)
        for loop in part.edge_loops:
            if not len(loop):
                continue
            closed = loop if (loop[0] == loop[-1]).all() else np.vstack([loop, loop[0]])
            ax.plot(
                closed[:, 0],
                closed[:, 1],
                closed[:, 2],
                color=part.color,
                linewidth=1.3,
            )
        if show_labels:
            ax.text(
                part.centroid[0],
                part.centroid[1],
                part.centroid[2],
                part.part_id,
                color=part.color,
                fontsize=8,
            )
    for patch in overlap_patches:
        if patch.part_id not in selected:
            continue
        if len(patch.fill_vertices) and len(patch.fill_faces):
            ax.add_collection3d(
                Poly3DCollection(
                    patch.fill_vertices[patch.fill_faces],
                    facecolors=(1.0, 0.0, 0.0, 0.8),
                    edgecolors=(1.0, 0.0, 0.0, 0.95),
                    linewidths=0.2,
                )
            )
            bounds.append(patch.fill_vertices)
        for loop in patch.edge_loops:
            if not len(loop):
                continue
            closed = loop if (loop[0] == loop[-1]).all() else np.vstack([loop, loop[0]])
            ax.plot(
                closed[:, 0],
                closed[:, 1],
                closed[:, 2],
                color="#FF0000",
                linewidth=2.5,
            )
    if bounds:
        cloud = np.vstack(bounds)
        mins, maxs = cloud.min(axis=0), cloud.max(axis=0)
        center = (mins + maxs) * 0.5
        radius = max(float(np.max(maxs - mins)) * 0.55, 1.0)
        ax.set_xlim(center[0] - radius, center[0] + radius)
        ax.set_ylim(center[1] - radius, center[1] + radius)
        ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_zlabel("Z (mm)")
    ax.view_init(elev=22, azim=38)
    fig.tight_layout()
    return fig


def _render_overlay(run_dir: str, metrics: Dict[str, Any], key: str) -> None:
    """Show 3D overlay with display controls."""
    with st.expander("Display settings", expanded=False):
        c1, c2, c3, c4, c5 = st.columns(5)
        space_label = c1.selectbox(
            "Space", ["Original", "Normalized"], index=0, key=f"{key}-sp"
        )
        max_faces = c2.slider(
            "Face cap", 2000, 40000, DEFAULT_MAX_MESH_FACES, 1000, key=f"{key}-fc"
        )
        mesh_op = c3.slider("Mesh opacity", 0.05, 1.0, 0.30, 0.05, key=f"{key}-mo")
        part_op = c4.slider("Part opacity", 0.05, 1.0, 0.55, 0.05, key=f"{key}-po")
        labels = c5.checkbox("Labels", True, key=f"{key}-lb")

    space = "original" if space_label == "Original" else "normalized"
    try:
        scene = _cached_scene(run_dir, space, max_faces)
    except Exception as exc:
        st.warning(f"Overlay unavailable: {exc}")
        return
    if not scene.parts:
        st.warning("No part geometry for overlay.")
        return

    part_ids = [p.part_id for p in scene.parts]
    visible = st.multiselect(
        "Visible parts", part_ids, default=part_ids, key=f"{key}-vp"
    )
    debug = metrics.get("debug", {}) if isinstance(metrics, dict) else {}
    overlap_details = (
        debug.get("plane_overlap_details", []) if isinstance(debug, dict) else []
    )
    design_payload_path = Path(run_dir) / "artifacts" / "design_first_principles.json"
    part_payloads = (
        _part_payload_map(str(design_payload_path))
        if design_payload_path.exists()
        else {}
    )
    overlap_patches = _build_overlap_patches(part_payloads, overlap_details, scene)
    visible_set = set(visible)
    visible_overlap_regions = sum(
        1 for patch in overlap_patches if patch.part_id in visible_set
    )

    if go is not None:
        fig = _plotly_figure(
            scene,
            visible_set,
            mesh_op,
            part_op,
            labels,
            overlap_patches=overlap_patches,
        )
        st.plotly_chart(fig, use_container_width=True, key=f"{key}-fig")
    else:
        fig = _matplotlib_figure(
            scene,
            visible_set,
            mesh_op,
            part_op,
            labels,
            overlap_patches=overlap_patches,
        )
        st.info("Plotly unavailable; showing static matplotlib 3D overlay.")
        st.pyplot(fig, clear_figure=True, use_container_width=True)
    st.caption(
        f"Run: {scene.run_id} | Space: {scene.space} | "
        f"Mesh faces: {len(scene.mesh_faces)} | Parts: {len(scene.parts)}"
    )
    if overlap_patches:
        st.caption(
            f"Overlap highlight: {visible_overlap_regions} visible overlap region(s) in red."
        )
    if scene.warnings:
        with st.expander("Overlay warnings"):
            for w in scene.warnings:
                st.write(f"- {w}")


# ---------------------------------------------------------------------------
# Phase step-through debugger
# ---------------------------------------------------------------------------


@st.cache_data(show_spinner=False)
def _cached_snapshot_scene(
    run_dir: str, snapshot_path: str, space: str, max_faces: int
) -> OverlaySceneData:
    return build_overlay_scene(
        run_dir,
        space=space,
        max_mesh_faces=max_faces,
        design_json_override=snapshot_path,
    )


def _render_phase_stepper(run_dir: str, key: str) -> None:
    """Show a slider to step through pipeline phase snapshots."""
    snapshots_dir = Path(run_dir) / "artifacts" / "snapshots"
    if not snapshots_dir.is_dir():
        return

    snapshot_files = sorted(snapshots_dir.glob("phase_*.json"))
    if not snapshot_files:
        return

    # Load metadata for slider labels
    snapshot_meta = []
    for sf in snapshot_files:
        try:
            meta = read_json(sf)
            label = meta.get("phase_label", sf.stem)
            count = meta.get("part_count", "?")
            diagnostics = meta.get("diagnostics", {})
            overlap = (
                diagnostics.get("plane_overlap", {})
                if isinstance(diagnostics, dict)
                else {}
            )
            overlap_pairs = (
                int(overlap.get("plane_overlap_pairs", 0) or 0)
                if isinstance(overlap, dict)
                else 0
            )
            overlap_details = (
                overlap.get("plane_overlap_details", [])
                if isinstance(overlap, dict)
                else []
            )
            snapshot_meta.append(
                (str(sf), label, count, overlap_pairs, overlap_details)
            )
        except Exception:
            snapshot_meta.append((str(sf), sf.stem, "?", 0, []))

    if len(snapshot_meta) < 2:
        return

    with st.expander("Phase Step-Through", expanded=False):
        labels = [
            f"{i}: {label} ({count} parts)"
            for i, (_, label, count, _, _) in enumerate(snapshot_meta)
        ]
        selected_label = st.select_slider(
            "Pipeline phase",
            options=labels,
            value=labels[-1],
            key=f"{key}-phase-sl",
        )
        selected_idx = labels.index(selected_label)
        snapshot_path, _, _, overlap_pairs, overlap_details = snapshot_meta[
            selected_idx
        ]

        with st.container():
            c1, c2, c3 = st.columns(3)
            space_label = c1.selectbox(
                "Space", ["Original", "Normalized"], index=0, key=f"{key}-ps-sp"
            )
            max_faces = c2.slider(
                "Face cap",
                2000,
                40000,
                DEFAULT_MAX_MESH_FACES,
                1000,
                key=f"{key}-ps-fc",
            )
            part_op = c3.slider(
                "Part opacity", 0.05, 1.0, 0.55, 0.05, key=f"{key}-ps-po"
            )

        space = "original" if space_label == "Original" else "normalized"
        try:
            scene = _cached_snapshot_scene(run_dir, snapshot_path, space, max_faces)
        except Exception as exc:
            st.warning(f"Phase overlay unavailable: {exc}")
            return

        if not scene.parts:
            st.info("No parts in this phase.")
            return

        all_ids = {p.part_id for p in scene.parts}
        part_payloads = _part_payload_map(snapshot_path)
        overlap_patches = _build_overlap_patches(part_payloads, overlap_details, scene)
        if go is not None:
            fig = _plotly_figure(
                scene,
                all_ids,
                0.30,
                part_op,
                True,
                overlap_patches=overlap_patches,
            )
            st.plotly_chart(fig, use_container_width=True, key=f"{key}-ps-fig")
        else:
            fig = _matplotlib_figure(
                scene,
                all_ids,
                0.30,
                part_op,
                True,
                overlap_patches=overlap_patches,
            )
            st.pyplot(fig, clear_figure=True, use_container_width=True)

        st.caption(
            f"Phase: {snapshot_meta[selected_idx][1]} | Parts: {len(scene.parts)} | "
            f"Overlap pairs: {overlap_pairs} | Overlap regions: {len(overlap_patches)}"
        )


# ---------------------------------------------------------------------------
# Artifact / SVG gallery
# ---------------------------------------------------------------------------


def _render_artifacts(run_dir: str, key: str) -> None:
    files = artifact_files(run_dir)
    if not files:
        st.caption("No artifacts.")
        return
    previewable = [
        p
        for p in files
        if p.suffix.lower() in PREVIEW_IMAGE_EXTENSIONS or p.suffix.lower() == ".svg"
    ]
    if previewable:
        for path in previewable[:12]:
            rel = path.relative_to(Path(run_dir))
            if path.suffix.lower() == ".svg":
                with st.expander(f"SVG: {rel}", expanded=False):
                    try:
                        st.components.v1.html(
                            path.read_text(encoding="utf-8"), height=460, scrolling=True
                        )
                    except OSError as exc:
                        st.warning(f"Could not load {rel}: {exc}")
            else:
                st.image(str(path), caption=str(rel), use_container_width=True)
    for path in files:
        rel = path.relative_to(Path(run_dir))
        with path.open("rb") as f:
            st.download_button(
                f"Download {rel}",
                data=f.read(),
                file_name=path.name,
                key=f"{key}-dl-{rel}",
            )


# ---------------------------------------------------------------------------
# Violations table
# ---------------------------------------------------------------------------


def _render_violations(metrics: Dict[str, Any]) -> None:
    violations = metrics.get("violations", [])
    if not violations:
        return
    st.subheader("Violations")
    st.dataframe(violations, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Metrics row
# ---------------------------------------------------------------------------


def _render_metrics_row(metrics: Dict[str, Any]) -> None:
    quality = metrics.get("quality_metrics", {})
    counts = metrics.get("counts", {})
    debug = metrics.get("debug", {})
    c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
    c1.metric("Status", (metrics.get("status") or "n/a").upper())
    c2.metric("Score", f"{quality.get('overall_score', 0):.3f}")
    c3.metric("Parts", str(quality.get("part_count", counts.get("components", 0))))
    c4.metric(
        "Errors",
        str(
            sum(
                1 for v in metrics.get("violations", []) if v.get("severity") == "error"
            )
        ),
    )
    c5.metric("Hausdorff", f"{quality.get('hausdorff_mm', 0):.1f} mm")
    c6.metric("Normal err", f"{quality.get('normal_error_deg', 0):.1f} deg")
    coverage = debug.get("coverage_ratio_unique_faces")
    c7.metric("Coverage", f"{coverage:.2f}" if coverage is not None else "n/a")


# ---------------------------------------------------------------------------
# Tab 1: Suite Review
# ---------------------------------------------------------------------------


def _thumbnail_grid(rows: List[Dict[str, Any]], cols: int = 3) -> None:
    """Show overlay screenshots in a grid if available."""
    thumbs = []
    for row in rows:
        run_dir = str(row.get("run_dir", "")).strip()
        if not run_dir:
            continue
        screenshot = Path(run_dir) / "artifacts" / "overlay_screenshot.png"
        if screenshot.exists():
            thumbs.append((row.get("case_id", ""), str(screenshot)))
    if not thumbs:
        return
    st.subheader("Overlay Thumbnails")
    col_objs = st.columns(cols)
    for i, (case_id, img_path) in enumerate(thumbs):
        col_objs[i % cols].image(img_path, caption=case_id, use_container_width=True)


def _suite_compare(suites: List[Dict[str, Any]], suites_dir: str) -> None:
    if len(suites) < 2:
        return
    st.subheader("Suite Comparison")
    suite_ids = [str(s["suite_run_id"]) for s in suites]
    c1, c2 = st.columns(2)
    cand_id = c1.selectbox("Candidate", suite_ids, index=0, key="cmp-cand")
    base_id = c2.selectbox(
        "Baseline", suite_ids, index=min(1, len(suite_ids) - 1), key="cmp-base"
    )
    if cand_id == base_id:
        st.caption("Select different suites to compare.")
        return

    regressions_only = st.checkbox("Show regressions only", value=False, key="cmp-reg")
    cand_rows = read_json(Path(suites_dir) / cand_id / "results.json").get("rows", [])
    base_rows = read_json(Path(suites_dir) / base_id / "results.json").get("rows", [])
    cand_map = suite_rows_by_case(cand_rows if isinstance(cand_rows, list) else [])
    base_map = suite_rows_by_case(base_rows if isinstance(base_rows, list) else [])

    deltas: List[Dict[str, Any]] = []
    for case_id, cand in cand_map.items():
        base = base_map.get(case_id)
        if not base:
            continue
        cs = safe_float(cand.get("overall_score"))
        bs = safe_float(base.get("overall_score"))
        ce = int(cand.get("errors", 0) or 0)
        be = int(base.get("errors", 0) or 0)
        score_delta = round(cs - bs, 4) if cs is not None and bs is not None else None
        if regressions_only and score_delta is not None and score_delta >= 0:
            continue
        deltas.append(
            {
                "case_id": case_id,
                "cand_status": cand.get("status"),
                "base_status": base.get("status"),
                "score_delta": score_delta,
                "error_delta": ce - be,
            }
        )
    if not deltas:
        st.caption("No overlapping cases (or no regressions).")
        return
    st.dataframe(
        sorted(
            deltas,
            key=lambda r: (
                1e9 if r["score_delta"] is None else r["score_delta"],
                -r["error_delta"],
            ),
        ),
        use_container_width=True,
        hide_index=True,
    )


def _render_launch_panel(suites_dir: str) -> None:
    """Collapsible panel for launching a suite run and showing progress."""
    with st.expander("Launch Suite", expanded="suite_process" in st.session_state):
        suite_file = st.text_input(
            "Suite file", value="benchmarks/mesh_suite.json", key="launch-suite-file"
        )
        if st.button(
            "Launch", key="launch-btn", disabled="suite_process" in st.session_state
        ):
            suite_path = Path(suite_file)
            if not suite_path.is_absolute():
                suite_path = ROOT / suite_path
            if not suite_path.is_file():
                st.error(f"Suite file not found: {suite_path}")
            else:
                proc = subprocess.Popen(
                    [
                        str(ROOT / "venv" / "bin" / "python3"),
                        str(ROOT / "scripts" / "run_mesh_suite.py"),
                        "--suite-file",
                        str(suite_path),
                        "--suites-dir",
                        suites_dir,
                    ],
                    cwd=str(ROOT),
                )
                st.session_state["suite_process"] = proc
                # Wait briefly for the suite dir to appear (manifest written before loop)
                import time

                time.sleep(0.5)
                # Find the newest suite dir (the one we just launched)
                sd = Path(suites_dir)
                if sd.exists():
                    dirs = sorted(
                        [d for d in sd.iterdir() if d.is_dir() and d.name != "latest"],
                        key=lambda d: d.stat().st_mtime,
                        reverse=True,
                    )
                    if dirs:
                        st.session_state["suite_dir"] = str(dirs[0])
                st.rerun()

        if "suite_process" in st.session_state:
            _render_suite_progress()


@st.fragment(run_every="3s")
def _render_suite_progress() -> None:
    """Polling fragment that shows progress while a suite is running."""
    proc: subprocess.Popen = st.session_state.get("suite_process")
    suite_dir: str = st.session_state.get("suite_dir", "")
    if proc is None:
        return

    finished = proc.poll() is not None

    if suite_dir:
        progress = read_suite_progress(suite_dir)
        total = progress["total"]
        completed = progress["completed"]
        rows = progress["rows"]
    else:
        total, completed, rows = 0, 0, []

    if total > 0:
        st.progress(completed / total)
        if finished:
            if proc.returncode == 0:
                st.success(f"Suite complete: {completed}/{total} cases.")
            else:
                st.error(f"Suite finished with errors (exit code {proc.returncode}).")
        else:
            st.caption(f"Running case {completed + 1} of {total}...")
    elif finished:
        st.warning("Suite process exited before writing results.")

    if rows:
        mini = [
            {
                "case_id": r.get("case_id"),
                "status": status_badge(str(r.get("status", ""))),
                "score": r.get("overall_score"),
            }
            for r in rows
        ]
        st.dataframe(mini, use_container_width=True, hide_index=True)

    if finished:
        del st.session_state["suite_process"]
        st.session_state.pop("suite_dir", None)
        st.rerun()


def _render_suite_review(suites_dir: str) -> None:
    _render_launch_panel(suites_dir)

    suites = list_suite_runs(suites_dir)
    if not suites:
        st.info("No suite runs found.")
        return

    suite_labels = [
        (
            f"{s['suite_run_id']}  ({s['case_count']} cases, mean={s['mean_score']:.3f})"
            if s["mean_score"] is not None
            else f"{s['suite_run_id']}  ({s['case_count']} cases)"
        )
        for s in suites
    ]
    idx = st.selectbox(
        "Suite run",
        range(len(suites)),
        format_func=lambda i: suite_labels[i],
        key="suite-sel",
    )
    selected = suites[idx]
    suite_dir = Path(str(selected["suite_dir"]))

    # Summary
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Cases", selected["case_count"])
    c2.metric("Success", selected["successes"])
    c3.metric("Partial", selected["partials"])
    c4.metric("Fail", selected["fails"])
    c5.metric(
        "Mean Score",
        (
            f"{selected['mean_score']:.3f}"
            if selected["mean_score"] is not None
            else "n/a"
        ),
    )

    results = read_json(suite_dir / "results.json")
    rows = results.get("rows", [])
    rows = rows if isinstance(rows, list) else []

    # Thumbnail grid
    _thumbnail_grid(rows)

    # Case table
    case_table = [
        {
            "case_id": r.get("case_id"),
            "status": status_badge(str(r.get("status", ""))),
            "score": r.get("overall_score"),
            "hausdorff_mm": r.get("hausdorff_mm"),
            "parts": r.get("part_count"),
            "errors": r.get("errors"),
        }
        for r in rows
    ]

    event = st.dataframe(
        case_table,
        use_container_width=True,
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
        key="suite-cases",
    )
    sel = _selected_rows(event)
    if not rows:
        return

    case_idx = max(0, min(sel[0] if sel else 0, len(rows) - 1))
    row = rows[case_idx]
    run_dir = str(row.get("run_dir", "")).strip()

    st.subheader(f"Case: {row.get('case_id', 'unknown')}")
    if run_dir and Path(run_dir).exists():
        metrics = read_json(Path(run_dir) / "metrics.json")
        _render_metrics_row(metrics)
        _render_overlay(run_dir, metrics, key=f"sc-{case_idx}")
        _render_phase_stepper(run_dir, key=f"sc-ps-{case_idx}")
        _render_violations(metrics)
        with st.expander("SVG / Artifacts"):
            _render_artifacts(run_dir, key=f"sa-{case_idx}")

        # Failure modes
        debug = metrics.get("debug", {}) if isinstance(metrics, dict) else {}
        code_counts: Dict[str, int] = {}
        for v in metrics.get("violations", []):
            code = str(v.get("code", "")).strip()
            if code:
                code_counts[code] = code_counts.get(code, 0) + 1
        modes = infer_failure_modes(
            str(row.get("status", "")).lower(),
            code_counts,
            debug if isinstance(debug, dict) else {},
        )
        if modes:
            with st.expander("Failure modes"):
                for m in modes:
                    st.write(f"- {m}")
    else:
        st.warning(f"Run directory not found: {run_dir}")

    # Suite comparison
    _suite_compare(suites, suites_dir)


# ---------------------------------------------------------------------------
# Tab 2: Run Detail
# ---------------------------------------------------------------------------


def _render_run_detail(runs_dir: str) -> None:
    runs = list_runs(runs_dir)
    if not runs:
        st.info("No runs found.")
        return

    run_table = [
        {
            "run_id": r["run_id"],
            "status": status_badge(str(r.get("status", ""))),
            "design": r.get("design_name"),
            "created_utc": r.get("created_utc"),
        }
        for r in runs
    ]

    event = st.dataframe(
        run_table,
        use_container_width=True,
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
        key="run-table",
    )
    sel = _selected_rows(event)
    if not runs:
        return

    run_idx = max(0, min(sel[0] if sel else 0, len(runs) - 1))
    run = runs[run_idx]
    run_dir = str(run.get("run_dir", ""))

    if not run_dir or not Path(run_dir).exists():
        st.warning("Run directory missing.")
        return

    metrics = read_json(Path(run_dir) / "metrics.json")

    st.subheader(f"Run: {run['run_id']}")
    _render_metrics_row(metrics)
    _render_overlay(run_dir, metrics, key=f"rd-{run_idx}")
    _render_phase_stepper(run_dir, key=f"rd-ps-{run_idx}")
    _render_violations(metrics)
    with st.expander("SVG / Artifacts"):
        _render_artifacts(run_dir, key=f"ra-{run_idx}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    st.set_page_config(page_title="Furniture Review", layout="wide")
    st.title("Furniture Review")

    c1, c2 = st.columns(2)
    runs_dir = c1.text_input("Runs directory", value="runs")
    suites_dir = c2.text_input("Suites directory", value="runs/suites")

    tab_suite, tab_run = st.tabs(["Suite Review", "Run Detail"])
    with tab_suite:
        _render_suite_review(suites_dir)
    with tab_run:
        _render_run_detail(runs_dir)


if __name__ == "__main__":
    main()
