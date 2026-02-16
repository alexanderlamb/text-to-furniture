"""Overlay geometry utilities for visualizing run outputs in 3D."""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import trimesh
from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import triangulate

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]

_PART_COLORS = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
]

# Realistic material colors for rendering
_MATERIAL_COLORS: Dict[str, Tuple[str, str]] = {
    # (face_color, edge_color)
    "plywood_baltic_birch": ("#D4B896", "#8B7355"),
    "mdf": ("#C4A882", "#7A6B5D"),
    "hardboard": ("#8B7355", "#5C4A32"),
    "acrylic_clear": ("#E8F4FD", "#B0C4DE"),
    "acrylic_black": ("#2D2D2D", "#1A1A1A"),
    "aluminum_5052": ("#C0C0C0", "#808080"),
    "steel_mild": ("#A8A8A8", "#696969"),
    "steel_stainless_304": ("#D0D0D0", "#909090"),
}

_DEFAULT_MATERIAL_COLOR = ("#D4B896", "#8B7355")  # Baltic Birch default


@dataclass
class NormalizationParams:
    scale: float
    anchor_raw: np.ndarray


@dataclass
class OverlayPart:
    part_id: str
    color: str
    fill_vertices: np.ndarray
    fill_faces: np.ndarray
    edge_loops: List[np.ndarray]
    centroid: np.ndarray
    edge_color: str = "#8B7355"


@dataclass
class OverlaySceneData:
    run_id: str
    run_dir: str
    source_mesh_path: str
    space: str
    mesh_vertices: np.ndarray
    mesh_faces: np.ndarray
    parts: List[OverlayPart]
    normalization: NormalizationParams
    warnings: List[str]


def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _resolve_path(run_dir: Path, candidate: str) -> Path:
    p = Path(candidate)
    search = [p]
    if not p.is_absolute():
        search.append(ROOT / p)
        search.append(run_dir / p)

    for option in search:
        if option.exists():
            return option
    raise FileNotFoundError(f"Unable to resolve path from run: {candidate}")


def _extract_step3_input(manifest: Dict[str, Any]) -> Dict[str, Any]:
    config = manifest.get("config", {})
    if not isinstance(config, dict):
        return {}
    direct = config.get("step3_input")
    if isinstance(direct, dict):
        return direct
    pipeline_cfg = config.get("pipeline", {})
    if isinstance(pipeline_cfg, dict):
        nested = pipeline_cfg.get("step3_input")
        if isinstance(nested, dict):
            return nested
    return {}


def _load_mesh(mesh_path: Path) -> trimesh.Trimesh:
    scene_or_mesh = trimesh.load(str(mesh_path))
    if isinstance(scene_or_mesh, trimesh.Scene):
        mesh = scene_or_mesh.to_mesh()
        if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) == 0:
            meshes = [
                g
                for g in scene_or_mesh.geometry.values()
                if isinstance(g, trimesh.Trimesh)
            ]
            if not meshes:
                raise ValueError(f"No triangle mesh found in scene: {mesh_path}")
            mesh = max(meshes, key=lambda m: len(m.faces))
    elif isinstance(scene_or_mesh, trimesh.Trimesh):
        mesh = scene_or_mesh
    else:
        raise ValueError(f"Unsupported mesh object: {type(scene_or_mesh)}")
    out = mesh.copy()
    out.remove_unreferenced_vertices()
    return out


def load_run_overlay_inputs(
    run_dir: str,
    design_json_override: Optional[str] = None,
) -> Dict[str, Any]:
    run_path = Path(run_dir)
    manifest_path = run_path / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")

    manifest = _read_json(manifest_path)

    mesh_candidate = str(manifest.get("input_mesh", "")).strip()
    if mesh_candidate:
        mesh_path = _resolve_path(run_path, mesh_candidate)
    else:
        mesh_files = sorted((run_path / "input").glob("*"))
        if not mesh_files:
            raise FileNotFoundError(f"No input mesh found in {run_path / 'input'}")
        mesh_path = mesh_files[0]

    if design_json_override:
        design = _read_json(Path(design_json_override))
    else:
        artifacts = manifest.get("artifacts", {})
        design_json = artifacts.get("design_json")
        if not design_json:
            raise FileNotFoundError(
                f"Manifest missing artifacts.design_json: {manifest_path}"
            )
        design_path = _resolve_path(run_path, design_json)
        design = _read_json(design_path)

    return {
        "manifest": manifest,
        "design": design,
        "mesh_path": mesh_path,
        "step3_input": _extract_step3_input(manifest),
    }


def compute_normalization_params(
    mesh_vertices: np.ndarray, step3_input: Dict[str, Any]
) -> NormalizationParams:
    min_corner = np.min(mesh_vertices, axis=0)
    max_corner = np.max(mesh_vertices, axis=0)
    extents = max_corner - min_corner
    mesh_height = float(extents[2])

    auto_scale = bool(step3_input.get("auto_scale", True))
    target_height_mm = float(step3_input.get("target_height_mm", 750.0))
    scale = target_height_mm / mesh_height if auto_scale and mesh_height > 1e-6 else 1.0
    center_xy_raw = (min_corner[:2] + max_corner[:2]) * 0.5
    min_z_raw = min_corner[2]
    anchor_raw = np.array([center_xy_raw[0], center_xy_raw[1], min_z_raw], dtype=float)
    return NormalizationParams(scale=scale, anchor_raw=anchor_raw)


def original_to_normalized(
    points: np.ndarray, params: NormalizationParams
) -> np.ndarray:
    arr = np.asarray(points, dtype=float)
    return params.scale * (arr - params.anchor_raw)


def normalized_to_original(
    points: np.ndarray, params: NormalizationParams
) -> np.ndarray:
    arr = np.asarray(points, dtype=float)
    scale = params.scale if abs(params.scale) > 1e-12 else 1.0
    return arr / scale + params.anchor_raw


def _rotation_matrix_xyz(rotation_xyz: Sequence[float]) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    euler_angles = np.asarray(rotation_xyz, dtype=float)
    return Rotation.from_euler("xyz", euler_angles).as_matrix()


def _clean_polygon(polygon: Polygon) -> Optional[Polygon]:
    if polygon.is_empty:
        return None
    clean = polygon if polygon.is_valid else polygon.buffer(0)
    if clean.is_empty:
        return None
    if isinstance(clean, MultiPolygon):
        clean = max(clean.geoms, key=lambda g: g.area)
    if clean.area <= 1e-6:
        return None
    return clean


def triangulate_part_polygon(
    outline_2d: Sequence[Sequence[float]],
    cutouts_2d: Optional[Sequence[Sequence[Sequence[float]]]] = None,
) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
    if len(outline_2d) < 3:
        return np.zeros((0, 2), dtype=float), np.zeros((0, 3), dtype=int), []

    shell = np.asarray(outline_2d, dtype=float)
    if np.allclose(shell[0], shell[-1]):
        shell = shell[:-1]
    holes: List[np.ndarray] = []
    if cutouts_2d:
        for cutout in cutouts_2d:
            if len(cutout) < 3:
                continue
            cut = np.asarray(cutout, dtype=float)
            if np.allclose(cut[0], cut[-1]):
                cut = cut[:-1]
            holes.append(cut)

    polygon = _clean_polygon(
        Polygon(shell=shell, holes=[h.tolist() for h in holes if len(h) >= 3])
    )
    if polygon is None:
        return np.zeros((0, 2), dtype=float), np.zeros((0, 3), dtype=int), []

    vertices: List[List[float]] = []
    faces: List[List[int]] = []
    for tri in triangulate(polygon):
        if tri.is_empty or tri.area <= 1e-9:
            continue
        if not polygon.covers(tri.representative_point()):
            continue
        coords = np.asarray(list(tri.exterior.coords)[:3], dtype=float)
        base = len(vertices)
        vertices.extend(coords.tolist())
        faces.append([base, base + 1, base + 2])

    edge_loops = [np.asarray(polygon.exterior.coords[:-1], dtype=float)]
    edge_loops.extend(
        np.asarray(interior.coords[:-1], dtype=float) for interior in polygon.interiors
    )

    return (
        np.asarray(vertices, dtype=float),
        np.asarray(faces, dtype=int),
        edge_loops,
    )


def _transform_local_points(
    points_2d: np.ndarray, rotation_3d: Sequence[float], position_3d: Sequence[float]
) -> np.ndarray:
    if len(points_2d) == 0:
        return np.zeros((0, 3), dtype=float)
    local = np.column_stack([points_2d, np.zeros(len(points_2d), dtype=float)])
    rot = _rotation_matrix_xyz(rotation_3d)
    pos = np.asarray(position_3d, dtype=float)
    return (rot @ local.T).T + pos


def _transform_local_3d_points(
    points_3d: np.ndarray, rotation_3d: Sequence[float], position_3d: Sequence[float]
) -> np.ndarray:
    if len(points_3d) == 0:
        return np.zeros((0, 3), dtype=float)
    rot = _rotation_matrix_xyz(rotation_3d)
    pos = np.asarray(position_3d, dtype=float)
    return (rot @ points_3d.T).T + pos


def _extrude_triangulated_polygon(
    verts_2d: np.ndarray,
    faces_2d: np.ndarray,
    edge_loops_2d: List[np.ndarray],
    thickness_mm: float,
) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
    """Extrude a 2D triangulated polygon into a 3D slab in local coords.

    Outer face at z=0 (mesh surface), inner face at z=-thickness_mm.
    Returns (vertices_3d, faces, edge_loops_3d) in local coordinates.
    """
    n_fill = len(verts_2d)

    # Outer and inner fill faces
    outer_fill = np.column_stack([verts_2d, np.zeros(n_fill)])
    inner_fill = np.column_stack([verts_2d, np.full(n_fill, -thickness_mm)])

    all_verts: List[np.ndarray] = [outer_fill, inner_fill]
    all_faces: List[np.ndarray] = [
        faces_2d.copy(),
        faces_2d[:, ::-1] + n_fill,  # reversed winding for inner face
    ]

    base_idx = 2 * n_fill

    # Side walls from edge loops
    for loop in edge_loops_2d:
        n_loop = len(loop)
        if n_loop < 2:
            continue
        outer_edge = np.column_stack([loop, np.zeros(n_loop)])
        inner_edge = np.column_stack([loop, np.full(n_loop, -thickness_mm)])
        all_verts.append(outer_edge)
        all_verts.append(inner_edge)

        side_tris = []
        for k in range(n_loop):
            k_next = (k + 1) % n_loop
            o0 = base_idx + k
            o1 = base_idx + k_next
            i0 = base_idx + n_loop + k
            i1 = base_idx + n_loop + k_next
            side_tris.append([o0, o1, i0])
            side_tris.append([i0, o1, i1])
        all_faces.append(np.array(side_tris, dtype=int))
        base_idx += 2 * n_loop

    verts_out = np.vstack(all_verts) if all_verts else np.zeros((0, 3), dtype=float)
    faces_out = np.vstack(all_faces) if all_faces else np.zeros((0, 3), dtype=int)

    # Edge loops: outer and inner copy of each
    loops_out: List[np.ndarray] = []
    for loop in edge_loops_2d:
        loops_out.append(np.column_stack([loop, np.zeros(len(loop))]))
        loops_out.append(np.column_stack([loop, np.full(len(loop), -thickness_mm)]))

    return verts_out, faces_out, loops_out


def transform_part_outline_to_3d(
    outline_2d: Sequence[Sequence[float]],
    rotation_3d: Sequence[float],
    position_3d: Sequence[float],
) -> np.ndarray:
    if len(outline_2d) < 3:
        return np.zeros((0, 3), dtype=float)
    points = np.asarray(outline_2d, dtype=float)
    if np.allclose(points[0], points[-1]):
        points = points[:-1]
    return _transform_local_points(points, rotation_3d, position_3d)


def _limit_mesh_faces(
    vertices: np.ndarray, faces: np.ndarray, max_mesh_faces: int
) -> Tuple[np.ndarray, np.ndarray]:
    if max_mesh_faces <= 0 or len(faces) <= max_mesh_faces:
        return vertices, faces
    rng = np.random.default_rng(seed=0)
    idx = np.sort(rng.choice(len(faces), size=max_mesh_faces, replace=False))
    sampled_faces = faces[idx]
    used_vertices = np.unique(sampled_faces.reshape(-1))
    remap = np.full(len(vertices), -1, dtype=int)
    remap[used_vertices] = np.arange(len(used_vertices), dtype=int)
    return vertices[used_vertices], remap[sampled_faces]


def _part_color(part_id: str) -> str:
    digest = hashlib.md5(part_id.encode("utf-8")).hexdigest()
    slot = int(digest[:8], 16) % len(_PART_COLORS)
    return _PART_COLORS[slot]


def _convert_space(
    points: np.ndarray, params: NormalizationParams, space: str
) -> np.ndarray:
    if space == "normalized":
        return points
    return normalized_to_original(points, params)


def build_overlay_scene(
    run_dir: str,
    space: str = "original",
    max_mesh_faces: int = 8000,
    design_json_override: Optional[str] = None,
) -> OverlaySceneData:
    if space not in {"original", "normalized"}:
        raise ValueError(f"Unsupported space: {space}")

    payload = load_run_overlay_inputs(
        run_dir, design_json_override=design_json_override
    )
    manifest = payload["manifest"]
    design = payload["design"]
    mesh_path: Path = payload["mesh_path"]
    step3_input = payload["step3_input"]

    mesh = _load_mesh(mesh_path)
    raw_vertices = np.asarray(mesh.vertices, dtype=float)
    raw_faces = np.asarray(mesh.faces, dtype=int)
    params = compute_normalization_params(raw_vertices, step3_input)
    normalized_vertices = original_to_normalized(raw_vertices, params)
    mesh_vertices = normalized_vertices if space == "normalized" else raw_vertices
    mesh_vertices, mesh_faces = _limit_mesh_faces(
        mesh_vertices, raw_faces, max_mesh_faces
    )

    warnings: List[str] = []
    parts: List[OverlayPart] = []
    for idx, part_payload in enumerate(design.get("parts", [])):
        part_id = str(part_payload.get("part_id", f"part_{idx:02d}"))
        outline_2d = part_payload.get("outline_2d") or []
        cutouts_2d = part_payload.get("cutouts_2d") or []
        rotation_3d = part_payload.get("rotation_3d") or [0.0, 0.0, 0.0]
        # Use origin_3d for outline placement (outline coords are relative to it).
        # Fall back to position_3d for older data that doesn't have origin_3d.
        position_3d = part_payload.get("origin_3d") or part_payload.get("position_3d") or [0.0, 0.0, 0.0]
        thickness_mm = float(part_payload.get("thickness_mm", 0.0))
        material_key = str(part_payload.get("material_key", ""))
        face_color = _part_color(part_id)
        _, edge_color = _MATERIAL_COLORS.get(
            material_key, _DEFAULT_MATERIAL_COLOR
        )

        verts_2d, faces_2d, edge_loops_2d = triangulate_part_polygon(
            outline_2d, cutouts_2d
        )
        if len(verts_2d) == 0 or len(faces_2d) == 0:
            warnings.append(f"Skipped part {part_id}: non-triangulable outline")
            continue

        if thickness_mm > 0:
            # Extrude into 3D slab with material thickness
            local_verts, local_faces, local_loops = _extrude_triangulated_polygon(
                verts_2d, faces_2d, edge_loops_2d, thickness_mm
            )
            fill_vertices = _transform_local_3d_points(
                local_verts, rotation_3d, position_3d
            )
            fill_faces = local_faces
            edge_loops = [
                _transform_local_3d_points(loop, rotation_3d, position_3d)
                for loop in local_loops
                if len(loop) >= 2
            ]
        else:
            # Flat rendering fallback
            fill_vertices = _transform_local_points(verts_2d, rotation_3d, position_3d)
            fill_faces = faces_2d
            edge_loops = [
                _transform_local_points(loop, rotation_3d, position_3d)
                for loop in edge_loops_2d
                if len(loop) >= 2
            ]
        if len(fill_vertices) == 0:
            warnings.append(f"Skipped part {part_id}: empty transformed geometry")
            continue

        fill_vertices = _convert_space(fill_vertices, params, space)
        edge_loops = [_convert_space(loop, params, space) for loop in edge_loops]
        centroid = np.mean(fill_vertices, axis=0)

        parts.append(
            OverlayPart(
                part_id=part_id,
                color=face_color,
                fill_vertices=fill_vertices,
                fill_faces=fill_faces,
                edge_loops=edge_loops,
                centroid=centroid,
                edge_color=edge_color,
            )
        )

    return OverlaySceneData(
        run_id=str(manifest.get("run_id", Path(run_dir).name)),
        run_dir=str(run_dir),
        source_mesh_path=str(mesh_path),
        space=space,
        mesh_vertices=mesh_vertices,
        mesh_faces=mesh_faces,
        parts=parts,
        normalization=params,
        warnings=warnings,
    )


def render_overlay_screenshot(
    run_dir: str,
    output_path: str,
    *,
    space: str = "original",
    max_mesh_faces: int = 8000,
    mesh_opacity: float = 0.30,
    part_opacity: float = 0.55,
    elev: float = 22.0,
    azim: float = 38.0,
    dpi: int = 150,
) -> str:
    """Render a static overlay screenshot to *output_path* (PNG).

    Uses matplotlib with the Agg backend so it works headless.
    Returns the absolute path of the written file.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    scene = build_overlay_scene(run_dir, space=space, max_mesh_faces=max_mesh_faces)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    bounds: List[np.ndarray] = []

    v, f = scene.mesh_vertices, scene.mesh_faces
    if len(v) > 0 and len(f) > 0:
        ax.add_collection3d(Poly3DCollection(
            v[f],
            facecolors=(0.65, 0.65, 0.65, mesh_opacity),
            edgecolors=(0.65, 0.65, 0.65, mesh_opacity * 0.2),
            linewidths=0.05,
        ))
        bounds.append(v)

    for part in scene.parts:
        if len(part.fill_vertices) > 0 and len(part.fill_faces) > 0:
            c = part.color.lstrip("#")
            if len(c) == 6:
                rgba = (int(c[0:2], 16) / 255, int(c[2:4], 16) / 255, int(c[4:6], 16) / 255, part_opacity)
            else:
                rgba = (0.2, 0.6, 0.9, part_opacity)
            ax.add_collection3d(Poly3DCollection(
                part.fill_vertices[part.fill_faces],
                facecolors=rgba,
                edgecolors=(*rgba[:3], min(1.0, part_opacity + 0.15)),
                linewidths=0.1,
            ))
            bounds.append(part.fill_vertices)
        for loop in part.edge_loops:
            if len(loop) == 0:
                continue
            closed = loop if (loop[0] == loop[-1]).all() else np.vstack([loop, loop[0]])
            ax.plot(closed[:, 0], closed[:, 1], closed[:, 2], color=part.edge_color, linewidth=1.0)

    if bounds:
        cloud = np.vstack(bounds)
        mins, maxs = cloud.min(axis=0), cloud.max(axis=0)
        center = (mins + maxs) * 0.5
        radius = max(float(np.max(maxs - mins)) * 0.55, 1.0)
        ax.set_xlim(center[0] - radius, center[0] + radius)
        ax.set_ylim(center[1] - radius, center[1] + radius)
        ax.set_zlim(center[2] - radius, center[2] + radius)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.view_init(elev=elev, azim=azim)
    fig.tight_layout()

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out), dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info("Overlay screenshot saved: %s", out)
    return str(out.resolve())
