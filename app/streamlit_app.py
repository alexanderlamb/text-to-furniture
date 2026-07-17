"""Step 1 dashboard for OpenSCAD panelization runs."""

from __future__ import annotations

import json
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st

from data import artifact_files, list_runs, read_json

try:
    import plotly.graph_objects as go
except ModuleNotFoundError:
    go = None

try:
    from shapely.geometry import MultiPolygon, Polygon
    from shapely.ops import triangulate
except ModuleNotFoundError:
    MultiPolygon = None
    Polygon = None
    triangulate = None


HYBRID_STRATEGY_COLORS = {
    "planar_skin": "#2f80ed",
    "contour_stack": "#f2994a",
    "waffle_ribs": "#8f7a1f",
    "voxel_blocks": "#27ae60",
    "unassigned": "#7a8694",
}

HYBRID_STRATEGY_ORDER = (
    "planar_skin",
    "contour_stack",
    "waffle_ribs",
    "voxel_blocks",
)

HYBRID_STRATEGY_LABELS = {
    "planar_skin": "Planar Skin",
    "contour_stack": "Contour Stack",
    "waffle_ribs": "Waffle Ribs",
    "voxel_blocks": "Voxel Blocks",
    "hybrid": "Hybrid",
}

_BOX_TRIANGLES = np.asarray(
    [
        [0, 1, 2],
        [0, 2, 3],
        [4, 6, 5],
        [4, 7, 6],
        [0, 4, 5],
        [0, 5, 1],
        [1, 5, 6],
        [1, 6, 2],
        [2, 6, 7],
        [2, 7, 3],
        [3, 7, 4],
        [3, 4, 0],
    ],
    dtype=int,
)

_BOX_EDGE_PAIRS = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 0),
    (4, 5),
    (5, 6),
    (6, 7),
    (7, 4),
    (0, 4),
    (1, 5),
    (2, 6),
    (3, 7),
)


def _python_executable() -> str:
    preferred = ROOT / "venv" / "bin" / "python3"
    return str(preferred) if preferred.exists() else sys.executable


def _run_step1(
    mesh_path: str,
    design_name: str,
    runs_dir: str,
    material_key: str,
    thickness_mm: float | None,
    part_budget: int,
    auto_scale: bool,
    target_height_mm: float,
) -> subprocess.CompletedProcess[str]:
    cmd: List[str] = [
        _python_executable(),
        str(ROOT / "scripts" / "generate_openscad_step1.py"),
        "--mesh",
        mesh_path,
        "--name",
        design_name,
        "--runs-dir",
        runs_dir,
        "--material-key",
        material_key,
        "--part-budget",
        str(part_budget),
        "--target-height-mm",
        str(target_height_mm),
    ]
    if thickness_mm is not None:
        cmd.extend(["--thickness-mm", str(thickness_mm)])
    if not auto_scale:
        cmd.append("--no-auto-scale")

    return subprocess.run(cmd, capture_output=True, text=True)


def _load_artifact(
    manifest: Dict[str, Any], run_dir: Path, key: str, fallback: str
) -> Dict[str, Any]:
    artifact_path = manifest.get("artifacts", {}).get(key)
    if artifact_path:
        return read_json(Path(artifact_path))
    return read_json(run_dir / "artifacts" / fallback)


def _relation_counts(capsule: Dict[str, Any]) -> Dict[str, int]:
    relations = capsule.get("relations", [])
    if not isinstance(relations, list):
        return {}
    counter: Counter[str] = Counter()
    for relation in relations:
        if not isinstance(relation, dict):
            continue
        relation_class = str(relation.get("class", "unknown"))
        counter[relation_class] += 1
    return dict(counter)


def _render_centers_plot(capsule: Dict[str, Any]) -> None:
    if go is None:
        st.info("Install plotly to render 3D part centers.")
        return

    parts = capsule.get("parts", [])
    if not isinstance(parts, list) or not parts:
        st.info("No parts available in spatial capsule.")
        return

    xs: List[float] = []
    ys: List[float] = []
    zs: List[float] = []
    labels: List[str] = []
    for part in parts:
        if not isinstance(part, dict):
            continue
        obb = part.get("obb", {})
        center = (
            obb.get("center", [0.0, 0.0, 0.0])
            if isinstance(obb, dict)
            else [0.0, 0.0, 0.0]
        )
        if not isinstance(center, list) or len(center) != 3:
            continue
        xs.append(float(center[0]))
        ys.append(float(center[1]))
        zs.append(float(center[2]))
        labels.append(str(part.get("part_id", "part")))

    if not xs:
        st.info("No plottable part centers.")
        return

    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=xs,
                y=ys,
                z=zs,
                mode="markers+text",
                marker={"size": 5},
                text=labels,
                textposition="top center",
            )
        ]
    )
    fig.update_layout(
        margin={"l": 0, "r": 0, "t": 20, "b": 0},
        scene={
            "xaxis_title": "X (mm)",
            "yaxis_title": "Y (mm)",
            "zaxis_title": "Z (mm)",
        },
    )
    st.plotly_chart(fig, use_container_width=True)


def _as_vec3(raw: Any, default: tuple[float, float, float]) -> np.ndarray:
    if not isinstance(raw, list) or len(raw) != 3:
        return np.asarray(default, dtype=float)
    try:
        return np.asarray([float(raw[0]), float(raw[1]), float(raw[2])], dtype=float)
    except (TypeError, ValueError):
        return np.asarray(default, dtype=float)


def _ring_from_points(raw: Any) -> np.ndarray:
    if not isinstance(raw, list) or len(raw) < 3:
        return np.zeros((0, 2), dtype=float)
    pts: list[list[float]] = []
    for point in raw:
        if not isinstance(point, list) or len(point) != 2:
            continue
        try:
            pts.append([float(point[0]), float(point[1])])
        except (TypeError, ValueError):
            continue
    if len(pts) < 3:
        return np.zeros((0, 2), dtype=float)
    arr = np.asarray(pts, dtype=float)
    if np.allclose(arr[0], arr[-1]):
        arr = arr[:-1]
    return arr if arr.shape[0] >= 3 else np.zeros((0, 2), dtype=float)


def _triangulate_polygon_2d(poly: Polygon) -> tuple[np.ndarray, np.ndarray]:
    vertices: list[list[float]] = []
    faces: list[list[int]] = []
    for tri in triangulate(poly):
        if tri.is_empty or tri.area <= 1e-9:
            continue
        if not poly.covers(tri.representative_point()):
            continue
        coords = np.asarray(list(tri.exterior.coords)[:3], dtype=float)
        if coords.shape != (3, 2):
            continue
        base = len(vertices)
        vertices.extend(coords.tolist())
        faces.append([base, base + 1, base + 2])
    if not vertices or not faces:
        return np.zeros((0, 2), dtype=float), np.zeros((0, 3), dtype=int)
    return np.asarray(vertices, dtype=float), np.asarray(faces, dtype=int)


def _build_local_panel_mesh(
    part: Dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray], float] | None:
    if Polygon is None or triangulate is None:
        return None

    thickness = float(part.get("thickness_mm", 0.0) or 0.0)
    if thickness <= 1e-6:
        return None

    outline = _ring_from_points(part.get("outline_2d"))
    if outline.shape[0] < 3:
        return None

    holes_raw = part.get("holes_2d", [])
    holes: list[list[tuple[float, float]]] = []
    if isinstance(holes_raw, list):
        for hole in holes_raw:
            ring = _ring_from_points(hole)
            if ring.shape[0] >= 3:
                holes.append([(float(p[0]), float(p[1])) for p in ring])

    poly = Polygon(
        [(float(p[0]), float(p[1])) for p in outline],
        holes=holes if holes else None,
    )
    if not poly.is_valid:
        poly = poly.buffer(0)
    if poly.is_empty:
        return None
    if MultiPolygon is not None and isinstance(poly, MultiPolygon):
        poly = max(poly.geoms, key=lambda g: float(g.area))

    tri_verts_2d, tri_faces = _triangulate_polygon_2d(poly)
    if tri_verts_2d.size == 0 or tri_faces.size == 0:
        return None

    n_base = tri_verts_2d.shape[0]
    bottom = np.column_stack((tri_verts_2d, np.zeros(n_base, dtype=float)))
    top = np.column_stack((tri_verts_2d, np.full(n_base, thickness, dtype=float)))
    vertices = np.vstack((bottom, top))

    faces: list[list[int]] = []
    for f in tri_faces:
        faces.append([int(f[0]), int(f[2]), int(f[1])])
        faces.append([int(f[0] + n_base), int(f[1] + n_base), int(f[2] + n_base)])

    rings_local: list[np.ndarray] = [np.asarray(poly.exterior.coords[:-1], dtype=float)]
    rings_local.extend(np.asarray(r.coords[:-1], dtype=float) for r in poly.interiors)

    for ring in rings_local:
        m = ring.shape[0]
        if m < 2:
            continue
        for i in range(m):
            p0 = ring[i]
            p1 = ring[(i + 1) % m]
            base = vertices.shape[0]
            quad = np.asarray(
                [
                    [float(p0[0]), float(p0[1]), 0.0],
                    [float(p1[0]), float(p1[1]), 0.0],
                    [float(p1[0]), float(p1[1]), thickness],
                    [float(p0[0]), float(p0[1]), thickness],
                ],
                dtype=float,
            )
            vertices = np.vstack((vertices, quad))
            faces.append([base, base + 1, base + 2])
            faces.append([base, base + 2, base + 3])

    return vertices, np.asarray(faces, dtype=int), rings_local, thickness


def _local_to_world(
    local_vertices: np.ndarray,
    origin: np.ndarray,
    basis_u: np.ndarray,
    basis_v: np.ndarray,
    basis_n: np.ndarray,
) -> np.ndarray:
    if local_vertices.size == 0:
        return np.zeros((0, 3), dtype=float)
    return (
        origin[None, :]
        + local_vertices[:, [0]] * basis_u[None, :]
        + local_vertices[:, [1]] * basis_v[None, :]
        + local_vertices[:, [2]] * basis_n[None, :]
    )


def _part_color(index: int) -> str:
    palette = [
        "#0077B6",
        "#2A9D8F",
        "#E76F51",
        "#E9C46A",
        "#457B9D",
        "#F4A261",
        "#1D3557",
        "#8AB17D",
        "#6D597A",
        "#3D405B",
    ]
    return palette[index % len(palette)]


def _hybrid_strategy_color(strategy_id: str) -> str:
    if strategy_id in HYBRID_STRATEGY_COLORS:
        return HYBRID_STRATEGY_COLORS[strategy_id]
    digest = abs(hash(strategy_id)) % 360
    return f"hsl({digest} 64% 42%)"


def _hybrid_evaluation_path(run: Mapping[str, Any]) -> Path:
    manifest = run.get("manifest", {})
    artifacts = manifest.get("artifacts", {}) if isinstance(manifest, dict) else {}
    evaluation_json = (
        artifacts.get("evaluation_json") if isinstance(artifacts, dict) else None
    )
    if evaluation_json:
        return Path(str(evaluation_json))
    return (
        Path(str(run.get("run_dir", "")))
        / "artifacts"
        / "hybrid_evaluation"
        / "evaluation.json"
    )


def _load_hybrid_evaluation(run: Mapping[str, Any]) -> Dict[str, Any]:
    return read_json(_hybrid_evaluation_path(run))


def _hybrid_rows_by_mesh(evaluation: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    rows = evaluation.get("rows", [])
    if not isinstance(rows, list):
        return {}
    by_mesh: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        mesh_name = str(row.get("mesh") or Path(str(row.get("mesh_path", ""))).name)
        if mesh_name:
            by_mesh[mesh_name] = row
    return by_mesh


def _run_display_name(run: Mapping[str, Any]) -> str:
    run_id = str(run.get("run_id") or Path(str(run.get("run_dir", ""))).name)
    design_name = str(run.get("design_name") or "")
    return (
        f"{run_id} · {design_name}" if design_name and design_name != run_id else run_id
    )


def _load_hybrid_plan(row: Mapping[str, Any]) -> Dict[str, Any]:
    path_text = str(row.get("hybrid_plan") or "").strip()
    return read_json(Path(path_text)) if path_text else {}


def _hybrid_plan_path(row: Mapping[str, Any]) -> Path | None:
    path_text = str(row.get("hybrid_plan") or "").strip()
    return Path(path_text) if path_text else None


def _source_strategy_plan_path(row: Mapping[str, Any], strategy_id: str) -> Path | None:
    hybrid_plan_path = _hybrid_plan_path(row)
    if hybrid_plan_path is None:
        return None
    return hybrid_plan_path.parent / "source_strategies" / strategy_id / "plan.json"


def _load_source_strategy_plan(
    row: Mapping[str, Any], strategy_id: str
) -> Dict[str, Any]:
    path = _source_strategy_plan_path(row, strategy_id)
    return read_json(path) if path is not None else {}


def _strategy_label(strategy_id: str) -> str:
    return HYBRID_STRATEGY_LABELS.get(
        strategy_id, strategy_id.replace("_", " ").title()
    )


def _float_triplet(raw: Any) -> np.ndarray | None:
    if not isinstance(raw, list) or len(raw) != 3:
        return None
    try:
        return np.asarray([float(raw[0]), float(raw[1]), float(raw[2])], dtype=float)
    except (TypeError, ValueError):
        return None


def _box_vertices(aabb_min: Sequence[float], aabb_max: Sequence[float]) -> np.ndarray:
    lo = np.asarray(aabb_min, dtype=float)
    hi = np.asarray(aabb_max, dtype=float)
    return np.asarray(
        [
            [lo[0], lo[1], lo[2]],
            [hi[0], lo[1], lo[2]],
            [hi[0], hi[1], lo[2]],
            [lo[0], hi[1], lo[2]],
            [lo[0], lo[1], hi[2]],
            [hi[0], lo[1], hi[2]],
            [hi[0], hi[1], hi[2]],
            [lo[0], hi[1], hi[2]],
        ],
        dtype=float,
    )


def _box_center(aabb_min: Sequence[float], aabb_max: Sequence[float]) -> np.ndarray:
    return (np.asarray(aabb_min, dtype=float) + np.asarray(aabb_max, dtype=float)) * 0.5


def _add_box_trace(
    fig: Any,
    *,
    aabb_min: Sequence[float],
    aabb_max: Sequence[float],
    color: str,
    name: str,
    hover: str,
    opacity: float,
    show_edges: bool,
    show_legend: bool,
    legend_group: str,
) -> None:
    vertices = _box_vertices(aabb_min, aabb_max)
    fig.add_trace(
        go.Mesh3d(
            x=vertices[:, 0],
            y=vertices[:, 1],
            z=vertices[:, 2],
            i=_BOX_TRIANGLES[:, 0],
            j=_BOX_TRIANGLES[:, 1],
            k=_BOX_TRIANGLES[:, 2],
            color=color,
            opacity=opacity,
            name=name,
            legendgroup=legend_group,
            showlegend=show_legend,
            flatshading=True,
            hovertemplate=hover + "<extra></extra>",
            showscale=False,
        )
    )
    if not show_edges:
        return

    xs: list[float | None] = []
    ys: list[float | None] = []
    zs: list[float | None] = []
    for start, end in _BOX_EDGE_PAIRS:
        xs.extend([float(vertices[start, 0]), float(vertices[end, 0]), None])
        ys.extend([float(vertices[start, 1]), float(vertices[end, 1]), None])
        zs.extend([float(vertices[start, 2]), float(vertices[end, 2]), None])
    fig.add_trace(
        go.Scatter3d(
            x=xs,
            y=ys,
            z=zs,
            mode="lines",
            line={"color": "#111111", "width": 2},
            hoverinfo="skip",
            showlegend=False,
        )
    )


def _assignment_strategy_by_region(plan: Mapping[str, Any]) -> Dict[str, str]:
    assignments = plan.get("assignments", [])
    if not isinstance(assignments, list):
        return {}
    result: Dict[str, str] = {}
    for assignment in assignments:
        if not isinstance(assignment, dict):
            continue
        region_id = str(assignment.get("region_id") or "")
        strategy_id = str(assignment.get("strategy_id") or "")
        if region_id and strategy_id:
            result[region_id] = strategy_id
    return result


def _part_source_strategy(part: Mapping[str, Any]) -> str:
    metadata = part.get("metadata", {})
    if isinstance(metadata, dict):
        return str(
            metadata.get("source_strategy_id") or part.get("strategy_id") or "unknown"
        )
    return str(part.get("strategy_id") or "unknown")


def _part_regions(part: Mapping[str, Any]) -> str:
    metadata = part.get("metadata", {})
    if not isinstance(metadata, dict):
        return ""
    region_ids = metadata.get("hybrid_region_ids") or metadata.get("hybrid_region_id")
    if isinstance(region_ids, list):
        return ", ".join(str(item) for item in region_ids)
    return str(region_ids or "")


def _hybrid_scene_items(
    plan: Mapping[str, Any], view_mode: str
) -> list[dict[str, Any]]:
    if view_mode == "Regions":
        strategy_by_region = _assignment_strategy_by_region(plan)
        regions = plan.get("regions", [])
        items: list[dict[str, Any]] = []
        if not isinstance(regions, list):
            return items
        for region in regions:
            if not isinstance(region, dict):
                continue
            region_id = str(region.get("region_id") or "region")
            strategy_id = strategy_by_region.get(region_id, "unassigned")
            kind = str(region.get("kind") or "")
            items.append(
                {
                    "id": region_id,
                    "strategy": strategy_id,
                    "aabb_min": region.get("aabb_min"),
                    "aabb_max": region.get("aabb_max"),
                    "hover": (
                        f"<b>{region_id}</b><br>kind={kind}<br>"
                        f"assigned={strategy_id}<br>"
                        f"volume={float(region.get('volume_mm3', 0.0) or 0.0):.0f} mm³"
                    ),
                }
            )
        return items

    parts = plan.get("parts", [])
    items = []
    if not isinstance(parts, list):
        return items
    for part in parts:
        if not isinstance(part, dict):
            continue
        part_id = str(part.get("part_id") or "part")
        strategy_id = _part_source_strategy(part)
        items.append(
            {
                "id": part_id,
                "strategy": strategy_id,
                "aabb_min": part.get("aabb_min"),
                "aabb_max": part.get("aabb_max"),
                "hover": (
                    f"<b>{part_id}</b><br>strategy={strategy_id}<br>"
                    f"kind={part.get('kind', '')}<br>regions={_part_regions(part)}<br>"
                    f"volume={float(part.get('volume_mm3', 0.0) or 0.0):.0f} mm³"
                ),
            }
        )
    return items


def _render_hybrid_3d_plot(
    plan: Mapping[str, Any],
    *,
    view_mode: str,
    opacity: float,
    show_edges: bool,
    show_labels: bool,
    plot_key: str,
    height: int | None = None,
) -> None:
    if go is None:
        st.info("Install plotly to render hybrid 3D previews.")
        return
    items = _hybrid_scene_items(plan, view_mode)
    if not items:
        st.info(f"No {view_mode.lower()} available in hybrid plan.")
        return

    fig = go.Figure()
    labels_x: list[float] = []
    labels_y: list[float] = []
    labels_z: list[float] = []
    labels_text: list[str] = []
    legend_seen: set[str] = set()
    rendered = 0

    for item in items:
        aabb_min = _float_triplet(item.get("aabb_min"))
        aabb_max = _float_triplet(item.get("aabb_max"))
        if aabb_min is None or aabb_max is None or np.any(aabb_max <= aabb_min):
            continue
        strategy_id = str(item["strategy"])
        color = _hybrid_strategy_color(strategy_id)
        show_legend = strategy_id not in legend_seen
        legend_seen.add(strategy_id)
        _add_box_trace(
            fig,
            aabb_min=aabb_min,
            aabb_max=aabb_max,
            color=color,
            name=strategy_id,
            hover=str(item["hover"]),
            opacity=opacity,
            show_edges=show_edges,
            show_legend=show_legend,
            legend_group=strategy_id,
        )
        center = _box_center(aabb_min, aabb_max)
        labels_x.append(float(center[0]))
        labels_y.append(float(center[1]))
        labels_z.append(float(center[2]))
        labels_text.append(str(item["id"]))
        rendered += 1

    if rendered == 0:
        st.info(f"No valid {view_mode.lower()} AABBs to render.")
        return

    if show_labels:
        fig.add_trace(
            go.Scatter3d(
                x=labels_x,
                y=labels_y,
                z=labels_z,
                mode="text",
                text=labels_text,
                textposition="top center",
                showlegend=False,
                hoverinfo="skip",
            )
        )

    layout: dict[str, Any] = {
        "margin": {"l": 0, "r": 0, "t": 8, "b": 0},
        "scene": {
            "xaxis_title": "X (mm)",
            "yaxis_title": "Y (mm)",
            "zaxis_title": "Z (mm)",
            "aspectmode": "data",
        },
        "legend": {"orientation": "h", "yanchor": "bottom", "y": 1.02},
    }
    if height is not None:
        layout["height"] = height
    fig.update_layout(**layout)
    st.plotly_chart(fig, use_container_width=True, key=plot_key)


def _format_strategy_mix(value: Any) -> str:
    if not isinstance(value, dict):
        return ""
    return ", ".join(f"{key}: {value[key]}" for key in sorted(value))


def _plan_overall_score(plan: Mapping[str, Any]) -> float:
    scores = plan.get("scores", {})
    raw = plan.get("overall_score")
    if raw is None and isinstance(scores, dict):
        raw = scores.get("overall")
    try:
        return float(raw or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _plan_count(plan: Mapping[str, Any], key: str, fallback_collection: str) -> int:
    raw = plan.get(key)
    if raw is None:
        collection = plan.get(fallback_collection, [])
        return len(collection) if isinstance(collection, list) else 0
    try:
        return int(raw or 0)
    except (TypeError, ValueError):
        collection = plan.get(fallback_collection, [])
        return len(collection) if isinstance(collection, list) else 0


def _hybrid_part_counts_by_strategy(plan: Mapping[str, Any]) -> Counter[str]:
    parts = plan.get("parts", [])
    counter: Counter[str] = Counter()
    if not isinstance(parts, list):
        return counter
    for part in parts:
        if isinstance(part, dict):
            counter[_part_source_strategy(part)] += 1
    return counter


def _hybrid_volume_by_strategy(plan: Mapping[str, Any]) -> Counter[str]:
    parts = plan.get("parts", [])
    counter: Counter[str] = Counter()
    if not isinstance(parts, list):
        return counter
    for part in parts:
        if not isinstance(part, dict):
            continue
        try:
            volume = float(part.get("volume_mm3", 0.0) or 0.0)
        except (TypeError, ValueError):
            volume = 0.0
        counter[_part_source_strategy(part)] += volume
    return counter


def _hybrid_assignment_counts_by_strategy(plan: Mapping[str, Any]) -> Counter[str]:
    assignments = plan.get("assignments", [])
    counter: Counter[str] = Counter()
    if not isinstance(assignments, list):
        return counter
    for assignment in assignments:
        if not isinstance(assignment, dict):
            continue
        strategy_id = str(assignment.get("strategy_id") or "unassigned")
        counter[strategy_id] += 1
    return counter


def _strategy_board_summary_rows(
    source_plans: Mapping[str, Mapping[str, Any]],
    hybrid_plan: Mapping[str, Any],
) -> list[dict[str, Any]]:
    selected_counts = _hybrid_part_counts_by_strategy(hybrid_plan)
    assigned_counts = _hybrid_assignment_counts_by_strategy(hybrid_plan)
    selected_volumes = _hybrid_volume_by_strategy(hybrid_plan)

    rows: list[dict[str, Any]] = []
    for strategy_id in HYBRID_STRATEGY_ORDER:
        plan = source_plans.get(strategy_id, {})
        rows.append(
            {
                "strategy": _strategy_label(strategy_id),
                "source_status": str(plan.get("status") or "missing"),
                "source_score": round(_plan_overall_score(plan), 4),
                "source_parts": _plan_count(plan, "part_count", "parts"),
                "source_joints": _plan_count(plan, "joint_count", "joints"),
                "hybrid_regions": int(assigned_counts.get(strategy_id, 0)),
                "hybrid_selected_parts": int(selected_counts.get(strategy_id, 0)),
                "hybrid_selected_cm3": round(
                    float(selected_volumes.get(strategy_id, 0.0)) / 1000.0, 1
                ),
            }
        )
    return rows


def _hybrid_assignment_rows(plan: Mapping[str, Any]) -> list[dict[str, Any]]:
    assignments = plan.get("assignments", [])
    if not isinstance(assignments, list):
        return []
    rows: list[dict[str, Any]] = []
    for assignment in assignments:
        if not isinstance(assignment, dict):
            continue
        part_ids = assignment.get("part_ids", [])
        reason_codes = assignment.get("reason_codes", [])
        rows.append(
            {
                "region": assignment.get("region_id"),
                "chosen_strategy": _strategy_label(
                    str(assignment.get("strategy_id") or "unassigned")
                ),
                "fit_score": round(float(assignment.get("fit_score", 0.0) or 0.0), 4),
                "selected_parts": len(part_ids) if isinstance(part_ids, list) else 0,
                "reasons": (
                    ", ".join(str(code) for code in reason_codes[:4])
                    if isinstance(reason_codes, list)
                    else ""
                ),
            }
        )
    return rows


def _render_strategy_plan_card(
    *,
    title: str,
    plan: Mapping[str, Any],
    plot_key: str,
    opacity: float,
    show_edges: bool,
    show_labels: bool,
    height: int,
    caption: str = "",
) -> None:
    if not plan:
        st.markdown(f"#### {title}")
        st.warning("Missing plan artifact.")
        return

    part_count = _plan_count(plan, "part_count", "parts")
    joint_count = _plan_count(plan, "joint_count", "joints")
    status = str(plan.get("status") or "unknown")
    score = _plan_overall_score(plan)
    st.markdown(f"#### {title}")
    st.caption(
        f"status={status} · score={score:.3f} · parts={part_count} · joints={joint_count}"
    )
    if caption:
        st.caption(caption)
    _render_hybrid_3d_plot(
        plan,
        view_mode="Parts",
        opacity=opacity,
        show_edges=show_edges,
        show_labels=show_labels,
        plot_key=plot_key,
        height=height,
    )


def _render_hybrid_strategy_board(hybrid_runs: list[Dict[str, Any]]) -> None:
    st.subheader("Strategy Board")
    st.caption(
        "One mesh, five views: the four individual strategy plans, then the hybrid "
        "selection. These are part AABBs for fast comparison, not final cut geometry."
    )

    run_labels = [_run_display_name(run) for run in hybrid_runs]
    selected_run_label = st.selectbox("Run", run_labels, index=0, key="board-run")
    selected_run = hybrid_runs[run_labels.index(selected_run_label)]
    evaluation = _load_hybrid_evaluation(selected_run)
    rows_by_mesh = _hybrid_rows_by_mesh(evaluation)
    mesh_names = sorted(rows_by_mesh)
    if not mesh_names:
        st.info("No mesh rows found in selected hybrid evaluation.")
        return

    controls = st.columns([2, 1, 1, 1])
    mesh_name = controls[0].selectbox("Mesh", mesh_names, index=0, key="board-mesh")
    show_edges = controls[1].checkbox("Edges", value=True, key="board-edges")
    show_labels = controls[2].checkbox("Labels", value=False, key="board-labels")
    opacity = controls[3].slider(
        "Opacity",
        min_value=0.1,
        max_value=1.0,
        value=0.6,
        step=0.05,
        key="board-opacity",
    )

    row = rows_by_mesh[mesh_name]
    hybrid_plan = _load_hybrid_plan(row)
    source_plans = {
        strategy_id: _load_source_strategy_plan(row, strategy_id)
        for strategy_id in HYBRID_STRATEGY_ORDER
    }

    st.markdown("##### Individual outputs + hybrid pick")
    card_columns = st.columns([1, 1, 1, 1, 1.12])
    for column, strategy_id in zip(card_columns, HYBRID_STRATEGY_ORDER):
        with column:
            _render_strategy_plan_card(
                title=_strategy_label(strategy_id),
                plan=source_plans[strategy_id],
                plot_key=(
                    f"strategy-board-{selected_run.get('run_id')}-{mesh_name}-"
                    f"{strategy_id}"
                ),
                opacity=opacity,
                show_edges=show_edges,
                show_labels=show_labels,
                height=300,
                caption="Standalone output",
            )

    with card_columns[-1]:
        _render_strategy_plan_card(
            title="Hybrid Selection",
            plan=hybrid_plan,
            plot_key=f"strategy-board-{selected_run.get('run_id')}-{mesh_name}-hybrid",
            opacity=opacity,
            show_edges=show_edges,
            show_labels=show_labels,
            height=300,
            caption="Selected/reused pieces, colored by source strategy",
        )

    st.markdown("##### How the hybrid combined them")
    summary_rows = _strategy_board_summary_rows(source_plans, hybrid_plan)
    st.dataframe(summary_rows, use_container_width=True, hide_index=True)

    assignment_rows = _hybrid_assignment_rows(hybrid_plan)
    if assignment_rows:
        st.dataframe(assignment_rows, use_container_width=True, hide_index=True)

    with st.expander("Open raw selected row + source plan paths"):
        source_paths = {
            strategy_id: str(_source_strategy_plan_path(row, strategy_id) or "")
            for strategy_id in HYBRID_STRATEGY_ORDER
        }
        st.json({"row": row, "source_strategy_paths": source_paths})


def _render_hybrid_run_card(
    *,
    run: Mapping[str, Any],
    row: Mapping[str, Any],
    plan: Mapping[str, Any],
    view_mode: str,
    opacity: float,
    show_edges: bool,
    show_labels: bool,
    plot_key: str,
) -> None:
    st.subheader(_run_display_name(run))
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Score", f"{float(row.get('overall_score', 0.0) or 0.0):.3f}")
    c2.metric("Parts", int(row.get("parts", 0) or 0))
    c3.metric("Regions", int(row.get("regions", 0) or 0))
    c4.metric("Joints", int(row.get("joints", 0) or 0))
    st.caption(f"Mix: {_format_strategy_mix(row.get('strategy_mix'))}")
    flags = row.get("flags", [])
    if isinstance(flags, list) and flags:
        st.warning("Flags: " + ", ".join(str(flag) for flag in flags))
    _render_hybrid_3d_plot(
        plan,
        view_mode=view_mode,
        opacity=opacity,
        show_edges=show_edges,
        show_labels=show_labels,
        plot_key=plot_key,
    )


def _render_hybrid_compare(hybrid_runs: list[Dict[str, Any]]) -> None:
    st.header("Hybrid 3D Compare")
    st.caption(
        "Prototype viewer: renders generated region and selected-part AABBs, "
        "colored by source strategy. This is not exact mesh geometry yet."
    )

    hybrid_view = st.radio(
        "Hybrid view",
        ["Strategy Board", "Run Compare"],
        horizontal=True,
        key="hybrid-view",
    )
    if hybrid_view == "Strategy Board":
        _render_hybrid_strategy_board(hybrid_runs)
        return

    st.subheader("Run Compare")
    run_labels = [_run_display_name(run) for run in hybrid_runs]
    left_index = 1 if len(hybrid_runs) > 1 else 0
    right_index = 0
    controls = st.columns([2, 2, 1, 1, 1])
    left_label = controls[0].selectbox("Baseline run", run_labels, index=left_index)
    right_label = controls[1].selectbox("Candidate run", run_labels, index=right_index)
    view_mode = controls[2].radio("3D view", ["Parts", "Regions"], horizontal=True)
    show_edges = controls[3].checkbox("Edges", value=True)
    show_labels = controls[4].checkbox("Labels", value=False)
    opacity = st.slider("Opacity", min_value=0.1, max_value=1.0, value=0.6, step=0.05)

    left_run = hybrid_runs[run_labels.index(left_label)]
    right_run = hybrid_runs[run_labels.index(right_label)]
    left_eval = _load_hybrid_evaluation(left_run)
    right_eval = _load_hybrid_evaluation(right_run)
    left_rows = _hybrid_rows_by_mesh(left_eval)
    right_rows = _hybrid_rows_by_mesh(right_eval)
    mesh_names = sorted(set(left_rows) & set(right_rows))
    if not mesh_names:
        st.info("No overlapping mesh rows between selected runs.")
        return

    mesh_name = st.selectbox("Mesh", mesh_names, index=0)
    left_row = left_rows[mesh_name]
    right_row = right_rows[mesh_name]
    left_plan = _load_hybrid_plan(left_row)
    right_plan = _load_hybrid_plan(right_row)

    st.subheader("Delta")
    d1, d2, d3, d4 = st.columns(4)
    left_score = float(left_row.get("overall_score", 0.0) or 0.0)
    right_score = float(right_row.get("overall_score", 0.0) or 0.0)
    left_parts = int(left_row.get("parts", 0) or 0)
    right_parts = int(right_row.get("parts", 0) or 0)
    left_joints = int(left_row.get("joints", 0) or 0)
    right_joints = int(right_row.get("joints", 0) or 0)
    d1.metric("Score", f"{right_score:.3f}", delta=f"{right_score - left_score:+.3f}")
    d2.metric("Parts", right_parts, delta=right_parts - left_parts)
    d3.metric("Joints", right_joints, delta=right_joints - left_joints)
    d4.metric("Run status", str(right_eval.get("status", "unknown")).upper())

    col_left, col_right = st.columns(2)
    with col_left:
        _render_hybrid_run_card(
            run=left_run,
            row=left_row,
            plan=left_plan,
            view_mode=view_mode,
            opacity=opacity,
            show_edges=show_edges,
            show_labels=show_labels,
            plot_key=f"hybrid-left-{mesh_name}-{left_run.get('run_id')}-{view_mode}",
        )
    with col_right:
        _render_hybrid_run_card(
            run=right_run,
            row=right_row,
            plan=right_plan,
            view_mode=view_mode,
            opacity=opacity,
            show_edges=show_edges,
            show_labels=show_labels,
            plot_key=f"hybrid-right-{mesh_name}-{right_run.get('run_id')}-{view_mode}",
        )

    with st.expander("Selected row payloads"):
        st.json({"baseline": left_row, "candidate": right_row})


def _render_solids_plot(
    capsule: Dict[str, Any], *, opacity: float = 0.8, show_edges: bool = True
) -> None:
    if go is None:
        st.info("Install plotly to render 3D solids.")
        return
    if Polygon is None:
        st.info("Install shapely to render 3D solids.")
        return

    parts = capsule.get("parts", [])
    if not isinstance(parts, list) or not parts:
        st.info("No parts available in spatial capsule.")
        return

    fig = go.Figure()
    label_x: list[float] = []
    label_y: list[float] = []
    label_z: list[float] = []
    label_text: list[str] = []
    rendered = 0

    for idx, part in enumerate(parts):
        if not isinstance(part, dict):
            continue
        mesh_payload = _build_local_panel_mesh(part)
        if mesh_payload is None:
            continue
        local_vertices, faces, rings_local, thickness = mesh_payload
        if faces.size == 0:
            continue

        origin = _as_vec3(part.get("origin_3d"), (0.0, 0.0, 0.0))
        basis_u = _as_vec3(part.get("basis_u"), (1.0, 0.0, 0.0))
        basis_v = _as_vec3(part.get("basis_v"), (0.0, 1.0, 0.0))
        basis_n = _as_vec3(part.get("basis_n"), (0.0, 0.0, 1.0))

        world_vertices = _local_to_world(
            local_vertices, origin, basis_u, basis_v, basis_n
        )
        color = _part_color(idx)
        part_id = str(part.get("part_id", f"part_{idx:03d}"))

        fig.add_trace(
            go.Mesh3d(
                x=world_vertices[:, 0],
                y=world_vertices[:, 1],
                z=world_vertices[:, 2],
                i=faces[:, 0],
                j=faces[:, 1],
                k=faces[:, 2],
                color=color,
                opacity=opacity,
                name=part_id,
                flatshading=True,
                hovertemplate=f"{part_id}<extra></extra>",
                showscale=False,
            )
        )

        if show_edges:
            for ring in rings_local:
                if ring.shape[0] < 2:
                    continue
                ring_closed = np.vstack((ring, ring[0]))
                local_bottom = np.column_stack(
                    (ring_closed, np.zeros(ring_closed.shape[0], dtype=float))
                )
                local_top = np.column_stack(
                    (ring_closed, np.full(ring_closed.shape[0], thickness, dtype=float))
                )
                world_bottom = _local_to_world(
                    local_bottom, origin, basis_u, basis_v, basis_n
                )
                world_top = _local_to_world(
                    local_top, origin, basis_u, basis_v, basis_n
                )
                fig.add_trace(
                    go.Scatter3d(
                        x=world_bottom[:, 0],
                        y=world_bottom[:, 1],
                        z=world_bottom[:, 2],
                        mode="lines",
                        line={"color": "#111111", "width": 2},
                        hoverinfo="skip",
                        showlegend=False,
                    )
                )
                fig.add_trace(
                    go.Scatter3d(
                        x=world_top[:, 0],
                        y=world_top[:, 1],
                        z=world_top[:, 2],
                        mode="lines",
                        line={"color": "#111111", "width": 2},
                        hoverinfo="skip",
                        showlegend=False,
                    )
                )

        obb = part.get("obb", {})
        center_raw = obb.get("center") if isinstance(obb, dict) else None
        if isinstance(center_raw, list) and len(center_raw) == 3:
            center = _as_vec3(center_raw, (0.0, 0.0, 0.0))
        else:
            center = np.mean(world_vertices, axis=0)
        label_x.append(float(center[0]))
        label_y.append(float(center[1]))
        label_z.append(float(center[2]))
        label_text.append(part_id)
        rendered += 1

    if rendered == 0:
        st.info("No valid panel solids to render.")
        return

    fig.add_trace(
        go.Scatter3d(
            x=label_x,
            y=label_y,
            z=label_z,
            mode="text",
            text=label_text,
            textposition="top center",
            showlegend=False,
            hoverinfo="skip",
        )
    )
    fig.update_layout(
        margin={"l": 0, "r": 0, "t": 20, "b": 0},
        scene={
            "xaxis_title": "X (mm)",
            "yaxis_title": "Y (mm)",
            "zaxis_title": "Z (mm)",
            "aspectmode": "data",
        },
    )
    st.plotly_chart(fig, use_container_width=True)


def main() -> None:
    st.set_page_config(page_title="Furniture Fabrication Dashboard", layout="wide")
    st.title("Furniture Fabrication Dashboard")

    with st.sidebar:
        runs_dir = st.text_input("Runs dir", value=str(ROOT / "runs"))
        dashboard_mode = st.radio(
            "Dashboard",
            options=["Hybrid 3D Compare", "OpenSCAD Step 1"],
            index=0,
        )

        st.divider()
        st.subheader("Launch Step 1 Run")
        mesh_path = st.text_input(
            "Mesh path", value=str(ROOT / "benchmarks" / "meshes" / "01_box.stl")
        )
        design_name = st.text_input("Run name", value="step1_dashboard")
        material_key = st.text_input("Material key", value="plywood_baltic_birch")
        thickness_mm_raw = st.text_input("Preferred thickness mm (optional)", value="")
        part_budget = st.number_input("Part budget", min_value=1, value=18, step=1)
        auto_scale = st.checkbox("Auto scale", value=True)
        target_height_mm = st.number_input(
            "Target height mm", min_value=1.0, value=750.0, step=10.0
        )

        if st.button("Run Step 1", use_container_width=True):
            parsed_thickness = None
            thickness_text = thickness_mm_raw.strip()
            if thickness_text:
                try:
                    parsed_thickness = float(thickness_text)
                except ValueError:
                    st.error("Thickness must be numeric.")
                    st.stop()

            proc = _run_step1(
                mesh_path=mesh_path,
                design_name=design_name,
                runs_dir=runs_dir,
                material_key=material_key,
                thickness_mm=parsed_thickness,
                part_budget=int(part_budget),
                auto_scale=bool(auto_scale),
                target_height_mm=float(target_height_mm),
            )
            if proc.returncode != 0:
                st.error("Run failed")
                st.code(proc.stderr or "No stderr")
            else:
                st.success("Run completed")
                st.code(proc.stdout)

    runs = list_runs(runs_dir)
    if not runs:
        st.info("No runs found.")
        return

    if dashboard_mode == "Hybrid 3D Compare":
        hybrid_runs = [run for run in runs if run.get("is_hybrid_eval")]
        if not hybrid_runs:
            st.info("No hybrid benchmark evaluation runs found yet.")
            return
        _render_hybrid_compare(hybrid_runs)
        return

    step1_runs = [run for run in runs if run.get("is_step1")]
    legacy_runs = [
        run for run in runs if not run.get("is_step1") and not run.get("is_hybrid_eval")
    ]
    hybrid_runs = [run for run in runs if run.get("is_hybrid_eval")]

    if legacy_runs:
        st.warning(f"{len(legacy_runs)} legacy runs hidden (not Step 1 strategy).")
    if hybrid_runs:
        st.info(
            f"{len(hybrid_runs)} hybrid evaluation runs hidden. "
            "Switch Dashboard to Hybrid 3D Compare to inspect them."
        )

    if not step1_runs:
        st.info("No Step 1 runs found yet.")
        return

    run_ids = [str(run["run_id"]) for run in step1_runs]
    selected_run_id = st.selectbox("Run", options=run_ids, index=0)
    selected = next(run for run in step1_runs if str(run["run_id"]) == selected_run_id)

    run_dir = Path(selected["run_dir"])
    manifest = selected.get("manifest", {})
    metrics = selected.get("metrics", {})
    design = _load_artifact(
        manifest, run_dir, "design_json", "design_step1_openscad.json"
    )
    capsule = _load_artifact(
        manifest, run_dir, "spatial_capsule", "spatial_capsule_step1.json"
    )

    counts = metrics.get("counts", {}) if isinstance(metrics, dict) else {}
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Status", str(metrics.get("status", "unknown")).upper())
    col2.metric("Panels", int(counts.get("panels", 0) or 0))
    col3.metric("Families", int(counts.get("selected_families", 0) or 0))
    col4.metric("Trims", int(counts.get("trim_pairs", 0) or 0))
    col5.metric("Errors", int(counts.get("violations_error", 0) or 0))

    tab_overview, tab_panels, tab_trim, tab_spatial, tab_audit, tab_files = st.tabs(
        ["Overview", "Panels", "Trim", "Spatial", "Audit", "Files"]
    )

    with tab_overview:
        st.subheader("Manifest")
        st.json(manifest)
        st.subheader("Metrics")
        st.json(metrics)
        selection_debug = (
            metrics.get("debug", {}).get("selection", {})
            if isinstance(metrics, dict)
            else {}
        )
        if isinstance(selection_debug, dict) and selection_debug:
            st.subheader("Selection Summary")
            st.dataframe(
                [
                    {
                        "mode": selection_debug.get("selection_mode"),
                        "shell_budget_spent": selection_debug.get(
                            "budget_spent_shell",
                            selection_debug.get("budget_spent_pass1"),
                        ),
                        "interior_budget_spent": selection_debug.get(
                            "budget_spent_interior",
                            selection_debug.get("budget_spent_pass2"),
                        ),
                        "shell_coverage": selection_debug.get(
                            "shell_face_coverage_ratio",
                            selection_debug.get("pass1_face_coverage_ratio"),
                        ),
                        "final_coverage": selection_debug.get(
                            "final_face_coverage_ratio"
                        ),
                        "cavity_count": selection_debug.get("cavity_count"),
                        "blocked_shell_conflicts": selection_debug.get(
                            "blocked_shell_conflicts"
                        ),
                        "blocked_interior_conflicts": selection_debug.get(
                            "blocked_interior_conflicts"
                        ),
                        "thin_gap_single_panel_count": selection_debug.get(
                            "thin_gap_single_panel_count"
                        ),
                        "selected_layers_total": selection_debug.get(
                            "selected_panel_layers_total"
                        ),
                    }
                ],
                use_container_width=True,
                hide_index=True,
            )

    with tab_panels:
        panels = design.get("panels", []) if isinstance(design, dict) else []
        if not isinstance(panels, list) or not panels:
            st.info("No panels in design payload.")
        else:
            st.write(f"Panel count: {len(panels)}")
            st.dataframe(
                [
                    {
                        "panel_id": panel.get("panel_id"),
                        "family_id": panel.get("family_id"),
                        "thickness_mm": panel.get("thickness_mm"),
                        "area_mm2": panel.get("area_mm2"),
                        "source_face_count": panel.get("source_face_count"),
                    }
                    for panel in panels
                    if isinstance(panel, dict)
                ],
                use_container_width=True,
                hide_index=True,
            )

    with tab_trim:
        trim_debug = (
            metrics.get("debug", {}).get("trim", {})
            if isinstance(metrics, dict)
            else {}
        )
        if isinstance(trim_debug, dict) and trim_debug:
            tc1, tc2, tc3 = st.columns(3)
            tc1.metric(
                "Pairs evaluated", int(trim_debug.get("trim_pairs_evaluated", 0))
            )
            tc2.metric("Pairs applied", int(trim_debug.get("trim_pairs_applied", 0)))
            total_area = sum(
                float(d.get("loss_a_mm2", 0))
                for d in trim_debug.get("trim_pair_details", [])
                if isinstance(d, dict) and d.get("trimmed") == d.get("panel_a")
            ) + sum(
                float(d.get("loss_b_mm2", 0))
                for d in trim_debug.get("trim_pair_details", [])
                if isinstance(d, dict) and d.get("trimmed") == d.get("panel_b")
            )
            tc3.metric("Total area trimmed", f"{total_area:.0f} mm²")

        trim_decisions_raw = (
            design.get("trim_decisions", []) if isinstance(design, dict) else []
        )
        if isinstance(trim_decisions_raw, list) and trim_decisions_raw:
            st.subheader("Trim Decisions")
            st.dataframe(
                [
                    {
                        "trimmed": td.get("trimmed_panel_id"),
                        "receiving": td.get("receiving_panel_id"),
                        "loss_trimmed_mm2": td.get("loss_trimmed_mm2"),
                        "loss_receiving_mm2": td.get("loss_receiving_mm2"),
                        "dihedral_deg": td.get("dihedral_angle_deg"),
                        "reason": td.get("direction_reason"),
                    }
                    for td in trim_decisions_raw
                    if isinstance(td, dict)
                ],
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info(
                "No trim decisions recorded (trim may be disabled or no intersections found)."
            )

        violations_raw = (
            design.get("violations", []) if isinstance(design, dict) else []
        )
        perp_violations = [
            v
            for v in violations_raw
            if isinstance(v, dict) and v.get("code") == "perpendicular_panel_overlap"
        ]
        if perp_violations:
            st.subheader(f"Perpendicular Overlap Violations ({len(perp_violations)})")
            st.dataframe(
                [
                    {
                        "panel_id": v.get("panel_id"),
                        "severity": v.get("severity"),
                        "penetration_mm": v.get("value"),
                        "message": v.get("message"),
                    }
                    for v in perp_violations
                ],
                use_container_width=True,
                hide_index=True,
            )

    with tab_spatial:
        relation_counts = _relation_counts(capsule)
        if relation_counts:
            st.write("Relation class counts")
            st.json(relation_counts)
        mode = st.radio(
            "Spatial view",
            options=["Panel solids", "Part centers"],
            horizontal=True,
            index=0,
        )
        if mode == "Panel solids":
            edge_on = st.checkbox("Show panel edges", value=True)
            opacity = st.slider(
                "Solid opacity", min_value=0.2, max_value=1.0, value=0.8, step=0.05
            )
            _render_solids_plot(
                capsule, opacity=float(opacity), show_edges=bool(edge_on)
            )
        else:
            _render_centers_plot(capsule)

    with tab_audit:
        artifacts = manifest.get("artifacts", {}) if isinstance(manifest, dict) else {}
        checkpoints = (
            artifacts.get("checkpoints", []) if isinstance(artifacts, dict) else []
        )
        st.write(f"Checkpoints: {len(checkpoints)}")
        if checkpoints:
            rows = []
            for checkpoint_path in checkpoints:
                checkpoint = read_json(Path(checkpoint_path))
                rows.append(
                    {
                        "phase_index": checkpoint.get("phase_index"),
                        "phase_name": checkpoint.get("phase_name"),
                        "payload_sha256": checkpoint.get("payload_sha256"),
                        "timestamp_utc": checkpoint.get("timestamp_utc"),
                    }
                )
            st.dataframe(rows, use_container_width=True, hide_index=True)

        decision_log_path = artifacts.get("decision_log")
        if decision_log_path:
            log_path = Path(decision_log_path)
            if log_path.exists():
                decisions: List[Dict[str, Any]] = []
                for line in log_path.read_text(encoding="utf-8").splitlines()[:200]:
                    if line.strip():
                        try:
                            decisions.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
                st.write(f"Decision records: {len(decisions)} (showing first 200)")
                st.dataframe(
                    [
                        {
                            "seq": d.get("seq"),
                            "phase_index": d.get("phase_index"),
                            "decision_type": d.get("decision_type"),
                            "selected": d.get("selected"),
                        }
                        for d in decisions
                    ],
                    use_container_width=True,
                    hide_index=True,
                )

    with tab_files:
        files = artifact_files(str(run_dir))
        st.write(f"Artifacts: {len(files)} files")
        for file_path in files:
            st.code(str(file_path))


if __name__ == "__main__":
    main()
