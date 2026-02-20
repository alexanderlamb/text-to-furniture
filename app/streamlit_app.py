"""Step 1 dashboard for OpenSCAD panelization runs."""

from __future__ import annotations

import json
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

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
    st.set_page_config(page_title="OpenSCAD Step 1 Dashboard", layout="wide")
    st.title("OpenSCAD Step 1 Dashboard")

    with st.sidebar:
        st.subheader("Launch Step 1 Run")
        runs_dir = st.text_input("Runs dir", value=str(ROOT / "runs"))
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

    step1_runs = [run for run in runs if run.get("is_step1")]
    legacy_runs = [run for run in runs if not run.get("is_step1")]

    if legacy_runs:
        st.warning(f"{len(legacy_runs)} legacy runs hidden (not Step 1 strategy).")

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
            tc1.metric("Pairs evaluated", int(trim_debug.get("trim_pairs_evaluated", 0)))
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
            st.info("No trim decisions recorded (trim may be disabled or no intersections found).")

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
