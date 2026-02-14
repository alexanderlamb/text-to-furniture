#!/usr/bin/env python3
"""
Visualize mesh decomposition pipeline stages.

Shows:
  1. Original mesh
  2. Dominant orientations (normal vectors)
  3. Voxel grid (surface shell)
  4. Selected slabs overlaid on mesh
  5. Joint connections

Usage:
    python scripts/visualize_decomposition.py --input model.stl
    python scripts/visualize_decomposition.py --input model.obj --height 500 --max-slabs 8
"""
import sys
import os
import argparse
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.colors as mcolors

from mesh_decomposer import (
    DecompositionConfig, load_mesh, find_dominant_orientations,
    voxelize, generate_candidates, select_slabs, infer_joints,
    build_design, optimize_arrangement, _slab_local_axes, SlabCandidate,
)
from materials import MATERIALS


# Colour palette for slabs
SLAB_COLORS = list(mcolors.TABLEAU_COLORS.values())


def _mesh_wireframe(ax, mesh, alpha=0.08, color="gray"):
    """Draw mesh as a semi-transparent triangulated surface."""
    verts = mesh.vertices
    faces = mesh.faces
    # Sub-sample faces for large meshes
    max_faces = 4000
    if len(faces) > max_faces:
        idx = np.random.choice(len(faces), max_faces, replace=False)
        faces = faces[idx]
    tri = verts[faces]
    poly = Poly3DCollection(tri, alpha=alpha, edgecolor="none", facecolor=color)
    ax.add_collection3d(poly)


def _set_axes_equal(ax, mesh):
    """Set equal aspect ratio for 3D axes based on mesh bounds."""
    bounds = mesh.bounds
    center = (bounds[0] + bounds[1]) / 2.0
    extent = (bounds[1] - bounds[0]).max() / 2.0 * 1.1
    ax.set_xlim(center[0] - extent, center[0] + extent)
    ax.set_ylim(center[1] - extent, center[1] + extent)
    ax.set_zlim(center[2] - extent, center[2] + extent)
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_zlabel("Z (mm)")


def _draw_slab(ax, slab, color, alpha=0.35, label=None):
    """Draw a slab as a coloured rectangular prism."""
    u, v = _slab_local_axes(slab.normal)
    n = slab.normal / np.linalg.norm(slab.normal)
    hw, hh, ht = slab.width / 2, slab.height / 2, slab.thickness / 2
    o = slab.origin

    # 8 corners
    corners = []
    for su in (-1, 1):
        for sv in (-1, 1):
            for sn in (-1, 1):
                corners.append(o + u * su * hw + v * sv * hh + n * sn * ht)
    corners = np.array(corners)

    # 6 faces (each is 4 corner indices)
    face_idx = [
        [0, 1, 3, 2], [4, 5, 7, 6],  # -n, +n
        [0, 1, 5, 4], [2, 3, 7, 6],  # -v, +v
        [0, 2, 6, 4], [1, 3, 7, 5],  # -u, +u
    ]
    quads = [[corners[i] for i in f] for f in face_idx]
    poly = Poly3DCollection(quads, alpha=alpha, facecolor=color,
                            edgecolor=color, linewidth=0.5)
    ax.add_collection3d(poly)

    if label:
        ax.text(o[0], o[1], o[2], label, fontsize=7, ha="center",
                color="black", weight="bold")


def plot_step1_mesh(ax, mesh):
    """Step 1: Original mesh."""
    ax.set_title("1. Original Mesh", fontsize=11, weight="bold")
    _mesh_wireframe(ax, mesh, alpha=0.15, color="steelblue")
    _set_axes_equal(ax, mesh)


def plot_step2_orientations(ax, mesh, orientations):
    """Step 2: Dominant orientation normals."""
    ax.set_title("2. Dominant Orientations", fontsize=11, weight="bold")
    _mesh_wireframe(ax, mesh, alpha=0.06)
    _set_axes_equal(ax, mesh)

    center = (mesh.bounds[0] + mesh.bounds[1]) / 2.0
    extent = (mesh.bounds[1] - mesh.bounds[0]).max() * 0.35

    for i, (normal, area) in enumerate(orientations[:12]):
        # Arrow length proportional to area (clamped)
        max_area = orientations[0][1] if orientations[0][1] > 0 else 1.0
        length = extent * (0.3 + 0.7 * min(area / max_area, 1.0))
        end = center + normal * length
        color = SLAB_COLORS[i % len(SLAB_COLORS)]
        ax.quiver(center[0], center[1], center[2],
                  normal[0] * length, normal[1] * length, normal[2] * length,
                  color=color, arrow_length_ratio=0.15, linewidth=2)

    ax.text2D(0.02, 0.02, f"{len(orientations)} orientations",
              transform=ax.transAxes, fontsize=8, color="gray")


def plot_step3_voxels(ax, mesh, voxel_grid):
    """Step 3: Surface shell voxels."""
    ax.set_title("3. Surface Shell Voxels", fontsize=11, weight="bold")
    _set_axes_equal(ax, mesh)

    ii, jj, kk = np.where(voxel_grid.matrix)
    if len(ii) == 0:
        return

    centres = (voxel_grid.origin
               + np.column_stack([ii, jj, kk]) * voxel_grid.pitch
               + voxel_grid.pitch / 2.0)

    # Sub-sample for plotting speed
    max_pts = 6000
    if len(centres) > max_pts:
        idx = np.random.choice(len(centres), max_pts, replace=False)
        centres = centres[idx]

    ax.scatter(centres[:, 0], centres[:, 1], centres[:, 2],
               s=1, alpha=0.3, c="mediumseagreen", marker=".")

    ax.text2D(0.02, 0.02,
              f"{voxel_grid.total_filled} shell voxels  "
              f"(pitch={voxel_grid.pitch}mm)",
              transform=ax.transAxes, fontsize=8, color="gray")


def plot_step4_slabs(ax, mesh, slabs, voxel_grid):
    """Step 4: Selected slabs overlaid on mesh."""
    ax.set_title("4. Selected Slabs", fontsize=11, weight="bold")
    _mesh_wireframe(ax, mesh, alpha=0.05)
    _set_axes_equal(ax, mesh)

    for i, slab in enumerate(slabs):
        color = SLAB_COLORS[i % len(SLAB_COLORS)]
        _draw_slab(ax, slab, color, alpha=0.40, label=f"S{i}")

    cov = voxel_grid.coverage_fraction() * 100
    ax.text2D(0.02, 0.02,
              f"{len(slabs)} slabs, {cov:.0f}% coverage",
              transform=ax.transAxes, fontsize=8, color="gray")


def plot_step5_joints(ax, mesh, slabs, joints):
    """Step 5: Joint connections between slabs."""
    ax.set_title("5. Joints", fontsize=11, weight="bold")
    _mesh_wireframe(ax, mesh, alpha=0.04)
    _set_axes_equal(ax, mesh)

    for i, slab in enumerate(slabs):
        color = SLAB_COLORS[i % len(SLAB_COLORS)]
        _draw_slab(ax, slab, color, alpha=0.25)

    # Map slab name to index
    name_to_idx = {f"slab_{i}": i for i in range(len(slabs))}
    joint_colors = {
        "tab_slot": "red",
        "butt": "orange",
        "through_bolt": "dodgerblue",
    }

    for j in joints:
        ia = name_to_idx.get(j.component_a)
        ib = name_to_idx.get(j.component_b)
        if ia is None or ib is None:
            continue
        a_pos = slabs[ia].origin
        b_pos = slabs[ib].origin
        jcolor = joint_colors.get(j.joint_type.value, "gray")
        ax.plot([a_pos[0], b_pos[0]],
                [a_pos[1], b_pos[1]],
                [a_pos[2], b_pos[2]],
                color=jcolor, linewidth=2, alpha=0.8)

    # Legend
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], color=c, lw=2, label=t)
               for t, c in joint_colors.items()]
    ax.legend(handles=handles, loc="upper right", fontsize=7)

    ax.text2D(0.02, 0.02, f"{len(joints)} joints",
              transform=ax.transAxes, fontsize=8, color="gray")


def plot_step6_design(ax, mesh, design):
    """Step 6: Final FurnitureDesign components."""
    ax.set_title("6. FurnitureDesign Output", fontsize=11, weight="bold")
    _mesh_wireframe(ax, mesh, alpha=0.04)
    _set_axes_equal(ax, mesh)

    for i, comp in enumerate(design.components):
        color = SLAB_COLORS[i % len(SLAB_COLORS)]
        dims = comp.get_dimensions()
        # Reconstruct a slab-like object for drawing
        from mesh_decomposer import normal_to_rotation
        # Reverse: rotation -> normal (Z axis after rotation)
        rx, ry, rz = comp.rotation
        # Build rotation matrix from Euler angles (XYZ intrinsic)
        cx, sx = np.cos(rx), np.sin(rx)
        cy, sy = np.cos(ry), np.sin(ry)
        cz, sz = np.cos(rz), np.sin(rz)
        Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
        Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
        Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
        R = Rz @ Ry @ Rx
        normal = R[:, 2]  # Z column = thickness direction

        dummy_slab = SlabCandidate(
            normal=normal,
            position=0.0,
            thickness=comp.thickness,
            width=dims[0],
            height=dims[1],
            material_key=comp.material,
            rotation=comp.rotation.copy(),
            origin=comp.position.copy(),
        )
        _draw_slab(ax, dummy_slab, color, alpha=0.40,
                   label=f"{comp.name}\n{comp.type.value}")


def run_pipeline_with_visualization(filepath, config, output_dir, optimize):
    """Run the full pipeline and visualize each step."""
    print(f"Loading mesh: {filepath}")
    mesh = load_mesh(filepath, config)

    print("Finding dominant orientations...")
    orientations = find_dominant_orientations(mesh, config)
    print(f"  {len(orientations)} orientations found")

    print("Voxelizing...")
    vgrid = voxelize(mesh, config)
    print(f"  {vgrid.total_filled} surface voxels")

    print("Generating candidates...")
    candidates = generate_candidates(mesh, orientations, config)
    print(f"  {len(candidates)} candidates")

    print("Selecting slabs (greedy)...")
    selected = select_slabs(candidates, vgrid, config)
    print(f"  {len(selected)} slabs, {vgrid.coverage_fraction()*100:.1f}% coverage")

    if optimize and config.optimize_iterations > 0 and selected:
        print(f"Optimizing ({config.optimize_iterations} iterations)...")
        selected = optimize_arrangement(selected, mesh, vgrid, config)

    print("Inferring joints...")
    joints = infer_joints(selected)
    print(f"  {len(joints)} joints")

    print("Building design...")
    design = build_design(selected, joints)
    is_valid, errors = design.validate()
    print(f"  Valid: {is_valid}")

    # Create the 6-panel figure
    fig = plt.figure(figsize=(18, 11))
    fig.suptitle(
        f"Mesh Decomposition: {Path(filepath).name}  "
        f"({len(design.components)} components, {len(joints)} joints)",
        fontsize=13, weight="bold",
    )

    panels = [
        (plot_step1_mesh, (mesh,)),
        (plot_step2_orientations, (mesh, orientations)),
        (plot_step3_voxels, (mesh, vgrid)),
        (plot_step4_slabs, (mesh, selected, vgrid)),
        (plot_step5_joints, (mesh, selected, joints)),
        (plot_step6_design, (mesh, design)),
    ]

    for idx, (plot_fn, plot_args) in enumerate(panels):
        ax = fig.add_subplot(2, 3, idx + 1, projection="3d")
        ax.set_box_aspect([1, 1, 1])
        plot_fn(ax, *plot_args)

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # Save to output dir if provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        png_path = os.path.join(output_dir, "decomposition_steps.png")
        fig.savefig(png_path, dpi=150, bbox_inches="tight")
        print(f"\nSaved figure to {png_path}")

    plt.show(block=True)
    # If plt.show() returns immediately (non-interactive terminal),
    # fall back to a manual event loop to keep the window alive.
    try:
        while plt.get_fignums():
            fig.canvas.flush_events()
            import time
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass

    return design


def main():
    parser = argparse.ArgumentParser(
        description="Visualize mesh decomposition pipeline stages.",
    )
    parser.add_argument("--input", required=True,
                        help="Path to input mesh file")
    parser.add_argument("--output", default=None,
                        help="Output directory for saving the figure")
    parser.add_argument("--material", default="plywood_baltic_birch",
                        choices=list(MATERIALS.keys()))
    parser.add_argument("--height", type=float, default=750.0,
                        help="Target height in mm")
    parser.add_argument("--max-slabs", type=int, default=10)
    parser.add_argument("--coverage-target", type=float, default=0.80)
    parser.add_argument("--no-optimize", action="store_true")
    parser.add_argument("--voxel-res", type=float, default=5.0,
                        help="Voxel resolution in mm (larger = faster)")
    parser.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    config = DecompositionConfig(
        default_material=args.material,
        target_height_mm=args.height,
        max_slabs=args.max_slabs,
        coverage_target=args.coverage_target,
        voxel_resolution_mm=args.voxel_res,
    )

    run_pipeline_with_visualization(
        args.input, config, args.output, optimize=not args.no_optimize,
    )


if __name__ == "__main__":
    main()
