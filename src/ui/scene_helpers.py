"""
Helpers for rendering 3D meshes and decomposition slabs into NiceGUI scenes.

NiceGUI scene API notes:
- gltf(url) and stl(url) are methods on the Scene object, not on Group.
- Objects created inside a `with scene.group()` context are parented to that group.
- Scaling is done via .scale(s) chained call, not a constructor param.
- three.js uses Y-up coordinate system.
"""

import math
import os

from nicegui import ui

from furniture import FurnitureDesign

# Tableau-inspired color palette for slab visualization
SLAB_COLORS = [
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
    "#aec7e8",
    "#ffbb78",
    "#98df8a",
    "#ff9896",
    "#c5b0d5",
]

# Conversion factor: project uses mm, three.js scenes use metres
MM_TO_M = 0.001


def normalize_mesh_for_viewer(mesh_path: str, output_dir: str) -> str:
    """Load a mesh, orient it Y-up, place bottom on y=0, and export a new GLB.

    This bakes the correct orientation directly into the geometry so
    three.js renders it correctly without needing runtime transforms.

    Returns the path to the normalized GLB file.
    """
    import trimesh
    import numpy as np

    scene = trimesh.load(mesh_path)

    # Flatten scene to a single mesh with all transforms applied
    if isinstance(scene, trimesh.Scene):
        mesh = scene.to_mesh()
    else:
        mesh = scene

    verts = mesh.vertices  # (N, 3)

    # Determine which axis is "up" — the axis with the smallest extent
    # is most likely the height for furniture (tables, shelves, etc.)
    extents = verts.max(axis=0) - verts.min(axis=0)
    up_axis = int(np.argmin(extents))

    if up_axis == 0:
        # X is up → rotate +90° around Z to make Y up
        rot = trimesh.transformations.rotation_matrix(math.pi / 2, [0, 0, 1])
        mesh.apply_transform(rot)
    elif up_axis == 2:
        # Z is up → rotate -90° around X to make Y up
        rot = trimesh.transformations.rotation_matrix(-math.pi / 2, [1, 0, 0])
        mesh.apply_transform(rot)
    # up_axis == 1 → Y is already up, no rotation needed

    # Check if mesh is upside-down: for furniture, the "top" surface typically
    # has more faces near the Y-max. If face normals at the top mostly point
    # down, flip it.
    face_centers = mesh.triangles_center
    face_normals = mesh.face_normals
    y_max = mesh.vertices[:, 1].max()
    y_min = mesh.vertices[:, 1].min()
    y_range = y_max - y_min
    # Look at faces in the top 20% of the mesh
    top_mask = face_centers[:, 1] > (y_max - 0.2 * y_range)
    if top_mask.sum() > 0:
        avg_top_normal_y = face_normals[top_mask, 1].mean()
        if avg_top_normal_y < -0.3:
            # Top faces point down → mesh is upside down, rotate 180° around Z
            rot = trimesh.transformations.rotation_matrix(math.pi, [0, 0, 1])
            mesh.apply_transform(rot)

    # Translate so bottom sits on y=0, centered on x=0 and z=0
    bounds = mesh.bounds  # [[min_x, min_y, min_z], [max_x, max_y, max_z]]
    center_x = (bounds[0][0] + bounds[1][0]) / 2
    center_z = (bounds[0][2] + bounds[1][2]) / 2
    y_min = bounds[0][1]
    mesh.apply_translation([-center_x, -y_min, -center_z])

    # Export normalized GLB
    basename = os.path.splitext(os.path.basename(mesh_path))[0]
    out_path = os.path.join(output_dir, f"{basename}_normalized.glb")
    mesh.export(out_path, file_type="glb")
    return out_path


def render_mesh(scene: ui.scene, mesh_url: str, ext: str) -> None:
    """Load a mesh into a NiceGUI scene.

    GLB meshes should be pre-normalized via normalize_mesh_for_viewer()
    so they're already Y-up with bottom on y=0.
    STL files from the decomposer are in mm with Z-up.
    """
    with scene:
        if ext in ("glb", "gltf"):
            scene.gltf(mesh_url)
        elif ext == "stl":
            with scene.group().rotate(-math.pi / 2, 0, 0):
                scene.stl(mesh_url).scale(MM_TO_M).material(color="#bbbbbb")


def render_manufacturing_parts(scene: ui.scene, mfg_summary: dict,
                               viewer_mesh_path: str = "") -> None:
    """Render manufacturing pipeline parts as colored 3D boxes.

    The decomposition operates in Z-up mm space (auto-scaled), while the
    viewer shows a Y-up mesh in its original units (typically metres).
    We compute a scale factor from decomp bounds to viewer bounds so the
    parts overlay the mesh correctly.
    """
    import trimesh
    import numpy as np

    components = mfg_summary.get("components", [])
    if not components:
        return

    decomp_bounds = mfg_summary.get("decomp_bounds")
    if not decomp_bounds:
        return

    # Decomposition mesh: Z-up, mm, centered XY, bottom at Z=0
    decomp_min = np.array(decomp_bounds[0])
    decomp_max = np.array(decomp_bounds[1])
    decomp_extents = decomp_max - decomp_min  # [x_extent, y_extent, z_extent] in mm

    # Load the normalized viewer mesh to get its bounds
    # The normalized mesh is Y-up, centered XZ, bottom at Y=0
    viewer_extents = None
    if viewer_mesh_path:
        try:
            vmesh = trimesh.load(viewer_mesh_path)
            if isinstance(vmesh, trimesh.Scene):
                vmesh = vmesh.to_mesh()
            vbounds = vmesh.bounds
            viewer_extents = vbounds[1] - vbounds[0]  # [x, y, z] in viewer units
        except Exception:
            pass

    if viewer_extents is None:
        return

    # The decomp Z-height maps to viewer Y-height.
    # Compute per-axis scale factors: decomp mm → viewer units.
    # After Z-up → Y-up rotation: decomp X→viewer X, decomp Y→viewer Z, decomp Z→viewer Y
    if decomp_extents[2] < 0.01:
        return
    scale = viewer_extents[1] / decomp_extents[2]  # fallback uniform scale
    scale_x = viewer_extents[0] / decomp_extents[0] if decomp_extents[0] > 0.01 else scale
    scale_y = viewer_extents[2] / decomp_extents[1] if decomp_extents[1] > 0.01 else scale
    scale_z = viewer_extents[1] / decomp_extents[2] if decomp_extents[2] > 0.01 else scale

    scales = [scale_x, scale_y, scale_z]
    if min(scales) > 0 and max(scales) / min(scales) > 1.02:
        import logging as _logging
        _logging.getLogger(__name__).warning(
            "Per-axis scale spread: %.4f / %.4f / %.4f (max/min=%.3f)",
            scale_x, scale_y, scale_z, max(scales) / min(scales))

    dim_scale = (scale_x + scale_y + scale_z) / 3.0

    with scene:
        # Parts are in decomp Z-up space.  We apply the scale, then
        # the Z-up → Y-up rotation so they align with the viewer mesh.
        with scene.group().rotate(-math.pi / 2, 0, 0):
            for i, cd in enumerate(components):
                color = SLAB_COLORS[i % len(SLAB_COLORS)]
                w = cd["width"] * dim_scale
                h = cd["height"] * dim_scale
                t = cd["thickness"] * dim_scale

                px = cd["position"][0] * scale_x
                py = cd["position"][1] * scale_y
                pz = cd["position"][2] * scale_z

                rx = cd["rotation"][0]
                ry = cd["rotation"][1]
                rz = cd["rotation"][2]

                (
                    scene.box(w, h, t)
                    .material(color=color, opacity=0.65)
                    .move(px, py, pz)
                    .rotate(rx, ry, rz)
                )


def render_slabs(scene: ui.scene, design: FurnitureDesign) -> None:
    """Render each component of a FurnitureDesign as a colored 3D box.

    Component.position is the slab's world-space origin (center).
    Component.profile gives width/height; thickness is separate.
    """
    with scene:
        # Root group with Z-up -> Y-up rotation
        with scene.group().rotate(-math.pi / 2, 0, 0):
            for i, comp in enumerate(design.components):
                color = SLAB_COLORS[i % len(SLAB_COLORS)]
                dims = comp.get_dimensions()
                w_m = dims[0] * MM_TO_M
                h_m = dims[1] * MM_TO_M
                t_m = comp.thickness * MM_TO_M

                px = float(comp.position[0]) * MM_TO_M
                py = float(comp.position[1]) * MM_TO_M
                pz = float(comp.position[2]) * MM_TO_M

                rx = float(comp.rotation[0])
                ry = float(comp.rotation[1])
                rz = float(comp.rotation[2])

                (
                    scene.box(w_m, h_m, t_m)
                    .material(color=color, opacity=0.75)
                    .move(px, py, pz)
                    .rotate(rx, ry, rz)
                )
