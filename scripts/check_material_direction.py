#!/usr/bin/env python3
"""Determine the correct material direction for each part by comparing
with the mesh geometry."""
import sys
sys.path.insert(0, "src")
import numpy as np
import trimesh
from pipeline import run_pipeline_from_mesh
import step3_first_principles as s3

mesh_path = "benchmarks/meshes/08_shelf_unit.stl"
mesh = trimesh.load(mesh_path)

# Get mesh vertices that belong to specific faces
result = run_pipeline_from_mesh(mesh_path, design_name="matdir_check")
parts = list(result.step3_output.parts.values())

print(f"Mesh bounds: {mesh.bounds}")
print(f"Mesh extents: {mesh.extents}")
print()

# For each part, check: do the mesh face vertices extend in the +normal or -normal
# direction from origin_3d?
for p in parts:
    n = s3._rotation_to_normal(p.rotation_3d)
    prof = p.profile
    face_indices = p.source_faces

    if not face_indices:
        print(f"{p.part_id}: no source faces")
        continue

    # Get all unique vertices from source faces
    face_verts = mesh.vertices[mesh.faces[face_indices].flatten()]

    # Project onto normal axis
    d_surface = np.dot(n, prof.origin_3d)
    d_vals = np.dot(face_verts, n) - d_surface

    # Get mesh vertices NOT on this part's face but nearby
    all_d = np.dot(mesh.vertices, n) - d_surface

    print(f"{p.part_id}: n=[{n[0]:.1f},{n[1]:.1f},{n[2]:.1f}], origin={prof.origin_3d}")
    print(f"  Face vertex d-range: [{d_vals.min():.4f}, {d_vals.max():.4f}] (should be ~0)")
    print(f"  All mesh vertices d-range: [{all_d.min():.4f}, {all_d.max():.4f}]")
    print(f"  Mesh extends {-all_d.min():.1f}mm in +normal dir, {all_d.max():.1f}mm in -normal dir from face")

    # The material is on the side where OTHER mesh faces are (the interior)
    # Face vertices should be at d≈0. Interior is where most other vertices are.
    interior_count_neg = np.sum(all_d < -0.1)  # anti-normal direction
    interior_count_pos = np.sum(all_d > 0.1)   # normal direction
    print(f"  Vertices in anti-normal (d<-0.1): {interior_count_neg}")
    print(f"  Vertices in +normal (d>0.1): {interior_count_pos}")
    direction = "anti-normal (d<0)" if interior_count_neg > interior_count_pos else "+normal (d>0)"
    print(f"  Material direction: {direction}")
    print()
