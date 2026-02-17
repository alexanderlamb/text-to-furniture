#!/usr/bin/env python3
"""Verify the slab direction convention used by _slab_penetration."""
import sys
sys.path.insert(0, "src")
import numpy as np
from pipeline import run_pipeline_from_mesh
import step3_first_principles as s3

mesh_path = "benchmarks/meshes/08_shelf_unit.stl"
result = run_pipeline_from_mesh(mesh_path, design_name="slab_check")
parts = list(result.step3_output.parts.values())

print("=== Slab Convention Check ===\n")
for p in parts:
    prof = p.profile
    n = s3._rotation_to_normal(p.rotation_3d)

    # Get all 3D vertices from the profile
    coords = list(prof.outline.exterior.coords)
    pts_3d = [prof.project_2d_to_3d(u, v) for u, v in coords]
    pts_3d = np.array(pts_3d)

    # Profile vertices are on one face. The other face is offset by ±thickness in normal direction.
    face_a = pts_3d  # at profile plane
    face_b_normal = pts_3d + n * p.thickness_mm   # offset in +normal direction
    face_b_antinormal = pts_3d - n * p.thickness_mm  # offset in -normal direction

    # Compute bboxes
    all_normal = np.vstack([face_a, face_b_normal])
    all_anti = np.vstack([face_a, face_b_antinormal])

    bbox_normal_min = all_normal.min(axis=0)
    bbox_normal_max = all_normal.max(axis=0)
    bbox_anti_min = all_anti.min(axis=0)
    bbox_anti_max = all_anti.max(axis=0)

    # Compare with position_3d convention
    # The design JSON uses: p3 = R @ [x,y,0] + position_3d, p3_top = p3 + normal * thickness
    # So the two faces are at profile plane and profile plane + normal * thickness

    print(f"{p.part_id}: normal=[{n[0]:.1f},{n[1]:.1f},{n[2]:.1f}], pos={np.array(p.position_3d)}, origin={prof.origin_3d}")
    print(f"  profile plane bbox: [{pts_3d.min(0)[0]:.1f},{pts_3d.min(0)[1]:.1f},{pts_3d.min(0)[2]:.1f}] to [{pts_3d.max(0)[0]:.1f},{pts_3d.max(0)[1]:.1f},{pts_3d.max(0)[2]:.1f}]")
    print(f"  if material extends +normal: [{bbox_normal_min[0]:.1f},{bbox_normal_min[1]:.1f},{bbox_normal_min[2]:.1f}] to [{bbox_normal_max[0]:.1f},{bbox_normal_max[1]:.1f},{bbox_normal_max[2]:.1f}]")
    print(f"  if material extends -normal: [{bbox_anti_min[0]:.1f},{bbox_anti_min[1]:.1f},{bbox_anti_min[2]:.1f}] to [{bbox_anti_max[0]:.1f},{bbox_anti_max[1]:.1f},{bbox_anti_max[2]:.1f}]")

    # Check _slab_penetration direction: it uses d<0 = anti-normal side
    # So it treats origin as the surface and material in anti-normal direction
    # This means the slab from _slab_penetration perspective is:
    # [origin_plane, origin_plane + anti_normal * thickness]
    # = [origin_plane, origin_plane - normal * thickness]
    print(f"  _slab_penetration assumes material at: profile_plane - normal * thickness")
    print(f"    i.e., anti-normal direction from profile plane")
    print()

# Now check the actual mesh to see which direction is correct
import trimesh
mesh = trimesh.load(mesh_path)
print("=== Mesh face analysis ===\n")
print(f"  Mesh bounds: x=[{mesh.bounds[0][0]:.1f},{mesh.bounds[1][0]:.1f}], y=[{mesh.bounds[0][1]:.1f},{mesh.bounds[1][1]:.1f}], z=[{mesh.bounds[0][2]:.1f},{mesh.bounds[1][2]:.1f}]")
print(f"  Mesh size: {mesh.bounds[1] - mesh.bounds[0]}")
