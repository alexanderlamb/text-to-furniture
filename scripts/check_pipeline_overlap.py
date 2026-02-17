#!/usr/bin/env python3
"""Run the actual pipeline on shelf_unit and inspect overlap detection."""
import sys
sys.path.insert(0, "src")

import numpy as np
from pipeline import run_pipeline_from_mesh
import step3_first_principles as s3

# Run the pipeline
mesh_path = "benchmarks/meshes/08_shelf_unit.stl"
result = run_pipeline_from_mesh(mesh_path, design_name="debug_overlap")

s3out = result.step3_output
parts = list(s3out.parts.values())  # ManufacturingPart objects

print(f"Parts: {len(parts)}\n")

# Check origin_3d vs position_3d
print("=== origin_3d vs position_3d ===\n")
for p in parts:
    prof = p.profile
    o3d = prof.origin_3d if prof.origin_3d is not None else "None"
    p3d = np.array(p.position_3d)
    if prof.origin_3d is not None:
        diff = np.linalg.norm(prof.origin_3d - p3d)
        print(f"  {p.part_id}: origin_3d={prof.origin_3d}, position_3d={p3d}, diff={diff:.4f}")
    else:
        print(f"  {p.part_id}: origin_3d=None, position_3d={p3d}")

# Now manually call _slab_penetration for the overlapping pairs
print("\n=== _slab_penetration for overlapping pairs ===\n")

def check_pair(parts, i, j):
    pa, pb = parts[i], parts[j]
    prof_a, prof_b = pa.profile, pb.profile
    n_a = s3._rotation_to_normal(pa.rotation_3d)
    n_b = s3._rotation_to_normal(pb.rotation_3d)

    # Slab origin = outward face (position_3d + normal * thickness)
    slab_origin_b = np.array(pb.position_3d) + n_b * pb.thickness_mm
    slab_origin_a = np.array(pa.position_3d) + n_a * pa.thickness_mm
    pen_a_into_b = s3._slab_penetration(prof_a, n_b, slab_origin_b, pb.thickness_mm)
    pen_b_into_a = s3._slab_penetration(prof_b, n_a, slab_origin_a, pa.thickness_mm)
    pair_max = min(pen_a_into_b, pen_b_into_a)

    print(f"  {pa.part_id} vs {pb.part_id}:")
    print(f"    pen_{pa.part_id}_into_{pb.part_id} = {pen_a_into_b:.4f}mm")
    print(f"    pen_{pb.part_id}_into_{pa.part_id} = {pen_b_into_a:.4f}mm")
    print(f"    pair_max = min({pen_a_into_b:.4f}, {pen_b_into_a:.4f}) = {pair_max:.4f}mm")
    print(f"    detected as overlap: {'YES' if pair_max > 0.01 else 'NO'}")

    # Also dump the raw d-values for understanding
    d_surface_b = float(np.dot(n_b, prof_b.origin_3d))
    coords_a = list(prof_a.outline.exterior.coords)
    d_vals = []
    for u, v in coords_a:
        pt3d = prof_a.project_2d_to_3d(u, v)
        d_vals.append(float(np.dot(n_b, pt3d)) - d_surface_b)
    print(f"    A vertices d-range in B's frame: [{min(d_vals):.3f}, {max(d_vals):.3f}]")
    print(f"    B slab interior: [{-pb.thickness_mm + 0.5:.3f}, {-0.5:.3f}]")
    print()

# Check the pairs that my bbox script found overlapping
pairs_to_check = [(2, 5), (2, 7), (3, 5), (3, 7), (3, 8)]
for i, j in pairs_to_check:
    check_pair(parts, i, j)

# Also run the full _compute_plane_overlap and report
print("=== Full _compute_plane_overlap result ===\n")
overlap_result = s3._compute_plane_overlap(parts)
for k, v in overlap_result.items():
    if not isinstance(v, list):
        print(f"  {k}: {v}")
    elif v:
        print(f"  {k}: {v}")
    else:
        print(f"  {k}: []")
