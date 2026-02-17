#!/usr/bin/env python3
"""Check for 3D volumetric overlaps between parts in a run's design output.

Loads the design JSON, reconstructs each part's 3D mesh (extruded outline),
and checks all pairs for intersection volume.
"""
import json
import sys
import numpy as np
from pathlib import Path
from itertools import combinations

def euler_to_rotation_matrix(rx, ry, rz):
    """Intrinsic XYZ Euler angles to rotation matrix (Rz @ Ry @ Rx)."""
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx

def part_to_3d_bbox(part):
    """Compute the axis-aligned 3D bounding box of a part after placement."""
    outline = np.array(part["outline_2d"])
    pos = np.array(part["position_3d"])
    rot = part["rotation_3d"]
    thickness = part["thickness_mm"]
    R = euler_to_rotation_matrix(*rot)

    # Normal is R @ [0,0,1]
    normal = R @ np.array([0.0, 0.0, 1.0])

    # 2D outline points in 3D: R @ [x, y, 0] + pos
    pts_2d = outline[:-1] if np.allclose(outline[0], outline[-1]) else outline
    pts_3d = []
    for p in pts_2d:
        p3 = R @ np.array([p[0], p[1], 0.0]) + pos
        pts_3d.append(p3)
        # Also the extruded point (offset by thickness along normal)
        pts_3d.append(p3 + normal * thickness)

    pts_3d = np.array(pts_3d)
    bbox_min = pts_3d.min(axis=0)
    bbox_max = pts_3d.max(axis=0)
    return bbox_min, bbox_max, pts_3d

def part_to_3d_vertices(part):
    """Get all 3D vertices (both faces of extruded slab)."""
    outline = np.array(part["outline_2d"])
    pos = np.array(part["position_3d"])
    rot = part["rotation_3d"]
    thickness = part["thickness_mm"]
    R = euler_to_rotation_matrix(*rot)
    normal = R @ np.array([0.0, 0.0, 1.0])

    pts_2d = outline[:-1] if np.allclose(outline[0], outline[-1]) else outline
    face_bottom = []
    face_top = []
    for p in pts_2d:
        p3 = R @ np.array([p[0], p[1], 0.0]) + pos
        face_bottom.append(p3)
        face_top.append(p3 + normal * thickness)

    return np.array(face_bottom), np.array(face_top), normal

def bbox_overlap(min1, max1, min2, max2, tol=0.1):
    """Check if two AABBs overlap (with tolerance)."""
    overlap_min = np.maximum(min1, min2)
    overlap_max = np.minimum(max1, max2)
    overlap_size = overlap_max - overlap_min
    if np.all(overlap_size > tol):
        return overlap_size
    return None

def check_slab_intersection(part_a, part_b, tol=0.5):
    """Check if two planar slabs actually intersect in 3D.

    Projects each part's outline vertices onto the other part's slab
    (normal direction) to check for material penetration, then checks
    lateral overlap in the slab plane.
    """
    face_a_bot, face_a_top, normal_a = part_to_3d_vertices(part_a)
    face_b_bot, face_b_top, normal_b = part_to_3d_vertices(part_b)

    pos_a = np.array(part_a["position_3d"])
    pos_b = np.array(part_b["position_3d"])
    t_a = part_a["thickness_mm"]
    t_b = part_b["thickness_mm"]

    # Check A's vertices against B's slab
    # B's slab: dot(point - pos_b, normal_b) in [0, t_b]
    a_in_b = []
    for pt in np.vstack([face_a_bot, face_a_top]):
        d = np.dot(pt - pos_b, normal_b)
        if -tol < d < t_b + tol:
            a_in_b.append((pt, d))

    # Check B's vertices against A's slab
    b_in_a = []
    for pt in np.vstack([face_b_bot, face_b_top]):
        d = np.dot(pt - pos_a, normal_a)
        if -tol < d < t_a + tol:
            b_in_a.append((pt, d))

    return a_in_b, b_in_a


def main():
    if len(sys.argv) < 2:
        print("Usage: check_overlaps.py <design_json>")
        sys.exit(1)

    design_path = Path(sys.argv[1])
    with open(design_path) as f:
        design = json.load(f)

    parts = design["parts"]
    print(f"Loaded {len(parts)} parts\n")

    # Print part summary
    for p in parts:
        bbox_min, bbox_max, _ = part_to_3d_bbox(p)
        rot = p["rotation_3d"]
        R = euler_to_rotation_matrix(*rot)
        normal = R @ np.array([0.0, 0.0, 1.0])
        print(f"  {p['part_id']}: pos={np.array(p['position_3d'])}, "
              f"normal=[{normal[0]:.3f},{normal[1]:.3f},{normal[2]:.3f}], "
              f"t={p['thickness_mm']:.2f}mm")
        print(f"    bbox: [{bbox_min[0]:.1f},{bbox_min[1]:.1f},{bbox_min[2]:.1f}] "
              f"to [{bbox_max[0]:.1f},{bbox_max[1]:.1f},{bbox_max[2]:.1f}]")

    print(f"\n{'='*80}")
    print("Checking all {0} pairs for overlap...\n".format(len(parts) * (len(parts)-1) // 2))

    overlap_count = 0
    for i, j in combinations(range(len(parts)), 2):
        pa, pb = parts[i], parts[j]
        bbox_min_a, bbox_max_a, _ = part_to_3d_bbox(pa)
        bbox_min_b, bbox_max_b, _ = part_to_3d_bbox(pb)

        overlap_size = bbox_overlap(bbox_min_a, bbox_max_a, bbox_min_b, bbox_max_b)
        if overlap_size is not None:
            # AABB overlap detected — do detailed slab check
            a_in_b, b_in_a = check_slab_intersection(pa, pb, tol=0.5)

            # Filter to actual penetration (not just touching)
            a_penetrating = [(pt, d) for pt, d in a_in_b
                           if 0.5 < d < pa["thickness_mm"] - 0.5 or
                              0.5 < d < pb["thickness_mm"] - 0.5]
            b_penetrating = [(pt, d) for pt, d in b_in_a
                           if 0.5 < d < pb["thickness_mm"] - 0.5 or
                              0.5 < d < pa["thickness_mm"] - 0.5]

            # Get normals to understand relationship
            rot_a = pa["rotation_3d"]
            rot_b = pb["rotation_3d"]
            R_a = euler_to_rotation_matrix(*rot_a)
            R_b = euler_to_rotation_matrix(*rot_b)
            n_a = R_a @ np.array([0.0, 0.0, 1.0])
            n_b = R_b @ np.array([0.0, 0.0, 1.0])
            dot = abs(np.dot(n_a, n_b))

            relationship = "parallel" if dot > 0.95 else ("perpendicular" if dot < 0.05 else f"angled({dot:.3f})")

            print(f"BBOX OVERLAP: {pa['part_id']} <-> {pb['part_id']} ({relationship})")
            print(f"  overlap size: [{overlap_size[0]:.2f}, {overlap_size[1]:.2f}, {overlap_size[2]:.2f}] mm")
            print(f"  A vertices in B's slab: {len(a_in_b)} (penetrating: {len(a_penetrating)})")
            print(f"  B vertices in A's slab: {len(b_in_a)} (penetrating: {len(b_penetrating)})")

            if a_in_b:
                depths_a = [d for _, d in a_in_b]
                print(f"    A in B depths: min={min(depths_a):.3f}, max={max(depths_a):.3f} "
                      f"(B thickness={pb['thickness_mm']:.2f})")
            if b_in_a:
                depths_b = [d for _, d in b_in_a]
                print(f"    B in A depths: min={min(depths_b):.3f}, max={max(depths_b):.3f} "
                      f"(A thickness={pa['thickness_mm']:.2f})")

            if a_penetrating or b_penetrating:
                print(f"  *** REAL VOLUMETRIC OVERLAP ***")
                overlap_count += 1
            else:
                print(f"  (surface contact only — no material overlap)")
            print()

    print(f"{'='*80}")
    print(f"Total real overlaps: {overlap_count}")
    if overlap_count == 0:
        print("No volumetric overlaps detected.")
    else:
        print(f"WARNING: {overlap_count} pair(s) have material overlap!")


if __name__ == "__main__":
    main()
