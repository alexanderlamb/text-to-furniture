#!/usr/bin/env python3
"""Detailed overlap analysis: show exactly which parts overlap and why."""
import json
import sys
import numpy as np
from pathlib import Path

def euler_to_rotation_matrix(rx, ry, rz):
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx

def part_info(p):
    rot = p["rotation_3d"]
    R = euler_to_rotation_matrix(*rot)
    normal = R @ np.array([0.0, 0.0, 1.0])

    outline = np.array(p["outline_2d"])
    pts_2d = outline[:-1] if np.allclose(outline[0], outline[-1]) else outline
    pts_3d_bot = []
    pts_3d_top = []
    for pt in pts_2d:
        p3 = R @ np.array([pt[0], pt[1], 0.0]) + np.array(p["position_3d"])
        pts_3d_bot.append(p3)
        pts_3d_top.append(p3 + normal * p["thickness_mm"])
    return {
        "normal": normal,
        "R": R,
        "pts_bot": np.array(pts_3d_bot),
        "pts_top": np.array(pts_3d_top),
    }

def main():
    design_path = Path(sys.argv[1])
    with open(design_path) as f:
        design = json.load(f)

    parts = design["parts"]

    # Identify stacked groups
    print("=== Part Layout ===\n")
    for p in parts:
        info = part_info(p)
        n = info["normal"]
        all_pts = np.vstack([info["pts_bot"], info["pts_top"]])
        bbox_min = all_pts.min(axis=0)
        bbox_max = all_pts.max(axis=0)

        # Determine orientation
        if abs(n[0]) > 0.9:
            orient = "YZ-plane (vertical wall, x-facing)"
            slab_range = f"x=[{bbox_min[0]:.1f}, {bbox_max[0]:.1f}]"
            span = f"y=[{bbox_min[1]:.1f},{bbox_max[1]:.1f}] z=[{bbox_min[2]:.1f},{bbox_max[2]:.1f}]"
        elif abs(n[1]) > 0.9:
            orient = "XZ-plane (vertical wall, y-facing)"
            slab_range = f"y=[{bbox_min[1]:.1f}, {bbox_max[1]:.1f}]"
            span = f"x=[{bbox_min[0]:.1f},{bbox_max[0]:.1f}] z=[{bbox_min[2]:.1f},{bbox_max[2]:.1f}]"
        elif abs(n[2]) > 0.9:
            orient = "XY-plane (horizontal shelf, z-facing)"
            slab_range = f"z=[{bbox_min[2]:.1f}, {bbox_max[2]:.1f}]"
            span = f"x=[{bbox_min[0]:.1f},{bbox_max[0]:.1f}] y=[{bbox_min[1]:.1f},{bbox_max[1]:.1f}]"
        else:
            orient = "angled"
            slab_range = "?"
            span = "?"

        meta = p.get("metadata", {})
        stack_info = ""
        if meta.get("stack_layer_count", 1) > 1 or meta.get("stack_layer_index", 0) > 0:
            stack_info = f" [stack group={meta.get('stack_group_id','?')}, layer={meta.get('stack_layer_index',0)}/{meta.get('stack_layer_count',1)}]"

        print(f"  {p['part_id']}: {orient}")
        print(f"    slab: {slab_range} (t={p['thickness_mm']:.2f}mm)")
        print(f"    span: {span}")
        print(f"    cutouts: {len(p.get('cutouts_2d', []))}{stack_info}")
        print()

    # Now check the specific overlapping pairs
    print("=== Overlap Detail ===\n")

    overlapping = [
        (2, 5, "inner-left-wall vs bottom-shelf-layer2"),
        (2, 7, "inner-left-wall vs top-shelf-layer2"),
        (3, 5, "inner-right-wall vs bottom-shelf-layer2"),
        (3, 7, "inner-right-wall vs top-shelf-layer2"),
        (3, 8, "inner-right-wall vs middle-shelf"),
    ]

    for i, j, desc in overlapping:
        pa, pb = parts[i], parts[j]
        info_a, info_b = part_info(pa), part_info(pb)

        all_a = np.vstack([info_a["pts_bot"], info_a["pts_top"]])
        all_b = np.vstack([info_b["pts_bot"], info_b["pts_top"]])

        n_b = info_b["normal"]
        pos_b = np.array(pb["position_3d"])

        # Project A's vertices onto B's normal axis
        depths_a_in_b = np.dot(all_a - pos_b, n_b)

        print(f"  {pa['part_id']} vs {pb['part_id']} ({desc})")
        print(f"    {pb['part_id']} slab: d=0 to d={pb['thickness_mm']:.2f} along normal")
        print(f"    {pa['part_id']} vertex depths in {pb['part_id']}'s slab:")
        print(f"      min={depths_a_in_b.min():.3f}, max={depths_a_in_b.max():.3f}")
        inside = (depths_a_in_b > 0.5) & (depths_a_in_b < pb['thickness_mm'] - 0.5)
        print(f"      vertices inside slab: {inside.sum()} / {len(depths_a_in_b)}")
        if inside.sum() > 0:
            pen_depths = depths_a_in_b[inside]
            print(f"      penetration depths: {pen_depths}")
        print()


if __name__ == "__main__":
    main()
