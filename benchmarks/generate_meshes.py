#!/usr/bin/env python3
"""Generate benchmark meshes of increasing complexity.

All meshes are single connected watertight volumes suitable for
flat-pack decomposition testing. Run once to populate benchmarks/meshes/.
"""

from pathlib import Path

import numpy as np
import trimesh
from trimesh.creation import box, cylinder

OUT = Path(__file__).parent / "meshes"
OUT.mkdir(exist_ok=True)


def save(mesh: trimesh.Trimesh, name: str) -> None:
    assert mesh.is_watertight, f"{name} is not watertight"
    assert mesh.is_volume, f"{name} is not a volume"
    path = OUT / f"{name}.stl"
    mesh.export(str(path))
    print(f"  {name}: {len(mesh.vertices)} verts, {len(mesh.faces)} faces, vol={mesh.volume:.0f} mm³ → {path.name}")


def make_box():
    """1. Simple rectangular box — the trivial case (6 planes)."""
    return box(extents=[400, 300, 250])


def make_tall_cabinet():
    """2. Tall narrow box — tests high aspect ratio panels."""
    return box(extents=[300, 150, 700])


def make_l_bracket():
    """3. L-shaped bracket — two perpendicular slabs joined."""
    a = box(extents=[300, 200, 20])  # horizontal
    b = box(extents=[20, 200, 250])  # vertical
    b.apply_translation([-140, 0, 135])
    return trimesh.boolean.union([a, b])


def make_t_beam():
    """4. T-shaped cross section extruded — three perpendicular planes."""
    flange = box(extents=[300, 150, 20])  # top
    web = box(extents=[20, 150, 200])  # vertical
    web.apply_translation([0, 0, -110])
    return trimesh.boolean.union([flange, web])


def make_u_channel():
    """5. U-shaped channel — open top, three inner walls."""
    outer = box(extents=[300, 200, 250])
    cutout = box(extents=[260, 200, 230])
    cutout.apply_translation([0, 0, 20])
    return trimesh.boolean.difference([outer, cutout])


def make_step_stool():
    """6. Two-step staircase — multiple horizontal/vertical planes."""
    base = box(extents=[300, 250, 20])  # bottom
    step1 = box(extents=[300, 125, 120])
    step1.apply_translation([0, 62.5, 70])
    step2 = box(extents=[300, 125, 60])
    step2.apply_translation([0, -62.5, 40])
    return trimesh.boolean.union([base, step1, step2])


def make_h_beam():
    """7. H cross section — parallel flanges + connecting web."""
    top = box(extents=[200, 150, 20])
    top.apply_translation([0, 0, 140])
    bottom = box(extents=[200, 150, 20])
    bottom.apply_translation([0, 0, -140])
    web = box(extents=[20, 150, 280])
    return trimesh.boolean.union([top, bottom, web])


def make_shelf_unit():
    """8. Open shelf unit — two sides + three shelves, single volume."""
    left = box(extents=[20, 200, 500])
    left.apply_translation([-190, 0, 0])
    right = box(extents=[20, 200, 500])
    right.apply_translation([190, 0, 0])
    bottom = box(extents=[400, 200, 20])
    bottom.apply_translation([0, 0, -240])
    mid = box(extents=[400, 200, 20])
    top = box(extents=[400, 200, 20])
    top.apply_translation([0, 0, 240])
    return trimesh.boolean.union([left, right, bottom, mid, top])


def make_desk():
    """9. Simple desk — top surface + two side panels + back panel."""
    top_panel = box(extents=[600, 350, 20])
    top_panel.apply_translation([0, 0, 340])
    left_side = box(extents=[20, 350, 330])
    left_side.apply_translation([-290, 0, 175])
    right_side = box(extents=[20, 350, 330])
    right_side.apply_translation([290, 0, 175])
    back = box(extents=[600, 20, 330])
    back.apply_translation([0, 165, 175])
    return trimesh.boolean.union([top_panel, left_side, right_side, back])


def make_table_with_stretchers():
    """10. Table with 4 legs + top + side stretchers — most complex."""
    top = box(extents=[500, 350, 20])
    top.apply_translation([0, 0, 400])
    legs = []
    for sx, sy in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
        leg = box(extents=[30, 30, 390])
        leg.apply_translation([sx * 220, sy * 145, 195])
        legs.append(leg)
    # Side stretchers connecting legs
    s1 = box(extents=[20, 290, 30])
    s1.apply_translation([-220, 0, 80])
    s2 = box(extents=[20, 290, 30])
    s2.apply_translation([220, 0, 80])
    return trimesh.boolean.union([top] + legs + [s1, s2])


def make_v_bracket():
    """11. V-bracket — two plates meeting at 60 degrees, non-perpendicular join."""
    angle = np.radians(30)
    plate = box(extents=[20, 200, 250])

    p1 = plate.copy()
    rot1 = trimesh.transformations.rotation_matrix(angle, [0, 1, 0])
    p1.apply_transform(rot1)
    p1.apply_translation([60, 0, 100])

    p2 = plate.copy()
    rot2 = trimesh.transformations.rotation_matrix(-angle, [0, 1, 0])
    p2.apply_transform(rot2)
    p2.apply_translation([-60, 0, 100])

    base = box(extents=[160, 200, 20])
    return trimesh.boolean.union([p1, p2, base])


def make_angled_flange():
    """12. Compound-angle flange — plate with a flange tilted in two axes."""
    base = box(extents=[300, 200, 20])

    flange = box(extents=[20, 200, 150])
    rx = trimesh.transformations.rotation_matrix(np.radians(25), [1, 0, 0])
    ry = trimesh.transformations.rotation_matrix(np.radians(15), [0, 1, 0])
    flange.apply_transform(rx @ ry)
    flange.apply_translation([-140, 20, 80])

    return trimesh.boolean.union([base, flange])


def make_trapezoidal_tray():
    """13. Trapezoidal tray — bottom plate + 4 walls angled inward at 12 deg."""
    bottom = box(extents=[300, 200, 15])
    wall_h = 120
    wall_t = 15
    tilt = np.radians(12)

    front = box(extents=[300, wall_t, wall_h])
    front.apply_transform(trimesh.transformations.rotation_matrix(tilt, [1, 0, 0]))
    front.apply_translation([0, -90, 55])

    back = box(extents=[300, wall_t, wall_h])
    back.apply_transform(trimesh.transformations.rotation_matrix(-tilt, [1, 0, 0]))
    back.apply_translation([0, 90, 55])

    left = box(extents=[wall_t, 200, wall_h])
    left.apply_transform(trimesh.transformations.rotation_matrix(-tilt, [0, 1, 0]))
    left.apply_translation([-140, 0, 55])

    right = box(extents=[wall_t, 200, wall_h])
    right.apply_transform(trimesh.transformations.rotation_matrix(tilt, [0, 1, 0]))
    right.apply_translation([140, 0, 55])

    return trimesh.boolean.union([bottom, front, back, left, right])


GENERATORS = [
    ("01_box", make_box),
    ("02_tall_cabinet", make_tall_cabinet),
    ("03_l_bracket", make_l_bracket),
    ("04_t_beam", make_t_beam),
    ("05_u_channel", make_u_channel),
    ("06_step_stool", make_step_stool),
    ("07_h_beam", make_h_beam),
    ("08_shelf_unit", make_shelf_unit),
    ("09_desk", make_desk),
    ("10_table_with_stretchers", make_table_with_stretchers),
    ("11_v_bracket", make_v_bracket),
    ("12_angled_flange", make_angled_flange),
    ("13_trapezoidal_tray", make_trapezoidal_tray),
]


if __name__ == "__main__":
    print(f"Generating {len(GENERATORS)} benchmark meshes → {OUT}/")
    for name, gen_fn in GENERATORS:
        mesh = gen_fn()
        save(mesh, name)
    print("Done.")
