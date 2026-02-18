from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from shapely.geometry import Polygon

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

sys.path.insert(0, str(ROOT / "src"))

from geometry_primitives import PartProfile2D
from step3_first_principles import ManufacturingPart, RegionType
from src_v2.spatial_capsule import (
    emit_spatial_capsule_from_snapshot_payload,
    emit_spatial_capsule_from_manufacturing_parts,
    write_spatial_capsule_json,
)


def _make_part(
    part_id: str,
    position_xyz: tuple[float, float, float],
    size_xy: tuple[float, float] = (100.0, 40.0),
) -> ManufacturingPart:
    w, h = size_xy
    outline = Polygon([(0.0, 0.0), (w, 0.0), (w, h), (0.0, h)])
    profile = PartProfile2D(
        outline=outline,
        material_key="plywood_baltic_birch",
        thickness_mm=6.35,
        basis_u=np.array([1.0, 0.0, 0.0]),
        basis_v=np.array([0.0, 1.0, 0.0]),
        origin_3d=np.array([0.0, 0.0, 0.0]),
    )
    return ManufacturingPart(
        part_id=part_id,
        material_key="plywood_baltic_birch",
        thickness_mm=6.35,
        profile=profile,
        region_type=RegionType.PLANAR_CUT,
        position_3d=np.array(position_xyz, dtype=float),
        rotation_3d=np.array([0.0, 0.0, 0.0]),
        source_area_mm2=float(w * h),
        source_faces=[0, 1],
        metadata={},
    )


def test_emit_spatial_capsule_contains_parts_and_relations():
    part_a = _make_part("part_a", (0.0, 0.0, 0.0))
    part_b = _make_part("part_b", (300.0, 0.0, 0.0))

    capsule = emit_spatial_capsule_from_manufacturing_parts([part_a, part_b])
    payload = capsule.to_dict()

    assert payload["schema_version"] == "v2.spatial_capsule.1"
    assert len(payload["parts"]) == 2
    assert len(payload["relations"]) == 1
    assert payload["relations"][0]["class"] == "disjoint"


def test_emit_spatial_capsule_detects_overlap_and_jointed():
    part_a = _make_part("part_a", (0.0, 0.0, 0.0))
    part_b = _make_part("part_b", (30.0, 0.0, 0.0))  # overlaps part_a
    part_c = _make_part("part_c", (100.2, 0.0, 0.0))  # near-contact with part_a

    capsule = emit_spatial_capsule_from_manufacturing_parts(
        [part_a, part_b, part_c],
        joint_pairs=[("part_a", "part_c")],
        contact_tolerance_mm=0.5,
    )
    rel = {(r.part_a, r.part_b): r for r in capsule.relations}
    rel_by_sorted = {tuple(sorted((a, b))): r for (a, b), r in rel.items()}

    assert rel_by_sorted[("part_a", "part_b")].relation_class == "overlapping"
    assert rel_by_sorted[("part_a", "part_b")].penetration_mm > 0.0
    assert rel_by_sorted[("part_a", "part_c")].relation_class == "jointed"


def test_write_spatial_capsule_json(tmp_path: Path):
    part_a = _make_part("part_a", (0.0, 0.0, 0.0))
    capsule = emit_spatial_capsule_from_manufacturing_parts([part_a])
    out = write_spatial_capsule_json(tmp_path / "capsule.json", capsule)

    assert out.exists()
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "v2.spatial_capsule.1"
    assert payload["units"] == "mm"
    assert len(payload["parts"]) == 1


def test_emit_spatial_capsule_from_snapshot_payload():
    snapshot = {
        "phase_label": "Final",
        "phase_index": 4,
        "parts": [
            {
                "part_id": "part_a",
                "thickness_mm": 6.35,
                "outline_2d": [[0.0, 0.0], [100.0, 0.0], [100.0, 40.0], [0.0, 40.0]],
                "cutouts_2d": [],
                "position_3d": [0.0, 0.0, 0.0],
                "origin_3d": [0.0, 0.0, 0.0],
                "rotation_3d": [0.0, 0.0, 0.0],
            },
            {
                "part_id": "part_b",
                "thickness_mm": 6.35,
                "outline_2d": [[0.0, 0.0], [100.0, 0.0], [100.0, 40.0], [0.0, 40.0]],
                "cutouts_2d": [],
                "position_3d": [100.2, 0.0, 0.0],
                "origin_3d": [100.2, 0.0, 0.0],
                "rotation_3d": [0.0, 0.0, 0.0],
            },
        ],
        "joints": [{"part_a": "part_a", "part_b": "part_b"}],
    }

    capsule = emit_spatial_capsule_from_snapshot_payload(
        snapshot, contact_tolerance_mm=0.5
    )
    assert len(capsule.parts) == 2
    assert len(capsule.relations) == 1
    assert capsule.relations[0].relation_class == "jointed"
