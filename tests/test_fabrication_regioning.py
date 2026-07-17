from __future__ import annotations

from pathlib import Path

import pytest
import trimesh

from fabrication.context import build_fabrication_context
from fabrication.contracts import FabricationConfig, HybridRegion
from fabrication.regioning import (
    region_payload,
    regioning_debug_payload,
    regionize_mesh,
)


def _export_mesh(mesh: trimesh.Trimesh, tmp_path: Path, name: str) -> str:
    path = tmp_path / f"{name}.stl"
    mesh.export(path)
    return str(path)


def _signature(regions: list[HybridRegion]):
    return [
        (
            region.region_id,
            region.kind,
            tuple(round(value, 6) for value in region.aabb_min),
            tuple(round(value, 6) for value in region.aabb_max),
            round(region.volume_mm3, 6),
            round(region.surface_area_mm2, 6),
            region.metadata["occupancy_proxy"]["method"],
        )
        for region in regions
    ]


def test_regionize_mesh_generates_deterministic_hybrid_regions(box_mesh_file: str):
    context = build_fabrication_context(
        FabricationConfig(mesh_path=box_mesh_file, auto_scale=False)
    )

    regions_a = regionize_mesh(context, max_regions=4)
    regions_b = regionize_mesh(context, max_regions=4)

    assert _signature(regions_a) == _signature(regions_b)
    assert len(regions_a) == 4
    assert all(isinstance(region, HybridRegion) for region in regions_a)
    assert {region.kind for region in regions_a} == {"blocky_band"}
    assert sum(region.volume_mm3 for region in regions_a) == pytest.approx(
        context.mesh_volume_mm3
    )

    for index, region in enumerate(regions_a):
        assert region.region_id == f"region_x_band_{index:02d}"
        assert all(lo <= hi for lo, hi in zip(region.aabb_min, region.aabb_max))
        assert region.volume_mm3 > 0.0
        assert region.surface_area_mm2 > 0.0
        assert region.metadata["algorithm"] == "aabb_observed_axis_bands_v1"
        assert region.metadata["band_index"] == index
        assert region.metadata["band_count"] == 4
        assert region.metadata["split_axis"] == "x"
        assert region.metadata["classification_reasons"]
        assert region.metadata["occupancy_proxy"]["sample_count"] > 0
        assert 0.0 <= region.metadata["occupancy_proxy"]["fill_ratio"] <= 1.0
        assert region.metadata["debug"]["mesh_watertight"] is True


def test_regionize_mesh_respects_max_regions_and_classifies_flat_mesh(tmp_path: Path):
    mesh = trimesh.creation.box(extents=[160.0, 90.0, 6.0])
    mesh.apply_translation([0.0, 0.0, 3.0])
    context = build_fabrication_context(
        FabricationConfig(
            mesh_path=_export_mesh(mesh, tmp_path, "flat_panel"),
            auto_scale=False,
            preferred_thickness_mm=6.35,
        )
    )

    regions = regionize_mesh(context, max_regions=3)

    assert len(regions) == 3
    assert {region.kind for region in regions} == {"flat_band"}
    assert all(region.metadata["split_axis"] == "x" for region in regions)
    assert all(
        "mesh_has_thin_axis" in region.metadata["classification_reasons"]
        for region in regions
    )


def test_regionize_mesh_uses_component_regions_for_disconnected_geometry(
    tmp_path: Path,
):
    left = trimesh.creation.box(extents=[40.0, 20.0, 20.0])
    left.apply_translation([-80.0, -45.0, 10.0])
    right = trimesh.creation.box(extents=[40.0, 20.0, 20.0])
    right.apply_translation([80.0, 45.0, 10.0])
    mesh = trimesh.util.concatenate([left, right])
    context = build_fabrication_context(
        FabricationConfig(
            mesh_path=_export_mesh(mesh, tmp_path, "separated_blocks"),
            auto_scale=False,
        )
    )

    regions = regionize_mesh(context, max_regions=2)

    assert len(regions) == 2
    mesh_y_extent = context.mesh.bounds[1][1] - context.mesh.bounds[0][1]
    assert [region.region_id for region in regions] == [
        "region_component_00",
        "region_component_01",
    ]
    for region in regions:
        y_extent = region.aabb_max[1] - region.aabb_min[1]
        assert y_extent < mesh_y_extent * 0.5
        assert region.kind == "blocky_band"
        assert region.metadata["algorithm"] == "connected_component_regions_v0"
        assert region.metadata["bounds_method"] == "connected_component_aabb"
        assert region.metadata["component_count"] == 1
        assert region.metadata["total_component_count"] == 2
        assert region.metadata["observed_point_count"] > 0


@pytest.mark.filterwarnings("ignore:invalid value encountered in divide:RuntimeWarning")
def test_regionize_mesh_tightens_connected_band_bounds_to_observed_geometry(
    tmp_path: Path,
):
    vertices = []
    faces = []

    def add_quad(points):
        start = len(vertices)
        vertices.extend(points)
        faces.extend(
            [
                [start, start + 1, start + 2],
                [start, start + 2, start + 3],
            ]
        )

    add_quad([[-100, -55, 0], [-50, -55, 0], [-50, -5, 0], [-100, -5, 0]])
    add_quad([[-100, -5, 0], [-50, -5, 0], [-50, 5, 0], [-100, 5, 0]])
    add_quad([[-50, -5, 0], [50, -5, 0], [50, 5, 0], [-50, 5, 0]])
    add_quad([[50, -5, 0], [100, -5, 0], [100, 5, 0], [50, 5, 0]])
    add_quad([[50, 5, 0], [100, 5, 0], [100, 55, 0], [50, 55, 0]])
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    mesh.merge_vertices()
    context = build_fabrication_context(
        FabricationConfig(
            mesh_path=_export_mesh(mesh, tmp_path, "connected_lobes"),
            auto_scale=False,
        )
    )

    regions = regionize_mesh(context, max_regions=2)

    assert len(regions) == 2
    mesh_y_extent = context.mesh.bounds[1][1] - context.mesh.bounds[0][1]
    assert {region.metadata["algorithm"] for region in regions} == {
        "aabb_observed_axis_bands_v1"
    }
    for region in regions:
        y_extent = region.aabb_max[1] - region.aabb_min[1]
        assert y_extent < mesh_y_extent * 0.75
        assert region.metadata["bounds_method"] == "observed_geometry_transverse_aabb"
        assert region.metadata["observed_point_count"] > 0
        assert 0.0 <= region.metadata["transverse_shrink_ratio"] <= 1.0
        assert region.metadata["coarse_aabb_max"][1] - region.metadata[
            "coarse_aabb_min"
        ][1] == pytest.approx(mesh_y_extent)


def test_region_payload_helpers_include_region_debug(box_mesh_file: str):
    context = build_fabrication_context(
        FabricationConfig(mesh_path=box_mesh_file, auto_scale=False)
    )
    regions = regionize_mesh(context, max_regions=2)

    payload = region_payload(regions[0])
    debug_payload = regioning_debug_payload(context, regions)

    assert payload["region_id"] == regions[0].region_id
    assert payload["extents_mm"] == [100.0, 140.0, 120.0]
    assert payload["metadata"]["debug"]["region_extents_mm"] == [
        100.0,
        140.0,
        120.0,
    ]
    assert debug_payload["algorithm"] == "aabb_observed_axis_bands_v1"
    assert debug_payload["region_count"] == 2
    assert debug_payload["mesh"]["volume_mm3"] == pytest.approx(context.mesh_volume_mm3)
    assert len(debug_payload["regions"]) == 2


def test_regionize_mesh_rejects_invalid_region_count(box_mesh_file: str):
    context = build_fabrication_context(
        FabricationConfig(mesh_path=box_mesh_file, auto_scale=False)
    )

    with pytest.raises(ValueError, match="max_regions"):
        regionize_mesh(context, max_regions=0)
