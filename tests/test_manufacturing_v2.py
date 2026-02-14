"""Tests for manufacturing_decomposer_v2.py — end-to-end v2 pipeline."""
import os
import tempfile

import numpy as np
import pytest
import trimesh

from manufacturing_decomposer_v2 import (
    ManufacturingDecompositionConfigV2,
    decompose_manufacturing_v2,
    _classify_slab_component,
    _empty_result,
)
from slab_candidates import Slab3D, Slab3DConfig
from slab_selector import SlabSelectionConfig
from furniture import ComponentType, JointType


@pytest.fixture
def box_mesh_file(box_mesh):
    """Write a box mesh to a temp file."""
    with tempfile.NamedTemporaryFile(suffix=".stl", delete=False) as f:
        box_mesh.export(f.name)
        yield f.name
    os.unlink(f.name)


@pytest.fixture
def table_mesh_file(table_mesh):
    """Write a table mesh to a temp file."""
    with tempfile.NamedTemporaryFile(suffix=".stl", delete=False) as f:
        table_mesh.export(f.name)
        yield f.name
    os.unlink(f.name)


class TestDecomposeV2:
    def test_box_mesh_end_to_end(self, box_mesh_file):
        """End-to-end v2 decomposition on a box mesh."""
        config = ManufacturingDecompositionConfigV2(
            slab_candidates=Slab3DConfig(min_slab_area_mm2=50.0),
            slab_selection=SlabSelectionConfig(
                coverage_target=0.30,
                max_slabs=8,
                min_coverage_contribution=0.01,
                dfm_check=False,
            ),
            auto_scale=False,
        )
        result = decompose_manufacturing_v2(box_mesh_file, config)

        assert result.design is not None
        assert len(result.design.components) > 0
        assert result.score is not None
        assert result.score.part_count > 0

        # Validate design
        is_valid, errors = result.design.validate()
        assert is_valid, f"Design validation failed: {errors}"

    def test_table_mesh_end_to_end(self, table_mesh_file):
        """End-to-end v2 decomposition on a table mesh."""
        config = ManufacturingDecompositionConfigV2(
            slab_candidates=Slab3DConfig(min_slab_area_mm2=50.0),
            slab_selection=SlabSelectionConfig(
                coverage_target=0.20,
                max_slabs=10,
                min_coverage_contribution=0.01,
                dfm_check=False,
            ),
            auto_scale=False,
        )
        result = decompose_manufacturing_v2(table_mesh_file, config)

        assert result.design is not None
        assert len(result.design.components) > 0

    def test_box_mesh_volume_fill_mode(self, box_mesh_file):
        """v2 should support volume_fill objective mode."""
        config = ManufacturingDecompositionConfigV2(
            slab_candidates=Slab3DConfig(min_slab_area_mm2=50.0),
            slab_selection=SlabSelectionConfig(
                objective_mode="volume_fill",
                target_volume_fill=0.20,
                max_slabs=8,
                min_volume_contribution=0.001,
                plane_penalty_weight=0.0,
                dfm_check=False,
            ),
            auto_scale=False,
        )
        result = decompose_manufacturing_v2(box_mesh_file, config)
        assert len(result.design.components) > 0
        debug = result.design.metadata.get("decomposition_debug", {})
        assert debug.get("objective_mode") == "volume_fill"
        assert debug.get("volume_fill_fraction", 0.0) > 0.0

    def test_result_has_parts_dict(self, box_mesh_file):
        """Result should contain a parts dict with PartProfile2D entries."""
        config = ManufacturingDecompositionConfigV2(
            slab_candidates=Slab3DConfig(min_slab_area_mm2=50.0),
            slab_selection=SlabSelectionConfig(
                coverage_target=0.30,
                max_slabs=5,
                min_coverage_contribution=0.01,
                dfm_check=False,
            ),
            auto_scale=False,
        )
        result = decompose_manufacturing_v2(box_mesh_file, config)

        assert len(result.parts) > 0
        for name, profile in result.parts.items():
            assert not profile.outline.is_empty
            assert profile.thickness_mm > 0

    def test_result_has_assembly_steps(self, box_mesh_file):
        """Result should contain assembly steps."""
        config = ManufacturingDecompositionConfigV2(
            slab_candidates=Slab3DConfig(min_slab_area_mm2=50.0),
            slab_selection=SlabSelectionConfig(
                coverage_target=0.30,
                max_slabs=5,
                min_coverage_contribution=0.01,
                dfm_check=False,
            ),
            auto_scale=False,
        )
        result = decompose_manufacturing_v2(box_mesh_file, config)
        # Assembly steps should match component count
        assert len(result.assembly_steps) == len(result.design.components)

    def test_manufacturing_json_export(self, box_mesh_file):
        """to_manufacturing_json should work on v2 results."""
        config = ManufacturingDecompositionConfigV2(
            slab_candidates=Slab3DConfig(min_slab_area_mm2=50.0),
            slab_selection=SlabSelectionConfig(
                coverage_target=0.30,
                max_slabs=5,
                min_coverage_contribution=0.01,
                dfm_check=False,
            ),
            auto_scale=False,
        )
        result = decompose_manufacturing_v2(box_mesh_file, config)
        json_data = result.to_manufacturing_json()
        assert "parts" in json_data
        assert "metrics" in json_data
        assert len(json_data["parts"]) == len(result.design.components)


class TestDecompositionDebugTelemetry:
    """Verify decomposition_debug metadata is populated."""

    def test_debug_metadata_populated(self, box_mesh_file):
        """design.metadata should contain decomposition_debug with key fields."""
        config = ManufacturingDecompositionConfigV2(
            slab_candidates=Slab3DConfig(min_slab_area_mm2=50.0),
            slab_selection=SlabSelectionConfig(
                coverage_target=0.30,
                max_slabs=5,
                min_coverage_contribution=0.01,
                dfm_check=False,
            ),
            auto_scale=False,
        )
        result = decompose_manufacturing_v2(box_mesh_file, config)

        debug = result.design.metadata.get("decomposition_debug", {})
        assert "mesh_bounds_mm" in debug
        assert "mesh_extents_mm" in debug
        assert "candidate_count" in debug
        assert debug["candidate_count"] > 0
        assert "selected_slab_count" in debug
        assert debug["selected_slab_count"] > 0
        assert "coverage_fraction" in debug
        assert debug["coverage_fraction"] > 0
        assert "voxel_resolution_mm" in debug
        assert "intersection_count" in debug
        assert "joint_count" in debug
        assert "dfm_error_count" in debug
        assert "dfm_warning_count" in debug
        assert "pose_alignment_fix" in debug

    def test_source_histogram_populated(self, box_mesh_file):
        """selected_source_histogram should be present and non-empty."""
        config = ManufacturingDecompositionConfigV2(
            slab_candidates=Slab3DConfig(min_slab_area_mm2=50.0),
            slab_selection=SlabSelectionConfig(
                coverage_target=0.30,
                max_slabs=5,
                min_coverage_contribution=0.01,
                dfm_check=False,
            ),
            auto_scale=False,
        )
        result = decompose_manufacturing_v2(box_mesh_file, config)
        debug = result.design.metadata.get("decomposition_debug", {})

        assert "selected_source_histogram" in debug
        histogram = debug["selected_source_histogram"]
        assert isinstance(histogram, dict)
        assert len(histogram) > 0
        assert sum(histogram.values()) == debug["selected_slab_count"]

    def test_fit_ratios_populated(self, box_mesh_file):
        """fit_ratios should be present when components exist."""
        config = ManufacturingDecompositionConfigV2(
            slab_candidates=Slab3DConfig(min_slab_area_mm2=50.0),
            slab_selection=SlabSelectionConfig(
                coverage_target=0.30,
                max_slabs=5,
                min_coverage_contribution=0.01,
                dfm_check=False,
            ),
            auto_scale=False,
        )
        result = decompose_manufacturing_v2(box_mesh_file, config)
        debug = result.design.metadata.get("decomposition_debug", {})

        assert "fit_ratios" in debug
        fit_ratios = debug["fit_ratios"]
        assert len(fit_ratios) == 3
        # Box mesh: fit ratios should be within 1.15x
        for i, ratio in enumerate(fit_ratios):
            assert ratio <= 1.15, (
                f"Fit ratio axis {i} = {ratio:.3f} exceeds 1.15"
            )


class TestPartsAABBRegression:
    """Verify generated parts stay within reasonable bounds of mesh."""

    def test_box_parts_within_mesh_extents(self, box_mesh_file):
        """Part positions should stay within 1.5x of mesh extents."""
        config = ManufacturingDecompositionConfigV2(
            slab_candidates=Slab3DConfig(min_slab_area_mm2=50.0),
            slab_selection=SlabSelectionConfig(
                coverage_target=0.30,
                max_slabs=8,
                min_coverage_contribution=0.01,
                dfm_check=False,
            ),
            auto_scale=False,
        )
        result = decompose_manufacturing_v2(box_mesh_file, config)
        assert len(result.design.components) > 0

        # Get mesh bounds from debug telemetry
        debug = result.design.metadata.get("decomposition_debug", {})
        mesh_bounds = debug.get("mesh_bounds_mm")
        assert mesh_bounds is not None

        mesh_min = np.array(mesh_bounds[0])
        mesh_max = np.array(mesh_bounds[1])
        mesh_extents = mesh_max - mesh_min
        tolerance_factor = 1.5

        for comp in result.design.components:
            dims = comp.get_dimensions()
            max_dim = max(dims)
            for ax in range(3):
                # Part position should be within mesh bounds + tolerance
                margin = mesh_extents[ax] * (tolerance_factor - 1.0) + max_dim
                assert comp.position[ax] >= mesh_min[ax] - margin, (
                    f"{comp.name} pos[{ax}]={comp.position[ax]:.1f} "
                    f"too far below mesh min {mesh_min[ax]:.1f}"
                )
                assert comp.position[ax] <= mesh_max[ax] + margin, (
                    f"{comp.name} pos[{ax}]={comp.position[ax]:.1f} "
                    f"too far above mesh max {mesh_max[ax]:.1f}"
                )


class TestBasisToRotationRoundTrip:
    """Verify _basis_to_rotation Euler angles reconstruct the original matrix."""

    @staticmethod
    def _reconstruct(rx, ry, rz):
        cx, sx = np.cos(rx), np.sin(rx)
        cy, sy = np.cos(ry), np.sin(ry)
        cz, sz = np.cos(rz), np.sin(rz)
        Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
        Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
        Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
        return Rz @ Ry @ Rx

    @pytest.mark.parametrize("normal", [
        [0, 0, 1], [0, 0, -1],
        [1, 0, 0], [-1, 0, 0],
        [0, 1, 0], [0, -1, 0],
        [0.37, 0.91, -0.18],
        [0.707, 0, 0.707],
    ])
    def test_round_trip(self, normal):
        from geometry_primitives import _make_2d_basis
        from manufacturing_decomposer_v2 import _basis_to_rotation

        n = np.array(normal, dtype=float)
        n /= np.linalg.norm(n)
        u, v = _make_2d_basis(n)
        R_orig = np.column_stack([u, v, n])

        euler = _basis_to_rotation(u, v, n)
        R_rec = self._reconstruct(*euler)

        np.testing.assert_allclose(R_rec, R_orig, atol=1e-10,
            err_msg=f"Round-trip failed for normal={normal}")


class TestEmptyResult:
    def test_empty_result(self):
        result = _empty_result()
        assert result.design is not None
        assert len(result.parts) == 0
        assert result.score.overall_score == 0.0

    def test_empty_result_with_debug_telemetry(self, box_mesh):
        """_empty_result with mesh/config/failure_reason populates debug."""
        config = ManufacturingDecompositionConfigV2()
        result = _empty_result(
            mesh=box_mesh,
            candidates=[],
            config=config,
            failure_reason="no_candidates_generated",
        )
        debug = result.design.metadata.get("decomposition_debug", {})
        assert debug != {}, "decomposition_debug should not be empty"
        assert debug["failure_reason"] == "no_candidates_generated"
        assert "mesh_bounds_mm" in debug
        assert "mesh_extents_mm" in debug
        assert "mesh_faces" in debug
        assert debug["candidate_count"] == 0
        assert "objective_mode" in debug
        assert "mesh_is_watertight" in debug

    def test_empty_result_legacy_no_args(self):
        """Legacy _empty_result() with no args still works."""
        result = _empty_result()
        debug = result.design.metadata.get("decomposition_debug", {})
        # Should be empty dict when no args provided
        assert debug == {}

    def test_empty_result_with_selection(self, box_mesh):
        """_empty_result with selection populates selection debug."""
        from slab_selector import SlabSelection
        config = ManufacturingDecompositionConfigV2()
        selection = SlabSelection(
            selected_slabs=[],
            coverage_fraction=0.0,
            uncovered_points=np.array([]).reshape(0, 3),
            stop_reason="no_dfm_valid_candidates",
            dfm_rejected_count=5,
            candidate_count_after_dfm=0,
        )
        result = _empty_result(
            mesh=box_mesh,
            candidates=[None] * 5,  # dummy list
            config=config,
            selection=selection,
            failure_reason="no_slabs_selected",
        )
        debug = result.design.metadata.get("decomposition_debug", {})
        assert debug["failure_reason"] == "no_slabs_selected"
        assert debug["candidate_count"] == 5
        assert debug["selection_stop_reason"] == "no_dfm_valid_candidates"
        assert debug["selection_dfm_rejected_count"] == 5


class TestComponentClassification:
    def test_horizontal_slab_is_shelf(self):
        from geometry_primitives import _make_2d_basis
        from shapely.geometry import Polygon

        n = np.array([0, 0, 1.0])
        u, v = _make_2d_basis(n)
        slab = Slab3D(
            normal=n, origin=np.array([0, 0, 500.0]),
            thickness_mm=6.35, width_mm=400, height_mm=300,
            basis_u=u, basis_v=v,
            outline_2d=Polygon([(-200, -150), (200, -150), (200, 150), (-200, 150)]),
            material_key="plywood_baltic_birch", source="test",
        )
        bounds = np.array([[0, 0, 0], [0, 0, 750]])
        comp_type = _classify_slab_component(slab, 0, [slab], [], bounds)
        assert comp_type == ComponentType.SHELF

    def test_vertical_narrow_slab_is_leg(self):
        from geometry_primitives import _make_2d_basis
        from shapely.geometry import Polygon

        n = np.array([1.0, 0, 0])
        u, v = _make_2d_basis(n)
        slab = Slab3D(
            normal=n, origin=np.array([0, 0, 375.0]),
            thickness_mm=6.35, width_mm=40, height_mm=700,
            basis_u=u, basis_v=v,
            outline_2d=Polygon([(-20, -350), (20, -350), (20, 350), (-20, 350)]),
            material_key="plywood_baltic_birch", source="test",
        )
        bounds = np.array([[0, 0, 0], [0, 0, 750]])
        comp_type = _classify_slab_component(slab, 0, [slab], [], bounds)
        assert comp_type == ComponentType.LEG

    def test_vertical_wide_slab_is_panel(self):
        from geometry_primitives import _make_2d_basis
        from shapely.geometry import Polygon

        n = np.array([1.0, 0, 0])
        u, v = _make_2d_basis(n)
        slab = Slab3D(
            normal=n, origin=np.array([0, 0, 375.0]),
            thickness_mm=6.35, width_mm=400, height_mm=300,
            basis_u=u, basis_v=v,
            outline_2d=Polygon([(-200, -150), (200, -150), (200, 150), (-200, 150)]),
            material_key="plywood_baltic_birch", source="test",
        )
        bounds = np.array([[0, 0, 0], [0, 0, 750]])
        comp_type = _classify_slab_component(slab, 0, [slab], [], bounds)
        assert comp_type == ComponentType.PANEL


class TestLoadMeshViewerConsistency:
    """Verify load_mesh and normalize_mesh_for_viewer produce matching proportions."""

    def test_load_mesh_matches_viewer_proportions(self, tmp_path):
        """load_mesh and normalize_mesh_for_viewer should agree on axis ratios."""
        from mesh_decomposer import load_mesh, DecompositionConfig
        from ui.scene_helpers import normalize_mesh_for_viewer

        # Create a box with non-uniform extents, wrap in Scene with a rotation
        box = trimesh.creation.box(extents=[200, 100, 300])
        scene = trimesh.Scene()
        rot = trimesh.transformations.rotation_matrix(0.3, [0, 1, 0])
        scene.add_geometry(box, transform=rot)
        glb_path = str(tmp_path / "test.glb")
        scene.export(glb_path)

        decomp = load_mesh(glb_path, DecompositionConfig(auto_scale=False))
        viewer_path = normalize_mesh_for_viewer(glb_path, str(tmp_path))
        viewer = trimesh.load(viewer_path)
        if isinstance(viewer, trimesh.Scene):
            viewer = viewer.to_mesh()

        d_ext = decomp.bounds[1] - decomp.bounds[0]
        v_ext = viewer.bounds[1] - viewer.bounds[0]

        # After Z→Y mapping: d_ext[0]↔v[0], d_ext[1]↔v[2], d_ext[2]↔v[1]
        ratio_x = v_ext[0] / d_ext[0]
        ratio_y = v_ext[2] / d_ext[1]
        ratio_z = v_ext[1] / d_ext[2]
        # All ratios should be within 1% of each other
        ratios = [ratio_x, ratio_y, ratio_z]
        assert max(ratios) / min(ratios) < 1.01, (
            f"Axis ratios diverge: {ratios}"
        )
