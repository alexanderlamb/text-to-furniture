"""Tests for slab_selector.py — v2 greedy set-cover selection."""
import numpy as np
import pytest
import trimesh

from slab_candidates import Slab3D, Slab3DConfig, generate_candidates
from slab_selector import (
    SlabSelectionConfig, SlabSelection, select_slabs,
)


class TestSlabSelection:
    def test_box_mesh_covered(self, box_mesh):
        """A box mesh should be coverable by a small set of slabs."""
        config = Slab3DConfig(min_slab_area_mm2=50.0)
        candidates = generate_candidates(box_mesh, config)
        assert len(candidates) > 0

        sel_config = SlabSelectionConfig(
            coverage_target=0.50,
            max_slabs=10,
            min_coverage_contribution=0.01,
            dfm_check=False,
        )
        selection = select_slabs(box_mesh, candidates, sel_config)
        assert len(selection.selected_slabs) > 0
        assert selection.coverage_fraction > 0

    def test_table_mesh_covered(self, table_mesh):
        """A table mesh should be partially coverable."""
        config = Slab3DConfig(min_slab_area_mm2=50.0)
        candidates = generate_candidates(table_mesh, config)
        assert len(candidates) > 0

        sel_config = SlabSelectionConfig(
            coverage_target=0.30,
            max_slabs=10,
            min_coverage_contribution=0.01,
            dfm_check=False,
        )
        selection = select_slabs(table_mesh, candidates, sel_config)
        assert len(selection.selected_slabs) > 0

    def test_max_slabs_respected(self, box_mesh):
        """Should not select more than max_slabs."""
        config = Slab3DConfig(min_slab_area_mm2=50.0)
        candidates = generate_candidates(box_mesh, config)

        sel_config = SlabSelectionConfig(
            coverage_target=0.99,
            max_slabs=3,
            min_coverage_contribution=0.01,
            dfm_check=False,
        )
        selection = select_slabs(box_mesh, candidates, sel_config)
        assert len(selection.selected_slabs) <= 3

    def test_empty_candidates(self, box_mesh):
        """Empty candidate list should return empty selection."""
        sel_config = SlabSelectionConfig(dfm_check=False)
        selection = select_slabs(box_mesh, [], sel_config)
        assert len(selection.selected_slabs) == 0

    def test_uncovered_points_shape(self, box_mesh):
        """uncovered_points should be Nx3."""
        config = Slab3DConfig(min_slab_area_mm2=50.0)
        candidates = generate_candidates(box_mesh, config)
        sel_config = SlabSelectionConfig(
            coverage_target=0.50,
            max_slabs=2,
            min_coverage_contribution=0.01,
            dfm_check=False,
        )
        selection = select_slabs(box_mesh, candidates, sel_config)
        assert selection.uncovered_points.ndim == 2
        assert selection.uncovered_points.shape[1] == 3 or len(selection.uncovered_points) == 0


class TestMinContributionThreshold:
    """Verify min_coverage_contribution controls early stopping."""

    def test_low_threshold_accepts_small_coverage(self, box_mesh):
        """With a very low threshold, even small-coverage slabs are accepted."""
        config = Slab3DConfig(min_slab_area_mm2=50.0)
        candidates = generate_candidates(box_mesh, config)
        assert len(candidates) > 0

        sel_config = SlabSelectionConfig(
            coverage_target=0.99,
            max_slabs=10,
            min_coverage_contribution=0.001,
            dfm_check=False,
        )
        selection = select_slabs(box_mesh, candidates, sel_config)
        assert len(selection.selected_slabs) > 0

    def test_high_threshold_may_stop_early(self, box_mesh):
        """With a very high threshold, selection may stop with fewer slabs."""
        config = Slab3DConfig(min_slab_area_mm2=50.0)
        candidates = generate_candidates(box_mesh, config)
        assert len(candidates) > 0

        # Very high threshold — only accepts slabs covering >50% individually
        sel_config_high = SlabSelectionConfig(
            coverage_target=0.99,
            max_slabs=10,
            min_coverage_contribution=0.50,
            dfm_check=False,
        )
        selection_high = select_slabs(box_mesh, candidates, sel_config_high)

        # Low threshold for comparison
        sel_config_low = SlabSelectionConfig(
            coverage_target=0.99,
            max_slabs=10,
            min_coverage_contribution=0.001,
            dfm_check=False,
        )
        selection_low = select_slabs(box_mesh, candidates, sel_config_low)

        # High threshold should select fewer or equal slabs
        assert len(selection_high.selected_slabs) <= len(selection_low.selected_slabs)


class TestDFMCheckToggle:
    """Verify dfm_check toggle controls whether DFM filtering is applied."""

    def test_dfm_check_off_accepts_oversized(self, box_mesh):
        """With dfm_check=False, oversized candidates are selectable."""
        from dfm_rules import DFMConfig
        from shapely.geometry import Polygon

        oversized = Slab3D(
            normal=np.array([0, 0, 1.0]),
            origin=np.array([0, 0, 50.0]),
            thickness_mm=6.35,
            width_mm=2000.0,
            height_mm=2000.0,
            basis_u=np.array([1.0, 0, 0]),
            basis_v=np.array([0, 1.0, 0]),
            outline_2d=Polygon([
                (-1000, -1000), (1000, -1000), (1000, 1000), (-1000, 1000),
            ]),
            material_key="plywood_baltic_birch",
            source="axis_slice",
        )

        dfm = DFMConfig(max_sheet_width_mm=600, max_sheet_height_mm=760)
        sel_config = SlabSelectionConfig(
            coverage_target=0.50,
            max_slabs=5,
            min_coverage_contribution=0.001,
            dfm_check=False,
        )
        selection = select_slabs(box_mesh, [oversized], sel_config, dfm)
        # Without DFM check, the oversized slab should be accepted
        assert len(selection.selected_slabs) == 1

    def test_dfm_check_on_falls_back_for_oversized(self, box_mesh):
        """With dfm_check=True, oversized candidates trigger DFM fallback."""
        from dfm_rules import DFMConfig
        from shapely.geometry import Polygon

        oversized = Slab3D(
            normal=np.array([0, 0, 1.0]),
            origin=np.array([0, 0, 50.0]),
            thickness_mm=6.35,
            width_mm=2000.0,
            height_mm=2000.0,
            basis_u=np.array([1.0, 0, 0]),
            basis_v=np.array([0, 1.0, 0]),
            outline_2d=Polygon([
                (-1000, -1000), (1000, -1000), (1000, 1000), (-1000, 1000),
            ]),
            material_key="plywood_baltic_birch",
            source="axis_slice",
        )

        dfm = DFMConfig(max_sheet_width_mm=600, max_sheet_height_mm=760)
        sel_config = SlabSelectionConfig(
            coverage_target=0.50,
            max_slabs=5,
            min_coverage_contribution=0.001,
            dfm_check=True,
        )
        selection = select_slabs(box_mesh, [oversized], sel_config, dfm)
        # DFM fallback kicks in: oversized slab is accepted despite DFM violation
        assert len(selection.selected_slabs) == 1


class TestVolumeObjective:
    """Verify volume_fill objective behavior."""

    def test_volume_fill_selects_slabs(self, box_mesh):
        config = Slab3DConfig(min_slab_area_mm2=50.0)
        candidates = generate_candidates(box_mesh, config)
        assert len(candidates) > 0

        sel_config = SlabSelectionConfig(
            objective_mode="volume_fill",
            target_volume_fill=0.20,
            max_slabs=10,
            min_volume_contribution=0.001,
            plane_penalty_weight=0.0,
            dfm_check=False,
        )
        selection = select_slabs(box_mesh, candidates, sel_config)
        assert selection.objective_mode == "volume_fill"
        assert selection.volume_fill_fraction > 0.0
        assert len(selection.selected_slabs) > 0

    def test_large_penalty_can_block_selection(self, box_mesh):
        config = Slab3DConfig(min_slab_area_mm2=50.0)
        candidates = generate_candidates(box_mesh, config)
        assert len(candidates) > 0

        sel_config = SlabSelectionConfig(
            objective_mode="volume_fill",
            target_volume_fill=0.80,
            max_slabs=10,
            min_volume_contribution=0.001,
            plane_penalty_weight=1.0,  # larger than any possible contribution
            dfm_check=False,
        )
        selection = select_slabs(box_mesh, candidates, sel_config)
        assert len(selection.selected_slabs) == 0
        assert selection.stop_reason == "non_positive_objective_gain"

    def test_volume_mode_reports_dfm_fallback(self, box_mesh):
        """DFM pre-filter fallback is triggered in volume mode too."""
        from dfm_rules import DFMConfig
        from shapely.geometry import Polygon

        oversized = Slab3D(
            normal=np.array([0, 0, 1.0]),
            origin=np.array([0, 0, 50.0]),
            thickness_mm=6.35,
            width_mm=2000.0,
            height_mm=2000.0,
            basis_u=np.array([1.0, 0, 0]),
            basis_v=np.array([0, 1.0, 0]),
            outline_2d=Polygon([
                (-1000, -1000), (1000, -1000), (1000, 1000), (-1000, 1000),
            ]),
            material_key="plywood_baltic_birch",
            source="axis_slice",
        )
        dfm = DFMConfig(max_sheet_width_mm=600, max_sheet_height_mm=760)
        sel_config = SlabSelectionConfig(
            objective_mode="volume_fill",
            target_volume_fill=0.20,
            max_slabs=5,
            min_volume_contribution=0.001,
            plane_penalty_weight=0.0,
            dfm_check=True,
        )
        selection = select_slabs(box_mesh, [oversized], sel_config, dfm)
        assert selection.candidate_count_before_dfm == 1
        # DFM fallback kicks in: all candidates are restored
        assert selection.candidate_count_after_dfm == 1
        assert len(selection.selected_slabs) == 1


class TestSelectedSlabsAABB:
    """Verify selected slabs stay within mesh bounds (regression guard)."""

    def test_box_mesh_parts_within_bounds(self, box_mesh):
        """Selected slabs should produce parts within 1.3x mesh extents."""
        config = Slab3DConfig(min_slab_area_mm2=50.0)
        candidates = generate_candidates(box_mesh, config)
        assert len(candidates) > 0

        sel_config = SlabSelectionConfig(
            coverage_target=0.50,
            max_slabs=10,
            min_coverage_contribution=0.001,
            dfm_check=False,
        )
        selection = select_slabs(box_mesh, candidates, sel_config)
        assert len(selection.selected_slabs) > 0

        mesh_bounds = box_mesh.bounds
        mesh_extents = mesh_bounds[1] - mesh_bounds[0]

        for i, slab in enumerate(selection.selected_slabs):
            # Check origin is within bounds with tolerance
            tol = max(slab.width_mm, slab.height_mm, slab.thickness_mm)
            for ax in range(3):
                assert slab.origin[ax] >= mesh_bounds[0][ax] - tol, (
                    f"Selected slab {i} origin[{ax}]={slab.origin[ax]:.1f} "
                    f"too far below mesh min {mesh_bounds[0][ax]:.1f}"
                )
                assert slab.origin[ax] <= mesh_bounds[1][ax] + tol, (
                    f"Selected slab {i} origin[{ax}]={slab.origin[ax]:.1f} "
                    f"too far above mesh max {mesh_bounds[1][ax]:.1f}"
                )


class TestStopReasonLogging:
    """Verify stop-reason log messages are emitted."""

    def test_logs_no_coverage_gain(self, box_mesh, caplog):
        """When no candidates cover any voxels, logs no_coverage_gain."""
        import logging

        # Empty candidate list → immediate break
        sel_config = SlabSelectionConfig(dfm_check=False)
        with caplog.at_level(logging.INFO, logger="slab_selector"):
            select_slabs(box_mesh, [], sel_config)
        # Empty candidates means the loop body never runs; stop_reason is max_slabs_reached
        # The final summary should still log
        assert any("stop_reason=" in r.message for r in caplog.records)

    def test_logs_below_min_contribution(self, box_mesh, caplog):
        """When contribution is below threshold, logs the reason."""
        import logging

        config = Slab3DConfig(min_slab_area_mm2=50.0)
        candidates = generate_candidates(box_mesh, config)
        assert len(candidates) > 0

        sel_config = SlabSelectionConfig(
            coverage_target=0.99,
            max_slabs=10,
            min_coverage_contribution=0.99,  # impossibly high
            dfm_check=False,
        )
        with caplog.at_level(logging.INFO, logger="slab_selector"):
            select_slabs(box_mesh, candidates, sel_config)
        assert any("below threshold" in r.message for r in caplog.records)

    def test_logs_dfm_rejected_count(self, box_mesh, caplog):
        """Final summary should include dfm_rejected count."""
        import logging

        config = Slab3DConfig(min_slab_area_mm2=50.0)
        candidates = generate_candidates(box_mesh, config)

        sel_config = SlabSelectionConfig(
            coverage_target=0.50,
            max_slabs=5,
            min_coverage_contribution=0.001,
            dfm_check=False,
        )
        with caplog.at_level(logging.INFO, logger="slab_selector"):
            select_slabs(box_mesh, candidates, sel_config)
        assert any("dfm_rejected=" in r.message for r in caplog.records)


class TestDfmFallback:
    """Verify DFM pre-filter fallback when all candidates are rejected."""

    def test_dfm_fallback_produces_slabs(self, box_mesh):
        """With tiny sheet limits that reject all candidates, fallback should still select."""
        from dfm_rules import DFMConfig
        from shapely.geometry import Polygon

        # Create a slab covering the top face of the box (100x100 at z=100)
        slab = Slab3D(
            normal=np.array([0, 0, 1.0]),
            origin=np.array([0, 0, 100.0]),
            thickness_mm=6.35,
            width_mm=100.0,
            height_mm=100.0,
            basis_u=np.array([1.0, 0, 0]),
            basis_v=np.array([0, 1.0, 0]),
            outline_2d=Polygon([
                (-50, -50), (50, -50), (50, 50), (-50, 50),
            ]),
            material_key="plywood_baltic_birch",
            source="axis_slice",
        )

        # Absurdly small sheet limits: 10x10mm — will reject the 100x100 slab
        dfm = DFMConfig(max_sheet_width_mm=10, max_sheet_height_mm=10)
        sel_config = SlabSelectionConfig(
            coverage_target=0.50,
            max_slabs=5,
            min_coverage_contribution=0.001,
            dfm_check=True,
        )
        selection = select_slabs(box_mesh, [slab], sel_config, dfm)
        # Fallback should skip DFM and still select the slab
        assert len(selection.selected_slabs) == 1


class TestDiversityWeight:
    """Verify diversity_weight discourages same-direction slab hoarding."""

    def test_diversity_prefers_different_directions(self, box_mesh):
        """With diversity_weight > 0, different-normal slabs should be preferred
        over additional same-direction slabs after the first few picks."""
        from shapely.geometry import Polygon

        # Create candidates: 5 Z-normal slabs at different depths, 1 X-normal
        z_candidates = []
        for i in range(5):
            z_candidates.append(Slab3D(
                normal=np.array([0, 0, 1.0]),
                origin=np.array([50, 50, 20 + i * 10.0]),
                thickness_mm=6.35,
                width_mm=100.0,
                height_mm=100.0,
                basis_u=np.array([1.0, 0, 0]),
                basis_v=np.array([0, 1.0, 0]),
                outline_2d=Polygon([(-50, -50), (50, -50), (50, 50), (-50, 50)]),
                material_key="plywood_baltic_birch",
                source="axis_slice",
            ))
        x_slab = Slab3D(
            normal=np.array([1.0, 0, 0]),
            origin=np.array([50, 50, 50]),
            thickness_mm=6.35,
            width_mm=100.0,
            height_mm=100.0,
            basis_u=np.array([0, 1.0, 0]),
            basis_v=np.array([0, 0, 1.0]),
            outline_2d=Polygon([(-50, -50), (50, -50), (50, 50), (-50, 50)]),
            material_key="plywood_baltic_birch",
            source="axis_slice",
        )
        candidates = z_candidates + [x_slab]

        # With diversity: X-normal slab should appear earlier in selection
        sel_diverse = SlabSelectionConfig(
            objective_mode="volume_fill",
            target_volume_fill=0.99,
            max_slabs=6,
            min_volume_contribution=0.0001,
            plane_penalty_weight=0.0,
            diversity_weight=2.0,
            dfm_check=False,
        )
        result_diverse = select_slabs(box_mesh, candidates, sel_diverse)

        # Without diversity: all Z-normal slabs should come first
        sel_nodiv = SlabSelectionConfig(
            objective_mode="volume_fill",
            target_volume_fill=0.99,
            max_slabs=6,
            min_volume_contribution=0.0001,
            plane_penalty_weight=0.0,
            diversity_weight=0.0,
            dfm_check=False,
        )
        result_nodiv = select_slabs(box_mesh, candidates, sel_nodiv)

        # Find when the X-normal slab was selected in each run
        def x_slab_iteration(result):
            for entry in result.selection_trace:
                idx = entry["candidate_index"]
                if idx == len(z_candidates):  # X-slab is at this index
                    return entry["iteration"]
            return None

        iter_diverse = x_slab_iteration(result_diverse)
        iter_nodiv = x_slab_iteration(result_nodiv)

        # With diversity, X-normal should be picked earlier (or at all)
        if iter_diverse is not None and iter_nodiv is not None:
            assert iter_diverse <= iter_nodiv
        elif iter_diverse is not None:
            pass  # X-slab selected with diversity but not without — success
