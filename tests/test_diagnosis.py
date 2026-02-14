"""Tests for the diagnosis module."""
import json
import os
import tempfile

import numpy as np
import pytest

from diagnosis import (
    DiagnosisReport,
    QualityGate,
    build_diagnosis_report,
    compare_reports,
    evaluate_quality_gates,
    generate_summary,
    report_to_json,
    report_to_markdown,
)
from scoring import DecompositionScore
from manufacturing_decomposer_v2 import (
    ManufacturingDecompositionConfigV2,
    ManufacturingDecompositionResult,
    decompose_manufacturing_v2,
)
from slab_selector import SlabSelectionConfig
from furniture import FurnitureDesign


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_synthetic_report(
    coverage=0.85,
    part_count=5,
    hausdorff_mm=20.0,
    dfm_errors=0,
    structural=0.80,
    overall_score=0.75,
    sim_stable=None,
    fit_ratios=None,
):
    """Build a synthetic DiagnosisReport for gate testing."""
    report = DiagnosisReport()
    report.version = "1.0.0"
    report.mesh_path = "test.glb"
    report.timestamp = "2025-01-01T00:00:00Z"
    report.config_snapshot = {
        "max_slabs": 15,
        "coverage_target": 0.80,
        "min_coverage_contribution": 0.002,
    }
    report.mesh_faces = 1000
    report.mesh_extents_mm = [100.0, 100.0, 100.0]
    report.parts = [
        {
            "name": f"part_{i}",
            "width_mm": 200.0,
            "height_mm": 100.0,
            "thickness_mm": 6.35,
            "material": "plywood_baltic_birch",
            "dfm_violations": [],
        }
        for i in range(part_count)
    ]
    report.scores = {
        "coverage_fraction": coverage,
        "hausdorff_mm": hausdorff_mm,
        "part_count": part_count,
        "dfm_violations_error": dfm_errors,
        "dfm_violations_warning": 0,
        "structural_plausibility": structural,
        "overall_score": overall_score,
    }
    if fit_ratios is not None:
        report.scores["fit_ratios"] = fit_ratios

    if sim_stable is not None:
        report.simulation = {
            "stable": sim_stable,
            "fell_over": not sim_stable,
            "height_drop": 0.0 if sim_stable else 0.1,
        }

    return report


# ---------------------------------------------------------------------------
# Quality gate tests
# ---------------------------------------------------------------------------

class TestQualityGateCoverage:
    def test_pass(self):
        report = _make_synthetic_report(coverage=0.85)
        evaluate_quality_gates(report)
        gate = next(g for g in report.quality_gates if g.name == "coverage")
        assert gate.status == "pass"
        assert gate.recommendations == []

    def test_warn(self):
        report = _make_synthetic_report(coverage=0.55)
        evaluate_quality_gates(report)
        gate = next(g for g in report.quality_gates if g.name == "coverage")
        assert gate.status == "warn"
        assert len(gate.recommendations) > 0

    def test_fail(self):
        report = _make_synthetic_report(coverage=0.30)
        evaluate_quality_gates(report)
        gate = next(g for g in report.quality_gates if g.name == "coverage")
        assert gate.status == "fail"
        assert len(gate.recommendations) > 0


class TestQualityGateVolumeFill:
    def test_pass(self):
        report = _make_synthetic_report()
        report.config_snapshot["selection_objective_mode"] = "volume_fill"
        report.config_snapshot["target_volume_fill"] = 0.55
        report.config_snapshot["min_volume_contribution"] = 0.002
        report.config_snapshot["plane_penalty_weight"] = 0.0005
        report.scores["volume_fill_fraction"] = 0.60

        evaluate_quality_gates(report)
        gate = next(g for g in report.quality_gates if g.name == "volume_fill")
        assert gate.status == "pass"

    def test_fail(self):
        report = _make_synthetic_report()
        report.config_snapshot["selection_objective_mode"] = "volume_fill"
        report.config_snapshot["target_volume_fill"] = 0.55
        report.config_snapshot["min_volume_contribution"] = 0.002
        report.config_snapshot["plane_penalty_weight"] = 0.0005
        report.scores["volume_fill_fraction"] = 0.10

        evaluate_quality_gates(report)
        gate = next(g for g in report.quality_gates if g.name == "volume_fill")
        assert gate.status == "fail"
        assert len(gate.recommendations) > 0


class TestQualityGatePartCount:
    def test_pass(self):
        report = _make_synthetic_report(part_count=5)
        evaluate_quality_gates(report)
        gate = next(g for g in report.quality_gates if g.name == "part_count")
        assert gate.status == "pass"

    def test_warn_low(self):
        report = _make_synthetic_report(part_count=2)
        evaluate_quality_gates(report)
        gate = next(g for g in report.quality_gates if g.name == "part_count")
        assert gate.status == "warn"

    def test_fail_zero(self):
        report = _make_synthetic_report(part_count=0)
        report.parts = []
        evaluate_quality_gates(report)
        gate = next(g for g in report.quality_gates if g.name == "part_count")
        assert gate.status == "fail"
        assert any("0 parts" in r for r in gate.recommendations)


class TestQualityGateHausdorff:
    def test_pass(self):
        report = _make_synthetic_report(hausdorff_mm=15.0)
        evaluate_quality_gates(report)
        gate = next(g for g in report.quality_gates if g.name == "hausdorff")
        assert gate.status == "pass"

    def test_warn(self):
        report = _make_synthetic_report(hausdorff_mm=45.0)
        evaluate_quality_gates(report)
        gate = next(g for g in report.quality_gates if g.name == "hausdorff")
        assert gate.status == "warn"

    def test_fail(self):
        report = _make_synthetic_report(hausdorff_mm=80.0)
        evaluate_quality_gates(report)
        gate = next(g for g in report.quality_gates if g.name == "hausdorff")
        assert gate.status == "fail"


class TestQualityGateDfmErrors:
    def test_pass(self):
        report = _make_synthetic_report(dfm_errors=0)
        evaluate_quality_gates(report)
        gate = next(g for g in report.quality_gates if g.name == "dfm_errors")
        assert gate.status == "pass"

    def test_warn(self):
        report = _make_synthetic_report(dfm_errors=2)
        evaluate_quality_gates(report)
        gate = next(g for g in report.quality_gates if g.name == "dfm_errors")
        assert gate.status == "warn"

    def test_fail(self):
        report = _make_synthetic_report(dfm_errors=5)
        evaluate_quality_gates(report)
        gate = next(g for g in report.quality_gates if g.name == "dfm_errors")
        assert gate.status == "fail"

    def test_with_violation_details(self):
        report = _make_synthetic_report(dfm_errors=2)
        report.parts = [
            {
                "name": "part_0",
                "width_mm": 200.0,
                "height_mm": 100.0,
                "thickness_mm": 6.35,
                "material": "plywood",
                "dfm_violations": [
                    {
                        "rule_name": "sheet_size_width",
                        "severity": "error",
                        "message": "Part too wide",
                        "value": 700.0,
                        "limit": 609.6,
                    },
                    {
                        "rule_name": "aspect_ratio",
                        "severity": "error",
                        "message": "Aspect ratio too high",
                        "value": 25.0,
                        "limit": 20.0,
                    },
                ],
            }
        ]
        evaluate_quality_gates(report)
        gate = next(g for g in report.quality_gates if g.name == "dfm_errors")
        assert len(gate.recommendations) >= 2


class TestQualityGateStructural:
    def test_pass(self):
        report = _make_synthetic_report(structural=0.80)
        evaluate_quality_gates(report)
        gate = next(g for g in report.quality_gates if g.name == "structural")
        assert gate.status == "pass"

    def test_fail(self):
        report = _make_synthetic_report(structural=0.30)
        evaluate_quality_gates(report)
        gate = next(g for g in report.quality_gates if g.name == "structural")
        assert gate.status == "fail"


class TestQualityGateStability:
    def test_no_sim(self):
        report = _make_synthetic_report(sim_stable=None)
        evaluate_quality_gates(report)
        names = [g.name for g in report.quality_gates]
        assert "stability" not in names

    def test_stable(self):
        report = _make_synthetic_report(sim_stable=True)
        evaluate_quality_gates(report)
        gate = next(g for g in report.quality_gates if g.name == "stability")
        assert gate.status == "pass"

    def test_unstable(self):
        report = _make_synthetic_report(sim_stable=False)
        evaluate_quality_gates(report)
        gate = next(g for g in report.quality_gates if g.name == "stability")
        assert gate.status == "fail"
        assert len(gate.recommendations) > 0


class TestQualityGateFitRatios:
    def test_pass(self):
        report = _make_synthetic_report(fit_ratios=[0.95, 1.0, 0.98])
        evaluate_quality_gates(report)
        gate = next(g for g in report.quality_gates if g.name == "fit_ratios")
        assert gate.status == "pass"

    def test_fail_extreme(self):
        report = _make_synthetic_report(fit_ratios=[0.2, 1.0, 3.0])
        evaluate_quality_gates(report)
        gate = next(g for g in report.quality_gates if g.name == "fit_ratios")
        assert gate.status == "fail"

    def test_no_ratios(self):
        report = _make_synthetic_report()
        evaluate_quality_gates(report)
        names = [g.name for g in report.quality_gates]
        assert "fit_ratios" not in names


# ---------------------------------------------------------------------------
# Overall status
# ---------------------------------------------------------------------------

class TestOverallStatus:
    def test_all_pass(self):
        report = _make_synthetic_report(
            coverage=0.85, part_count=5, hausdorff_mm=10.0,
            dfm_errors=0, structural=0.80,
        )
        evaluate_quality_gates(report)
        assert report.overall_status == "pass"

    def test_any_fail(self):
        report = _make_synthetic_report(coverage=0.30)
        evaluate_quality_gates(report)
        assert report.overall_status == "fail"

    def test_warn_no_fail(self):
        report = _make_synthetic_report(coverage=0.55)
        evaluate_quality_gates(report)
        assert report.overall_status in ("warn", "fail")


class TestRecommendationsPopulated:
    def test_recommendations_on_fail(self):
        report = _make_synthetic_report(
            coverage=0.30, hausdorff_mm=80.0, structural=0.30,
        )
        evaluate_quality_gates(report)
        assert len(report.recommended_actions) > 0
        assert len(report.top_issues) > 0


# ---------------------------------------------------------------------------
# Report builder — integration tests with mesh fixtures
# ---------------------------------------------------------------------------

class TestBuildReportBoxMesh:
    def test_box_mesh(self, box_mesh):
        """Full pipeline on box_mesh fixture, verify report structure."""
        # Write box mesh to a temp file
        with tempfile.NamedTemporaryFile(suffix=".stl", delete=False) as f:
            box_mesh.export(f.name)
            mesh_path = f.name

        try:
            config = ManufacturingDecompositionConfigV2(
                slab_selection=SlabSelectionConfig(
                    coverage_target=0.50,
                    max_slabs=10,
                ),
                target_height_mm=100.0,
                auto_scale=False,
            )
            mfg_result = decompose_manufacturing_v2(mesh_path, config)
            report = build_diagnosis_report(
                mesh_path=mesh_path,
                mesh=box_mesh,
                mfg_result=mfg_result,
                config=config,
            )

            # Structure checks
            assert report.version == "1.0.0"
            assert report.mesh_path == mesh_path
            assert report.mesh_faces > 0
            assert len(report.mesh_extents_mm) == 3
            assert report.overall_status in ("pass", "warn", "fail")
            assert len(report.quality_gates) >= 5
            assert isinstance(report.summary, str)
            assert len(report.summary) > 0
        finally:
            os.unlink(mesh_path)


class TestBuildReportTableMesh:
    def test_table_mesh(self, table_mesh):
        """Full pipeline on table_mesh fixture."""
        with tempfile.NamedTemporaryFile(suffix=".stl", delete=False) as f:
            table_mesh.export(f.name)
            mesh_path = f.name

        try:
            config = ManufacturingDecompositionConfigV2(
                slab_selection=SlabSelectionConfig(
                    coverage_target=0.50,
                    max_slabs=15,
                ),
                target_height_mm=700.0,
                auto_scale=False,
            )
            mfg_result = decompose_manufacturing_v2(mesh_path, config)
            report = build_diagnosis_report(
                mesh_path=mesh_path,
                mesh=table_mesh,
                mfg_result=mfg_result,
                config=config,
            )

            assert report.scores.get("part_count", 0) > 0
            assert report.overall_status in ("pass", "warn", "fail")
        finally:
            os.unlink(mesh_path)


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

class TestCompareReports:
    def test_improved(self):
        prev = _make_synthetic_report(coverage=0.50, hausdorff_mm=60.0, overall_score=0.40)
        curr = _make_synthetic_report(coverage=0.80, hausdorff_mm=20.0, overall_score=0.75)

        evaluate_quality_gates(prev)
        evaluate_quality_gates(curr)

        # Write prev to temp file
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(report_to_json(prev), f)
            prev_path = f.name

        try:
            compare_reports(curr, prev_path)
            assert curr.comparison is not None
            assert curr.comparison["verdict"] == "improved"
            assert curr.comparison["deltas"]["coverage_fraction"] > 0
        finally:
            os.unlink(prev_path)

    def test_regressed(self):
        prev = _make_synthetic_report(coverage=0.80, hausdorff_mm=20.0, overall_score=0.75)
        curr = _make_synthetic_report(coverage=0.50, hausdorff_mm=60.0, overall_score=0.40)

        evaluate_quality_gates(prev)
        evaluate_quality_gates(curr)

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(report_to_json(prev), f)
            prev_path = f.name

        try:
            compare_reports(curr, prev_path)
            assert curr.comparison is not None
            assert curr.comparison["verdict"] == "regressed"
        finally:
            os.unlink(prev_path)

    def test_unchanged(self):
        prev = _make_synthetic_report(coverage=0.80, hausdorff_mm=20.0, overall_score=0.75)
        curr = _make_synthetic_report(coverage=0.80, hausdorff_mm=20.0, overall_score=0.75)

        evaluate_quality_gates(prev)
        evaluate_quality_gates(curr)

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(report_to_json(prev), f)
            prev_path = f.name

        try:
            compare_reports(curr, prev_path)
            assert curr.comparison is not None
            assert curr.comparison["verdict"] == "unchanged"
        finally:
            os.unlink(prev_path)


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------

class TestReportJsonRoundtrip:
    def test_roundtrip(self):
        report = _make_synthetic_report(coverage=0.75, sim_stable=True)
        evaluate_quality_gates(report)
        generate_summary(report)

        d = report_to_json(report)
        json_str = json.dumps(d)  # must not raise
        loaded = json.loads(json_str)

        assert loaded["version"] == report.version
        assert loaded["overall_status"] == report.overall_status
        assert loaded["scores"]["coverage_fraction"] == report.scores["coverage_fraction"]
        assert len(loaded["quality_gates"]) == len(report.quality_gates)

    def test_inf_handling(self):
        """Verify inf values are serialized without error."""
        report = _make_synthetic_report(hausdorff_mm=float("inf"))
        evaluate_quality_gates(report)
        d = report_to_json(report)
        json_str = json.dumps(d)  # must not raise on inf
        assert "inf" in json_str.lower() or "Infinity" in json_str


class TestReportMarkdown:
    def test_markdown_output(self):
        report = _make_synthetic_report(coverage=0.85)
        evaluate_quality_gates(report)
        generate_summary(report)
        md = report_to_markdown(report)

        assert "# Diagnosis Report" in md
        assert "Quality Gates" in md
        assert "PASS" in md


# ---------------------------------------------------------------------------
# Selection trace integration
# ---------------------------------------------------------------------------

class TestSelectionTrace:
    def test_trace_populated_on_box(self, box_mesh):
        """Verify selection trace is populated after decomposition."""
        with tempfile.NamedTemporaryFile(suffix=".stl", delete=False) as f:
            box_mesh.export(f.name)
            mesh_path = f.name

        try:
            config = ManufacturingDecompositionConfigV2(
                slab_selection=SlabSelectionConfig(
                    coverage_target=0.50,
                    max_slabs=10,
                ),
                target_height_mm=100.0,
                auto_scale=False,
            )
            mfg_result = decompose_manufacturing_v2(mesh_path, config)
            debug = mfg_result.design.metadata.get("decomposition_debug", {})

            assert "selection_trace" in debug
            assert "selection_stop_reason" in debug
            assert isinstance(debug["selection_trace"], list)
            assert len(debug["selection_stop_reason"]) > 0

            # If any slabs were selected, verify trace entries
            if debug["selection_trace"]:
                entry = debug["selection_trace"][0]
                assert "iteration" in entry
                assert "source" in entry
                assert "coverage_after" in entry
                assert "dfm_rejected" in entry
        finally:
            os.unlink(mesh_path)
