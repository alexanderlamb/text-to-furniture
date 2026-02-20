from __future__ import annotations

from pathlib import Path

import numpy as np

from openscad_step1 import Step1Config, run_step1_pipeline
from openscad_step1.audit import AuditTrail
from openscad_step1.contracts import CandidateRegion, PanelFamily
from openscad_step1.pipeline import _build_panel_families, _select_families


def _run_once(mesh_path: str, out_dir: Path, run_id: str):
    artifacts = out_dir / run_id / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    config = Step1Config(
        mesh_path=mesh_path,
        design_name="step1_deterministic",
        auto_scale=False,
        part_budget_max=12,
    )
    audit = AuditTrail(run_id=run_id, artifacts_dir=artifacts)
    return run_step1_pipeline(
        config=config,
        run_id=run_id,
        artifacts_dir=artifacts,
        audit=audit,
    )


def test_step1_pipeline_is_deterministic(box_mesh_file: str, tmp_path: Path):
    result_a = _run_once(box_mesh_file, tmp_path, "run_a")
    result_b = _run_once(box_mesh_file, tmp_path, "run_b")

    assert result_a.status == result_b.status
    assert len(result_a.panels) == len(result_b.panels)
    assert result_a.openscad_code == result_b.openscad_code

    sig_a = [
        (
            panel.family_id,
            tuple(round(v, 6) for v in panel.origin_3d),
            tuple(round(v, 6) for v in panel.basis_n),
            round(panel.area_mm2, 6),
        )
        for panel in result_a.panels
    ]
    sig_b = [
        (
            panel.family_id,
            tuple(round(v, 6) for v in panel.origin_3d),
            tuple(round(v, 6) for v in panel.basis_n),
            round(panel.area_mm2, 6),
        )
        for panel in result_b.panels
    ]
    assert sig_a == sig_b


def test_step1_panel_frames_are_valid(box_mesh_file: str, tmp_path: Path):
    result = _run_once(box_mesh_file, tmp_path, "frame_check")
    assert result.panels

    for panel in result.panels:
        u = np.asarray(panel.basis_u, dtype=float)
        v = np.asarray(panel.basis_v, dtype=float)
        n = np.asarray(panel.basis_n, dtype=float)

        assert np.isclose(np.linalg.norm(u), 1.0, atol=1e-6)
        assert np.isclose(np.linalg.norm(v), 1.0, atol=1e-6)
        assert np.isclose(np.linalg.norm(n), 1.0, atol=1e-6)

        assert abs(float(np.dot(u, v))) < 1e-6
        assert abs(float(np.dot(u, n))) < 1e-6
        assert abs(float(np.dot(v, n))) < 1e-6

        assert panel.thickness_mm > 0.0
        assert panel.area_mm2 > 0.0


def _synthetic_family(
    family_id: str,
    faces: list[int],
    area: float,
    layers: int,
    representative_candidate_id: str,
    candidate_ids: list[str],
    estimated_gap_mm: float,
) -> PanelFamily:
    return PanelFamily(
        family_id=family_id,
        candidate_ids=candidate_ids,
        representative_candidate_id=representative_candidate_id,
        face_ids=faces,
        total_area_mm2=area,
        layer_count=layers,
        selected_layer_count=0,
        shell_layers_required=1,
        interior_layers_capacity=max(0, layers - 1),
        selected_shell_layers=0,
        selected_interior_layers=0,
        estimated_gap_mm=estimated_gap_mm,
        is_opposite_pair=True,
        axis_role="unassigned_pair",
        notes={},
    )


def _synthetic_candidate(
    candidate_id: str,
    normal: tuple[float, float, float],
    center: tuple[float, float, float],
) -> CandidateRegion:
    n = np.asarray(normal, dtype=float)
    n = n / np.linalg.norm(n)
    ref = np.asarray([0.0, 0.0, 1.0], dtype=float)
    if abs(float(np.dot(n, ref))) > 0.9:
        ref = np.asarray([1.0, 0.0, 0.0], dtype=float)
    u = np.cross(ref, n)
    u = u / np.linalg.norm(u)
    v = np.cross(n, u)
    v = v / np.linalg.norm(v)
    return CandidateRegion(
        candidate_id=candidate_id,
        normal=(float(n[0]), float(n[1]), float(n[2])),
        center_3d=center,
        basis_u=(float(u[0]), float(u[1]), float(u[2])),
        basis_v=(float(v[0]), float(v[1]), float(v[2])),
        plane_offset=float(np.dot(n, np.asarray(center))),
        outline_2d=[(-50.0, -40.0), (50.0, -40.0), (50.0, 40.0), (-50.0, 40.0)],
        holes_2d=[],
        area_mm2=8000.0,
        source_faces=[],
    )


def test_cavity_primary_selection_blocks_secondary_interior(tmp_path: Path):
    families = [
        _synthetic_family(
            "fam_a",
            [1, 2],
            area=1200.0,
            layers=6,
            representative_candidate_id="cand_a_rep",
            candidate_ids=["cand_a_rep", "cand_a_opp"],
            estimated_gap_mm=300.0,
        ),
        _synthetic_family(
            "fam_b",
            [3, 4],
            area=1100.0,
            layers=6,
            representative_candidate_id="cand_b_rep",
            candidate_ids=["cand_b_rep", "cand_b_opp"],
            estimated_gap_mm=300.0,
        ),
        _synthetic_family(
            "fam_c",
            [5, 6],
            area=700.0,
            layers=4,
            representative_candidate_id="cand_c_rep",
            candidate_ids=["cand_c_rep", "cand_c_opp"],
            estimated_gap_mm=120.0,
        ),
    ]
    candidate_map = {
        "cand_a_rep": _synthetic_candidate(
            "cand_a_rep", (0.0, 0.0, -1.0), (0.0, 0.0, 15.0)
        ),
        "cand_a_opp": _synthetic_candidate(
            "cand_a_opp", (0.0, 0.0, 1.0), (0.0, 0.0, -285.0)
        ),
        "cand_b_rep": _synthetic_candidate(
            "cand_b_rep", (0.0, 0.0, 1.0), (0.0, 0.0, -15.0)
        ),
        "cand_b_opp": _synthetic_candidate(
            "cand_b_opp", (0.0, 0.0, -1.0), (0.0, 0.0, 285.0)
        ),
        "cand_c_rep": _synthetic_candidate(
            "cand_c_rep", (1.0, 0.0, 0.0), (100.0, 0.0, 0.0)
        ),
        "cand_c_opp": _synthetic_candidate(
            "cand_c_opp", (-1.0, 0.0, 0.0), (-20.0, 0.0, 0.0)
        ),
    }

    cfg = Step1Config(
        mesh_path="synthetic.stl",
        auto_scale=False,
        part_budget_max=8,
    )
    artifacts = tmp_path / "synthetic_select" / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    audit = AuditTrail(run_id="synthetic_select", artifacts_dir=artifacts)
    selected, debug = _select_families(
        families=families,
        part_budget=8,
        config=cfg,
        candidate_map=candidate_map,
        material_thickness_mm=4.7498,
        audit=audit,
    )
    audit.finalize()

    assert len(selected) == 3
    assert all(family.selected_shell_layers >= 1 for family in selected)
    assert sum(family.selected_layer_count for family in selected) == 8
    assert float(debug["budget_spent_shell"]) == 3
    assert float(debug["budget_spent_interior"]) == 5
    assert float(debug["blocked_interior_conflicts"]) >= 0.0
    selected_by_id = {family.family_id: family for family in selected}
    assert selected_by_id["fam_b"].axis_role == "secondary"
    assert selected_by_id["fam_b"].selected_interior_layers == 0


def test_selection_is_deterministic_for_synthetic_case(tmp_path: Path):
    candidate_map = {
        "cand_a_rep": _synthetic_candidate(
            "cand_a_rep", (0.0, 0.0, -1.0), (0.0, 0.0, 20.0)
        ),
        "cand_a_opp": _synthetic_candidate(
            "cand_a_opp", (0.0, 0.0, 1.0), (0.0, 0.0, -280.0)
        ),
        "cand_b_rep": _synthetic_candidate(
            "cand_b_rep", (0.0, 0.0, 1.0), (0.0, 0.0, -20.0)
        ),
        "cand_b_opp": _synthetic_candidate(
            "cand_b_opp", (0.0, 0.0, -1.0), (0.0, 0.0, 280.0)
        ),
        "cand_c_rep": _synthetic_candidate(
            "cand_c_rep", (1.0, 0.0, 0.0), (120.0, 0.0, 0.0)
        ),
        "cand_c_opp": _synthetic_candidate(
            "cand_c_opp", (-1.0, 0.0, 0.0), (-120.0, 0.0, 0.0)
        ),
        "cand_d_rep": _synthetic_candidate(
            "cand_d_rep", (0.0, 1.0, 0.0), (0.0, 90.0, 0.0)
        ),
        "cand_d_opp": _synthetic_candidate(
            "cand_d_opp", (0.0, -1.0, 0.0), (0.0, -90.0, 0.0)
        ),
    }

    def run_once(run_id: str):
        fams = [
            _synthetic_family(
                "fam_a",
                [1, 2],
                area=1200.0,
                layers=5,
                representative_candidate_id="cand_a_rep",
                candidate_ids=["cand_a_rep", "cand_a_opp"],
                estimated_gap_mm=300.0,
            ),
            _synthetic_family(
                "fam_b",
                [3, 4],
                area=900.0,
                layers=5,
                representative_candidate_id="cand_b_rep",
                candidate_ids=["cand_b_rep", "cand_b_opp"],
                estimated_gap_mm=300.0,
            ),
            _synthetic_family(
                "fam_c",
                [5, 6],
                area=700.0,
                layers=5,
                representative_candidate_id="cand_c_rep",
                candidate_ids=["cand_c_rep", "cand_c_opp"],
                estimated_gap_mm=240.0,
            ),
            _synthetic_family(
                "fam_d",
                [7, 8],
                area=650.0,
                layers=5,
                representative_candidate_id="cand_d_rep",
                candidate_ids=["cand_d_rep", "cand_d_opp"],
                estimated_gap_mm=180.0,
            ),
        ]
        cfg_local = Step1Config(
            mesh_path="synthetic.stl",
            auto_scale=False,
            part_budget_max=9,
        )
        artifacts = tmp_path / run_id / "artifacts"
        artifacts.mkdir(parents=True, exist_ok=True)
        audit = AuditTrail(run_id=run_id, artifacts_dir=artifacts)
        selected_local, _ = _select_families(
            families=fams,
            part_budget=9,
            config=cfg_local,
            candidate_map=candidate_map,
            material_thickness_mm=4.7498,
            audit=audit,
        )
        audit.finalize()
        return [
            (
                family.family_id,
                int(family.selected_shell_layers),
                int(family.selected_interior_layers),
                family.axis_role,
            )
            for family in sorted(selected_local, key=lambda f: f.family_id)
        ]

    sig_a = run_once("deterministic_a")
    sig_b = run_once("deterministic_b")
    assert sig_a == sig_b


def test_build_panel_families_uses_thin_gap_single_shell(tmp_path: Path):
    candidates = [
        CandidateRegion(
            candidate_id="cand_000",
            normal=(0.0, 0.0, 1.0),
            center_3d=(0.0, 0.0, 4.0),
            basis_u=(1.0, 0.0, 0.0),
            basis_v=(0.0, 1.0, 0.0),
            plane_offset=4.0,
            outline_2d=[(-40.0, -30.0), (40.0, -30.0), (40.0, 30.0), (-40.0, 30.0)],
            holes_2d=[],
            area_mm2=4800.0,
            source_faces=[1, 2],
        ),
        CandidateRegion(
            candidate_id="cand_001",
            normal=(0.0, 0.0, -1.0),
            center_3d=(0.0, 0.0, -4.0),
            basis_u=(1.0, 0.0, 0.0),
            basis_v=(0.0, 1.0, 0.0),
            plane_offset=4.0,
            outline_2d=[(-40.0, -30.0), (40.0, -30.0), (40.0, 30.0), (-40.0, 30.0)],
            holes_2d=[],
            area_mm2=4800.0,
            source_faces=[3, 4],
        ),
    ]
    cfg = Step1Config(mesh_path="synthetic.stl", auto_scale=False)
    artifacts = tmp_path / "thin_gap" / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    audit = AuditTrail(run_id="thin_gap", artifacts_dir=artifacts)
    families = _build_panel_families(
        candidates=candidates,
        mesh_centroid=np.asarray([0.0, 0.0, 0.0], dtype=float),
        config=cfg,
        material_thickness_mm=4.7498,
        audit=audit,
    )
    audit.finalize()

    assert len(families) == 1
    family = families[0]
    assert family.is_opposite_pair
    assert family.shell_layers_required == 1
    assert family.interior_layers_capacity == 0


def test_build_panel_families_hollow_cavity_gate(tmp_path: Path):
    """A 300mm gap far exceeds max_bridgeable_mm — interior fill must be zero."""
    candidates = [
        CandidateRegion(
            candidate_id="cand_000",
            normal=(0.0, 0.0, 1.0),
            center_3d=(0.0, 0.0, 150.0),
            basis_u=(1.0, 0.0, 0.0),
            basis_v=(0.0, 1.0, 0.0),
            plane_offset=150.0,
            outline_2d=[(-50.0, -40.0), (50.0, -40.0), (50.0, 40.0), (-50.0, 40.0)],
            holes_2d=[],
            area_mm2=8000.0,
            source_faces=[1, 2],
        ),
        CandidateRegion(
            candidate_id="cand_001",
            normal=(0.0, 0.0, -1.0),
            center_3d=(0.0, 0.0, -150.0),
            basis_u=(1.0, 0.0, 0.0),
            basis_v=(0.0, 1.0, 0.0),
            plane_offset=150.0,
            outline_2d=[(-50.0, -40.0), (50.0, -40.0), (50.0, 40.0), (-50.0, 40.0)],
            holes_2d=[],
            area_mm2=8000.0,
            source_faces=[3, 4],
        ),
    ]
    cfg = Step1Config(mesh_path="synthetic.stl", auto_scale=False)
    artifacts = tmp_path / "hollow_gate" / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    audit = AuditTrail(run_id="hollow_gate", artifacts_dir=artifacts)
    families = _build_panel_families(
        candidates=candidates,
        mesh_centroid=np.asarray([0.0, 0.0, 0.0], dtype=float),
        config=cfg,
        material_thickness_mm=4.7498,
        audit=audit,
    )
    audit.finalize()

    assert len(families) == 1
    family = families[0]
    assert family.interior_layers_capacity == 0
    assert family.layer_count == 2
    assert family.notes["hollow_cavity_gate"] == 1
    assert family.notes["max_bridgeable_mm"] > 0


def test_build_panel_families_solid_gap_allows_interior(tmp_path: Path):
    """A 25mm gap is within max_bridgeable_mm — interior fill should be allowed."""
    candidates = [
        CandidateRegion(
            candidate_id="cand_000",
            normal=(0.0, 0.0, 1.0),
            center_3d=(0.0, 0.0, 12.5),
            basis_u=(1.0, 0.0, 0.0),
            basis_v=(0.0, 1.0, 0.0),
            plane_offset=12.5,
            outline_2d=[(-50.0, -40.0), (50.0, -40.0), (50.0, 40.0), (-50.0, 40.0)],
            holes_2d=[],
            area_mm2=8000.0,
            source_faces=[1, 2],
        ),
        CandidateRegion(
            candidate_id="cand_001",
            normal=(0.0, 0.0, -1.0),
            center_3d=(0.0, 0.0, -12.5),
            basis_u=(1.0, 0.0, 0.0),
            basis_v=(0.0, 1.0, 0.0),
            plane_offset=12.5,
            outline_2d=[(-50.0, -40.0), (50.0, -40.0), (50.0, 40.0), (-50.0, 40.0)],
            holes_2d=[],
            area_mm2=8000.0,
            source_faces=[3, 4],
        ),
    ]
    cfg = Step1Config(mesh_path="synthetic.stl", auto_scale=False)
    artifacts = tmp_path / "solid_gap" / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    audit = AuditTrail(run_id="solid_gap", artifacts_dir=artifacts)
    families = _build_panel_families(
        candidates=candidates,
        mesh_centroid=np.asarray([0.0, 0.0, 0.0], dtype=float),
        config=cfg,
        material_thickness_mm=4.7498,
        audit=audit,
    )
    audit.finalize()

    assert len(families) == 1
    family = families[0]
    assert family.interior_layers_capacity > 0
    assert family.layer_count > 2
    assert family.notes["hollow_cavity_gate"] == 0
