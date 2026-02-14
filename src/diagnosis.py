"""
Diagnosis module for mesh decomposition quality.

Consolidates all pipeline output into a single structured report with
quality gates, actionable recommendations, and run-to-run comparison.
Designed to be read by an LLM agent for autonomous diagnose-fix-rerun loops.
"""
import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np

from dfm_rules import DFMConfig, DFMViolation, check_part_dfm
from scoring import DecompositionScore

logger = logging.getLogger(__name__)

DIAGNOSIS_VERSION = "1.0.0"


@dataclass
class QualityGate:
    """A single pass/warn/fail quality check."""

    name: str
    status: str  # "pass" | "warn" | "fail"
    metric_value: float
    threshold_pass: float
    threshold_fail: float
    message: str
    recommendations: List[str] = field(default_factory=list)


@dataclass
class DiagnosisReport:
    """Complete diagnosis report for a decomposition run."""

    # Meta
    version: str = ""
    mesh_path: str = ""
    timestamp: str = ""
    config_snapshot: dict = field(default_factory=dict)

    # Mesh info
    mesh_faces: int = 0
    mesh_extents_mm: List[float] = field(default_factory=list)

    # Candidate generation summary
    candidate_generation: dict = field(default_factory=dict)

    # Selection trace
    selection_trace: List[dict] = field(default_factory=list)
    selection_stop_reason: str = ""

    # Per-part diagnostics
    parts: List[dict] = field(default_factory=list)

    # Aggregate scores
    scores: dict = field(default_factory=dict)

    # Simulation
    simulation: Optional[dict] = None

    # Quality gates
    quality_gates: List[QualityGate] = field(default_factory=list)

    # LLM-facing summary
    overall_status: str = ""
    summary: str = ""
    top_issues: List[str] = field(default_factory=list)
    recommended_actions: List[str] = field(default_factory=list)

    # Run comparison
    comparison: Optional[dict] = None


# ---------------------------------------------------------------------------
# Quality gate evaluation
# ---------------------------------------------------------------------------

def _gate_coverage(debug: dict, config_snapshot: dict) -> QualityGate:
    coverage = debug.get("coverage_fraction", 0.0)
    recs: List[str] = []
    if coverage >= 0.70:
        status = "pass"
    elif coverage >= 0.50:
        status = "warn"
    else:
        status = "fail"

    if status != "pass":
        mc = config_snapshot.get("min_coverage_contribution", 0.002)
        ms = config_snapshot.get("max_slabs", 15)
        recs.append(
            f"Lower min_coverage_contribution from {mc} to {mc * 0.5}"
        )
        recs.append(f"Increase max_slabs from {ms} to {ms + 5}")

    return QualityGate(
        name="coverage",
        status=status,
        metric_value=coverage,
        threshold_pass=0.70,
        threshold_fail=0.50,
        message=f"Surface coverage {coverage:.1%}",
        recommendations=recs,
    )


def _gate_volume_fill(debug: dict, config_snapshot: dict) -> QualityGate:
    volume_fill = debug.get("volume_fill_fraction", 0.0)
    target = float(config_snapshot.get("target_volume_fill", 0.55))
    pass_threshold = max(0.20, target * 0.9)
    fail_threshold = max(0.10, target * 0.5)

    recs: List[str] = []
    if volume_fill >= pass_threshold:
        status = "pass"
    elif volume_fill >= fail_threshold:
        status = "warn"
    else:
        status = "fail"

    if status != "pass":
        mvc = config_snapshot.get("min_volume_contribution", 0.002)
        penalty = config_snapshot.get("plane_penalty_weight", 0.0005)
        recs.append(
            f"Lower min_volume_contribution from {mvc} to {mvc * 0.5}"
        )
        recs.append(
            f"Lower plane_penalty_weight from {penalty} to {penalty * 0.5}"
        )
        recs.append("Increase max_slabs or add paired interior candidates")

    return QualityGate(
        name="volume_fill",
        status=status,
        metric_value=volume_fill,
        threshold_pass=pass_threshold,
        threshold_fail=fail_threshold,
        message=f"Interior volume fill {volume_fill:.1%}",
        recommendations=recs,
    )


def _gate_part_count(part_count: int) -> QualityGate:
    recs: List[str] = []
    if 3 <= part_count <= 15:
        status = "pass"
    elif (1 <= part_count <= 2) or (16 <= part_count <= 20):
        status = "warn"
    else:
        status = "fail"

    if part_count == 0:
        recs.append("0 parts selected: check candidate generation and coverage thresholds")
    elif part_count > 20:
        recs.append("Too many parts: increase min_coverage_contribution to reduce count")

    return QualityGate(
        name="part_count",
        status=status,
        metric_value=float(part_count),
        threshold_pass=15.0,
        threshold_fail=20.0,
        message=f"{part_count} parts selected",
        recommendations=recs,
    )


def _gate_hausdorff(hausdorff_mm: float) -> QualityGate:
    recs: List[str] = []
    if hausdorff_mm < 30.0:
        status = "pass"
    elif hausdorff_mm < 60.0:
        status = "warn"
    else:
        status = "fail"

    if status != "pass":
        recs.append("High hausdorff distance: missing surface coverage")
        recs.append("Add candidate sources or lower area threshold")

    return QualityGate(
        name="hausdorff",
        status=status,
        metric_value=hausdorff_mm,
        threshold_pass=30.0,
        threshold_fail=60.0,
        message=f"Hausdorff distance {hausdorff_mm:.1f} mm",
        recommendations=recs,
    )


def _gate_dfm_errors(error_count: int, per_part_violations: List[dict]) -> QualityGate:
    recs: List[str] = []
    if error_count == 0:
        status = "pass"
    elif error_count <= 3:
        status = "warn"
    else:
        status = "fail"

    if error_count > 0:
        # Collect unique violation details
        seen = set()
        for part in per_part_violations:
            for v in part.get("dfm_violations", []):
                if v.get("severity") == "error" and v.get("rule_name") not in seen:
                    seen.add(v["rule_name"])
                    recs.append(
                        f"DFM error on {part['name']}: {v['message']}"
                    )

    return QualityGate(
        name="dfm_errors",
        status=status,
        metric_value=float(error_count),
        threshold_pass=0.0,
        threshold_fail=3.0,
        message=f"{error_count} DFM errors",
        recommendations=recs,
    )


def _gate_structural(structural_plausibility: float) -> QualityGate:
    recs: List[str] = []
    if structural_plausibility >= 0.75:
        status = "pass"
    elif structural_plausibility >= 0.50:
        status = "warn"
    else:
        status = "fail"

    if status != "pass":
        if structural_plausibility < 0.5:
            recs.append("Missing horizontal surfaces (shelves/seats)")
            recs.append("No joints detected: lower joint_proximity_mm")
            recs.append("Check if assembly graph is disconnected")

    return QualityGate(
        name="structural",
        status=status,
        metric_value=structural_plausibility,
        threshold_pass=0.75,
        threshold_fail=0.50,
        message=f"Structural plausibility {structural_plausibility:.2f}",
        recommendations=recs,
    )


def _gate_stability(sim_result: Optional[dict]) -> Optional[QualityGate]:
    if sim_result is None:
        return None
    stable = sim_result.get("stable", True)
    status = "pass" if stable else "fail"
    recs: List[str] = []
    if not stable:
        if sim_result.get("fell_over", False):
            recs.append("Furniture fell over: check leg count and base width")
        if sim_result.get("height_drop", 0) > 0.05:
            recs.append("Significant height drop: reduce overhang or add supports")

    return QualityGate(
        name="stability",
        status=status,
        metric_value=1.0 if stable else 0.0,
        threshold_pass=1.0,
        threshold_fail=0.5,
        message="Stable" if stable else "Unstable",
        recommendations=recs,
    )


def _gate_fit_ratios(fit_ratios: Optional[List[float]]) -> Optional[QualityGate]:
    if not fit_ratios:
        return None
    worst = max(max(abs(r - 1.0) for r in fit_ratios), 0.0)
    any_extreme = any(r < 0.3 or r > 2.0 for r in fit_ratios)
    any_moderate = any((0.3 <= r < 0.5) or (1.5 < r <= 2.0) for r in fit_ratios)

    if any_extreme:
        status = "fail"
    elif any_moderate:
        status = "warn"
    else:
        status = "pass"

    recs: List[str] = []
    if status != "pass":
        recs.append("Parts don't span mesh correctly: check scaling and pose alignment")

    return QualityGate(
        name="fit_ratios",
        status=status,
        metric_value=worst,
        threshold_pass=0.5,
        threshold_fail=1.0,
        message=f"Fit ratio deviation {worst:.2f} (ratios: {[f'{r:.2f}' for r in fit_ratios]})",
        recommendations=recs,
    )


def _gate_candidate_generation(debug: dict) -> Optional[QualityGate]:
    """Check if candidate generation failed entirely."""
    failure_reason = debug.get("failure_reason", "")
    if not failure_reason:
        return None

    recs: List[str] = []
    recs.append("Lower min_slab_area_mm2 to capture smaller features")
    if not debug.get("mesh_is_watertight", True):
        recs.append("Mesh is non-watertight: try mesh repair or remeshing before decomposition")
    recs.append("Try dfm_check=False to skip DFM pre-filtering")

    dfm_rejected = debug.get("selection_dfm_rejected_count", 0)
    candidate_count = debug.get("candidate_count", 0)
    if failure_reason == "no_slabs_selected" and dfm_rejected > 0:
        recs.append(
            f"DFM rejected {dfm_rejected}/{candidate_count} candidates: "
            "check if mesh exceeds material sheet limits"
        )

    return QualityGate(
        name="candidate_generation",
        status="fail",
        metric_value=0.0,
        threshold_pass=1.0,
        threshold_fail=0.0,
        message=f"Candidate generation failed: {failure_reason}",
        recommendations=recs,
    )


def evaluate_quality_gates(report: DiagnosisReport) -> None:
    """Evaluate all quality gates and populate report fields."""
    gates: List[QualityGate] = []

    # Check for candidate generation failure first
    debug = report.candidate_generation
    cand_gate = _gate_candidate_generation(debug)
    if cand_gate is not None:
        gates.append(cand_gate)

    config_snapshot = report.config_snapshot
    objective_mode = config_snapshot.get("selection_objective_mode", "surface_coverage")
    if objective_mode == "volume_fill":
        gates.append(_gate_volume_fill(
            {"volume_fill_fraction": report.scores.get("volume_fill_fraction", 0.0)},
            config_snapshot,
        ))
    else:
        gates.append(_gate_coverage(
            {"coverage_fraction": report.scores.get("coverage_fraction", 0.0)},
            config_snapshot,
        ))
    gates.append(_gate_part_count(report.scores.get("part_count", 0)))
    gates.append(_gate_hausdorff(report.scores.get("hausdorff_mm", float("inf"))))
    gates.append(_gate_dfm_errors(
        report.scores.get("dfm_violations_error", 0),
        report.parts,
    ))
    gates.append(_gate_structural(report.scores.get("structural_plausibility", 0.0)))

    stability_gate = _gate_stability(report.simulation)
    if stability_gate is not None:
        gates.append(stability_gate)

    fit_gate = _gate_fit_ratios(report.scores.get("fit_ratios"))
    if fit_gate is not None:
        gates.append(fit_gate)

    report.quality_gates = gates

    # Overall status
    statuses = [g.status for g in gates]
    if "fail" in statuses:
        report.overall_status = "fail"
    elif "warn" in statuses:
        report.overall_status = "warn"
    else:
        report.overall_status = "pass"

    # Top issues: collect all non-pass gate messages
    report.top_issues = [
        f"[{g.status.upper()}] {g.name}: {g.message}"
        for g in gates
        if g.status != "pass"
    ]

    # Recommended actions: flatten gate recommendations, deduplicate
    seen: set = set()
    actions: List[str] = []
    for g in gates:
        for rec in g.recommendations:
            if rec not in seen:
                seen.add(rec)
                actions.append(rec)
    report.recommended_actions = actions


def generate_summary(report: DiagnosisReport) -> None:
    """Generate a natural language summary for the report."""
    n_parts = report.scores.get("part_count", 0)
    objective_mode = report.config_snapshot.get("selection_objective_mode", "surface_coverage")
    coverage = report.scores.get("coverage_fraction", 0.0)
    volume_fill = report.scores.get("volume_fill_fraction", 0.0)
    hausdorff = report.scores.get("hausdorff_mm", float("inf"))

    if objective_mode == "volume_fill":
        lines = [
            f"Decomposition produced {n_parts} parts with {volume_fill:.0%} interior volume fill "
            f"and {hausdorff:.1f} mm Hausdorff distance.",
        ]
    else:
        lines = [
            f"Decomposition produced {n_parts} parts with {coverage:.0%} surface coverage "
            f"and {hausdorff:.1f} mm Hausdorff distance.",
        ]

    n_fail = sum(1 for g in report.quality_gates if g.status == "fail")
    n_warn = sum(1 for g in report.quality_gates if g.status == "warn")
    if n_fail:
        lines.append(f"{n_fail} quality gate(s) failed.")
    elif n_warn:
        lines.append(f"All gates passed but {n_warn} have warnings.")
    else:
        lines.append("All quality gates passed.")

    if report.recommended_actions:
        lines.append(
            f"Top recommendation: {report.recommended_actions[0]}"
        )

    report.summary = " ".join(lines)


# ---------------------------------------------------------------------------
# Report builder
# ---------------------------------------------------------------------------

def build_diagnosis_report(
    mesh_path: str,
    mesh,  # trimesh.Trimesh
    mfg_result,  # ManufacturingDecompositionResult
    sim_result=None,  # Optional[SimulationResult]
    config=None,  # Optional[ManufacturingDecompositionConfigV2]
) -> DiagnosisReport:
    """Build a complete diagnosis report from pipeline outputs.

    Args:
        mesh_path: Path to the input mesh file.
        mesh: The loaded trimesh mesh.
        mfg_result: Result from decompose_manufacturing_v2().
        sim_result: Optional simulation result.
        config: Optional decomposition config for snapshot.

    Returns:
        Fully populated DiagnosisReport.
    """
    report = DiagnosisReport()
    report.version = DIAGNOSIS_VERSION
    report.mesh_path = mesh_path
    report.timestamp = datetime.now(timezone.utc).isoformat()

    # Config snapshot
    config_snap: dict = {}
    if config is not None:
        config_snap = {
            "max_slabs": config.slab_selection.max_slabs,
            "selection_objective_mode": config.slab_selection.objective_mode,
            "coverage_target": config.slab_selection.coverage_target,
            "target_volume_fill": config.slab_selection.target_volume_fill,
            "min_coverage_contribution": config.slab_selection.min_coverage_contribution,
            "min_volume_contribution": config.slab_selection.min_volume_contribution,
            "plane_penalty_weight": config.slab_selection.plane_penalty_weight,
            "voxel_resolution_mm": config.slab_selection.voxel_resolution_mm,
            "dfm_check": config.slab_selection.dfm_check,
            "default_material": config.default_material,
            "target_height_mm": config.target_height_mm,
            "joint_proximity_mm": config.joint_proximity_mm,
            "pose_alignment_fix": config.pose_alignment_fix,
        }
    report.config_snapshot = config_snap

    # Mesh info
    report.mesh_faces = int(len(mesh.faces))
    extents = (mesh.bounds[1] - mesh.bounds[0]).tolist()
    report.mesh_extents_mm = extents

    # Pull decomposition_debug from design metadata
    debug = mfg_result.design.metadata.get("decomposition_debug", {})

    # Candidate generation summary
    report.candidate_generation = {
        "total": debug.get("candidate_count", 0),
        "by_source": debug.get("selected_source_histogram", {}),
        "after_dedupe": debug.get("candidate_count", 0),
    }
    if "failure_reason" in debug:
        report.candidate_generation["failure_reason"] = debug["failure_reason"]
    if "selection_dfm_rejected_count" in debug:
        report.candidate_generation["selection_dfm_rejected_count"] = debug["selection_dfm_rejected_count"]
    if "candidate_count" in debug:
        report.candidate_generation["candidate_count"] = debug["candidate_count"]
    if "mesh_is_watertight" in debug:
        report.candidate_generation["mesh_is_watertight"] = debug["mesh_is_watertight"]

    # Selection trace
    report.selection_trace = debug.get("selection_trace", [])
    report.selection_stop_reason = debug.get("selection_stop_reason", "")

    # Per-part diagnostics with DFM violations
    parts_list: List[dict] = []
    dfm_config = DFMConfig()
    if config is not None:
        dfm_config = config.dfm

    for name, profile in mfg_result.parts.items():
        comp = mfg_result.design.get_component(name)
        violations = check_part_dfm(profile, dfm_config)
        bounds = profile.outline.bounds
        part_info: dict = {
            "name": name,
            "width_mm": bounds[2] - bounds[0],
            "height_mm": bounds[3] - bounds[1],
            "thickness_mm": profile.thickness_mm,
            "material": profile.material_key,
            "dfm_violations": [
                {
                    "rule_name": v.rule_name,
                    "severity": v.severity,
                    "message": v.message,
                    "value": v.value,
                    "limit": v.limit,
                }
                for v in violations
            ],
        }
        if comp is not None:
            part_info["component_type"] = comp.type.value
            part_info["position"] = comp.position.tolist()
        parts_list.append(part_info)
    report.parts = parts_list

    # Aggregate scores
    score = mfg_result.score
    report.scores = {
        "hausdorff_mm": score.hausdorff_mm,
        "mean_distance_mm": score.mean_distance_mm,
        "part_count": score.part_count,
        "dfm_violations_error": score.dfm_violations_error,
        "dfm_violations_warning": score.dfm_violations_warning,
        "structural_plausibility": score.structural_plausibility,
        "overall_score": score.overall_score,
        "objective_mode": debug.get("objective_mode", config_snap.get("selection_objective_mode", "surface_coverage")),
        "coverage_fraction": debug.get("coverage_fraction", 0.0),
        "volume_fill_fraction": debug.get("volume_fill_fraction", 0.0),
    }
    if "fit_ratios" in debug:
        report.scores["fit_ratios"] = debug["fit_ratios"]

    # Simulation
    if sim_result is not None:
        report.simulation = sim_result.to_dict()

    # Evaluate quality gates
    evaluate_quality_gates(report)
    generate_summary(report)

    return report


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

def compare_reports(
    current: DiagnosisReport,
    previous_path: str,
) -> None:
    """Compare current report against a previous report and fill comparison.

    Args:
        current: The current DiagnosisReport (will be mutated).
        previous_path: Path to a previous diagnosis_report.json.
    """
    with open(previous_path, "r") as f:
        prev_data = json.load(f)

    prev_scores = prev_data.get("scores", {})
    curr_scores = current.scores

    deltas: Dict[str, float] = {}
    for key in ("coverage_fraction", "volume_fill_fraction", "hausdorff_mm", "part_count",
                "structural_plausibility", "overall_score",
                "dfm_violations_error"):
        prev_val = prev_scores.get(key, 0.0)
        curr_val = curr_scores.get(key, 0.0)
        # Handle inf
        if prev_val == float("inf") and curr_val == float("inf"):
            deltas[key] = 0.0
        elif prev_val == float("inf"):
            deltas[key] = -1000.0  # large improvement
        elif curr_val == float("inf"):
            deltas[key] = 1000.0  # large regression
        else:
            deltas[key] = curr_val - prev_val

    # Determine verdict
    # "improved" if objective fraction went up OR hausdorff went down OR overall_score went up
    improved_signals = 0
    regressed_signals = 0

    if deltas.get("coverage_fraction", 0) > 0.01:
        improved_signals += 1
    elif deltas.get("coverage_fraction", 0) < -0.01:
        regressed_signals += 1

    if deltas.get("volume_fill_fraction", 0) > 0.01:
        improved_signals += 1
    elif deltas.get("volume_fill_fraction", 0) < -0.01:
        regressed_signals += 1

    if deltas.get("hausdorff_mm", 0) < -1.0:
        improved_signals += 1
    elif deltas.get("hausdorff_mm", 0) > 1.0:
        regressed_signals += 1

    if deltas.get("overall_score", 0) > 0.01:
        improved_signals += 1
    elif deltas.get("overall_score", 0) < -0.01:
        regressed_signals += 1

    if improved_signals > regressed_signals:
        verdict = "improved"
    elif regressed_signals > improved_signals:
        verdict = "regressed"
    else:
        verdict = "unchanged"

    current.comparison = {
        "previous_path": previous_path,
        "previous_overall_status": prev_data.get("overall_status", "unknown"),
        "current_overall_status": current.overall_status,
        "deltas": deltas,
        "verdict": verdict,
    }


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------

def _make_serializable(obj: Any) -> Any:
    """Recursively convert numpy types to Python native types."""
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_make_serializable(v) for v in obj]
    if isinstance(obj, QualityGate):
        return {
            "name": obj.name,
            "status": obj.status,
            "metric_value": _make_serializable(obj.metric_value),
            "threshold_pass": obj.threshold_pass,
            "threshold_fail": obj.threshold_fail,
            "message": obj.message,
            "recommendations": obj.recommendations,
        }
    if isinstance(obj, float) and (obj == float("inf") or obj == float("-inf")):
        return str(obj)
    return obj


def report_to_json(report: DiagnosisReport) -> dict:
    """Convert a DiagnosisReport to a JSON-serializable dict."""
    d = {
        "version": report.version,
        "mesh_path": report.mesh_path,
        "timestamp": report.timestamp,
        "config_snapshot": report.config_snapshot,
        "mesh_faces": report.mesh_faces,
        "mesh_extents_mm": report.mesh_extents_mm,
        "candidate_generation": report.candidate_generation,
        "selection_trace": report.selection_trace,
        "selection_stop_reason": report.selection_stop_reason,
        "parts": report.parts,
        "scores": report.scores,
        "simulation": report.simulation,
        "quality_gates": [_make_serializable(g) for g in report.quality_gates],
        "overall_status": report.overall_status,
        "summary": report.summary,
        "top_issues": report.top_issues,
        "recommended_actions": report.recommended_actions,
        "comparison": report.comparison,
    }
    return _make_serializable(d)


def report_to_markdown(report: DiagnosisReport) -> str:
    """Render a DiagnosisReport as a markdown string for stdout."""
    lines: List[str] = []
    lines.append(f"# Diagnosis Report — {report.overall_status.upper()}")
    lines.append("")
    lines.append(report.summary)
    lines.append("")

    # Mesh info
    lines.append(f"**Mesh:** {report.mesh_path}")
    lines.append(f"**Faces:** {report.mesh_faces}")
    ext = report.mesh_extents_mm
    if ext:
        lines.append(f"**Extents:** {ext[0]:.0f} x {ext[1]:.0f} x {ext[2]:.0f} mm")
    lines.append("")

    # Scores
    lines.append("## Scores")
    for key, val in report.scores.items():
        lines.append(f"- {key}: {val}")
    lines.append("")

    # Quality gates
    lines.append("## Quality Gates")
    for g in report.quality_gates:
        icon = {"pass": "PASS", "warn": "WARN", "fail": "FAIL"}[g.status]
        lines.append(f"- [{icon}] **{g.name}**: {g.message}")
        for rec in g.recommendations:
            lines.append(f"  - {rec}")
    lines.append("")

    # Parts summary
    lines.append(f"## Parts ({len(report.parts)})")
    for p in report.parts:
        n_err = sum(1 for v in p.get("dfm_violations", []) if v.get("severity") == "error")
        n_warn = sum(1 for v in p.get("dfm_violations", []) if v.get("severity") == "warning")
        dfm_str = ""
        if n_err or n_warn:
            dfm_str = f" (DFM: {n_err}E/{n_warn}W)"
        lines.append(
            f"- {p['name']}: {p['width_mm']:.0f}x{p['height_mm']:.0f}x{p['thickness_mm']:.1f} mm"
            f" [{p.get('material', '?')}]{dfm_str}"
        )
    lines.append("")

    # Top issues
    if report.top_issues:
        lines.append("## Top Issues")
        for issue in report.top_issues:
            lines.append(f"- {issue}")
        lines.append("")

    # Recommended actions
    if report.recommended_actions:
        lines.append("## Recommended Actions")
        for i, action in enumerate(report.recommended_actions, 1):
            lines.append(f"{i}. {action}")
        lines.append("")

    # Comparison
    if report.comparison:
        lines.append("## Comparison with Previous Run")
        lines.append(f"- Verdict: **{report.comparison['verdict']}**")
        lines.append(f"- Previous status: {report.comparison['previous_overall_status']}")
        lines.append(f"- Current status: {report.comparison['current_overall_status']}")
        for key, delta in report.comparison.get("deltas", {}).items():
            sign = "+" if delta > 0 else ""
            lines.append(f"- {key}: {sign}{delta:.4f}")
        lines.append("")

    # Selection trace
    if report.selection_trace:
        lines.append("## Selection Trace")
        lines.append(f"Stop reason: {report.selection_stop_reason}")
        for entry in report.selection_trace:
            rejected = entry.get("dfm_rejected", False)
            tag = " [DFM REJECTED]" if rejected else ""
            lines.append(
                f"- Iter {entry.get('iteration', '?')}: "
                f"{entry.get('source', '?')} "
                f"{entry.get('width_mm', 0):.0f}x{entry.get('height_mm', 0):.0f} mm, "
                f"+{entry.get('new_voxels', 0)} voxels, "
                f"coverage {entry.get('coverage_after', 0):.1%}{tag}"
            )
        lines.append("")

    return "\n".join(lines)
