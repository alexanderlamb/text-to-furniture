"""Single-path pipeline: mesh -> first-principles step3 -> run artifacts."""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field, replace
from typing import Dict, List, Optional

from dxf_exporter import design_to_dxf, design_to_nested_dxf
from materials import MATERIALS
from run_protocol import (
    copy_input_mesh,
    prepare_run_dir,
    update_latest_pointer,
    write_json,
    write_text,
)
from step3_first_principles import Step3Input, Step3Output, decompose_first_principles
from svg_exporter import design_to_nested_svg, design_to_svg

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    runs_dir: str = "runs"
    export_svg: bool = True
    export_nested_svg: bool = True
    export_dxf: bool = True
    export_nested_dxf: bool = True
    step3_input: Optional[Step3Input] = None


@dataclass
class PipelineResult:
    run_id: str
    run_dir: str
    design_json_path: str
    metrics_path: str
    summary_path: str
    manifest_path: str
    mesh_input_path: str
    svg_paths: List[str] = field(default_factory=list)
    nested_svg_path: Optional[str] = None
    dxf_paths: List[str] = field(default_factory=list)
    nested_dxf_path: Optional[str] = None
    step3_output: Optional[Step3Output] = None


def run_pipeline_from_mesh(
    mesh_path: str,
    design_name: str = "design",
    config: Optional[PipelineConfig] = None,
) -> PipelineResult:
    if not os.path.isfile(mesh_path):
        raise FileNotFoundError(f"Mesh file not found: {mesh_path}")

    if config is None:
        config = PipelineConfig()

    started = time.perf_counter()
    paths = prepare_run_dir(config.runs_dir, design_name)
    copied_mesh = copy_input_mesh(mesh_path, paths.input_dir)

    if config.step3_input is None:
        step3_input = Step3Input(mesh_path=str(copied_mesh), design_name=design_name)
    else:
        step3_input = replace(
            config.step3_input,
            mesh_path=str(copied_mesh),
            design_name=design_name,
        )

    logger.info("Running first-principles strategy for %s", copied_mesh)
    step3 = decompose_first_principles(step3_input)
    design = step3.design
    design.name = design_name

    design_json_path = paths.artifacts_dir / "design_first_principles.json"
    write_json(design_json_path, step3.to_manufacturing_json())

    # Write phase snapshots for step-through debugging
    phase_snapshots = step3.debug.get("phase_snapshots", [])
    snapshot_paths: List[str] = []
    if phase_snapshots:
        snapshots_dir = paths.artifacts_dir / "snapshots"
        snapshots_dir.mkdir(parents=True, exist_ok=True)
        for snap in phase_snapshots:
            label_slug = (
                snap["phase_label"]
                .lower()
                .replace(" ", "_")
                .replace("(", "")
                .replace(")", "")
            )
            fname = f"phase_{snap['phase_index']:02d}_{label_slug}.json"
            snap_path = snapshots_dir / fname
            write_json(snap_path, snap)
            snapshot_paths.append(str(snap_path))

    # Compact machine-readable debug trace for agents/tools.
    debug_trace_path = paths.artifacts_dir / "debug_trace.json"
    phase_diagnostics = []
    if isinstance(phase_snapshots, list):
        for snap in phase_snapshots:
            if not isinstance(snap, dict):
                continue
            phase_diagnostics.append(
                {
                    "phase_index": snap.get("phase_index"),
                    "phase_label": snap.get("phase_label"),
                    "part_count": snap.get("part_count"),
                    "diagnostics": snap.get("diagnostics", {}),
                }
            )
    write_json(
        debug_trace_path,
        {
            "run_id": paths.run_id,
            "phase_snapshot_paths": snapshot_paths,
            "phase_diagnostics": phase_diagnostics,
            "step3_debug": {
                k: v for k, v in step3.debug.items() if k != "phase_snapshots"
            },
        },
    )

    svg_paths: List[str] = []
    nested_svg_path = None
    if config.export_svg:
        svg_dir = paths.artifacts_dir / "svg"
        svg_paths = design_to_svg(design, str(svg_dir))
        if config.export_nested_svg:
            nested_svg_path = str(svg_dir / f"{design_name}_nested.svg")
            design_to_nested_svg(design, nested_svg_path)

    dxf_paths: List[str] = []
    nested_dxf_path = None
    if config.export_dxf and step3.parts:
        dxf_dir = paths.artifacts_dir / "dxf"
        dxf_parts = [(name, part.profile) for name, part in step3.parts.items()]
        dxf_paths = design_to_dxf(dxf_parts, str(dxf_dir))

        if config.export_nested_dxf:
            material_key = next(iter(step3.parts.values())).material_key
            sheet_w, sheet_h = MATERIALS[material_key].max_size_mm
            nested_dxf_path = str(dxf_dir / f"{design_name}_nested.dxf")
            design_to_nested_dxf(
                dxf_parts,
                nested_dxf_path,
                sheet_width_mm=sheet_w,
                sheet_height_mm=sheet_h,
            )

    elapsed = time.perf_counter() - started

    metrics_payload: Dict[str, object] = {
        "run_id": paths.run_id,
        "status": step3.status,
        "elapsed_s": round(elapsed, 3),
        "quality_metrics": asdict(step3.quality_metrics),
        "violations": [asdict(v) for v in step3.violations],
        "debug": {k: v for k, v in step3.debug.items() if k != "phase_snapshots"},
        "counts": {
            "components": len(design.components),
            "joints": len(design.assembly.joints),
            "svg_files": len(svg_paths),
            "dxf_files": len(dxf_paths),
        },
    }
    write_json(paths.metrics_path, metrics_payload)

    summary = _build_summary(step3, paths.run_id, elapsed, svg_paths, dxf_paths)
    write_text(paths.summary_path, summary)

    # Optional overlay screenshot (best-effort, requires matplotlib).
    try:
        from overlay_viewer import render_overlay_screenshot

        screenshot_path = str(paths.artifacts_dir / "overlay_screenshot.png")
        render_overlay_screenshot(str(paths.run_dir), screenshot_path)
    except Exception as exc:
        logger.debug("Overlay screenshot skipped: %s", exc)

    manifest = {
        "run_id": paths.run_id,
        "strategy": "first_principles_step3",
        "design_name": design_name,
        "input_mesh": str(copied_mesh),
        "status": step3.status,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "config": {
            "pipeline": asdict(config),
            "step3_input": asdict(step3_input),
        },
        "artifacts": {
            "design_json": str(design_json_path),
            "metrics": str(paths.metrics_path),
            "summary": str(paths.summary_path),
            "debug_trace": str(debug_trace_path),
            "phase_snapshots": snapshot_paths,
            "svg": svg_paths,
            "nested_svg": nested_svg_path,
            "dxf": dxf_paths,
            "nested_dxf": nested_dxf_path,
        },
    }
    write_json(paths.manifest_path, manifest)
    update_latest_pointer(config.runs_dir, paths.run_dir)

    return PipelineResult(
        run_id=paths.run_id,
        run_dir=str(paths.run_dir),
        design_json_path=str(design_json_path),
        metrics_path=str(paths.metrics_path),
        summary_path=str(paths.summary_path),
        manifest_path=str(paths.manifest_path),
        mesh_input_path=str(copied_mesh),
        svg_paths=svg_paths,
        nested_svg_path=nested_svg_path,
        dxf_paths=dxf_paths,
        nested_dxf_path=nested_dxf_path,
        step3_output=step3,
    )


def _build_summary(
    output: Step3Output,
    run_id: str,
    elapsed_s: float,
    svg_paths: List[str],
    dxf_paths: List[str],
) -> str:
    err = sum(1 for v in output.violations if v.severity == "error")
    warn = sum(1 for v in output.violations if v.severity == "warning")

    lines = [
        f"# Run {run_id}",
        "",
        f"- Status: **{output.status.upper()}**",
        f"- Duration: {elapsed_s:.2f}s",
        f"- Parts: {output.quality_metrics.part_count}",
        f"- Joints: {len(output.joints)}",
        f"- Hausdorff: {output.quality_metrics.hausdorff_mm:.2f} mm",
        f"- Normal error: {output.quality_metrics.normal_error_deg:.2f} deg",
        f"- Overall score: {output.quality_metrics.overall_score:.3f}",
        f"- Violations: {err} errors, {warn} warnings",
        f"- SVG files: {len(svg_paths)}",
        f"- DXF files: {len(dxf_paths)}",
        "",
        "## Key Violations",
    ]

    if not output.violations:
        lines.append("- None")
    else:
        for v in output.violations[:12]:
            part = f" ({v.part_id})" if v.part_id else ""
            lines.append(f"- [{v.severity}] {v.code}{part}: {v.message}")

    return "\n".join(lines) + "\n"
