"""
Per-session state management for the web UI.

Two dataclasses:
- RunSlot: one per comparison column (decomposition config + results)
- AppState: session root (mesh path, two comparison slots)
"""

from dataclasses import dataclass, field
from typing import Optional, List

from mesh_decomposer import DecompositionConfig
from pipeline import PipelineResult


@dataclass
class RunSlot:
    """State for one comparison column."""

    config: DecompositionConfig = field(default_factory=DecompositionConfig)
    result: Optional[PipelineResult] = None
    running: bool = False
    error: Optional[str] = None
    svg_content: Optional[str] = None
    run_simulation: bool = True
    mfg_summary: Optional[dict] = None
    progress_file: Optional[str] = None  # temp file for subprocess progress updates
    debug_report: Optional[dict] = None
    debug_report_path: Optional[str] = None


@dataclass
class AppState:
    """Root session state."""

    mesh_path: Optional[str] = None
    mesh_filename: Optional[str] = None
    mesh_serving_url: Optional[str] = None
    slots: List[RunSlot] = field(default_factory=lambda: [RunSlot(), RunSlot()])
