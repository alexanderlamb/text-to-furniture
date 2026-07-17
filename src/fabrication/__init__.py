"""Prototype fabrication strategy tournament API."""

from fabrication.contracts import FabricationConfig, FabricationPlan
from fabrication.context import FabricationContext, build_fabrication_context
from fabrication.tournament import (
    available_strategies,
    run_tournament,
    write_tournament_artifacts,
)
from fabrication.hybrid import run_hybrid_composition, write_hybrid_artifacts

__all__ = [
    "FabricationConfig",
    "FabricationContext",
    "FabricationPlan",
    "available_strategies",
    "build_fabrication_context",
    "run_hybrid_composition",
    "run_tournament",
    "write_hybrid_artifacts",
    "write_tournament_artifacts",
]
