"""
Greedy slab selection for v2 manufacturing pipeline.

Supports two objectives:
1) surface_coverage: maximize shell voxel coverage (legacy behavior)
2) volume_fill: maximize interior voxel fill with a per-plane penalty
"""
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import trimesh
from scipy.ndimage import binary_erosion

from slab_candidates import Slab3D, slab3d_to_part_profile
from dfm_rules import DFMConfig, check_part_dfm

logger = logging.getLogger(__name__)


@dataclass
class SlabSelectionConfig:
    """Configuration for slab selection."""
    coverage_target: float = 0.80
    target_volume_fill: float = 0.55
    max_slabs: int = 20
    min_coverage_contribution: float = 0.002
    min_volume_contribution: float = 0.001
    plane_penalty_weight: float = 0.0005
    diversity_weight: float = 2.0
    voxel_resolution_mm: float = 5.0
    dfm_check: bool = True
    objective_mode: str = "surface_coverage"  # "surface_coverage" | "volume_fill"


@dataclass
class SlabSelection:
    """Result of slab selection."""
    selected_slabs: List[Slab3D]
    coverage_fraction: float
    uncovered_points: np.ndarray  # remaining objective voxels not covered
    selection_trace: List[dict] = field(default_factory=list)
    stop_reason: str = ""
    objective_mode: str = "surface_coverage"
    volume_fill_fraction: float = 0.0
    surface_coverage_fraction: float = 0.0
    candidate_count_before_dfm: int = 0
    candidate_count_after_dfm: int = 0
    dfm_rejected_count: int = 0


def select_slabs(
    mesh: trimesh.Trimesh,
    candidates: List[Slab3D],
    config: Optional[SlabSelectionConfig] = None,
    dfm_config: Optional[DFMConfig] = None,
) -> SlabSelection:
    """Select a slab set under the configured objective."""
    if config is None:
        config = SlabSelectionConfig()

    if config.objective_mode not in ("surface_coverage", "volume_fill"):
        raise ValueError(
            f"Unknown objective_mode '{config.objective_mode}'. "
            "Expected 'surface_coverage' or 'volume_fill'.",
        )

    candidate_pairs, dfm_reject_count = _prefilter_candidates_by_dfm(
        candidates, config, dfm_config,
    )

    if not candidate_pairs and dfm_reject_count > 0:
        logger.warning(
            "DFM pre-filter rejected all %d candidates — falling back to unfiltered",
            dfm_reject_count,
        )
        candidate_pairs = list(enumerate(candidates))
        dfm_reject_count = 0  # reset since we're skipping the filter

    if not candidate_pairs:
        stop_reason = "no_dfm_valid_candidates" if config.dfm_check else "no_candidates"
        logger.info(
            "Selection complete: slabs=0 mode=%s fraction=0.0%% dfm_rejected=%d stop_reason=%s",
            config.objective_mode, dfm_reject_count, stop_reason,
        )
        return SlabSelection(
            selected_slabs=[],
            coverage_fraction=0.0,
            uncovered_points=np.array([]).reshape(0, 3),
            stop_reason=stop_reason,
            objective_mode=config.objective_mode,
            volume_fill_fraction=0.0,
            surface_coverage_fraction=0.0,
            candidate_count_before_dfm=len(candidates),
            candidate_count_after_dfm=0,
            dfm_rejected_count=dfm_reject_count,
        )

    if config.objective_mode == "volume_fill":
        vgrid: _VoxelGridBase = _FilledVoxelGrid(mesh, config.voxel_resolution_mm)
        objective_target = config.target_volume_fill
    else:
        vgrid = _SurfaceVoxelGrid(mesh, config.voxel_resolution_mm)
        objective_target = config.coverage_target

    logger.info(
        "Objective grid: mode=%s shape=%s target_voxels=%d candidates=%d->%d",
        config.objective_mode, vgrid.shape, vgrid.total_target,
        len(candidates), len(candidate_pairs),
    )

    if vgrid.total_target == 0:
        logger.info(
            "Selection complete: slabs=0 mode=%s fraction=100.0%% dfm_rejected=%d stop_reason=empty_target_grid",
            config.objective_mode, dfm_reject_count,
        )
        return SlabSelection(
            selected_slabs=[],
            coverage_fraction=1.0,
            uncovered_points=np.array([]).reshape(0, 3),
            stop_reason="empty_target_grid",
            objective_mode=config.objective_mode,
            volume_fill_fraction=1.0 if config.objective_mode == "volume_fill" else 0.0,
            surface_coverage_fraction=1.0 if config.objective_mode == "surface_coverage" else 0.0,
            candidate_count_before_dfm=len(candidates),
            candidate_count_after_dfm=len(candidate_pairs),
            dfm_rejected_count=dfm_reject_count,
        )

    selected: List[Slab3D] = []
    selected_normals: List[np.ndarray] = []
    remaining = list(range(len(candidate_pairs)))
    stop_reason = "max_slabs_reached"
    trace: List[dict] = []

    for _ in range(config.max_slabs):
        current_fraction = vgrid.coverage_fraction()
        if current_fraction >= objective_target:
            stop_reason = (
                "volume_target_met"
                if config.objective_mode == "volume_fill"
                else "coverage_target_met"
            )
            break

        best_score = -1.0
        best_new_voxels = -1
        best_idx = -1
        best_rem_pos = -1

        for rem_pos, cand_idx in enumerate(remaining):
            _, candidate = candidate_pairs[cand_idx]
            new_voxels = vgrid.count_new_coverage_slab(candidate)

            # Diminishing returns for same-direction slabs
            diversity_discount = 1.0
            if config.diversity_weight > 0 and selected_normals:
                cn = candidate.normal / max(np.linalg.norm(candidate.normal), 1e-12)
                n_similar = sum(
                    1 for sn in selected_normals
                    if abs(float(cn @ sn)) > 0.9
                )
                diversity_discount = 1.0 / (1.0 + config.diversity_weight * n_similar)

            if config.objective_mode == "volume_fill":
                contribution = new_voxels / max(vgrid.total_target, 1)
                score = contribution * diversity_discount - config.plane_penalty_weight
            else:
                score = float(new_voxels) * diversity_discount

            if score > best_score:
                best_score = float(score)
                best_new_voxels = int(new_voxels)
                best_idx = cand_idx
                best_rem_pos = rem_pos

        if best_idx < 0 or best_new_voxels <= 0:
            stop_reason = (
                "no_volume_gain"
                if config.objective_mode == "volume_fill"
                else "no_coverage_gain"
            )
            logger.info(
                "Stopping selection: best_new_voxels=%d remaining=%d",
                best_new_voxels, len(remaining),
            )
            break

        fraction = best_new_voxels / max(vgrid.total_target, 1)
        if config.objective_mode == "volume_fill":
            if fraction < config.min_volume_contribution:
                stop_reason = "below_min_volume_contribution"
                logger.info(
                    "Stopping selection: volume contribution %.4f below threshold %.4f",
                    fraction, config.min_volume_contribution,
                )
                break
            if best_score <= 0.0:
                stop_reason = "non_positive_objective_gain"
                logger.info(
                    "Stopping selection: objective gain %.6f <= 0 "
                    "(contribution %.4f, penalty %.4f)",
                    best_score, fraction, config.plane_penalty_weight,
                )
                break
        elif fraction < config.min_coverage_contribution:
            stop_reason = "below_min_contribution"
            logger.info(
                "Stopping selection: coverage contribution %.4f below threshold %.4f",
                fraction, config.min_coverage_contribution,
            )
            break

        original_idx, chosen = candidate_pairs[best_idx]
        vgrid.mark_covered_slab(chosen)
        selected.append(chosen)
        selected_normals.append(
            chosen.normal / max(np.linalg.norm(chosen.normal), 1e-12)
        )
        remaining.pop(best_rem_pos)

        post_fraction = float(vgrid.coverage_fraction())
        trace.append({
            "iteration": len(selected),
            "candidate_index": int(original_idx),
            "source": chosen.source,
            "width_mm": float(chosen.width_mm),
            "height_mm": float(chosen.height_mm),
            "thickness_mm": float(chosen.thickness_mm),
            "new_voxels": int(best_new_voxels),
            "contribution_fraction": float(fraction),
            "objective_score": float(best_score),
            "coverage_after": post_fraction,
            "objective_mode": config.objective_mode,
            "dfm_rejected": False,
        })

        logger.info(
            "Selected slab %d (%s): %.0fx%.0f mm, +%d voxels "
            "(objective=%.6f, total %.1f%%)",
            len(selected), chosen.source,
            chosen.width_mm, chosen.height_mm,
            best_new_voxels, best_score, post_fraction * 100,
        )

    uncovered = vgrid.get_uncovered_positions()
    current_fraction = float(vgrid.coverage_fraction())
    volume_fill = current_fraction if config.objective_mode == "volume_fill" else 0.0
    surface_cov = current_fraction if config.objective_mode == "surface_coverage" else 0.0

    logger.info(
        "Selection complete: slabs=%d mode=%s fraction=%.1f%% dfm_rejected=%d stop_reason=%s",
        len(selected), config.objective_mode, current_fraction * 100,
        dfm_reject_count, stop_reason,
    )

    return SlabSelection(
        selected_slabs=selected,
        coverage_fraction=current_fraction,
        uncovered_points=uncovered,
        selection_trace=trace,
        stop_reason=stop_reason,
        objective_mode=config.objective_mode,
        volume_fill_fraction=float(volume_fill),
        surface_coverage_fraction=float(surface_cov),
        candidate_count_before_dfm=len(candidates),
        candidate_count_after_dfm=len(candidate_pairs),
        dfm_rejected_count=dfm_reject_count,
    )


def _prefilter_candidates_by_dfm(
    candidates: List[Slab3D],
    config: SlabSelectionConfig,
    dfm_config: Optional[DFMConfig],
) -> Tuple[List[Tuple[int, Slab3D]], int]:
    """Filter out candidates with DFM errors before greedy selection."""
    if not candidates:
        return [], 0
    if not config.dfm_check:
        return list(enumerate(candidates)), 0

    kept: List[Tuple[int, Slab3D]] = []
    rejected = 0
    for idx, cand in enumerate(candidates):
        profile = slab3d_to_part_profile(cand)
        violations = check_part_dfm(profile, dfm_config)
        errors = [v for v in violations if v.severity == "error"]
        if errors:
            rejected += 1
            continue
        kept.append((idx, cand))
    return kept, rejected


class _VoxelGridBase:
    """Voxel grid with a boolean target mask for objective coverage."""

    def __init__(self, mesh: trimesh.Trimesh, resolution_mm: float, use_surface_shell: bool):
        self.pitch = resolution_mm
        vox = mesh.voxelized(pitch=resolution_mm)
        vox.fill()
        filled = vox.matrix.copy()
        self.origin = vox.transform[:3, 3].copy()

        if use_surface_shell:
            eroded = binary_erosion(filled)
            self.target = filled & ~eroded
        else:
            self.target = filled

        self.covered = np.zeros_like(self.target, dtype=bool)
        self.total_target = int(self.target.sum())

    @property
    def shape(self):
        return self.target.shape

    def coverage_fraction(self) -> float:
        if self.total_target == 0:
            return 1.0
        return float((self.target & self.covered).sum()) / self.total_target

    def count_new_coverage_slab(self, slab: Slab3D) -> int:
        mask = self._slab_mask(slab)
        return int((mask & self.target & ~self.covered).sum())

    def mark_covered_slab(self, slab: Slab3D):
        mask = self._slab_mask(slab)
        self.covered |= (mask & self.target)

    def get_uncovered_positions(self) -> np.ndarray:
        uncov = self.target & ~self.covered
        ii, jj, kk = np.where(uncov)
        if len(ii) == 0:
            return np.array([]).reshape(0, 3)
        return self.origin + np.column_stack([ii, jj, kk]) * self.pitch + self.pitch / 2

    def _slab_mask(self, slab: Slab3D) -> np.ndarray:
        n = slab.normal / np.linalg.norm(slab.normal)
        hw = slab.width_mm / 2.0
        hh = slab.height_mm / 2.0
        # Use at least one voxel pitch for thickness
        ht = max(slab.thickness_mm / 2.0, self.pitch)

        ii, jj, kk = np.where(self.target)
        if len(ii) == 0:
            return np.zeros_like(self.target, dtype=bool)

        centres = self.origin + np.column_stack([ii, jj, kk]) * self.pitch + self.pitch / 2
        diff = centres - slab.origin

        proj_u = diff @ slab.basis_u
        proj_v = diff @ slab.basis_v
        proj_n = diff @ n

        inside = (
            (np.abs(proj_u) <= hw)
            & (np.abs(proj_v) <= hh)
            & (np.abs(proj_n) <= ht)
        )

        mask = np.zeros_like(self.target, dtype=bool)
        mask[ii[inside], jj[inside], kk[inside]] = True
        return mask


class _SurfaceVoxelGrid(_VoxelGridBase):
    """Objective grid using the mesh shell."""

    def __init__(self, mesh: trimesh.Trimesh, resolution_mm: float):
        super().__init__(mesh, resolution_mm, use_surface_shell=True)


class _FilledVoxelGrid(_VoxelGridBase):
    """Objective grid using the filled mesh volume."""

    def __init__(self, mesh: trimesh.Trimesh, resolution_mm: float):
        super().__init__(mesh, resolution_mm, use_surface_shell=False)
