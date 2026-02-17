# Debug Artifacts for Agent Feedback

This project now writes step-by-step diagnostics intended for machine parsing.

## Primary files per run

- `runs/<run_id>/metrics.json`
  - `debug.plane_overlap_*`: final overlap metrics and pair details.
  - `debug.plane_overlap_regions`: final overlap region polygons (2D + 3D boundary).
  - `debug.step2_plane_overlap_*`: overlap metrics captured specifically at Phase 2.
  - `debug.step2_plane_overlap_regions`: overlap regions at Phase 2.
  - `debug.step2_trim_debug`: trim search/decision trace from Phase 2.
  - `debug.intersection_events`: pairwise keep/drop reasons from intersection filtering.
  - `debug.intersection_part_decisions`: kept/dropped decisions per candidate part.
  - `debug.intersection_reindex_map`: pre/post part-id mapping after selection reindex.

- `runs/<run_id>/artifacts/debug_trace.json`
  - Compact aggregation of per-phase diagnostics + final `step3_debug`.
  - `phase_diagnostics[*].diagnostics.part_geometry`: per-part area/bounds summaries.
  - `phase_diagnostics[*].diagnostics.plane_overlap`: overlap data for each phase.
  - `phase_diagnostics[*].diagnostics.trim_decisions`: detailed Phase 2 trim logic.

- `runs/<run_id>/artifacts/snapshots/phase_*.json`
  - Full step-through payload with part geometry for each pipeline phase.
  - `diagnostics` field mirrors phase-level debug (overlap, part stats, trim decisions).

## Suite-level outputs

- `runs/suites/<suite_id>/results.csv`
  - Added summary columns:
    - `plane_overlap_pairs`
    - `plane_overlap_region_count`
    - `plane_overlap_total_mm`
    - `step2_plane_overlap_pairs`
    - `step2_plane_overlap_region_count`
    - `step2_trim_search_mode`
    - `step2_trim_minor_pairs_count`
    - `step2_trim_significant_pairs_count`

## Notes

- Overlap regions are directional (A-into-B and B-into-A), so region counts can exceed pair counts.
- Region boundaries are stored as polygons on the source part plane (`outline_2d`) plus projected world-space boundary (`outline_3d`).
