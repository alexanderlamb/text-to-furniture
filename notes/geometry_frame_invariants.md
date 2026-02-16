# Geometry Frame Invariants (Step 0 / Phase Snapshots)

Date: 2026-02-16

## Context

While debugging `13_trapezoidal_tray` step-0 geometry, we hit mirrored and offset long-side artifacts:

- one run: long side looked short in `+x`
- another run: `+y` side short on `+x`, `-y` side short on `-x`

The mesh itself was symmetric; the bug was in part-frame handling.

## Root Cause

Two frame-consistency mistakes in `src/step3_first_principles.py`:

1. `_merge_planar_pair` canonicalized the chosen normal (`_canonicalize_normal`) but kept the chosen profile frame (`outline_2d`, `basis_u`, `basis_v`, `origin_3d`) unchanged.
2. `_select_candidates` anchored `position_3d` from candidate centroid (`cand.position_3d`) even though `outline_2d` is defined relative to `origin_3d`.

Both issues break a core invariant: rotation/normal and outline coordinates must be interpreted in the same local frame anchor.

## Correct Invariant

For planar parts, these must be frame-consistent:

- `outline_2d`
- `basis_u`, `basis_v`
- `origin_3d`
- `normal` / `rotation_3d`
- placement anchor used to transform `outline_2d` into world space

If you flip/re-canonicalize normal, you must also rotate/remap the 2D frame and outline.
If you do not remap frame data, keep the original chosen normal.

## Fix Applied

- Keep merged planar normal in chosen profile frame (no canonical flip).
- Build planar `position_3d` from `origin_3d` (fallback to centroid only if origin is missing).

## Practical Debug Check

When step-0 looks asymmetric:

1. Transform `outline_2d` with `rotation_3d + position_3d` and compare against mesh section on that plane.
2. Repeat with `rotation_3d + origin_3d`.
3. If these differ significantly, anchor/frame mismatch is present.

For `20260216_011651_design`, this check reproduced the opposite-direction side offsets.
For `20260216_012555_design`, both long sides aligned with mesh section bounds (within ~0.1 mm tolerance).
