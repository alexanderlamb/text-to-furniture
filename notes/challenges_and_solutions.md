# Challenges & Solutions: Mesh-to-Parts Decomposition

## The Overall Goal

Imagine you have a 3D model of a piece of furniture — a bookshelf, a desk, a cabinet, a stool. You want to actually build it out of real material. Specifically, you want to build it from flat sheets of plywood (or steel, or aluminum) that get cut into shapes by a CNC machine, then assembled together.

The CNC machine (SendCutSend's service) can cut *any* 2D shape from a flat sheet — circles, curves, L-shapes, whatever. The only constraint is that everything must be flat. The machine can't bend material or carve 3D surfaces. So the fundamental problem is:

**Given a 3D shape, figure out how to break it down into a set of flat pieces that, when cut from sheet stock and assembled together, recreate the original object as faithfully as possible.**

This means the system has to:
1. Look at the 3D mesh and figure out which parts of it are flat surfaces
2. Decide which of those flat surfaces to turn into actual cut parts (within a budget — you don't want 50 parts)
3. Make sure the parts don't physically overlap each other (can't have two sheets occupying the same space)
4. Validate basic manufacturing constraints (parts fit on the sheet, features aren't too small)
5. Output a parametric OpenSCAD assembly model for visual verification

**Not yet implemented:** Joint synthesis (tabs/slots/cross-laps), DXF/SVG cut-file generation, CNC toolpath packaging.

---

## The Pipeline: What Happens Step by Step

The system processes a mesh through six phases, each building on the previous one:

```
Phase 0: Load mesh → auto-scale to target height → center at origin
Phase 1: Extract candidates → find all flat regions from mesh facets
Phase 2: Merge coplanar → combine faces on the same plane into larger panels
Phase 3: Build families → pair opposite faces, assign cavity roles, select under budget
Phase 4: Instantiate panels → emit physical panel layers with 3D positions
Phase 5: Trim overlaps → slab-based half-plane cuts to resolve perpendicular conflicts
Phase 6: Validate → check DFM rules, emit OpenSCAD + spatial capsule + audit trail
```

Each phase writes a checkpoint to disk with counts, metrics, invariants, and a SHA256 hash chain linking to the previous phase.

---

## The Big Challenges

### 1. Finding the Flat Parts Inside a 3D Shape

**What we're trying to accomplish:** A 3D mesh is just thousands of tiny triangles stuck together. We need to identify which groups of triangles form meaningful flat surfaces — surfaces that could become physical sheet-cut parts.

**The problem in detail:** The mesh library (trimesh) gives us "facets" — groups of adjacent triangles that are coplanar. Each facet has a normal vector and an area. But many facets are too small to be useful parts, and the same physical surface might be broken into multiple facets if the mesh topology isn't clean.

**How we solve it — `_extract_candidates` (pipeline.py):**

For each facet in the mesh:
1. Check if its area is above the minimum threshold (default 450mm²)
2. Get its normal vector and derive an orthonormal basis `(basis_u, basis_v)` from it
3. Compute `center_3d` as the mean of all vertices in the facet
4. **Project all the facet's triangles into a 2D plane** and union them together via Shapely. This preserves the actual shape of the surface — an L-shaped surface stays L-shaped, not rectangularized by convex hull.
5. Create a `CandidateRegion` storing the 2D outline, 3D position, normal, basis vectors, source face indices, and area.

**Why triangle union matters:** The CNC machine can cut any 2D shape. By unioning the actual triangles, we preserve concavities, notches, and complex outlines that a convex hull would fill in.

---

### 2. Merging Coplanar Candidates

**What we're trying to accomplish:** The raw extraction often produces multiple separate candidates that should be one part — for example, a shelf splitting a side wall into two halves produces two facet groups on the same plane.

**How we solve it — `_merge_coplanar_candidates` (pipeline.py):**

Group candidates whose normals match (within 2.5°) and whose plane offsets match (within 1mm). For each group:
1. Pick the first candidate as `reference`
2. Reproject all other candidates' 2D polygons into the reference's coordinate frame via `_reproject_candidate_polygon_to_reference` (source 2D → world 3D → reference 2D)
3. Union all projected polygons
4. **Convex hull bridging:** If the union is a MultiPolygon (gap between patches from perpendicular members), check if the convex hull fill ratio > 80% and size balance > 25%. If so, use the convex hull to bridge the gap.
5. Set `center_3d = reference.center_3d` — this must match the 2D coordinate origin, not the averaged center of all source candidates

**Key bug we fixed:** The original code set `center_3d` to the average of all source candidate centers. But the 2D outline was projected relative to `reference.center_3d`. This mismatch shifted merged panels in 3D space — inner side panels of the shelf-unit stuck up 180mm above their correct position.

---

### 3. Pairing Opposite Faces Into Families

**What we're trying to accomplish:** A box has six faces, but the top and bottom are on the same axis — they face opposite directions and are separated by the box's height. Rather than treating them as independent candidates, we pair them into a "family" that knows about the gap between them. This enables stacking — filling the gap with multiple sheets of material.

**How we solve it — `_build_panel_families` (pipeline.py):**

Two candidates are paired if:
- Normals point in opposite directions (dot product < -cos(12°))
- Areas are similar (ratio > 0.55)
- Gap is at least 80% of material thickness
- Lateral offset is small relative to panel size

For each pair, compute:
- **Layer count:** `gap / thickness`, quantized with rounding bias, capped at `max_stack_layers` (6)
- **Shell vs interior:** Thin gaps (< 2*thickness + clearance) get 1 shell layer; wider gaps get 2 shell layers + interior capacity
- **Hollow cavity gate:** If gap > max_stack_layers × thickness, no interior fill (cavity too deep to bridge)
- **Representative:** The candidate with a more outward-facing center becomes the representative

Unmatched candidates become single-layer families.

---

### 4. Cavity Roles and Overlap Prevention

**What we're trying to accomplish:** When multiple families share physical space along the same axis, their stacked layers can collide. We need to prevent this.

**How we solve it — `_assign_cavity_roles` + `_build_family_conflict_edges` (pipeline.py):**

1. **Conflict detection:** For every pair of opposite-pair families with anti-parallel normals, reproject one into the other's 2D frame and check for overlap. If overlap ratio ≥ 5%, record a conflict edge with `max_rep_layers` (how many representative-side layers fit in the separation gap).

2. **Connected components:** Conflicting families form "cavities." Within each cavity, the family with the largest gap (then area, then face count) is designated **primary**; others are **secondary**. Only primary families get interior layers.

3. **Allocation check:** Before allocating any layer (shell or interior), `_can_allocate_representative_layers` verifies that the total representative-side layers for all conflicting families don't exceed the physical space between them.

---

### 5. Selecting Which Families to Keep — The Part Budget

**What we're trying to accomplish:** We might have 15+ families but a budget of 18 panels. We need to pick the set of families and layer counts that best covers the mesh surface.

**The two-pass selection algorithm — `_select_families` (pipeline.py):**

**Pass 1 — Shell reservation (coverage-first):**
Iteratively pick the best unselected family. Score: `0.75 × unique_face_gain_norm + 0.25 × area_norm`, divided by shell cost. Families that would cause overlap conflicts are blocked. Families covering zero new faces get a 0.35× penalty. Stops when budget exhausted.

**Pass 2 — Interior layer allocation (primary owners only):**
Add one interior layer at a time to primary-axis families. Score: `(0.85 × area_norm + 0.15 × gap_norm) × decay`. Decay schedule: [1.0, 0.65, 0.45, 0.30, 0.20] penalizes deeper layers. Same overlap check gates each increment.

---

### 6. Parts Overlap Where They Meet — The Trim Problem

**What we're trying to accomplish:** Panels at perpendicular junctions extend into each other's material space. We need to trim one panel at each junction so they fit without conflict.

**Why this is the hardest problem in the system:** This is a multi-body collision resolution problem. Every design choice cascades — trimming part A affects its area, which affects whether it should have been trimmed at all. And you're doing this across potentially dozens of pairs simultaneously.

**6a. Which panel to trim?**

For each overlapping pair, simulate both trim directions using `_effective_loss()`:
1. Compute the half-plane cut polygon for each direction
2. Subtract it from the target outline
3. Run `_clean_polygon()` on the result (keeps only the largest connected piece)
4. Measure actual area lost

If losses differ by >15%, trim the one with less loss. If within 15%, use structural tiebreaker: prefer trimming smaller-area panel, then horizontal-normal panel, then higher panel_id.

**6b. How to detect the overlap?**

Each panel occupies a "slab" — from `origin_3d` (anti-normal face) to `origin_3d + basis_n × thickness` (outward surface). The **slab intrusion** (`_panel_slab_intrusion` in trim.py) computes what region of panel A's outline lies inside panel B's material slab:

1. Project B's slab normal into A's 2D coordinate frame → linear equation `a*u + b*v`
2. Clip A's outline to the strip where that projection falls between B's slab faces
3. Include asymmetric thickness compensation for non-perpendicular angles

**6c. How to cut without fragmenting?**

Subtracting a thin strip that crosses the full width of a part splits it into two disconnected pieces — catastrophic. `_clean_polygon` would silently discard half the part.

**Solution — half-plane cuts (`_panel_slab_cut`):** Instead of subtracting the thin strip, compute a half-plane starting at the slab entry face (closest to the part being trimmed) extending to infinity. Subtracting a half-plane from a connected polygon always produces a connected polygon — it mathematically cannot fragment.

**6d. Phantom pair prevention**

Two panels whose planes intersect might not actually overlap if they're spatially separated. Three gates prevent false trims:

1. **Parallel skip:** Panels with nearly-parallel normals (dot > 0.95) are skipped
2. **Area gate:** Intrusion area must exceed `trim_min_intrusion_area_mm2` (1.0 mm²)
3. **Cross-projection gate (`_panel_intrusion_overlaps_target`):** Project the intrusion centroid from src's 2D → world 3D → dst's 2D. If it doesn't land within/near dst's outline, the intrusion is a phantom.

**6e. Compounding destruction — the loss budget**

A single panel might need trimming against 5+ neighbors. Without limits, cumulative loss could destroy the panel.

**Solution — incremental loss budget (40% max):**
1. Collect all pending trim polygons for each panel
2. Sort by estimated loss (smallest first — edge trims before mid-panel cuts)
3. Apply sequentially, checking cumulative loss after each
4. Stop when budget exceeded

**6f. Order-independence — collect-then-subtract**

Sequential trimming is order-dependent because each trim changes geometry for subsequent trims.

**Solution — two-phase processing:**
1. **Phase 1 (collect):** Walk all pairs, compute trim directions and cut polygons against ORIGINAL outlines. Store pending cuts without modifying geometry.
2. **Phase 2 (apply):** For each panel, sort pending cuts by loss, apply with budget. Deterministic because sort order is well-defined and inputs are original outlines.

---

### 7. Getting the Slab Geometry Right — Coordinate Conventions

**What we're trying to accomplish:** Every panel exists in two coordinate systems simultaneously. It has a 2D coordinate frame (for its outline polygon) and a 3D position/orientation in the assembly. The trim system needs to convert between these frames correctly.

**The key convention:**
- `origin_3d` = anti-normal face (back of the sheet)
- `origin_3d + basis_n × thickness` = outward surface (front of the sheet)
- 2D point `(u, v)` in outline maps to world as: `origin_3d + u × basis_u + v × basis_v`
- `center_3d` on a CandidateRegion is the origin of the 2D coordinate system — where (0,0) maps to in 3D

**Why `center_3d` must match the 2D origin:** The merged candidate's `outline_2d` is computed by `_reproject_candidate_polygon_to_reference`, which projects all vertices relative to `reference.center_3d`. If `center_3d` is set to something else (like the average of all source centers), there's a mismatch between where the 2D coordinates think they are and where the panel is placed in 3D.

---

### 8. Validation — DFM and Overlap Detection

**What we're trying to accomplish:** Every panel must be checked against manufacturing constraints before the design is usable.

**Current validation (`_validate_panels` in pipeline.py):**

For each panel:
- **Sheet size:** Panel bbox must fit within SendCutSend max sheet dimensions
- **Minimum feature:** Shortest dimension must exceed 3.175mm (1/8" CNC baseline)
- **Minimum area:** Must exceed `min_region_area_mm2`
- **Minimum clearance:** Shapely `minimum_clearance` check

Overlap detection:
- **Parallel panel overlap:** For panels with nearly-identical normals, project one into the other's 2D frame, check for 2D overlap AND normal-axis penetration → **error**
- **Perpendicular panel overlap:** Same slab intrusion math as the trim system, but requires BOTH panels to mutually intrude (post-trim, correctly trimmed junctions don't trigger) → **warning**

---

## The Benchmark Suite

13 test meshes of increasing complexity, from `01_box` (6 faces) to `13_trapezoidal_tray` (non-orthogonal trapezoid with angled walls).

All meshes live in `benchmarks/meshes/` and can be regenerated with `benchmarks/generate_meshes.py`. Run all 13 with:

```bash
for f in benchmarks/meshes/*.stl; do
  venv/bin/python3 scripts/generate_openscad_step1.py --mesh "$f" --name "batch-$(basename "$f" .stl)"
done
```

Key test cases:
- **01_box:** Simplest case. If this fails, everything is broken.
- **04_t_beam, 05_u_channel:** T-junction and U-channel trim direction testing
- **07_h_beam:** Multiple coplanar merges + cavity role assignment
- **08_shelf_unit:** Coplanar merge across perpendicular members (convex hull bridging)
- **11_v_bracket, 12_angled_flange, 13_trapezoidal_tray:** Non-orthogonal angles — the hardest cases for trim math

---

## What's Not Yet Implemented

| Feature | Status | Notes |
|---------|--------|-------|
| Joint synthesis (tabs/slots/cross-laps) | Not started | Requires contact line computation + joint type classification |
| Outline growth (mesh cross-section) | Not started | Extend panels to full coplanar extent |
| DXF/SVG cut-file export | Not started | Needs nested layout packing |
| Quality scoring formula | Not started | Composite fidelity/connectivity/overlap/violation score |
| CNC toolpath packaging | Not started | Complete manufacturing package for SendCutSend |
