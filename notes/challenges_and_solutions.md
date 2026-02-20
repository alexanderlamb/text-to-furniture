# Challenges & Solutions: Mesh-to-Parts Decomposition

## The Overall Goal

Imagine you have a 3D model of a piece of furniture — a bookshelf, a desk, a cabinet, a stool. You want to actually build it out of real material. Specifically, you want to build it from flat sheets of plywood (or steel, or aluminum) that get cut into shapes by a CNC machine, then assembled together.

The CNC machine (SendCutSend's service) can cut *any* 2D shape from a flat sheet — circles, curves, L-shapes, whatever. The only constraint is that everything must be flat. The machine can't bend material or carve 3D surfaces. So the fundamental problem is:

**Given a 3D shape, figure out how to break it down into a set of flat pieces that, when cut from sheet stock and assembled together, recreate the original object as faithfully as possible.**

This means the system has to:
1. Look at the 3D mesh and figure out which parts of it are flat surfaces
2. Decide which of those flat surfaces to turn into actual cut parts (within a budget — you don't want 50 parts)
3. Make sure the parts don't physically overlap each other (can't have two sheets occupying the same space)
4. Figure out how the parts connect to each other (tabs, slots, laps — the physical interlocking geometry)
5. Make sure every part is actually manufacturable (features aren't too small for the cutting tool, material won't break, etc.)
6. Output cut files (SVG/DXF) that can go directly to the CNC service

The output is a complete manufacturing package: cut files you send to SendCutSend, they cut the parts, ship them to you, and you assemble the furniture.

---

## The Pipeline: What Happens Step by Step

The system processes a mesh through a series of phases, each building on the previous one. Here is the full flow, and then we'll dive deep into the challenges at each step.

```
1. Load mesh → normalize scale
2. Extract candidates → find all flat regions
3. Merge coplanar → combine faces on the same plane
4. Collapse opposite pairs → top+bottom of box → single stacked region
5. Select best candidates → greedy coverage within part budget
6. Grow outlines → expand to full coplanar extent via mesh cross-section
7. Trim overlaps → slab subtraction to resolve physical conflicts
8. Synthesize joints → figure out where/how parts connect
9. Apply joint geometry → cut tabs, slots, cross-laps into the parts
10. Post-joint cleanup → resolve any residual overlaps from joint geometry
11. Validate DFM → check all parts against manufacturing constraints
12. Score quality → single composite number for the decomposition
13. Export → SVG/DXF cut files + debug artifacts
```

Each phase writes a snapshot to disk so you can see exactly what the geometry looked like at every step, which is critical for debugging when things go wrong.

---

## The Big Challenges

### 1. Finding the Flat Parts Inside a 3D Shape

**What we're trying to accomplish:** A 3D mesh is just thousands of tiny triangles stuck together. We need to identify which groups of triangles form meaningful flat surfaces — surfaces that could become physical sheet-cut parts. The goal is to find every planar region on the mesh that's worth considering as a part candidate.

**The problem in detail:** The mesh library (trimesh) gives us "facets" — groups of adjacent triangles that are coplanar (lie on the same plane). Each facet has a normal vector (which direction the face points) and an area. But:

- Many facets are too small to be useful parts (the mesh might have tiny triangles at edges or transitions). We filter out anything below 500mm² by default.
- The same physical surface might be broken into multiple facets if the mesh topology isn't clean. Two separate groups of triangles on the same plane need to be recognized as one surface.
- Some surfaces are curved (like the inside of a bowl), which can't become a flat sheet. These "unsupported" faces are counted but skipped, and if too many faces are unsupported, the system raises a warning.

**How we solve it — candidate extraction (`_extract_candidates`):**

For each facet in the mesh:
1. Check if its area is above the minimum threshold (500mm²)
2. Get its normal vector (which way the face points)
3. **Project all the facet's triangles into a 2D plane** and union them together. This is a critical design choice — we use triangle union, not convex hull. A convex hull would turn an L-shaped surface into a rectangle, losing the concavity. Triangle union preserves the actual shape of the surface, including notches, angles, and curves.
4. If the triangle union fails (degenerate geometry), fall back to convex hull as a last resort.
5. Create a `_RegionCandidate` that stores the 2D outline, the 3D position, the normal, the source face indices, and metadata.

For faces that aren't flat, we also extract "bend candidates" — curved regions that could potentially be formed by bending sheet metal. This only applies to materials like steel and aluminum that can be bent with controlled forming. Plywood can't bend.

**Why triangle union matters:** Consider an L-shaped surface on a mesh. If you take the convex hull of all its vertices, you get a rectangle — the convex hull fills in the notch. But the CNC machine can cut an L-shape just as easily as a rectangle. By using triangle union (taking each triangle as a tiny polygon and unioning them all together), we get the actual L-shaped outline. The CNC machine can cut any shape, so we should preserve the real geometry.

---

### 2. Merging and Collapsing — Reducing Redundant Candidates

**What we're trying to accomplish:** The raw candidate extraction often produces multiple separate candidates that should really be one part. Before we start selecting which parts to keep, we need to consolidate the candidate list so we're making clean choices.

**Two levels of merging:**

**Level 1 — Coplanar merge (`_merge_coplanar_candidates`):** Multiple facets might lie on the exact same plane but not be physically adjacent on the mesh. For example, a shelf might have three separate facet groups all on the same horizontal plane at the same height. These should be one candidate, not three. We group candidates whose normals match (dot product > 0.99) and whose plane offsets match (within 1mm), then re-project all their combined faces into a single unified outline.

**Level 2 — Opposite-face collapse (`_collapse_planar_face_pairs`):** A box has six faces, but the top and bottom are on the same axis — they face opposite directions and are separated by the box's height. Rather than treating them as two independent part candidates, we "collapse" them into one candidate that knows about the gap between them. This enables *stacking* — the system can fill that gap with multiple sheets of material. For example, if a box wall is 20mm thick and the sheet material is 5mm thick, we can stack 4 sheets.

**The subtle part — determining the pair:** Two candidates are an "opposite planar pair" if:
- Their normals point in opposite directions (dot product < -0.96)
- Their areas are similar (within 30% relative difference)
- Their bounding dimensions are similar (within 25%)
- They're separated in the normal direction (at least 0.5mm apart)
- They're not too far apart (less than 40mm or 45% of the largest dimension)
- They're roughly aligned laterally (not offset sideways by more than 15% of the largest dimension)

When collapsing a pair, we also classify which face is "exterior" (faces open space, determined by ray casting) vs "interior" (faces into the mesh volume). This matters for stack alignment — when placing multiple sheets, we want the outermost sheet flush with the exterior surface so the outside of the furniture looks clean.

---

### 3. Selecting Which Parts to Keep — The Part Budget

**What we're trying to accomplish:** After candidate generation and merging, we might have 15-20 candidates. But we have a budget of (say) 10 parts. We need to pick the set of parts that best covers the original mesh surface. This is essentially a weighted set-cover problem — each candidate "covers" a set of mesh faces, and we want to maximize total coverage within our budget.

**The greedy coverage algorithm (`_greedy_coverage_rank`):**

Instead of sorting by raw area (which would pick the largest parts first, even if they overlap), we use a greedy set-cover approach:

1. Start with no faces covered.
2. For each remaining candidate, compute its "marginal gain" — how many *uncovered* faces it would add, as a fraction of its total faces, multiplied by its area score.
3. Pick the candidate with the highest marginal gain. Mark its faces as covered.
4. Repeat until all candidates are ranked.

This naturally prioritizes candidates that cover unique parts of the mesh. A large candidate that overlaps with already-selected parts gets a lower marginal gain than a medium candidate that covers an entirely new region.

**Stacking:** For collapsed pairs (where the gap between opposite faces allows multiple sheets), we compute how many layers to stack using `_target_stack_layers`. If a wall is 20mm thick and sheets are 4.76mm (3/16" plywood), we might stack 4 layers. Each extra layer counts against the part budget but has diminishing returns — the second layer gets 65% of the first layer's gain, the third gets 55%, etc.

**Thin-side suppression:** Some candidates represent the narrow edges of a shape — for example, the 4.76mm-thick edge of a shelf. These "thin sides" use a whole budget slot but contribute very little surface coverage. The system detects these by comparing the candidate's narrowest dimension to the material thickness, and either penalizes their score (so they're less likely to be selected) or drops them entirely if they would waste too much budget.

**Where the sheets go in 3D space:** Once we know how many layers to place for each selected region, we need to compute the exact 3D position of each sheet. Two alignment modes:
- **Exterior flush:** The outermost sheet is placed flush with the exterior face. Additional sheets stack inward. This makes the outside surface clean.
- **Centered:** Sheets are centered within the member gap. Used when we can't determine which face is exterior.

**Post-selection intersection filter:** After selection, we check for parts that physically intersect in ways that aren't joint-related. Parts with too many unresolvable intersections get dropped. This prevents the assembly from including parts that could never physically coexist.

---

### 4. Growing Outlines to Cover the Full Surface

**What we're trying to accomplish:** The initial candidate outlines only cover the mesh faces that were exactly coplanar. But a wall panel should extend all the way to its edges — where it meets the floor, the ceiling, and the adjacent walls. The outline needs to grow from "just the coplanar triangles" to "the full extent of the planar surface within the mesh volume."

**The problem in detail:** Consider a rectangular box. The front face might only have 4 triangles that form the facet. But the physical wall of the box extends from the left edge to the right edge, from top to bottom. If we don't grow the outline, we'd cut a part that's slightly smaller than the full wall, leaving gaps at the corners where walls meet.

**How we solve it — mesh cross-section (`_compute_coplanar_extent`):**

1. Take the part's plane (defined by its position and normal vector).
2. Slice the mesh at that plane using `trimesh.section()`. This gives us the exact contour of where the mesh volume intersects the plane — essentially a cross-section of the 3D object.
3. Project that 3D contour into the part's 2D coordinate frame.
4. Keep only the connected component that overlaps the original outline (so we don't accidentally include a detached region from another part of the mesh).
5. Union this with the original outline to create the grown outline.

**Edge case — inset slicing:** Sometimes the part's plane sits exactly on a mesh face boundary. The cross-section at that exact position might be degenerate (a line instead of a polygon). So we try slicing at slight offsets in both directions (±0.1mm from the plane) and pick whichever yields more geometry.

**Preserving holes:** If the original outline had interior holes (like a window cutout), the growth step must preserve them. After growing the exterior, we re-punch the original holes back into the new outline.

---

### 5. Parts Overlap Where They Meet — The Trim Problem

**What we're trying to accomplish:** After growing outlines, parts will overlap where they physically meet. Two walls at a corner, a shelf meeting a side panel, a desk leg meeting the tabletop — at all these junctions, both parts extend into the same physical space. We need to trim one of them back so they fit together without conflict. The goal is to resolve every overlap while preserving as much of each part's surface area as possible.

**Why this is the hardest problem in the system:** This is fundamentally a multi-body collision resolution problem with geometric constraints. Every design choice cascades — trimming part A affects its area, which affects its usefulness, which might affect whether it should have been trimmed at all. And you're doing this across potentially dozens of part pairs simultaneously.

The specific sub-challenges:

**5a. Which part to trim?**

When parts A and B overlap, you could trim A or trim B. The wrong choice can be catastrophic. Consider a T-junction: a long shelf meets a short vertical divider. If you trim the shelf, you cut a notch out of a large important part to accommodate a small part. If you trim the divider, you just shorten it slightly — much better. But a naive "smaller area gets trimmed" heuristic fails at corner junctions where both parts are the same size.

**Current solution — effective loss simulation:** For each overlapping pair, simulate both trim directions. Use `_effective_loss()` which:
1. Computes the half-plane cut polygon for each direction
2. Subtracts it from the target outline
3. Runs `_clean_polygon()` on the result (which keeps only the largest connected piece)
4. Measures how much area was actually lost

The direction with lower effective loss wins. This handles T-junctions (trim the abutting part), corners (trim whichever loses less), and asymmetric overlaps (trim the side where the cut polygon is smaller).

**5b. How to geometrically represent the overlap?**

Each part occupies a "slab" in 3D space — a flat region with a specific thickness. The slab goes from `position_3d` (the anti-normal face, i.e., the back of the sheet) to `position_3d + normal * thickness` (the outer surface, i.e., the front of the sheet). Material fills the space between these two parallel planes.

The **slab intrusion polygon** (`_slab_intrusion_polygon`) answers: "what region of part A's outline lies inside part B's material slab?" It works by:
1. Taking part B's slab normal and projecting it into part A's 2D coordinate frame
2. Computing the linear equation `a*x + b*y + c0` that gives the signed distance from any point in A's 2D frame to B's slab faces
3. Clipping A's outline to the strip where that distance falls between B's inner face and outer face (with a small tolerance buffer to avoid including exactly-on-face vertices)

The result is a thin strip polygon (or set of polygons) in A's 2D frame representing the exact overlap.

**5c. How to cut without fragmenting?**

The intrusion polygon is a thin strip. If you subtract a thin strip that crosses the full width of a part, it splits the part into two disconnected pieces. The `_clean_polygon` function would keep only the larger piece, silently discarding the other half — a catastrophic silent failure.

**Solution — half-plane cuts (`_slab_cut_polygon`):** Instead of subtracting the thin strip, we compute a half-plane cut. The half-plane starts at the slab entry face (the face closest to the part being trimmed) and extends outward to infinity. Subtracting a half-plane from a connected polygon always produces a single connected polygon — it can never fragment.

The function figures out which side of the slab boundary the part's centroid is on, then cuts away the opposite side. This is equivalent to trimming the part back to the slab boundary.

**5d. What about phantom pairs?**

Two parts whose planes intersect in 3D might not actually overlap if they're on opposite ends of the mesh. For example, the left wall and right wall of a bookshelf are both vertical, they're perpendicular to the shelves, and their planes intersect somewhere in the middle. But the walls themselves are spatially separated — they never physically touch each other.

Three gates prevent phantom trims:

1. **Parallel skip:** Parts with nearly-parallel normals (cross product < 0.1) are skipped entirely. Parallel parts don't intersect.

2. **Slab penetration gate:** At least one part must have vertices that actually penetrate the other's material slab. `_slab_penetration()` projects A's outline vertices into 3D, then measures how deep they go into B's slab. If the maximum penetration across both directions is less than 0.01mm, the pair is skipped.

3. **Cross-projection gate (`_intrusion_overlaps_target`):** Even if the slab math says there's an intrusion strip, we check whether that strip is physically near the target part. We take the centroid of the intrusion polygon in A's 2D frame, project it through 3D space into B's 2D frame, and check whether it's within tolerance of B's outline. If it's far from B's outline, the intrusion is a phantom — A's material crosses B's slab plane, but in a region where B doesn't actually exist.

**5e. What about compounding destruction?**

A single part might need to be trimmed against 5 different neighbors. If each trim removes 15% of the part's area, the cumulative loss is 75% — the part is mostly gone. This can happen to back panels of bookshelves where every shelf, every side wall, and every top/bottom all request trims.

**Solution — incremental loss budget:** Each part has a maximum total trim budget of 40% of its original area. The algorithm:
1. Collects all pending trim polygons for each part
2. Sorts them by area (smallest first)
3. Applies them one by one, checking cumulative loss after each
4. Stops applying trims when the budget is exceeded

Smallest-first ordering means small edge trims (which are almost always correct and necessary) get applied first. Large mid-panel cuts (which are more likely to be destructive or wrong) get dropped when the budget runs out. This turns a potential 96.5% area loss (observed on shelf_unit's back panel before the budget was added) into a 36% loss.

**5f. Order-independence — the collect-then-subtract pattern:**

If you trim parts sequentially (trim pair 1, then pair 2, then pair 3...), the order matters because each trim changes the geometry that subsequent trims operate on. This makes results non-deterministic — changing the order you iterate through pairs produces different output.

The solution is two-phase processing:
1. **Phase 1 (collect):** Walk through all pairs, compute the trim direction and cut polygon for each, and store them in a dictionary keyed by part index. Don't modify any part geometry yet.
2. **Phase 2 (subtract):** For each part, take all its pending cut polygons, sort by area, and apply them incrementally with the loss budget.

Since each part's trims are computed against the original (pre-trim) outlines, the order of pair iteration in Phase 1 doesn't matter. And in Phase 2, the sort-by-area ordering is deterministic.

---

### 6. Getting the Slab Geometry Right — Coordinate System Conventions

**What we're trying to accomplish:** Every part exists in two coordinate systems simultaneously. It has a 2D coordinate frame (for its outline polygon) and a 3D position/orientation in the assembly. The trim system needs to convert between these frames and compute correct slab boundaries. Getting any convention wrong means trims happen at the wrong place.

**The two reference points that caused confusion:**

- `origin_3d`: The origin of the 2D coordinate system. This is the centroid of the projected faces — the point where 2D coordinates (0,0) maps to in 3D space. It's used for projecting 3D points into the part's 2D frame and back.

- `position_3d`: The actual 3D position of the sheet of material. For the slab convention, this is the anti-normal face (the "back" of the sheet). The outer surface is at `position_3d + normal * thickness`.

These can be quite different! For a tall wall panel, `origin_3d` might be at the centroid of the wall (halfway up), while `position_3d` is where the sheet's back face actually sits. Early code was computing slab boundaries from `origin_3d` instead of `position_3d`, which placed the slab in the wrong location — shifted by the distance between the centroid and the actual face.

**The fix:** The slab system now consistently uses:
```
slab_origin = position_3d + normal * thickness   (the outward surface)
```
The slab interior extends from `slab_origin` inward by `thickness` along the negative normal direction. This matches the physical reality: material occupies the space from the back face (`position_3d`) to the front face (`position_3d + normal * thickness`).

This single convention fix eliminated 74 out of 77 overlap pairs across the benchmark suite — nearly all detected "overlaps" were actually off-by-one-thickness errors from using the wrong reference point.

---

### 7. Joints — Connecting the Parts

**What we're trying to accomplish:** The decomposition produces a set of separate flat parts. For the final object to be structurally sound, these parts need to physically interlock. The system must figure out which parts should be connected, what type of joint to use, and generate the exact 2D geometry (tabs, slots, cross-lap cutouts) that gets cut into each part.

**7a. Finding which parts should be connected:**

Not every pair of parts needs a joint. Two shelves on opposite sides of a bookshelf don't connect to each other — they each connect to the side walls. The system identifies joint candidates by:

1. **Skipping same-stack pairs:** Parts in the same stack group (layers at the same position) don't need joints with each other — they're parallel sheets that just sit on top of each other.
2. **Skipping parallel parts:** Parts with nearly-parallel normals (dot product > 0.95) don't meet at an angle that allows tab/slot joints.
3. **Checking contact distance:** Using oriented bounding boxes (OBBs), measure whether the parts are close enough to be in contact. The contact tolerance is configurable (default 2mm). Parts separated by more than this aren't candidates.

**7b. Computing the contact line:**

Where two non-parallel planes meet is a line in 3D space. The system computes this plane-plane intersection line by solving the linear system of the two plane equations plus the intersection line direction (the cross product of the normals). The result is a point on the line and the line's direction.

This infinite line is then clipped to both parts' outlines by projecting it into each part's 2D frame and computing the intersection with the outline polygon. The result is two line segments — the contact line in part A's frame, and the contact line in part B's frame. These represent where the joint will be placed.

**7c. Classifying joint type:**

The system looks at where the contact line falls relative to each part's edges:

- **Tab-slot:** If the contact line is near the edge of part A but in the interior of part B, part A gets tabs that protrude outward, and part B gets matching slots. This is the most common case — a shelf edge meeting the interior of a side panel.
- **Cross-lap:** If the contact line is in the interior of both parts, both get complementary slots that interlock. This happens at T-junctions where one part passes through another.
- **Butt:** If the contact line is near the edge of both parts, or the contact is too short for meaningful joint geometry, fall back to a butt joint (no mechanical interlock — assembly relies on glue or fasteners).

The "near edge" threshold is `max(2 * thickness, 10mm)` — contact lines within this distance of the outline boundary count as "near edge."

**7d. Tab placement and geometry (`_apply_tab_slot`):**

Once we know which part gets tabs and which gets slots:

1. Place tabs along the contact line with configurable spacing (default 80mm between tab centers).
2. Tabs go at the ends first (they're most structurally important), then fill the middle.
3. Each tab is a rectangular protrusion: `tab_length_mm` (default 20mm) perpendicular to the contact edge, and width determined by the available contact length divided by tab count.
4. The tab extends outward from the part's edge — the outline is expanded at each tab position to add the protruding tab.
5. Matching rectangular slots are cut into the receiving part. Each slot is slightly oversized by the clearance amount (0.254mm per side, which is 0.010" — a standard slip-fit clearance).
6. Dogbone relief circles are added at every internal corner of every slot. CNC end mills are round, so they can't cut a perfect 90-degree internal corner. The dogbone is a small circle placed at the corner along the angle bisector, extending into the material. Its radius equals the minimum internal radius (1.588mm = 1/16").

**7e. Cross-lap geometry (`_apply_cross_lap`):**

Both parts get complementary slots that interlock like puzzle pieces:
1. Each slot starts from one end of the contact line and extends to the midpoint.
2. Slot width = the other part's thickness + clearance.
3. When assembled, the two slots interlock — each part slides into the other's slot from opposite sides.

**7f. Stacked parts and joints:**

When a region has multiple stacked layers, only the layer closest to the partner part gets joint geometry. The inner layers don't need their own tabs — they're held in place by being sandwiched between the outer layer and the structure.

---

### 8. Manufacturing Constraints (DFM) — Making Parts That Can Actually Be Cut

**What we're trying to accomplish:** The CNC machine has physical limits dictated by the cutting tool (end mill) and the material properties. Every part must be validated against these constraints before it can be manufactured. The goal is to catch problems early and score them so the system avoids producing unmanufacturable designs.

**The specific constraints and why they exist:**

**Minimum slot width (3.175mm = 1/8"):** The end mill has a physical diameter. It can't fit inside any opening narrower than its diameter. If we generate a slot that's 2mm wide, the CNC machine literally cannot cut it — the tool is too big to fit. This is the most common violation.

**Minimum internal radius (1.588mm = 1/16"):** When the end mill cuts an internal corner, the corner radius equals the tool radius. A perfectly sharp 90-degree corner is physically impossible — the tool always leaves a rounded corner. Dogbone relief compensates for this by adding circular reliefs that allow the mating part to fit, but parts with unexpectedly sharp internal corners (not at dogbone-relieved joints) get flagged.

**Minimum bridge width (3.175mm = 1/8"):** The material between two cutouts must be thick enough to not break during cutting. If two slots are only 1mm apart, the thin strip of material between them will snap when the CNC tool cuts through. We check the distance between every pair of cutouts, and between every cutout and the outer edge.

**Maximum aspect ratio (20:1):** Very long, thin parts will warp during cutting as internal stresses in the material are released. A part that's 500mm long and only 25mm wide might curl or twist. This is a warning, not an error — it's possible but risky.

**Sheet size limits:** Each material has a maximum sheet size (varies by material and supplier). Parts must fit within this. Parts that exceed the sheet size get split into tiles in an earlier phase.

**How validation works:** After all geometry is finalized (outlines, cutouts, tabs, slots, dogbones), every part is run through `check_part_dfm()` which checks all five constraints. Violations are classified as:
- **Error:** Must fix before manufacturing (tool physically can't do this)
- **Warning:** Should fix, might cause quality issues (warping risk, etc.)

The violation count feeds into the quality score, creating an incentive for the algorithm to produce designs that pass all checks.

---

### 9. Scoring — How We Measure Quality

**What we're trying to accomplish:** We need a single number between 0 and 1 that tells us how good a decomposition is. This number is used to compare different algorithm versions, track regressions across the benchmark suite, and identify which test cases need attention.

**The evolution of the scoring formula:**

The original formula was badly broken. It computed connectivity as `1 / (1 + joints)`, which meant *fewer joints = better score*. An assembly with zero joints (completely disconnected parts floating in space) would score perfectly on connectivity. It also had no penalties for overlapping parts or manufacturing violations. This formula actively rewarded bad designs.

**The current formula (`_compute_quality_metrics`):**

Four sub-scores, all between 0 and 1 (higher = better), weighted and combined:

**Fidelity (75% weight):** How closely do the parts match the original mesh?
- **Hausdorff component (80% of fidelity):** Based on unique face coverage ratio — what fraction of the mesh's surface area is covered by selected parts. The Hausdorff distance is approximated as `max_mesh_extent * (1 - coverage_ratio)`. Perfect coverage = 0 distance. Score is `1 / (1 + hausdorff / max_extent)`.
- **Normal component (20% of fidelity):** Average angular deviation between each part's normal and the normals of its source mesh faces. Parts facing the right direction score well; parts that are rotated or misaligned score poorly. Score is `1 / (1 + error_deg / 45)`.

**Connectivity (10% weight):** How well-connected is the assembly?
- A spanning tree of N structural members requires N-1 joints minimum.
- "Effective members" collapses stacked layers — a stack of 3 sheets counts as 1 member.
- Score = `min(1.0, actual_joints / expected_joints)`. Having the expected number of joints gives a perfect score. Having more is capped at 1.0.

**Overlap (10% weight):** Are parts physically colliding?
- Score = `1 / (1 + overlap_pairs)`. Zero overlap pairs = perfect. Each remaining pair degrades the score.

**Violations (5% weight):** Are parts manufacturable?
- Score = `1 / (1 + error_count)`. Only error-severity violations count (not warnings). Zero errors = perfect.

**Why these weights:** Fidelity dominates (75%) because the primary goal is faithfully reproducing the 3D shape. Connectivity and overlap split the next 10% equally because both represent "the assembly doesn't work" problems. Violations get 5% because they're important but usually fixable without changing the decomposition strategy.

---

### 10. Fragmentation from Geometry Operations

**What we're trying to accomplish:** Throughout the pipeline, we do lots of geometric operations — subtract this shape from that shape, intersect this polygon with that polygon, clip to a half-plane. Every one of these operations can potentially produce fragmented geometry (a single polygon splits into multiple disconnected pieces). We need to ensure that every part remains a single connected polygon.

**Why fragmentation is so dangerous:** The Shapely library's polygon operations (`.difference()`, `.intersection()`, `.union()`) can all produce `MultiPolygon` results — multiple disconnected polygons. Our `_clean_polygon()` function handles this by keeping only the largest connected component. But this is a lossy operation:

- If a thin strip crosses a part and splits it into two roughly equal halves, `_clean_polygon` silently discards half the part.
- If a cut separates a small tab from the main body, the tab vanishes.
- If floating-point precision issues create tiny slivers, those get cleaned up harmlessly — but genuine splits are destructive.

**The solutions applied throughout the system:**

1. **Half-plane cuts instead of strip subtraction:** This is the key insight in the slab trim system. Instead of subtracting the thin intrusion strip (which crosses the full width of a part and splits it), subtract a half-plane (everything on one side of a line). A half-plane operation on a connected polygon always produces a connected polygon (or empty). It mathematically cannot fragment.

2. **`_keep_connected()` for growth:** When growing outlines via cross-section, the result might include disconnected components from other parts of the mesh. `_keep_connected` keeps only the components that physically overlap the original outline — maintaining a single connected part.

3. **`_largest_polygon()` as a fallback:** Throughout the codebase, whenever a geometry operation might produce a `MultiPolygon`, we extract the largest polygon. This is a safety net, not a primary strategy — if it's being triggered on large pieces, something upstream is wrong.

4. **Loss budget as a safety net:** Even if a cut operation somehow fragments a part, the loss budget catches it. If the cleaned (largest component) result has lost more than 40% of the original area, the cut is dropped entirely.

---

### 11. Order-Independence — Making Results Deterministic

**What we're trying to accomplish:** The decomposition should produce the same output every time for the same input. If you change an unrelated line of code (like adding a log statement), the output shouldn't change. And when debugging, you need to be able to reproduce problems reliably.

**Why order matters:** Many algorithms iterate over pairs of parts. In Python, the order of dictionary iteration, list comprehension, or set operations can vary between runs or after seemingly unrelated code changes. If the trim algorithm processes pair (A,B) before pair (C,D), and those trims interact (both affect part B), the result depends on order.

**The collect-then-subtract pattern (used in trim system):**

1. **Phase 1:** Iterate all pairs and collect trim decisions. Each decision is just "part X should have polygon Y subtracted." No geometry is modified yet. The trim decisions are stored as `List[Tuple[part_index, cut_polygons, estimated_loss]]`.

2. **Phase 2:** Group pending trims by part index. For each part, sort its pending cuts by area (smallest first — deterministic). Apply them sequentially with the loss budget.

The Phase 1 collection is order-independent because it reads from original (unmodified) outlines. The Phase 2 application is deterministic because the sort ordering is well-defined. The overall result is the same regardless of which order pairs are processed in Phase 1.

**Where this pattern is NOT used (and the implications):**

- Joint geometry application iterates through joints in list order. Since each joint modifies part geometry (adding tabs, cutting slots), the order can technically affect which slots overlap with which tabs. In practice this rarely matters because joints affect local regions of parts, but it's a potential source of non-determinism.

- Post-joint overlap cleanup processes parts sequentially. This is bounded and rarely triggers (it's a safety net), so order effects are minimal.

---

### 12. Post-Joint Overlap Cleanup

**What we're trying to accomplish:** After applying joint geometry (tabs and slots), some parts might have new overlaps that didn't exist before. A tab protruding from part A might extend into the material slab of part C (not the receiving part B). The post-joint cleanup catches and resolves these residual overlaps.

**Why this is separate from the main trim:** The main trim system runs *before* joint geometry is applied. Joints change the outline geometry by adding tab protrusions and slot cutouts. These changes can create new overlaps that the trim system didn't anticipate.

**How it works:** Re-run the slab intrusion detection on the final geometry. For any remaining overlaps, apply bounded trims. The key word is "bounded" — this is a safety net, not a full re-trim. It should rarely trigger if the main trim system did its job correctly.

---

### 13. The Benchmark Suite — Measuring Progress

**What we're trying to accomplish:** With 13 test meshes of increasing complexity, we can measure whether algorithm changes help or hurt, and identify which geometric configurations still cause problems.

**The test meshes, in order of complexity:**

1. **01_box** (12 faces): The simplest possible case. Six faces, all perpendicular. If this fails, everything is broken.
2. **02_tall_cabinet** (12 faces): A box with extreme aspect ratio. Tests stack alignment and thin-side suppression.
3. **03_l_bracket**: An L-shaped profile. Tests concave outline extraction (triangle union, not convex hull).
4. **04_t_beam**: A T-shaped cross-section. Tests T-junction trim direction.
5. **05_u_channel**: A U-shaped channel. Tests multiple parallel faces with different offsets.
6. **06_step_stool**: Multiple horizontal surfaces at different heights. Tests multi-level stacking and trim interactions.
7. **07_bookshelf**: Multiple shelves between side walls. Tests many simultaneous trim pairs and joint density.
8. **08_shelf_unit**: Similar to bookshelf but more complex. Tests compounding trim destruction and loss budget.
9. **09_desk**: Horizontal surface with legs. Tests vertical-horizontal trim interactions.
10. **10_table_with_stretchers**: Table with structural supports. Tests complex joint networks.
11. **11_v_bracket**: Non-orthogonal angles. Tests trim math on non-90-degree intersections.
12. **12_angled_flange**: Non-orthogonal surface. Tests intrusion polygon computation at non-standard angles.
13. **13_trapezoidal_tray**: Non-rectangular, non-orthogonal. The hardest case — trapezoid with angled walls means every intersection is at a non-standard angle.

**Current performance:** Mean score 0.931 across all 13 cases. Best: tall_cabinet (1.000). Worst: trapezoidal_tray (0.803). The worst cases all involve non-orthogonal geometry — the algorithm was primarily developed and tuned for orthogonal (right-angle) intersections.

---

## Summary of the Evolution

| Version | Trim Approach | What Changed | Key Problem It Fixed |
|---------|--------------|--------------|---------------------|
| v1 | Half-plane at intersection line | Initial implementation | - |
| v2 | + Exhaustive bitmask search | Try all 2^N direction combinations | Fragmented parts from wrong direction choices |
| v3 | + Loss-based direction | Simulate both directions, pick lower loss | Wrong direction at T-junctions (smallest-area-wins failed) |
| v4 | + Shared-boundary gate | Skip pairs with <1mm shared boundary line | Phantom pairs between spatially separated parts (112/342 removed) |
| v5 | + Slab convention fix | Use position_3d, not origin_3d, for slab computation | Wrong reference point caused 74/77 false overlap detections |
| **v6 (current)** | **Slab subtraction + budget** | **Complete trim rewrite: slab intrusion, half-plane cuts, cross-projection gate, incremental budget** | **Non-fragmenting, order-independent, bounded loss, robust phantom detection** |

**What was removed in v6:** The entire previous trim system — `_trim_at_plane_intersections`, `_trim_part_against_plane`, `_trim_loss_fraction`, `_shared_boundary_length`, `_line_outline_segment`, `_group_pairs_by_stack`, `_apply_trim_combination`, and the exhaustive bitmask search. All replaced by ~220 lines in `_trim_by_slab_subtraction` plus helper functions.

---

## What's Still Hard

**Non-orthogonal geometry:** The trapezoidal tray (score 0.803) and angled flange (score 0.858) are the weakest cases. When surfaces meet at angles that aren't 90 degrees, the intrusion polygons are wider, the trim directions are less clear, and the loss calculations are tighter. The algorithm works correctly but produces worse results — more material is lost to trimming, and joints are harder to classify.

**Step stool (score 0.881):** Multiple horizontal surfaces at different heights create many trim pairs with subtle interactions. The loss budget helps, but the sheer number of simultaneous trims can still degrade quality.

**Joint fit validation:** Even when joints are geometrically correct in 2D, they might not assemble correctly in 3D. The system validates both 2D fit (does the tab cross-section fit in the slot?) and 3D assembly (do the parts align correctly when assembled?). Getting these checks right for all joint types across all geometry configurations is ongoing work.
