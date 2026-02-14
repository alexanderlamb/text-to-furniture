# Manufacturing-Aware Pipeline

This document explains how the manufacturing-aware decomposition pipeline works, from mesh input to CNC-ready DXF output. The pipeline is opt-in (activated via `--manufacturing-aware`) and coexists with the original voxel-based decomposition.

## Table of Contents

- [Overview](#overview)
- [Pipeline Stages](#pipeline-stages)
  - [1. Mesh Loading](#1-mesh-loading)
  - [2. Planar Patch Extraction](#2-planar-patch-extraction)
  - [3. Structural Segmentation](#3-structural-segmentation)
  - [4. Manufacturing Ladder](#4-manufacturing-ladder)
  - [5. Joint Synthesis](#5-joint-synthesis)
  - [6. DFM Validation](#6-dfm-validation)
  - [7. Scoring](#7-scoring)
  - [8. Assembly Sequence](#8-assembly-sequence)
  - [9. Export](#9-export)
- [Manufacturing Ladder Detail](#manufacturing-ladder-detail)
  - [Rung 0: Single Flat Plate](#rung-0-single-flat-plate)
  - [Rung 1: Orthogonal Plates](#rung-1-orthogonal-plates)
  - [Rung 2: Interlocking Ribs](#rung-2-interlocking-ribs-ribcage)
  - [Rung 3: Shell Panels](#rung-3-shell-panels)
- [Joint Types](#joint-types)
- [DFM Rules](#dfm-rules)
- [Scoring System](#scoring-system)
- [Configuration Reference](#configuration-reference)
- [Data Flow Diagram](#data-flow-diagram)

---

## Overview

The original pipeline uses a voxel-based greedy set-cover to produce simple rectangular slabs. It works well for quick approximations but lacks real joint geometry, DFM checks, and tolerance handling.

The manufacturing-aware pipeline replaces this with a multi-stage process:

1. Extract planar surfaces from the mesh using RANSAC
2. Classify surfaces into structural roles (tabletop, leg, panel, etc.)
3. Try progressively complex manufacturing strategies ("rungs") to find the simplest one that works
4. Synthesize actual joint geometry (tabs, slots, finger joints) with tolerances
5. Validate everything against SendCutSend CNC constraints
6. Export DXF files ready for manufacturing

The key insight is the **manufacturing ladder**: always try the simplest process first. A flat shelf only needs a single cut (Rung 0). A table needs orthogonal plates with tab-slot joints (Rung 1). A curved organic shape needs interlocking ribs (Rung 2). The ladder tries each rung in order and picks the simplest one that produces a valid, DFM-compliant result.

---

## Pipeline Stages

### 1. Mesh Loading

**Module:** `mesh_decomposer.load_mesh()`

The pipeline reuses the existing mesh loader, which handles STL, OBJ, GLB, and PLY formats. For GLB scenes with multiple meshes, it extracts the largest by face count. Processing steps:

- Merge duplicate vertices
- Fix face normals
- Fill holes (AI-generated meshes are often non-watertight)
- Scale to target height (default 750mm)
- Center at origin with bottom at z=0

### 2. Planar Patch Extraction

**Module:** `plane_extraction.py`

Identifies flat surfaces in the mesh using RANSAC (Random Sample Consensus) combined with region growing.

**Algorithm:**

```
while unassigned faces remain and area > threshold:
    1. RANSAC: randomly sample faces, use each face's normal as a candidate
       plane normal. Score by total area of inlier faces (faces within
       distance threshold whose normals align with the plane).
    2. Take the best plane found across all iterations.
    3. Region grow: starting from RANSAC inliers, expand along mesh face
       adjacency. A neighbor face is added if its normal aligns with the
       plane normal within the angle threshold.
    4. Project the grown region's faces onto the plane to get a 2D boundary
       polygon (via Shapely).
    5. Merge near-coplanar patches (same normal direction, close offset).
```

**Key parameters:**

| Parameter | Default | Effect |
|-----------|---------|--------|
| `ransac_threshold_mm` | 2.0 | Max distance from plane to count as inlier |
| `ransac_iterations` | 1000 | More iterations = better planes but slower |
| `min_patch_area_mm2` | 500.0 | Patches smaller than this are discarded |
| `merge_angle_threshold_deg` | 10.0 | Patches within this angle are merged |
| `region_grow_angle_deg` | 15.0 | Face normal must be within this of plane normal to grow |

**Output:** A list of `PlanarPatch` objects, each containing the plane equation, face indices, 2D boundary polygon, area, and 3D centroid.

### 3. Structural Segmentation

**Module:** `structural_segmenter.py`

Groups planar patches into structural segments and classifies their role based on orientation and geometry.

**Classification rules:**

| Role | Condition |
|------|-----------|
| `HORIZONTAL_SURFACE` | Normal within 30 degrees of vertical (Z axis) -- tabletops, shelves, seats |
| `VERTICAL_PANEL` | Normal near horizontal, width > 80mm -- side panels, backs |
| `LEG` | Normal near horizontal, narrow (< 80mm) and tall (> 200mm) |
| `BRACE` | Normal at 20-70 degrees from vertical -- diagonal supports |
| `ORGANIC` | Non-planar faces not assigned to any patch |

Adjacent patches with the same role are grouped into a single segment via flood-fill on the mesh face adjacency graph. The output includes segment bounding boxes, principal normals, and inter-segment adjacency.

### 4. Manufacturing Ladder

**Module:** `manufacturing_ladder.py`

For each structural segment, the ladder tries manufacturing rungs in order from simplest to most complex. It keeps the best-scoring candidate that passes DFM validation.

```
for each segment:
    try Rung 0 (single plate)     -> if valid, score it
    try Rung 1 (orthogonal plates) -> if valid, score it
    try Rung 2 (interlocking ribs) -> if valid, score it
    try Rung 3 (shell panels)      -> if valid, score it
    pick the candidate with the highest score
```

See [Manufacturing Ladder Detail](#manufacturing-ladder-detail) below for how each rung works.

### 5. Joint Synthesis

**Module:** `joint_synthesizer.py`

After the ladder produces parts, the joint synthesizer adds actual joint geometry -- tabs, slots, finger joints, bolt holes, and half-lap notches. All operations are Shapely polygon boolean operations:

- **Tabs:** `part.outline = part.outline.union(tab_rectangle)`
- **Slots:** `part.cutouts.append(slot_rectangle)` (then `part.outline.difference(slot)` at export)
- **Dogbone relief:** Circles at internal right-angle corners to accommodate CNC bit radius

Tolerances are built in: slot width = mating part thickness + 2x slip-fit clearance (default 0.010" / 0.254mm per side).

### 6. DFM Validation

**Module:** `dfm_rules.py`

Every part is checked against SendCutSend CNC routing constraints. Violations are classified as errors (must fix) or warnings (should fix).

See [DFM Rules](#dfm-rules) below for the full rule set.

### 7. Scoring

**Module:** `scoring.py`

Evaluates the decomposition quality with a composite score from 0 to 1:

- **Hausdorff distance** (30% weight): How far the assembled parts deviate from the original mesh surface
- **Part count** (20% weight): Fewer parts = simpler assembly
- **DFM compliance** (30% weight): Penalizes violations
- **Structural plausibility** (20% weight): Does the design have horizontal surfaces, vertical supports, and joint connectivity?

### 8. Assembly Sequence

**Module:** `assembly_sequence.py`

Determines the order to assemble the parts:

1. Start with the largest horizontal surface at the lowest z-position (the base)
2. BFS outward along the joint graph
3. Sort by z-position at each level (bottom-up, so gravity assists)
4. Generate step-by-step instructions with action type and notes

### 9. Export

**DXF export** (`dxf_exporter.py`): Produces DXF files via ezdxf with two layers:
- `CUT` (red, ACI color 1): Through-cut outlines and slot cutouts
- `ENGRAVE` (blue, ACI color 5): Labels, fold lines, engraving marks

Both individual part DXFs and a nested sheet layout DXF are generated.

**SVG export** (`svg_exporter.py`): Also produces SVG cut files (same as the original pipeline), now extended to render cutout features from `Component.features`.

**JSON export**: A manufacturing JSON with full part outlines, cutout coordinates, feature parameters, assembly graph, and scoring metrics.

---

## Manufacturing Ladder Detail

### Rung 0: Single Flat Plate

**Module:** `rung0_planar.py`

**When it applies:** A segment has one dominant planar patch covering >80% of the segment's total area.

**What it does:**
1. Takes the dominant patch's 2D boundary as the part outline
2. Picks the closest available SCS material thickness
3. Runs DFM check
4. If it passes, returns a single `PartProfile2D`

**Best for:** Shelves, tabletops, flat panels, simple rectangular components.

### Rung 1: Orthogonal Plates

**Module:** `rung1_orthogonal.py`

**When it applies:** A segment has 2+ planar patches with near-orthogonal normals (angle > 60 degrees between pairs).

**What it does:**
1. Creates one plate per patch at the closest SCS thickness
2. Identifies which pairs of plates need joints (orthogonal + adjacent in 3D)
3. Returns the parts with joint pair annotations for Phase 5 synthesis

**Best for:** Tables, shelves with sides, box-like furniture, L-shaped brackets.

### Rung 2: Interlocking Ribs (Ribcage)

**Module:** `rung2_ribcage.py`

**When it applies:** Organic or curved shapes without dominant planar surfaces. This is the fallback for shapes that don't have enough flat faces.

**What it does:**
1. Identifies the primary axis (longest bounding box dimension)
2. Slices the mesh perpendicular to the primary axis at evenly spaced intervals using `trimesh.section()`
3. Converts each cross-section path to a 2D Shapely polygon
4. Generates perpendicular secondary slices
5. At each intersection of primary and secondary ribs, creates half-lap notches (half-depth from opposite sides) so the ribs interlock

**Best for:** Curved chairs, organic sculptures, vases, any shape that can't be approximated with flat panels.

### Rung 3: Shell Panels

**Module:** `rung3_shell.py`

**When it applies:** Complex shapes with 3+ non-orthogonal planar regions that can be approximated as a faceted shell.

**What it does:**
1. Uses all planar patches (above a minimum area) as panel faces
2. Identifies adjacent panels and computes dihedral angles
3. For non-right-angle connections, generates L-bracket connector parts
4. Returns panels + connectors

**Best for:** Polyhedra, faceted designs, multi-angled enclosures.

---

## Joint Types

| Joint Type | Used By | Description |
|-----------|---------|-------------|
| **Tab-Slot** | Rung 1 | Rectangular tabs extend from Part A; matching slots (with clearance) are cut into Part B. Dogbone relief circles at slot corners. |
| **Finger Joint** | Rung 1, 3 | Alternating interlocking fingers along an edge. Part A gets odd fingers, Part B gets even. Both get matching slot cutouts. |
| **Through-Bolt** | Rung 1, 3 | Circular holes on both parts at bolt diameter (default M6). Used for parallel/stacked parts. |
| **Half-Lap** | Rung 2 | Notches cut to half the part height at crossing points. Primary ribs get top-down notches, secondary ribs get bottom-up notches. |
| **Butt** | Fallback | Simple edge-to-edge contact, no interlocking geometry. Requires external fasteners. |

### Joint Parameters

```
tab_width_mm:         30.0    Width of each tab
tab_depth_factor:     0.8     Tab depth as fraction of mating thickness
min_tabs_per_edge:    2       Minimum tabs per joint edge
finger_joint_pitch:   15.0mm  Alternating finger width
slip_fit_clearance:   0.254mm (0.010") per-side clearance
dogbone_radius:       1.6mm   (1/16") CNC bit radius relief
bolt_hole_diameter:   6.5mm   M6 clearance hole
```

---

## DFM Rules

All rules reference SendCutSend CNC routing constraints.

| Rule | Type | Default Limit | Description |
|------|------|---------------|-------------|
| Sheet size | Error | 24"x30" (plywood) | Part must fit within material sheet dimensions |
| Minimum slot width | Error | 1/8" (3.175mm) | Narrowest internal cutout dimension must exceed end mill diameter |
| Minimum bridge width | Error/Warning | 1/8" (3.175mm) | Remaining material between cutouts or between cutout and edge |
| Internal corner radius | Warning | 1/16" (1.6mm) | Sharp internal corners need dogbone relief for CNC bit |
| Aspect ratio | Warning | 20:1 | Very long thin parts risk warping during cutting |

### Dogbone Relief

CNC routers use round bits, so they cannot cut perfectly sharp internal corners. Dogbone relief adds small circles at each internal corner:

```
Before:            After dogbone:
+--------+         +---(O)---+
|        |         |         |
|  slot  |         |  slot   |
|        |         |         |
+--------+         +---(O)---+
```

The circle is placed on the bisector of the corner angle, with its center offset into the material by the bit radius. The algorithm:
1. For each corner with angle < 100 degrees
2. Compute the bisector direction (into material)
3. Place a circle at corner + bisector * radius
4. Union the circle with the slot polygon

---

## Scoring System

The composite score (0 to 1) combines four metrics:

### Hausdorff Distance (30% weight)

Measures how well the assembled flat-pack parts approximate the original mesh:
1. Sample N points on the original mesh surface (default 5000)
2. Build a KD-tree from the assembled part corner and face-center points
3. For each sample point, find the nearest assembled-part point
4. Report max distance (Hausdorff) and mean distance

Score: `1.0 - hausdorff_mm / max_acceptable_mm` (max = 50mm)

### Part Count (20% weight)

Fewer parts means simpler assembly and lower manufacturing cost.

Score: `1.0 - part_count / max_acceptable_parts` (max = 20)

### DFM Compliance (30% weight)

Penalizes manufacturing constraint violations.

Score: `1.0 - (errors * 0.2) - (warnings * 0.05)`

### Structural Plausibility (20% weight)

Checks four binary conditions:
1. Has at least one horizontal surface (table/shelf/seat)
2. Has vertical supports (legs/panels/braces)
3. Has joints connecting parts
4. All parts are reachable via joints (graph connectivity)

Score: fraction of checks that pass (0.0 to 1.0)

---

## Configuration Reference

The top-level configuration is `ManufacturingDecompositionConfig`:

```python
ManufacturingDecompositionConfig(
    plane_extraction=PlaneExtractionConfig(
        ransac_threshold_mm=2.0,
        ransac_iterations=1000,
        min_patch_area_mm2=500.0,
        merge_angle_threshold_deg=10.0,
        region_grow_angle_deg=15.0,
    ),
    segmentation=SegmentationConfig(
        horizontal_threshold_deg=30.0,
        leg_max_width_mm=80.0,
        leg_min_height_mm=200.0,
    ),
    dfm=DFMConfig(
        min_slot_width_inch=0.125,
        min_internal_radius_inch=0.063,
        slip_fit_clearance_inch=0.010,
        min_bridge_width_inch=0.125,
        max_aspect_ratio=20.0,
    ),
    ladder=LadderConfig(
        enable_rung0=True,
        enable_rung1=True,
        enable_rung2=True,
        enable_rung3=True,
    ),
    joint_synthesis=JointSynthesisConfig(
        tab_width_mm=30.0,
        slip_fit_clearance_mm=0.254,
        add_dogbones=True,
        dogbone_radius_mm=1.6,
    ),
    scoring=ScoringConfig(
        n_sample_points=5000,
        max_acceptable_hausdorff_mm=50.0,
    ),
    default_material="plywood_baltic_birch",
    target_height_mm=750.0,
)
```

---

## Data Flow Diagram

```
                         Input Mesh (STL/OBJ/GLB/PLY)
                                    |
                          load_mesh() + repair
                                    |
                         [trimesh.Trimesh object]
                                    |
                    extract_planar_patches() --- RANSAC + region grow
                                    |
                        [List[PlanarPatch]]
                                    |
                         segment_mesh() --- classify by normal + geometry
                                    |
                      [List[StructuralSegment]]
                                    |
                    +---------------+---------------+
                    |               |               |
               Segment A       Segment B       Segment C  ...
                    |               |               |
               run_ladder()    run_ladder()    run_ladder()
                    |               |               |
              Rung 0? 1? 2? 3?     ...             ...
                    |               |               |
              [ManufacturingCandidate per segment]
                    |
          synthesize_joints() --- tabs, slots, dogbones
                    |
          [Dict[str, PartProfile2D], List[Joint]]
                    |
          +----+----+----+
          |    |    |    |
        score  DFM  DXF  SVG
          |    |    |    |
    DecompositionScore   Export files
          |
    optimize_assembly_sequence()
          |
    [List[AssemblyStep]]
          |
    ManufacturingDecompositionResult
          |
    +-----+-----+
    |     |     |
  .dxf  .svg  .json
```
