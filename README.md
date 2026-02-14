# Text-to-Furniture

Generate flat-pack furniture designs from text descriptions, images, or 3D meshes. Cloud APIs produce 3D meshes, which are decomposed into flat rectangular components suitable for CNC cutting. The pipeline outputs SVG and DXF cut files ready for [SendCutSend](https://sendcutsend.com/) manufacturing.

## Goals

**End-to-end goal:** Turn a natural language description (or photo) of a piece of furniture into physical, manufacturable flat-pack parts that a person can assemble at home.

### Sequential steps

1. **Understand the desired furniture** -- Accept a human intent — a text prompt like "a modern coffee table" or a reference photo — and interpret it as a 3D object with proportions, surfaces, and structural purpose.

2. **Produce a 3D representation** -- Generate a 3D mesh that faithfully represents the described furniture. The mesh is a continuous surface model — the kind a 3D artist would make — not yet anything manufacturable.

3. **Decompose the 3D shape into flat parts** -- Approximate a freeform 3D surface as a small set of flat rectangular slabs. Each slab must be cuttable from a sheet of real material. This requires deciding: how many parts, what orientation/size/thickness for each, and how well the collection of flat parts approximates the original shape.

4. **Determine how parts connect** -- Infer where parts meet in 3D space and define joints (tabs, slots, fasteners) at those intersections. Joints must be geometrically valid and structurally sound.

5. **Validate manufacturability** -- Check that every part satisfies real-world manufacturing constraints: minimum feature sizes, available material thicknesses, maximum sheet dimensions, and relief cuts for interior corners.

6. **Validate structural integrity** -- Simulate whether the assembled furniture stands up, holds load, and doesn't tip over under expected use.

7. **Produce cut-ready output files** -- Export 2D cutting profiles — one per part — in a format a CNC service accepts. Send them to a laser/waterjet cutter, receive flat parts, assemble furniture.

### The core hard problem

Steps 1--2 are delegated to existing AI/API services. Step 7 is file formatting. Steps 5--6 are validation. **Step 3 is the fundamental research problem:** bridging continuous 3D geometry and discrete flat-stock manufacturing. It's a coverage/approximation problem with real physical constraints, and it's where most of the complexity lives.

## Features

- **Text-to-furniture** -- describe furniture in natural language, get CNC-ready cut files
- **Image-to-furniture** -- upload a photo and convert it to a flat-pack design
- **Local mesh support** -- start from an existing STL, OBJ, GLB, or PLY file (no API key required)
- **Two decomposition pipelines:**
  - **Voxel-based** (default) -- greedy set-cover over voxelized mesh using slab candidates
  - **Manufacturing-aware** (opt-in) -- RANSAC plane extraction, structural segmentation, manufacturing ladder (rungs 0--3), joint synthesis, DFM validation
- **Real material catalog** from SendCutSend (plywood, MDF, mild steel, aluminum, acrylic, and more)
- **Joint geometry synthesis** -- tab/slot, finger, through-bolt, and half-lap joints with dogbone relief
- **DFM validation** -- checks slot widths, internal radii, bridge widths, aspect ratios, and sheet sizes
- **Physics simulation** via PyBullet -- gravity stability testing, load testing, tip-over detection
- **Multiple export formats** -- SVG cut files, DXF files (manufacturing-aware mode), nested sheet layouts, design JSON
- **Assembly sequence generation** -- bottom-up ordering optimized for gravity-assisted assembly
- **Decomposition scoring** -- Hausdorff distance, structural plausibility, DFM compliance composite score

## Setup

### Prerequisites

- Python 3.12
- (Optional) API keys for cloud mesh generation:
  - [Tripo3D](https://www.tripo3d.ai/) -- set `TRIPO_API_KEY` environment variable
  - [Meshy](https://www.meshy.ai/) -- set `MESHY_API_KEY` environment variable

### Installation

```bash
# Clone the repository
git clone https://github.com/your-org/text-to-furniture.git
cd text-to-furniture

# Create virtual environment
python3.12 -m venv venv

# Install dependencies
venv/bin/pip install -r requirements.txt
```

### Environment Variables

Copy the example file and fill in your API keys:

```bash
cp .env.example .env
# Edit .env with your keys:
#   TRIPO_API_KEY=your_tripo_api_key_here
#   MESHY_API_KEY=your_meshy_api_key_here
```

## Usage

All commands use the virtual environment Python at `venv/bin/python3`.

### Generate furniture from a text prompt

```bash
venv/bin/python3 scripts/generate_furniture.py \
    --text "a modern coffee table" \
    --provider tripo
```

### Generate furniture from an image

```bash
venv/bin/python3 scripts/generate_furniture.py \
    --image photo.jpg \
    --provider meshy
```

### Generate from an existing mesh file (no API key needed)

```bash
venv/bin/python3 scripts/generate_furniture.py \
    --mesh model.glb
```

### Manufacturing-aware pipeline

The manufacturing-aware pipeline uses RANSAC plane extraction, structural segmentation, a manufacturing ladder, and joint synthesis with DFM validation. Opt in with `--manufacturing-aware`:

```bash
venv/bin/python3 scripts/generate_furniture.py \
    --mesh model.glb \
    --manufacturing-aware
```

This automatically enables DXF export. You can also enable DXF export explicitly:

```bash
venv/bin/python3 scripts/generate_furniture.py \
    --mesh model.glb \
    --manufacturing-aware \
    --export-dxf
```

### Decompose a mesh directly (lower-level)

```bash
venv/bin/python3 scripts/decompose_mesh.py --input model.stl
```

### Visualize a decomposition

```bash
MPLBACKEND=Agg venv/bin/python3 scripts/visualize_decomposition.py --input model.stl
```

### CLI options

| Flag | Description | Default |
|------|-------------|---------|
| `--text` | Text prompt describing furniture | -- |
| `--image` | Path to input image | -- |
| `--mesh` | Path to existing mesh file | -- |
| `--provider` | Cloud provider (`tripo` or `meshy`) | `tripo` |
| `--api-key` | API key override | env var |
| `--material` | SendCutSend material key | `plywood_baltic_birch` |
| `--height` | Target height in mm | `750` |
| `--max-slabs` | Max flat-pack components | `15` |
| `--coverage` | Voxel coverage target (0--1) | `0.80` |
| `--manufacturing-aware` | Use manufacturing-aware decomposition | `false` |
| `--export-dxf` | Export DXF cut files | `false` |
| `--no-simulate` | Skip physics simulation | `false` |
| `--no-svg` | Skip SVG export | `false` |
| `--no-optimize` | Skip decomposition optimization | `false` |
| `--output` | Output directory | `output` |
| `--name` | Design name | `furniture` |
| `-v` | Verbose logging | `false` |

### Output structure

```
output/<design-name>/
  design.json                    # Component dimensions, positions, joints
  design_manufacturing.json      # (manufacturing-aware only) Full manufacturing schema
  svg/
    slab_0.svg                   # Individual component cut files
    slab_1.svg
    ...
    <design-name>_nested.svg     # All components nested on a single sheet
  dxf/                           # (manufacturing-aware only)
    part_0.dxf
    part_1.dxf
    ...
    <design-name>_nested.dxf     # All parts nested on a single sheet
```

SVG files use red for cut lines and blue for engravings. DXF files use standard CUT and ENGRAVE layers with ACI colors 1 (red) and 5 (blue).

## Architecture

### Data flow

```
Text  --> [Tripo3D / Meshy API] --> 3D Mesh --+
Image --> [Tripo3D / Meshy API] --> 3D Mesh --+
Local mesh file -----------------------------|
                                              |
                      +-----------------------+
                      |
                      v
        +---------------------------+
        |       pipeline.py         |
        |  (use_manufacturing_aware |
        |   selects the path)       |
        +------+----------+--------+
               |          |
     +---------+          +----------+
     | Default pipeline              | Manufacturing-aware pipeline
     v                               v
  mesh_decomposer.py         manufacturing_decomposer.py
  - cluster normals           - RANSAC plane extraction
  - voxel set-cover           - structural segmentation
  - greedy slab selection     - manufacturing ladder (rungs 0-3)
  - local optimization        - joint synthesis
  - infer joints              - DFM validation
     |                               |
     +---------+          +----------+
               |          |
               v          v
        FurnitureDesign
               |
    +----------+----------+----------+
    |          |          |          |
    v          v          v          v
 SVG export  DXF export  Physics   Design JSON
 (svg_       (dxf_       sim
  exporter)   exporter)  (PyBullet)
```

### Module reference

All source modules are in `src/`. Imports use bare module names (e.g., `import furniture`, not `import src.furniture`). Scripts in `scripts/` add `src/` to `sys.path` before importing.

#### Core types

| Module | Description |
|--------|-------------|
| `furniture.py` | Foundation types: `Component`, `Joint`, `JointType`, `ComponentType`, `AssemblyGraph`, `FurnitureDesign`. All designs flow through `FurnitureDesign`. |
| `materials.py` | SendCutSend material catalog. `MATERIALS` dict mapping keys to `Material` dataclass (thicknesses, densities, max sheet sizes). Includes plywood, MDF, hardboard, mild steel, aluminum, stainless steel, acrylic, HDPE, polycarbonate, neoprene. |
| `geometry_primitives.py` | Shapely-based 2D geometry types: `PlanarPatch` (from mesh face extraction), `PartProfile2D` (CNC-ready 2D part with outline, cutouts, and features), plus conversion utilities between Shapely polygons and Component profiles. |

#### Cloud mesh providers

| Module | Description |
|--------|-------------|
| `mesh_provider.py` | Abstract `MeshProvider` base class, `ProviderConfig`, `GenerationResult`, exception hierarchy (`ProviderError`, `ProviderTimeoutError`, `ProviderAPIError`, `ProviderRateLimitError`). |
| `tripo_provider.py` | Tripo3D provider using the official `tripo3d` SDK. Async wrapped in sync interface. API key via `TRIPO_API_KEY` env var. |
| `meshy_provider.py` | Meshy provider using REST API via `requests`. Preview mode for text-to-3D. API key via `MESHY_API_KEY` env var. |

#### Pipeline orchestration

| Module | Description |
|--------|-------------|
| `pipeline.py` | Top-level orchestrator. `run_pipeline()` generates a mesh from text/image via a cloud provider and decomposes it. `run_pipeline_from_mesh()` starts from an existing mesh file. Returns `PipelineResult` with design, simulation results, SVG/DXF paths. Selects between voxel-based and manufacturing-aware decomposition via `PipelineConfig.use_manufacturing_aware`. |

#### Voxel-based decomposition (default)

| Module | Description |
|--------|-------------|
| `mesh_decomposer.py` | Converts 3D meshes to flat-pack designs: load and repair mesh, cluster face normals to find dominant orientations, generate candidate slabs at mesh surface positions, greedy voxel set-cover selection, optional local optimization (random perturbation), infer joints from slab proximity, build `FurnitureDesign`. Entry point: `decompose(filepath, config)`. |

#### Manufacturing-aware decomposition (opt-in)

| Module | Description |
|--------|-------------|
| `manufacturing_decomposer.py` | Top-level orchestrator for the manufacturing-aware pipeline. Chains together RANSAC extraction, segmentation, manufacturing ladder, joint synthesis, DFM validation, scoring, and assembly sequencing. Returns `ManufacturingDecompositionResult` with design, parts, DFM violations, score, and assembly steps. |
| `plane_extraction.py` | RANSAC-based planar patch extraction from 3D meshes. |
| `structural_segmenter.py` | Groups planar patches into structural segments (horizontal surfaces, vertical panels, legs, braces, organic shapes). |
| `manufacturing_ladder.py` | Tries manufacturing rungs 0--3 in order of simplicity for each structural segment: Rung 0 (single flat plate), Rung 1 (orthogonal plates with tab/slot joints), Rung 2 (interlocking ribs / waffle), Rung 3 (multiplanar shell panels with connectors). Returns the best-scoring DFM-valid candidate. |
| `dfm_rules.py` | Design-for-Manufacturing validation against SendCutSend CNC routing constraints: minimum slot widths (1/8" end mill), internal radii (1/16"), bridge widths, aspect ratios, sheet sizes. Also provides dogbone relief insertion for internal right-angle corners. |
| `joint_synthesizer.py` | Generates joint geometry (tab/slot, finger, through-bolt, half-lap) as Shapely polygon operations on `PartProfile2D` outlines. Adds dogbone relief to internal corners. Configurable tab width, clearance, bolt hole diameter. |
| `scoring.py` | Evaluates decomposition quality: Hausdorff distance from mesh surface to nearest part, structural plausibility (horizontal surfaces, vertical supports, joint connectivity), DFM compliance. Produces a weighted composite score (0--1). |
| `assembly_sequence.py` | Determines assembly order: starts from the largest horizontal surface, BFS along joints, bottom-up ordering for gravity-assisted assembly. |

#### Export

| Module | Description |
|--------|-------------|
| `svg_exporter.py` | CNC-ready SVG output. Generates individual component SVGs and a nested layout SVG that packs all components onto a single sheet. Red = cut lines, blue = engravings. |
| `dxf_exporter.py` | DXF export using `ezdxf` (R2010 format). Proper CUT and ENGRAVE layers. Individual part DXFs and nested sheet layout. Part labels with thickness annotations. Units: millimeters. |

#### Physics simulation

| Module | Description |
|--------|-------------|
| `simulator.py` | PyBullet physics simulation: stability testing under gravity, load testing, tip-over detection. Returns `SimulationResult` with stable/unstable verdict, position change, rotation change. |
| `urdf_generator.py` | Converts `FurnitureDesign` to URDF format for PyBullet simulation. |

#### Scripts

| Script | Description |
|--------|-------------|
| `scripts/generate_furniture.py` | Main CLI entry point for the full pipeline. |
| `scripts/decompose_mesh.py` | Lower-level script to decompose a mesh directly. |
| `scripts/visualize_decomposition.py` | Visualize decomposition results with matplotlib. |
| `scripts/run_ui.py` | Launch the NiceGUI web interface. |

### Units and conventions

- All dimensions are in **millimeters**
- Positions are **numpy arrays** (3D: x, y, z)
- Component profiles are 2D (x, y) closed polygons; thickness is a separate field
- Rotations are in **radians**
- `FurnitureDesign.validate()` checks component name uniqueness and joint reference integrity

### Available materials

The `MATERIALS` dictionary in `materials.py` contains SendCutSend stock with real thicknesses, densities, and maximum sheet sizes:

| Key | Material | Thicknesses (inch) | Max sheet |
|-----|----------|-------------------|-----------|
| `plywood_baltic_birch` | Baltic Birch Plywood | 1/8, 3/16, 1/4, 3/8, 1/2, 3/4 | 24 x 30 in |
| `mdf` | MDF | 1/8, 3/16, 1/4, 1/2 | 24 x 48 in |
| `hardboard` | Hardboard | 1/8 | 24 x 48 in |
| `mild_steel` | Mild Steel | 0.030 -- 0.500 | 48 x 96 in |
| `aluminum_5052` | Aluminum 5052 | 0.032 -- 0.250 | 48 x 96 in |
| `stainless_304` | Stainless Steel 304 | 0.030 -- 0.187 | 48 x 96 in |
| `acrylic_clear` | Acrylic Clear | 0.060 -- 0.472 | 24 x 48 in |
| `acrylic_black` | Acrylic Black | 0.118, 0.220 | 24 x 48 in |
| `hdpe` | HDPE | 1/4, 3/8, 1/2 | 24 x 48 in |
| `polycarbonate` | Polycarbonate | 0.118 -- 0.220 | 24 x 48 in |
| `neoprene` | Neoprene | 1/16, 1/8 | 24 x 36 in |

## Testing

The project has 44 tests covering geometry primitives, DFM rules, plane extraction, manufacturing ladder, joint synthesis, and DXF export.

```bash
# Run all tests
venv/bin/python3 -m pytest tests/

# Run with verbose output
venv/bin/python3 -m pytest tests/ -v

# Run a specific test file
venv/bin/python3 -m pytest tests/test_dfm_rules.py
```

## Code formatting

The project uses [Black](https://github.com/psf/black) for code formatting:

```bash
venv/bin/python3 -m black src/
```

## Dependencies

Listed in `requirements.txt`:

| Category | Packages |
|----------|----------|
| Physics simulation | `pybullet`, `numpy` |
| Geometry and CAD | `trimesh`, `shapely`, `svgwrite`, `ezdxf` |
| Cloud mesh providers | `tripo3d`, `requests` |
| Utilities | `matplotlib`, `scipy`, `tqdm` |
| Web UI | `nicegui` |
| Development | `pytest`, `black` |

## Contributing

1. Create a feature branch from `main`.
2. Make sure all 44 tests pass: `venv/bin/python3 -m pytest tests/`
3. Format code with Black: `venv/bin/python3 -m black src/`
4. Keep all dimensions in millimeters and positions as numpy arrays.
5. Use bare module imports (e.g., `import furniture`, not `import src.furniture`). Scripts should add `src/` to `sys.path`.
6. Run `FurnitureDesign.validate()` on any design before exporting.
7. For headless matplotlib rendering, set `MPLBACKEND=Agg`.

## License

(To be determined)
