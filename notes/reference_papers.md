# Reference Papers for Mesh-to-Parts Fabrication

This is a working reading map for the mesh-to-furniture prototype. The goal is not to read everything up front. Review each paper when the implementation reaches the trigger listed in `Review when`.

## How To Use This List

- Use the `Priority` column to decide what to read first when a design decision is blocked.
- Use `Review when` as the handoff trigger for future agents.
- Add notes under each paper after review: algorithms worth stealing, assumptions that do not fit sheet furniture, and implementation sketches.
- Prefer extracting concrete ideas into tests, metrics, or plan JSON fields rather than leaving them as research prose.

## Highest-Priority Papers

| Priority | Paper | Strategy area | Review when |
| --- | --- | --- | --- |
| P0 | [Platener: Low-Fidelity Fabrication of 3D Objects by Substituting 3D Print with Laser-Cut Plates](https://hcie.csail.mit.edu/research/platener/platener.html) | Hybrid sheet/print decomposition, plate graph, joints | Before changing the hybrid compositor, adding bent-sheet support, or designing a strategy graph with fabrication fallbacks. |
| P0 | [Chopper: Partitioning Models into 3D-Printable Parts](https://cdfg.mit.edu/publications/chopper-partitioning-models-3d-printable-parts) | Decomposition scoring, assemblability, connectors | Before replacing our regioning algorithm, scoring seams, or designing connector placement between regions. |
| P0 | [CofiFab: Coarse-to-Fine Fabrication of Large 3D Objects](https://orca.cardiff.ac.uk/id/eprint/98565/) | Coarse internal base plus fine shell, hybrid fabrication | Before implementing skeleton-plus-cladding, torsion-box/hollow-core behavior, or coarse internal support structures. |
| P0 | [Fabrication-aware Design with Intersecting Planar Pieces](https://infoscience.epfl.ch/entities/publication/df2a0cd0-330f-4392-80bd-869b78fc76eb) | Interlocking planar pieces, rigidity, assembly constraints | Before turning waffle/rib AABBs into real slotted cut geometry or adding rigidity checks. |
| P0 | [crdbrd: Shape Fabrication by Sliding Planar Slices](https://diglib.eg.org/items/66ad5473-f732-453b-bd94-03363603f275) | Constructible planar slice assemblies | Before adding insertion-order checks, slit feasibility checks, or assembly sequencing for rib systems. |
| P0 | [A Comprehensive Process of Reverse Engineering from 3D Meshes to CAD Models](https://www.sciencedirect.com/science/article/pii/S0010448513001012) | Mesh-to-B-Rep reverse engineering | Before converting mesh-derived regions into editable CAD surfaces, STEP/B-Rep output, or topology-consistent wires. |
| P0 | [Mesh2Brep: B-Rep Reconstruction via Robust Primitive Fitting and Intersection-Aware Constraints](https://pubmed.ncbi.nlm.nih.gov/40030873/) | Robust primitive fitting, valid B-Rep reconstruction | Before implementing primitive intersection constraints, fillet/blend handling, or editable CAD reconstruction from triangle meshes. |

## Decomposition and Regioning

### Chopper: Partitioning Models into 3D-Printable Parts

- Link: <https://cdfg.mit.edu/publications/chopper-partitioning-models-3d-printable-parts>
- Use for: part decomposition objectives, seam placement, assemblability, connector-aware cuts.
- Review when: our hybrid output has plausible parts but bad boundaries, too many regions, visible awkward seams, or unclear inter-region joints.
- What to extract: objective terms for part count, seam unobtrusiveness, structural soundness, and connector customizations.

### Approximate Convex Decomposition for 3D Meshes with Collision-Aware Concavity and Tree Search

- Link: <https://arxiv.org/abs/2205.02961>
- Project page: <https://colin97.github.io/CoACD/>
- Use for: cutting a complex solid mesh into almost-convex chunks before strategy assignment.
- Review when: component/axis band regioning starts failing on organic, concave, or branching geometry.
- What to extract: plane-cut search, concavity metrics, and a region quality measure based on interior and boundary error.

### Fabrication Oriented Shape Decomposition Using Polycube Mapping

- Link: <https://www.sciencedirect.com/science/article/pii/S0097849318301717>
- Use for: decomposing shapes into fabrication-friendly chunks with planar sides.
- Review when: we need regions that are more CAD-like than voxel bands but less expensive than full convex decomposition.
- What to extract: polycube-style normalization and planar-side constraints for subtractive/sheet-friendly part boundaries.

### Volume Decomposition for Two-Piece Rigid Casting

- Link: <https://vcg.isti.cnr.it/publication/2021/AMBCP21/>
- Use for: accessibility-aware volume partitioning.
- Review when: generated parts cannot be assembled, extracted, or joined because the assembly direction is ignored.
- What to extract: accessibility signals over the volume and energy terms for fabrication-aware partitioning.

## Mesh-to-CAD and Reverse Engineering

This section is about recovering a cleaner editable representation before fabrication. The useful split is:

- `mesh/point cloud -> B-Rep`: recover surfaces, edges, wires, and topology.
- `mesh/point cloud -> CAD program`: recover sketch/extrude/boolean code that can be edited and regenerated.

### A Comprehensive Process of Reverse Engineering from 3D Meshes to CAD Models

- Link: <https://www.sciencedirect.com/science/article/pii/S0010448513001012>
- PDF: <https://www.lirmm.fr/~wpuech/recherche/publications/RI/13_CAD.pdf>
- Use for: classic mesh-to-B-Rep reconstruction from triangle meshes exported from CAD.
- Review when: we need editable planes/cylinders/cones/spheres instead of raw triangle facets, or when exporting STEP/IGES-like solids becomes a goal.
- What to extract: primitive extraction, adjacency relation detection, intersection-curve/wire construction, and B-Rep topology assembly.

### Mesh2Brep: B-Rep Reconstruction via Robust Primitive Fitting and Intersection-Aware Constraints

- Link: <https://pubmed.ncbi.nlm.nih.gov/40030873/>
- Use for: high-quality B-Rep reconstruction from surface meshes with robust primitive fitting.
- Review when: simple primitive fitting breaks near blends/fillets, neighboring surfaces produce gaps, or we need constraints like tangency, collinearity, coplanarity, and valid intersections.
- What to extract: iterative outlier-resistant primitive fitting, intersection-aware constraints, blend recovery, and topology correction.

### ComplexGen: CAD Reconstruction by B-Rep Chain Complex Generation

- Link: <https://haopan.github.io/complexgen.html>
- Paper: <https://arxiv.org/abs/2205.14573>
- Use for: learning-plus-optimization B-Rep reconstruction from point clouds.
- Review when: hand-built primitive detection cannot robustly recover face/edge/vertex topology, especially on noisy or incomplete inputs.
- What to extract: representing CAD as a chain complex of faces, edges, vertices, and their incidence relations; global optimization under structural validity constraints.

### Point2CAD: Reverse Engineering CAD Models from 3D Point Clouds

- Link: <https://www.obukhov.ai/point2cad.html>
- Paper: <https://arxiv.org/abs/2312.04962>
- Code: <https://github.com/prs-eth/point2cad>
- Use for: reconstructing CAD surfaces, edges, and corners from segmented point clouds with a hybrid analytic-neural pipeline.
- Review when: we want to test an existing mesh/point-cloud-to-CAD reconstruction tool as an optional preprocessing path.
- What to extract: separation between surface segmentation and topology reconstruction, surface extension/intersection for clipping, and freeform surface fitting.

### ParSeNet: A Parametric Surface Fitting Network for 3D Point Clouds

- Link: <https://hippogriff.github.io/parsenet/>
- Paper: <https://arxiv.org/abs/2003.12181>
- Use for: decomposing point clouds into parametric patches, including primitives and B-spline patches.
- Review when: curvature-based regioning is not enough and we need semantic/parametric patch proposals.
- What to extract: patch segmentation, primitive-vs-B-spline classification, and learned priors for man-made shape decomposition.

### Supervised Fitting of Geometric Primitives to 3D Point Clouds

- Link: <https://openaccess.thecvf.com/content_CVPR_2019/html/Li_Supervised_Fitting_of_Geometric_Primitives_to_3D_Point_Clouds_CVPR_2019_paper.html>
- Paper: <https://arxiv.org/abs/1811.08988>
- Use for: primitive fitting from point clouds without per-input RANSAC tuning.
- Review when: we need a learned primitive proposal baseline for planes, cylinders, cones, and spheres.
- What to extract: variable-number primitive detection, differentiable model estimation, and point-to-primitive assignment.

### HPNet: Deep Primitive Segmentation Using Hybrid Representations

- Link: <https://arxiv.org/abs/2105.10620>
- Use for: primitive patch segmentation in point clouds.
- Review when: primitive fitting quality is limited more by bad segmentation than by the surface fit itself.
- What to extract: hybrid representation for primitive segmentation and cleaner instance grouping.

### Fusion 360 Gallery: A Dataset and Environment for Programmatic CAD Construction from Human Design Sequences

- Link: <https://www.research.autodesk.com/publications/fusion-360-gallery/>
- Paper: <https://arxiv.org/abs/2010.02392>
- Code/data: <https://github.com/AutodeskAILab/Fusion360GalleryDataset>
- Use for: learning or evaluating sketch/extrude CAD reconstruction.
- Review when: considering a CAD-program intermediate representation, especially one based on sketch profiles and extrusions.
- What to extract: simple sketch/extrude DSL, reconstruction metrics, and a gym-like environment for search over CAD construction sequences.

### DeepCAD: A Deep Generative Network for Computer-Aided Design Models

- Link: <https://arxiv.org/abs/2105.09492>
- Paper: <https://openaccess.thecvf.com/content/ICCV2021/papers/Wu_DeepCAD_A_Deep_Generative_Network_for_Computer-Aided_Design_Models_ICCV_2021_paper.pdf>
- Use for: CAD operation sequence generation.
- Review when: exploring learned CAD programs or autoencoding simple mechanical shapes into editable sequences.
- What to extract: tokenization of CAD operations and transformer-based sequence modeling for sketch/extrude programs.

### CAD-Recode: Reverse Engineering CAD Code from Point Clouds

- Link: <https://cad-recode.github.io/>
- Paper: <https://arxiv.org/abs/2412.14042>
- Code/model: <https://github.com/filaPro/cad-recode>
- Use for: translating point clouds into executable CadQuery Python code.
- Review when: we want an AI-assisted CAD reconstruction path where the output is inspectable/editable code rather than a black-box B-Rep.
- What to extract: CadQuery-as-output representation, point-cloud encoder plus code decoder, synthetic CAD program generation, and LLM-assisted editing of reconstructed CAD.

### CADReasoner: Iterative Program Editing for CAD Reverse Engineering

- Link: <https://arxiv.org/abs/2603.29847>
- Use for: iterative CAD reconstruction from geometry by comparing rendered output to the target and editing the program.
- Review when: we build an agentic loop that proposes CadQuery/OpenSCAD code, renders it, measures geometric error, and patches the program repeatedly.
- What to extract: discrepancy-driven refinement loop, multi-view render plus point-cloud conditioning, and scan-simulation evaluation.

### Scan2CAD: Learning CAD Model Alignment in RGB-D Scans

- Link: <https://openaccess.thecvf.com/content_CVPR_2019/html/Avetisyan_Scan2CAD_Learning_CAD_Model_Alignment_in_RGB-D_Scans_CVPR_2019_paper.html>
- Paper: <https://arxiv.org/abs/1811.11187>
- Use for: retrieving and aligning known CAD models to scanned/mesh inputs.
- Review when: exploring library-based reconstruction where furniture components come from a parametric/catalog library instead of being inferred from scratch.
- What to extract: CAD retrieval/alignment framing, 9-DoF pose fitting, and shape-library substitution ideas.

## Planar Sheet, Skin, and Plate Graphs

### Platener: Low-Fidelity Fabrication of 3D Objects by Substituting 3D Print with Laser-Cut Plates

- Link: <https://hcie.csail.mit.edu/research/platener/platener.html>
- Paper: <https://groups.csail.mit.edu/hcie/files/research-projects/platener/2015-chi-platener-paper.pdf>
- Use for: extracting straight/curved plates, classifying leftover regions, choosing joints, and mixing fabrication modes.
- Review when: implementing planar skin v2, bent sheet, plate graph data structures, or fallback from sheet parts to another strategy.
- What to extract: plate graph representation, node/edge fabrication labels, straight plate extraction from parallel planes, curved plate fallback behavior, and joint type selection.

### Fabrication-aware Design for Furniture with Planar Pieces

- Link: <https://arxiv.org/abs/2104.05052>
- Use for: flat-pack furniture abstractions, parameterized planar components, and user-facing fabrication guarantees.
- Review when: moving from mesh-derived panels toward editable furniture components, reusable joints, and design constraints.
- What to extract: component/connectivity graph, joint libraries, fabrication parameter handling, and manufacturability guarantees.

### Converting 3D Furniture Models to Fabricatable Parts and Connectors

- Link: <https://cir.nii.ac.jp/crid/1360016869219354368?lang=en>
- Use for: furniture-specific parsing of cabinets/tables into primitive parts and connectors.
- Review when: we want domain priors for cabinets, shelves, tables, or chairs instead of treating every mesh as generic geometry.
- What to extract: grammar-based primitive detection, structural analysis, contact graph generation, and connector inference.

### SketchChair: An All-in-one Chair Design System for End Users

- Link: <https://www.jst.go.jp/erato/igarashi/projects/sketchchair/tei2011.pdf>
- Use for: chair-specific sheet fabrication and validation UX.
- Review when: adding a chair-specialized workflow, sketch-to-parts mode, or stability checks for seating.
- What to extract: validation interaction patterns and full-sized sheet-material chair constraints.

## Waffle, Slice, and Rib Assemblies

### Fabrication-aware Design with Intersecting Planar Pieces

- Link: <https://infoscience.epfl.ch/entities/publication/df2a0cd0-330f-4392-80bd-869b78fc76eb>
- Use for: slotted planar assemblies that stay rigid without glue or screws.
- Review when: waffle/rib prototypes need real slots, insertion constraints, and structural validity.
- What to extract: constraint relation graph, rigidity constraints, slit construction rules, and optimization for manufacturable planar pieces.

### crdbrd: Shape Fabrication by Sliding Planar Slices

- Link: <https://diglib.eg.org/items/66ad5473-f732-453b-bd94-03363603f275>
- Use for: cardboard-style planar slices with guaranteed constructibility.
- Review when: rib parts intersect in ways that look fine in 3D but cannot be assembled physically.
- What to extract: extended BSP representation, insertion-order feasibility, and iterative slice addition heuristics.

### Field-Aligned Mesh Joinery

- Link: <https://vcg.isti.cnr.it/publication/2014/CPMS14/>
- Use for: slice/rib layouts that follow shape features rather than a world-axis grid.
- Review when: axis-aligned waffles look crude on organic or diagonal furniture forms.
- What to extract: cross-field-driven slice placement, manufacturing constraints for non-axis-aligned slits, and visual quality metrics.

### Automatic Paper Sliceform Design from 3D Solid Models

- Link: <https://dau.url.edu/handle/20.500.14342/5421>
- Use for: two-direction slotted sliceforms, foldability, stability, and physical realizability.
- Review when: implementing contour stack plus perpendicular ribs, deployable/fold-flat variants, or slot-label layouts.
- What to extract: stability conditions, slot/slit generation, physical-realizability tests, and automatic 2D layout labeling.

### Slices: A Shape-Proxy Based on Planar Sections

- Link: <https://graphics.stanford.edu/~niloy/research/slices/paper_docs/slices_siggA_11.pdf>
- Use for: choosing meaningful section planes based on perceptual/geometric features.
- Review when: deciding where to put a small number of cross-sections or ribs so the object still reads as the source shape.
- What to extract: feature-weighted plane selection, symmetry/orthogonality heuristics, and coverage scoring.

### Orthogonal Slicing for Additive Manufacturing

- Link: <https://hildebrand.beuth-hochschule.de/3dprinting/slices.pdf>
- Use for: decomposition into multiple slicing directions to reduce approximation error.
- Review when: single-axis contour stacking produces visually bad layers or misses thin features.
- What to extract: directional slicing error metrics and part decomposition by best local slicing direction.

## Developable and Bent-Sheet Surfaces

### D-Charts: Quasi-Developable Mesh Segmentation

- Link: <https://www.cs.ubc.ca/~vlady/dcharts/dcharts.htm>
- Use for: segmenting meshes into nearly developable charts suitable for sheet material patterns.
- Review when: adding bent sheet, kerf-bent plywood, fabric/leather, or unfoldable surface patches.
- What to extract: developability metric, chart-growing algorithm, and distortion-bound chart scoring.

### Developable Mesh Segmentation by Detecting Curve-like Features on Gauss Images

- Link: <https://www.sciencedirect.com/science/article/abs/pii/S0097849322001819>
- Use for: finding cylinders, cones, and tangent developables from triangle meshes.
- Review when: D-Charts-style near-developable patches are too loose and we need cleaner CAD-like patch classes.
- What to extract: Gauss image feature detection and exact developable patch approximation.

### LaserOrigami: Laser-Cutting 3D Objects

- Link: <https://hcie.csail.mit.edu/research/laserorigami/laserorigami.html>
- Use for: single-sheet cut-and-bend parts without manual assembly.
- Review when: exploring bent metal/acrylic as a strategy rather than treating every bend as a joint.
- What to extract: bend primitives, suspender/stretch concepts, and constraints for integrated cut-plus-bend fabrication.

## Hybrid, Skeleton, and Interior Structures

### CofiFab: Coarse-to-Fine Fabrication of Large 3D Objects

- Link: <https://orca.cardiff.ac.uk/id/eprint/98565/>
- Use for: combining a coarse laser-cut internal base with fine exterior shell pieces.
- Review when: implementing skeleton-plus-cladding, internal bases, or hybrid plans where structural and visual parts differ.
- What to extract: internal convex base optimization, nonorthogonal interlocking joint networks, shell partitioning, cost/aesthetic/stability objectives.

### Cost-effective Printing of 3D Objects with Skin-Frame Structures

- Link: <https://cir.nii.ac.jp/crid/1362544419422857216>
- Use for: lightweight internal frames under skins.
- Review when: deciding between solid laminate, hollow torsion box, honeycomb, ribs, and skin-frame structures.
- What to extract: multi-objective frame optimization, stability constraints, material reduction metrics, and strut sparsity.

### Medial Axis Tree: An Internal Supporting Structure for 3D Printing

- Link: <https://www.researchgate.net/publication/274096061_Medial_axis_tree_-_An_internal_supporting_structure_for_3D_printing>
- Use for: skeleton extraction as a structural backbone.
- Review when: skeleton-plus-cladding needs a better frame than arbitrary ribs or voxel blocks.
- What to extract: medial-axis-derived support trees and load-aware internal structures.

### WirePrint: 3D Printed Previews for Fast Prototyping

- Link: <https://hpi.de/baudisch/projects/wireprint.html>
- Use for: low-fidelity preview structures and fast ergonomic validation.
- Review when: building preview/export modes or when exact fabrication is too slow for iterative feedback.
- What to extract: wireframe approximation, direct 3D edge printing concepts, and low-fidelity evaluation framing.

### faBrickation: Fast 3D Printing of Functional Objects by Integrating Construction Kit Building Blocks

- Link: <https://hcie.csail.mit.edu/research/fabrickation/fabrickation.html>
- Use for: replacing low-detail volume with standard blocks while preserving high-detail functional regions.
- Review when: exploring voxel/block rebuilding, modular block fills, or mixed precision fabrication.
- What to extract: user-controlled fidelity regions, block substitution rules, and hybrid cost/time evaluation.

## ML and Learned Shape Abstraction

### Learning Fine-to-Coarse Cuboid Shape Abstraction

- Link: <https://arxiv.org/abs/2502.01855>
- Project page: <https://www.graphics.rwth-aachen.de/publication/03361/>
- Use for: learning cuboid primitives that summarize man-made shapes.
- Review when: heuristic voxel box merging stops scaling, or when we want learned primitive proposals for furniture-like objects.
- What to extract: fine-to-coarse primitive abstraction and evaluation metrics for compact cuboid coverage.

### Learning Cuboid Abstraction of 3D Shapes via Iterative Error Feedback

- Link: <https://www.sciencedirect.com/science/article/pii/S0010448521001032>
- Use for: unsupervised cuboid primitive fitting.
- Review when: we need block/cuboid proposals for voxel_blocks that are less stair-stepped than raw voxel merges.
- What to extract: iterative error feedback loop and primitive refinement objective.

### ShapeAssembly: Learning to Generate Programs for 3D Shape Structure Synthesis

- Link: <https://arxiv.org/abs/2009.08026>
- Use for: programmatic part assembly representations.
- Review when: considering AI-generated strategy plans, part grammars, or semantic furniture decomposition.
- What to extract: assembly-language representation and neural generation of structured part programs.

### Unsupervised 3D Shape Reconstruction by Part Retrieval and Assembly

- Link: <https://openaccess.thecvf.com/content/CVPR2023/papers/Xu_Unsupervised_3D_Shape_Reconstruction_by_Part_Retrieval_and_Assembly_CVPR_2023_paper.pdf>
- Use for: retrieving and placing parts to approximate a target shape.
- Review when: exploring library-based furniture components instead of deriving every part from scratch.
- What to extract: part retrieval, pose estimation, and unsupervised assembly loss ideas.

## Open Research Questions To Track

- How should we represent a common part graph shared across strategies so planar skin, waffle ribs, voxel blocks, and contour stacks can all be compared fairly?
- Should the pipeline go directly from mesh to fabrication strategies, or first reconstruct a cleaner CAD/B-Rep/primitive program for man-made objects?
- What is the minimum viable assemblability check for a hybrid output that mixes strategy families?
- When should the system warn that a mesh is a bad fit for sheet fabrication instead of producing a misleading plan?
- Can learned primitive proposals improve regioning without making the core pipeline dependent on ML?
- What UI evidence helps users trust a hybrid decision: score table, region colors, assembly sequence, cost/time estimate, or part count?

## Suggested Reading Order For The Current Prototype

1. Read Platener and Chopper before changing the hybrid compositor or region boundary scoring.
2. Read the mesh-to-CAD classics (`A Comprehensive Process`, `Mesh2Brep`) before adding B-Rep/STEP/exportable primitive reconstruction.
3. Read CAD-Recode and CADReasoner before trying an AI-agent loop that writes CadQuery/OpenSCAD from mesh evidence.
4. Read CofiFab before implementing skeleton/cladding or internal-base strategies.
5. Read Fabrication-aware Intersecting Planar Pieces and crdbrd before turning waffle/rib previews into real cut parts.
6. Read D-Charts before adding bent-sheet or unfoldable curved panel support.
7. Read the furniture grammar papers before building furniture-specific strategy presets.
8. Read ML/cuboid papers only after the heuristic block decomposition is clearly limiting quality.
