"""
End-to-end pipeline: Text/Image -> Cloud API -> 3D Mesh -> Flat-Pack Design -> SVG.

Two entry points:
  - run_pipeline(): Generate mesh from text/image via cloud provider, then decompose
  - run_pipeline_from_mesh(): Start from an existing mesh file (skip cloud generation)

Both produce a FurnitureDesign with optional physics simulation and SVG export.
"""
import os
import json
import logging
from typing import Optional, List
from dataclasses import dataclass, field

from mesh_provider import MeshProvider, GenerationResult
from mesh_decomposer import decompose, DecompositionConfig
from furniture import FurnitureDesign
from simulator import FurnitureSimulator, SimulationResult
from svg_exporter import design_to_svg, design_to_nested_svg
from manufacturing_decomposer_v2 import (
    ManufacturingDecompositionConfigV2,
    ManufacturingDecompositionResult,
    decompose_manufacturing_v2,
)
from dxf_exporter import design_to_dxf, design_to_nested_dxf

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Configuration for the full pipeline."""
    decomposition: DecompositionConfig = field(default_factory=DecompositionConfig)
    run_simulation: bool = True
    sim_duration: float = 3.0
    export_svg: bool = True
    export_nested_svg: bool = True
    optimize_decomposition: bool = True
    output_dir: str = "output"
    use_manufacturing_v2: bool = False
    manufacturing_v2_config: Optional[ManufacturingDecompositionConfigV2] = None
    export_dxf: bool = False


@dataclass
class PipelineResult:
    """Result from the full pipeline."""
    design: FurnitureDesign
    mesh_path: str
    generation: Optional[GenerationResult] = None
    simulation: Optional[SimulationResult] = None
    svg_paths: List[str] = field(default_factory=list)
    nested_svg_path: Optional[str] = None
    design_json_path: Optional[str] = None
    dxf_paths: List[str] = field(default_factory=list)
    manufacturing_result: Optional[ManufacturingDecompositionResult] = None

    @property
    def is_stable(self) -> bool:
        if self.simulation is None:
            return True  # Not tested
        return self.simulation.stable


def _run_post_mesh(
    mesh_path: str,
    design_name: str,
    config: PipelineConfig,
    generation: Optional[GenerationResult] = None,
) -> PipelineResult:
    """Shared logic after a mesh file is available."""
    output_dir = os.path.join(config.output_dir, design_name)
    os.makedirs(output_dir, exist_ok=True)

    manufacturing_result = None
    dxf_paths = []

    if config.use_manufacturing_v2:
        # V2 3D-first manufacturing pipeline
        v2_config = config.manufacturing_v2_config or ManufacturingDecompositionConfigV2(
            default_material=config.decomposition.default_material,
            target_height_mm=config.decomposition.target_height_mm,
            auto_scale=config.decomposition.auto_scale,
        )

        logger.info("Running v2 manufacturing decomposition: %s", mesh_path)
        manufacturing_result = decompose_manufacturing_v2(mesh_path, v2_config)
        design = manufacturing_result.design
        design.name = design_name

        logger.info(
            "V2 manufacturing decomposition: %d components, %d joints, score=%.2f",
            len(design.components),
            len(design.assembly.joints),
            manufacturing_result.score.overall_score,
        )

        # Save manufacturing JSON
        design_json_path = os.path.join(output_dir, "design_manufacturing.json")
        with open(design_json_path, "w") as f:
            json.dump(manufacturing_result.to_manufacturing_json(), f, indent=2)

        # Export DXF
        if config.export_dxf and manufacturing_result.parts:
            dxf_dir = os.path.join(output_dir, "dxf")
            dxf_parts = list(manufacturing_result.parts.items())
            dxf_paths = design_to_dxf(dxf_parts, dxf_dir)
            nested_dxf = os.path.join(dxf_dir, f"{design_name}_nested.dxf")
            design_to_nested_dxf(dxf_parts, nested_dxf)
            dxf_paths.append(nested_dxf)
            logger.info("Exported %d DXF files", len(dxf_paths))

    else:
        # Original voxel-based pipeline
        logger.info("Decomposing mesh: %s", mesh_path)
        design = decompose(
            filepath=mesh_path,
            config=config.decomposition,
            optimize=config.optimize_decomposition,
        )
        design.name = design_name
        logger.info(
            "Decomposed into %d components, %d joints",
            len(design.components), len(design.assembly.joints),
        )

        design_json_path = os.path.join(output_dir, "design.json")

    # Save design JSON (for both paths)
    if not config.use_manufacturing_v2:
        design_json_path = os.path.join(output_dir, "design.json")
        design_data = {
            "name": design.name,
            "components": [
                {
                    "name": c.name,
                    "type": c.type.value,
                    "width": float(c.profile[2][0]) if len(c.profile) >= 3 else 0,
                    "height": float(c.profile[2][1]) if len(c.profile) >= 3 else 0,
                    "thickness": float(c.thickness),
                    "position": [float(x) for x in c.position],
                    "rotation": [float(x) for x in c.rotation],
                    "material": c.material,
                }
                for c in design.components
            ],
            "joints": [
                {
                    "component_a": j.component_a,
                    "component_b": j.component_b,
                    "joint_type": j.joint_type.value,
                }
                for j in design.assembly.joints
            ],
        }
        with open(design_json_path, "w") as f:
            json.dump(design_data, f, indent=2)

    # Step 2: Optional physics simulation
    sim_result = None
    if config.run_simulation:
        logger.info("Running physics simulation...")
        sim = FurnitureSimulator(gui=False)
        try:
            sim_result = sim.simulate(design, duration=config.sim_duration)
            logger.info(
                "Simulation: %s (pos_change=%.4f, rot_change=%.4f)",
                "STABLE" if sim_result.stable else "UNSTABLE",
                sim_result.position_change,
                sim_result.rotation_change,
            )
        finally:
            sim.close()

    # Step 3: Export SVGs
    svg_paths = []
    nested_svg = None
    if config.export_svg:
        svg_dir = os.path.join(output_dir, "svg")
        svg_paths = design_to_svg(design, svg_dir)
        logger.info("Exported %d SVG cut files", len(svg_paths))
        if config.export_nested_svg:
            nested_svg = os.path.join(svg_dir, f"{design_name}_nested.svg")
            design_to_nested_svg(design, nested_svg)

    return PipelineResult(
        design=design,
        mesh_path=mesh_path,
        generation=generation,
        simulation=sim_result,
        svg_paths=svg_paths,
        nested_svg_path=nested_svg,
        design_json_path=design_json_path,
        dxf_paths=dxf_paths,
        manufacturing_result=manufacturing_result,
    )


def run_pipeline(
    provider: MeshProvider,
    prompt: str = "",
    image_path: str = "",
    design_name: str = "furniture",
    config: Optional[PipelineConfig] = None,
) -> PipelineResult:
    """Run the full text/image-to-furniture pipeline.

    Exactly one of prompt or image_path must be provided.

    Args:
        provider: Cloud mesh generation provider (Tripo, Meshy, etc.)
        prompt: Text description of the desired furniture.
        image_path: Path to an input image.
        design_name: Name for this design (used in output paths).
        config: Pipeline configuration.

    Returns:
        PipelineResult with design, simulation results, and SVG paths.
    """
    if config is None:
        config = PipelineConfig()

    output_dir = os.path.join(config.output_dir, design_name)
    os.makedirs(output_dir, exist_ok=True)
    mesh_path = os.path.join(output_dir, f"{design_name}.glb")

    # Generate mesh via cloud provider
    if prompt:
        logger.info("Generating mesh from text: %s", prompt)
        gen_result = provider.text_to_mesh(prompt, mesh_path)
    elif image_path:
        logger.info("Generating mesh from image: %s", image_path)
        gen_result = provider.image_to_mesh(image_path, mesh_path)
    else:
        raise ValueError("Either prompt or image_path must be provided")

    return _run_post_mesh(
        gen_result.mesh_path, design_name, config, generation=gen_result
    )


def run_pipeline_from_mesh(
    mesh_path: str,
    design_name: str = "furniture",
    config: Optional[PipelineConfig] = None,
) -> PipelineResult:
    """Run the pipeline from an existing mesh file (skip cloud generation).

    Useful for local meshes or debugging the decomposition step.

    Args:
        mesh_path: Path to a 3D mesh file (STL, OBJ, GLB, PLY).
        design_name: Name for this design.
        config: Pipeline configuration.

    Returns:
        PipelineResult with design, simulation results, and SVG paths.
    """
    if config is None:
        config = PipelineConfig()

    if not os.path.isfile(mesh_path):
        raise FileNotFoundError(f"Mesh file not found: {mesh_path}")

    return _run_post_mesh(mesh_path, design_name, config)
