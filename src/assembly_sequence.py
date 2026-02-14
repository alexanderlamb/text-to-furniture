"""
Assembly sequence optimization.

Determines the order in which flat-pack components should be assembled,
starting from the largest horizontal surface and working bottom-up so
gravity assists.
"""
import logging
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple

from furniture import FurnitureDesign, Component, ComponentType

logger = logging.getLogger(__name__)


@dataclass
class AssemblyStep:
    """A single step in the assembly sequence."""
    step_number: int
    component_name: str
    action: str            # "place", "attach", "fasten"
    attach_to: str = ""    # component being attached to
    notes: str = ""


def optimize_assembly_sequence(design: FurnitureDesign) -> List[AssemblyStep]:
    """Determine optimal assembly order for a furniture design.

    Strategy:
    1. Start with the largest horizontal surface (tabletop, shelf, seat)
    2. BFS from that base component along joints
    3. Prefer bottom-up ordering (lower z-position first)
    4. Gravity-assist: horizontal surfaces first, then vertical supports

    Args:
        design: The furniture design with components and joints.

    Returns:
        Ordered list of assembly steps.
    """
    if not design.components:
        return []

    # Build adjacency from joints
    adj: Dict[str, List[str]] = {}
    for comp in design.components:
        adj[comp.name] = []
    for joint in design.assembly.joints:
        if joint.component_a in adj and joint.component_b in adj:
            adj[joint.component_a].append(joint.component_b)
            adj[joint.component_b].append(joint.component_a)

    # Find the base component: largest horizontal surface at lowest z
    base = _find_base_component(design)
    if base is None:
        base = design.components[0]

    # BFS from base, prioritizing by z-position (bottom-up)
    steps: List[AssemblyStep] = []
    visited: Set[str] = set()

    # Place the base first
    steps.append(AssemblyStep(
        step_number=1,
        component_name=base.name,
        action="place",
        notes="Base component - place upside down on work surface",
    ))
    visited.add(base.name)

    # BFS queue: (component_name, parent_name)
    queue: List[Tuple[str, str]] = []
    for neighbor in adj.get(base.name, []):
        queue.append((neighbor, base.name))

    # Sort queue by z-position (attach bottom pieces first)
    comp_map = {c.name: c for c in design.components}

    while queue:
        # Sort by z-position to prefer bottom-up assembly
        queue.sort(key=lambda x: comp_map[x[0]].position[2] if x[0] in comp_map else 0)

        name, parent = queue.pop(0)
        if name in visited:
            continue

        visited.add(name)
        comp = comp_map.get(name)
        if comp is None:
            continue

        action = "attach"
        notes = f"Connect to {parent}"

        if comp.type == ComponentType.LEG:
            notes += " - align and insert tabs"
        elif comp.type in (ComponentType.PANEL, ComponentType.SHELF):
            notes += " - slide into slots"
        elif comp.type == ComponentType.BRACE:
            notes += " - cross-brace for stability"

        steps.append(AssemblyStep(
            step_number=len(steps) + 1,
            component_name=name,
            action=action,
            attach_to=parent,
            notes=notes,
        ))

        for neighbor in adj.get(name, []):
            if neighbor not in visited:
                queue.append((neighbor, name))

    # Add any unvisited components (disconnected)
    for comp in design.components:
        if comp.name not in visited:
            steps.append(AssemblyStep(
                step_number=len(steps) + 1,
                component_name=comp.name,
                action="place",
                notes="Disconnected component",
            ))

    logger.info("Assembly sequence: %d steps", len(steps))
    return steps


def _find_base_component(design: FurnitureDesign) -> Component:
    """Find the best base component for assembly.

    Prefers: largest horizontal surface at or near the bottom.
    """
    horizontal_types = {ComponentType.PANEL, ComponentType.SHELF, ComponentType.SEAT}

    # Candidates: horizontal surfaces
    candidates = [
        c for c in design.components
        if c.type in horizontal_types
    ]

    if not candidates:
        # Fallback: largest component
        return max(design.components, key=lambda c: c.get_dimensions()[0] * c.get_dimensions()[1])

    # Score: area / (1 + z_position) — prefer large and low
    def base_score(c: Component) -> float:
        dims = c.get_dimensions()
        area = dims[0] * dims[1]
        z = float(c.position[2])
        return area / (1.0 + max(z, 0.0))

    return max(candidates, key=base_score)
