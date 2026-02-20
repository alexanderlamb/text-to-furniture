"""Clean-slate Step 1: mesh -> parametric OpenSCAD panel model."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import trimesh
from shapely.geometry import GeometryCollection, MultiPolygon, Polygon
from shapely.ops import unary_union

from materials import MATERIALS
from openscad_step1.audit import AuditTrail, sha256_file
from openscad_step1.contracts import (
    CandidateRegion,
    PanelFamily,
    PanelInstance,
    Step1Config,
    Step1RunResult,
    Step1Violation,
    TrimDecision,
    to_vec2,
    to_vec3,
)
from openscad_step1.scad_writer import render_openscad
from openscad_step1.trim import detect_perpendicular_panel_overlaps, resolve_panel_trims


def run_step1_pipeline(
    *,
    config: Step1Config,
    run_id: str,
    artifacts_dir: Path,
    audit: Optional[AuditTrail] = None,
) -> Step1RunResult:
    if audit is None:
        audit = AuditTrail(run_id=run_id, artifacts_dir=artifacts_dir)

    mesh_path = Path(config.mesh_path)
    mesh_hash = sha256_file(mesh_path)
    mesh = _load_mesh(mesh_path)
    if mesh.is_empty:
        raise ValueError(f"Empty mesh: {mesh_path}")
    mesh.remove_unreferenced_vertices()

    mesh_norm, scale_factor = _normalize_mesh(mesh, config)
    bounds = mesh_norm.extents
    centroid = mesh_norm.bounding_box.centroid

    material, thickness_mm = _resolve_material(config)
    preflight_checkpoint = audit.write_checkpoint(
        phase_index=0,
        phase_name="preflight",
        counts={
            "vertices": int(len(mesh_norm.vertices)),
            "faces": int(len(mesh_norm.faces)),
            "facets": int(len(mesh_norm.facets)),
        },
        metrics={
            "scale_factor": float(scale_factor),
            "mesh_extent_x_mm": float(bounds[0]),
            "mesh_extent_y_mm": float(bounds[1]),
            "mesh_extent_z_mm": float(bounds[2]),
            "material_thickness_mm": float(thickness_mm),
        },
        invariants={
            "units": "mm",
            "coordinate_frame": "right_handed_z_up",
            "material_key": config.material_key,
        },
        outputs={},
        input_hashes={"mesh_sha256": mesh_hash},
    )

    raw_candidates = _extract_candidates(mesh_norm, config)
    audit.append_decision(
        phase_index=1,
        decision_type="candidate_extraction",
        entity_ids=[c.candidate_id for c in raw_candidates],
        alternatives=[
            {"name": "facet_planar_union", "cost": 0.0},
            {"name": "skip_small_regions", "cost": 0.0},
        ],
        selected="facet_planar_union",
        reason_codes=["deterministic", "geometry_first"],
        numeric_evidence={
            "candidate_count": float(len(raw_candidates)),
            "min_region_area_mm2": float(config.min_region_area_mm2),
        },
    )

    candidate_checkpoint = audit.write_checkpoint(
        phase_index=1,
        phase_name="candidate_generation",
        counts={"candidates_raw": len(raw_candidates)},
        metrics={
            "candidate_area_total_mm2": float(sum(c.area_mm2 for c in raw_candidates)),
        },
        invariants={"min_region_area_mm2": config.min_region_area_mm2},
        outputs={},
        input_hashes={"prev_checkpoint_sha256": preflight_checkpoint.payload_sha256},
    )

    merged_candidates = _merge_coplanar_candidates(
        raw_candidates=raw_candidates,
        config=config,
        audit=audit,
    )
    merged_checkpoint = audit.write_checkpoint(
        phase_index=2,
        phase_name="coplanar_merge",
        counts={
            "candidates_raw": len(raw_candidates),
            "candidates_merged": len(merged_candidates),
        },
        metrics={
            "merge_reduction_ratio": _safe_ratio(
                len(raw_candidates) - len(merged_candidates), len(raw_candidates)
            ),
        },
        invariants={
            "coplanar_angle_tol_deg": config.coplanar_angle_tol_deg,
            "coplanar_offset_tol_mm": config.coplanar_offset_tol_mm,
        },
        outputs={},
        input_hashes={"prev_checkpoint_sha256": candidate_checkpoint.payload_sha256},
    )

    families = _build_panel_families(
        candidates=merged_candidates,
        mesh_centroid=centroid,
        config=config,
        material_thickness_mm=thickness_mm,
        audit=audit,
    )
    candidate_map = {c.candidate_id: c for c in merged_candidates}
    selected_families, selection_debug = _select_families(
        families=families,
        part_budget=config.part_budget_max,
        config=config,
        candidate_map=candidate_map,
        material_thickness_mm=thickness_mm,
        audit=audit,
    )
    selected_face_ids = sorted(
        {face_id for family in selected_families for face_id in family.face_ids}
    )
    all_face_ids = sorted(
        {face_id for family in families for face_id in family.face_ids}
    )
    selection_checkpoint = audit.write_checkpoint(
        phase_index=3,
        phase_name="family_selection",
        counts={
            "families_total": len(families),
            "families_selected": len(selected_families),
            "panels_budget": config.part_budget_max,
            "panels_selected": int(
                sum(f.selected_layer_count for f in selected_families)
            ),
        },
        metrics={
            "face_coverage_ratio": _safe_ratio(
                len(selected_face_ids), len(all_face_ids)
            ),
            "face_coverage_ratio_pass1": float(
                selection_debug.get("shell_face_coverage_ratio", 0.0)
            ),
        },
        invariants={
            "selection_algorithm": str(config.selection_mode),
            "cavity_axis_policy": str(config.cavity_axis_policy),
            "shell_policy": str(config.shell_policy),
            "overlap_enforcement": str(config.overlap_enforcement),
            "thin_gap_clearance_mm": float(config.thin_gap_clearance_mm),
            "base_selection_weight_coverage": float(
                config.base_selection_weight_coverage
            ),
            "base_selection_weight_area": float(config.base_selection_weight_area),
        },
        outputs={
            "selected_family_ids": [f.family_id for f in selected_families],
            "selection_debug": selection_debug,
        },
        input_hashes={"prev_checkpoint_sha256": merged_checkpoint.payload_sha256},
    )

    panels = _build_panel_instances(
        selected_families=selected_families,
        candidate_map=candidate_map,
        material_thickness_mm=thickness_mm,
    )
    panel_checkpoint = audit.write_checkpoint(
        phase_index=4,
        phase_name="panel_instantiation",
        counts={"panels": len(panels)},
        metrics={
            "panel_area_total_mm2": float(sum(p.area_mm2 for p in panels)),
            "panel_layer_count_total": float(
                sum(f.selected_layer_count for f in selected_families)
            ),
        },
        invariants={"panel_frame": "origin+basis_vectors"},
        outputs={"panel_ids": [p.panel_id for p in panels]},
        input_hashes={"prev_checkpoint_sha256": selection_checkpoint.payload_sha256},
    )

    # Phase 5: Trim resolution
    if config.trim_enabled:
        panels, trim_decisions, trim_debug = resolve_panel_trims(
            panels=panels, config=config, audit=audit,
        )
        trim_checkpoint = audit.write_checkpoint(
            phase_index=5,
            phase_name="trim_resolution",
            counts={"panels": len(panels), "trim_pairs_applied": len(trim_decisions)},
            metrics={
                "total_area_trimmed_mm2": float(
                    sum(td.loss_trimmed_mm2 for td in trim_decisions)
                ),
            },
            invariants={"trim_max_loss_fraction": config.trim_max_loss_fraction},
            outputs={
                "trim_decisions": [
                    {
                        "trimmed": td.trimmed_panel_id,
                        "receiving": td.receiving_panel_id,
                        "loss_mm2": td.loss_trimmed_mm2,
                        "reason": td.direction_reason,
                    }
                    for td in trim_decisions
                ],
            },
            input_hashes={"prev_checkpoint_sha256": panel_checkpoint.payload_sha256},
        )
    else:
        trim_decisions: List[TrimDecision] = []
        trim_debug: Dict = {}
        trim_checkpoint = panel_checkpoint

    violations = _validate_panels(
        panels=panels,
        config=config,
        max_sheet_size_mm=material.max_size_mm,
        audit=audit,
    )
    status = _status_from_violations(violations)

    scad_code = render_openscad(
        panels=panels,
        material_thickness_mm=thickness_mm,
        design_name=config.design_name,
    )
    spatial_capsule = _build_spatial_capsule(panels)
    design_payload = _build_design_payload(
        run_id=run_id,
        config=config,
        mesh_hash=mesh_hash,
        scale_factor=scale_factor,
        mesh_bounds_mm=bounds,
        material_thickness_mm=thickness_mm,
        panel_families=families,
        selected_families=selected_families,
        panels=panels,
        violations=violations,
        status=status,
        selection_debug=selection_debug,
        trim_decisions=trim_decisions,
    )

    emit_checkpoint = audit.write_checkpoint(
        phase_index=6,
        phase_name="scad_emit_and_validation",
        counts={
            "panels": len(panels),
            "violations": len(violations),
            "violations_error": sum(1 for v in violations if v.severity == "error"),
        },
        metrics={
            "openscad_lines": float(len(scad_code.splitlines())),
            "spatial_relations": float(len(spatial_capsule.get("relations", []))),
        },
        invariants={
            "status": status,
            "spatial_capsule_schema": "openscad_step1.spatial_capsule.v1",
        },
        outputs={
            "design_schema": "openscad_step1.design.v1",
            "status": status,
        },
        input_hashes={"prev_checkpoint_sha256": trim_checkpoint.payload_sha256},
    )

    audit.finalize()
    checkpoints = [c.path for c in audit.checkpoints]

    return Step1RunResult(
        run_id=run_id,
        status=status,
        mesh_hash_sha256=mesh_hash,
        material_key=config.material_key,
        material_thickness_mm=thickness_mm,
        scale_factor=scale_factor,
        mesh_bounds_mm=to_vec3(bounds),
        panel_families=families,
        selected_families=selected_families,
        panels=panels,
        violations=violations,
        checkpoints=checkpoints,
        openscad_code=scad_code,
        design_payload=design_payload,
        spatial_capsule=spatial_capsule,
        decision_log_path=audit.decision_log_path,
        decision_hash_chain_path=audit.hash_chain_path,
        trim_decisions=trim_decisions,
        debug={
            "candidate_count_raw": len(raw_candidates),
            "candidate_count_merged": len(merged_candidates),
            "family_count": len(families),
            "selected_family_count": len(selected_families),
            "selection": selection_debug,
            "trim": trim_debug,
            "emit_checkpoint_sha256": emit_checkpoint.payload_sha256,
        },
    )


def _load_mesh(mesh_path: Path) -> trimesh.Trimesh:
    loaded = trimesh.load(mesh_path, force="mesh")
    if isinstance(loaded, trimesh.Scene):
        if not loaded.geometry:
            raise ValueError(f"Scene contains no geometry: {mesh_path}")
        meshes = [g for g in loaded.geometry.values() if isinstance(g, trimesh.Trimesh)]
        if not meshes:
            raise ValueError(f"Scene has no mesh geometry: {mesh_path}")
        return trimesh.util.concatenate(meshes)
    if not isinstance(loaded, trimesh.Trimesh):
        raise ValueError(f"Unsupported mesh type from {mesh_path}")
    return loaded


def _normalize_mesh(
    mesh: trimesh.Trimesh, config: Step1Config
) -> Tuple[trimesh.Trimesh, float]:
    mesh_out = mesh.copy()
    extents = mesh_out.extents
    scale_factor = 1.0
    if config.auto_scale and extents[2] > 1e-6:
        scale_factor = float(config.target_height_mm / extents[2])
        mesh_out.apply_scale(scale_factor)

    # Center the normalized mesh to stabilize plane offsets and transforms.
    center = mesh_out.bounding_box.centroid
    mesh_out.apply_translation(-center)
    return mesh_out, scale_factor


def _resolve_material(config: Step1Config):
    if config.material_key not in MATERIALS:
        raise ValueError(f"Unknown material key: {config.material_key}")
    material = MATERIALS[config.material_key]
    thicknesses = sorted(float(t) for t in material.thicknesses_mm)
    if not thicknesses:
        raise ValueError(f"Material has no thickness values: {config.material_key}")

    if config.preferred_thickness_mm is None:
        if len(thicknesses) == 1:
            selected = thicknesses[0]
        else:
            selected = thicknesses[1]
    else:
        target = float(config.preferred_thickness_mm)
        selected = min(thicknesses, key=lambda value: abs(value - target))
    return material, float(selected)


def _extract_candidates(
    mesh: trimesh.Trimesh, config: Step1Config
) -> List[CandidateRegion]:
    candidates: List[CandidateRegion] = []
    facets = mesh.facets
    facet_normals = np.asarray(mesh.facets_normal)

    for facet_index, face_ids in enumerate(facets):
        face_ids = np.asarray(face_ids, dtype=int)
        if face_ids.size == 0:
            continue
        area = float(mesh.area_faces[face_ids].sum())
        if area < config.min_region_area_mm2:
            continue

        normal = _unit_vector(facet_normals[facet_index])
        if normal is None:
            continue
        basis_u, basis_v = _basis_from_normal(normal)

        vertices = mesh.vertices[np.unique(mesh.faces[face_ids].reshape(-1))]
        center = np.mean(vertices, axis=0)

        triangles_2d: List[Polygon] = []
        for face_id in face_ids:
            tri_3d = mesh.vertices[mesh.faces[int(face_id)]]
            tri_2d = [
                _project_point(tri_3d[k], center, basis_u, basis_v) for k in range(3)
            ]
            triangle = Polygon(tri_2d)
            if triangle.area > 1e-6:
                triangles_2d.append(triangle)
        if not triangles_2d:
            continue

        merged = _largest_polygon(unary_union(triangles_2d))
        if merged is None:
            continue
        if merged.area < config.min_region_area_mm2:
            continue

        candidate = CandidateRegion(
            candidate_id=f"cand_{len(candidates):03d}",
            normal=to_vec3(normal),
            center_3d=to_vec3(center),
            basis_u=to_vec3(basis_u),
            basis_v=to_vec3(basis_v),
            plane_offset=float(np.dot(normal, center)),
            outline_2d=[to_vec2(p) for p in list(merged.exterior.coords)[:-1]],
            holes_2d=[
                [to_vec2(p) for p in list(hole.coords)[:-1]]
                for hole in merged.interiors
            ],
            area_mm2=float(merged.area),
            source_faces=[int(f) for f in sorted(face_ids.tolist())],
        )
        candidates.append(candidate)

    candidates.sort(key=lambda c: c.candidate_id)
    return candidates


def _merge_coplanar_candidates(
    *,
    raw_candidates: List[CandidateRegion],
    config: Step1Config,
    audit: AuditTrail,
) -> List[CandidateRegion]:
    if not raw_candidates:
        return []

    cos_tol = math.cos(math.radians(config.coplanar_angle_tol_deg))
    remaining = set(range(len(raw_candidates)))
    groups: List[List[int]] = []

    while remaining:
        seed = min(remaining)
        remaining.remove(seed)
        group = [seed]
        seed_c = raw_candidates[seed]
        seed_n = np.asarray(seed_c.normal, dtype=float)
        seed_offset = float(seed_c.plane_offset)

        for idx in sorted(list(remaining)):
            candidate = raw_candidates[idx]
            n = np.asarray(candidate.normal, dtype=float)
            if float(np.dot(seed_n, n)) < cos_tol:
                continue
            if (
                abs(candidate.plane_offset - seed_offset)
                > config.coplanar_offset_tol_mm
            ):
                continue
            group.append(idx)

        for idx in group[1:]:
            remaining.discard(idx)
        groups.append(sorted(group))

    merged: List[CandidateRegion] = []
    for group_index, group in enumerate(groups):
        if len(group) == 1:
            merged.append(raw_candidates[group[0]])
            continue

        reference = raw_candidates[group[0]]
        polygons: List[Polygon] = []
        source_faces: set[int] = set()

        for idx in group:
            candidate = raw_candidates[idx]
            projected = _reproject_candidate_polygon_to_reference(
                source=candidate, reference=reference
            )
            if projected is not None:
                polygons.append(projected)
            source_faces.update(candidate.source_faces)

        raw_union = unary_union(polygons)
        if isinstance(raw_union, MultiPolygon):
            # Coplanar face patches may be split by perpendicular members
            # (e.g. a shelf splitting a side wall into two halves).  Bridge
            # them with convex hull when: (a) components fill >80% of the
            # hull (gap is small), AND (b) the smallest component is >=25%
            # of the largest (ruling out tiny fragment artifacts).
            hull = raw_union.convex_hull
            fill = raw_union.area / hull.area if hull.area > 0 else 0
            areas = sorted([g.area for g in raw_union.geoms], reverse=True)
            balance = areas[-1] / areas[0] if areas[0] > 0 else 0
            if fill > 0.8 and balance > 0.25:
                unioned = _largest_polygon(hull)
            else:
                unioned = _largest_polygon(raw_union)
        else:
            unioned = _largest_polygon(raw_union)
        if unioned is None:
            merged.append(reference)
            continue

        center = np.asarray(reference.center_3d, dtype=float)
        merged_candidate = CandidateRegion(
            candidate_id=f"cand_m_{group_index:03d}",
            normal=reference.normal,
            center_3d=to_vec3(center),
            basis_u=reference.basis_u,
            basis_v=reference.basis_v,
            plane_offset=float(np.dot(np.asarray(reference.normal), center)),
            outline_2d=[to_vec2(p) for p in list(unioned.exterior.coords)[:-1]],
            holes_2d=[
                [to_vec2(p) for p in list(hole.coords)[:-1]]
                for hole in unioned.interiors
            ],
            area_mm2=float(unioned.area),
            source_faces=sorted(source_faces),
        )
        merged.append(merged_candidate)
        audit.append_decision(
            phase_index=2,
            decision_type="coplanar_merge",
            entity_ids=[raw_candidates[idx].candidate_id for idx in group],
            alternatives=[
                {"name": "merge", "cost": 0.0},
                {"name": "keep_separate", "cost": float(len(group) - 1)},
            ],
            selected="merge",
            reason_codes=["normal_and_offset_within_tolerance"],
            numeric_evidence={
                "group_size": float(len(group)),
                "merged_area_mm2": float(unioned.area),
            },
        )

    merged.sort(key=lambda c: c.candidate_id)
    return merged


def _build_panel_families(
    *,
    candidates: List[CandidateRegion],
    mesh_centroid: np.ndarray,
    config: Step1Config,
    material_thickness_mm: float,
    audit: AuditTrail,
) -> List[PanelFamily]:
    if not candidates:
        return []

    cos_pair = math.cos(math.radians(config.pair_angle_tol_deg))
    pair_options: List[Dict[str, float | int]] = []

    for i, cand_a in enumerate(candidates):
        na = np.asarray(cand_a.normal, dtype=float)
        ca = np.asarray(cand_a.center_3d, dtype=float)
        for j in range(i + 1, len(candidates)):
            cand_b = candidates[j]
            nb = np.asarray(cand_b.normal, dtype=float)
            cb = np.asarray(cand_b.center_3d, dtype=float)

            dot = float(np.dot(na, nb))
            if dot > -cos_pair:
                continue
            area_ratio = min(cand_a.area_mm2, cand_b.area_mm2) / max(
                cand_a.area_mm2, cand_b.area_mm2
            )
            if area_ratio < config.pair_min_area_ratio:
                continue

            delta = cb - ca
            gap_mm = abs(float(np.dot(na, delta)))
            if gap_mm < material_thickness_mm * 0.8:
                continue

            lateral = np.linalg.norm(delta - np.dot(delta, na) * na)
            max_dim = max(
                _candidate_max_dimension(cand_a), _candidate_max_dimension(cand_b)
            )
            lateral_limit = max(2.0, config.pair_lateral_fraction * max_dim)
            if float(lateral) > lateral_limit:
                continue

            score = area_ratio * (gap_mm / max(material_thickness_mm, 1e-6))
            score /= 1.0 + float(lateral)
            pair_options.append(
                {
                    "score": float(score),
                    "i": float(i),
                    "j": float(j),
                    "gap_mm": float(gap_mm),
                    "lateral_mm": float(lateral),
                    "area_ratio": float(area_ratio),
                    "dot": float(dot),
                }
            )

    pair_options.sort(
        key=lambda entry: (
            -float(entry["score"]),
            int(entry["i"]),
            int(entry["j"]),
        )
    )

    matched: set[int] = set()
    families: List[PanelFamily] = []
    family_index = 0

    for option in pair_options:
        i = int(option["i"])
        j = int(option["j"])
        if i in matched or j in matched:
            continue

        cand_i = candidates[i]
        cand_j = candidates[j]
        score_i = float(
            np.dot(
                np.asarray(cand_i.center_3d, dtype=float) - mesh_centroid,
                np.asarray(cand_i.normal, dtype=float),
            )
        )
        score_j = float(
            np.dot(
                np.asarray(cand_j.center_3d, dtype=float) - mesh_centroid,
                np.asarray(cand_j.normal, dtype=float),
            )
        )
        representative = cand_i if score_i >= score_j else cand_j
        raw_layers = float(option["gap_mm"]) / max(material_thickness_mm, 1e-6)
        quantized_layers = _quantize_layers(
            raw_layers=raw_layers,
            stack_roundup_bias=config.stack_roundup_bias,
            max_stack_layers=config.max_stack_layers,
        )
        clearance_mm = max(0.0, float(config.thin_gap_clearance_mm))
        dual_shell_min_mm = 2.0 * material_thickness_mm + clearance_mm
        shell_layers_required = 1 if float(option["gap_mm"]) < dual_shell_min_mm else 2
        interior_available_mm = (
            float(option["gap_mm"])
            - shell_layers_required * material_thickness_mm
            - clearance_mm
        )
        interior_capacity = max(
            0,
            int(
                math.floor(
                    max(0.0, interior_available_mm) / max(material_thickness_mm, 1e-6)
                )
            ),
        )
        total_capacity = min(
            max(1, int(config.max_stack_layers)),
            max(shell_layers_required, shell_layers_required + interior_capacity),
        )
        interior_capacity = max(0, total_capacity - shell_layers_required)

        # Hollow-cavity gate: only allow interior fill when sheets can
        # physically bridge the full gap as a solid laminated member.
        max_bridgeable_mm = config.max_stack_layers * material_thickness_mm
        if float(option["gap_mm"]) > max_bridgeable_mm:
            interior_capacity = 0
            total_capacity = shell_layers_required

        family = PanelFamily(
            family_id=f"fam_{family_index:03d}",
            candidate_ids=[cand_i.candidate_id, cand_j.candidate_id],
            representative_candidate_id=representative.candidate_id,
            face_ids=sorted(set(cand_i.source_faces).union(cand_j.source_faces)),
            total_area_mm2=float(max(cand_i.area_mm2, cand_j.area_mm2)),
            layer_count=total_capacity,
            shell_layers_required=shell_layers_required,
            interior_layers_capacity=interior_capacity,
            estimated_gap_mm=float(option["gap_mm"]),
            is_opposite_pair=True,
            axis_role="unassigned_pair",
            notes={
                "pair_score": float(option["score"]),
                "pair_area_ratio": float(option["area_ratio"]),
                "pair_lateral_mm": float(option["lateral_mm"]),
                "raw_layers": float(raw_layers),
                "quantized_layers": float(quantized_layers),
                "hollow_cavity_gate": int(float(option["gap_mm"]) > max_bridgeable_mm),
                "max_bridgeable_mm": float(max_bridgeable_mm),
            },
        )
        families.append(family)
        matched.add(i)
        matched.add(j)
        family_index += 1

        audit.append_decision(
            phase_index=3,
            decision_type="opposite_pair_collapse",
            entity_ids=[cand_i.candidate_id, cand_j.candidate_id],
            alternatives=[
                {
                    "name": "paired_family",
                    "cost": float(1.0 / max(option["score"], 1e-6)),
                },
                {"name": "keep_unpaired", "cost": 1.0},
            ],
            selected="paired_family",
            reason_codes=["opposite_normals", "similar_area", "bounded_lateral_offset"],
            numeric_evidence={
                "pair_score": float(option["score"]),
                "gap_mm": float(option["gap_mm"]),
                "layer_count": float(total_capacity),
                "shell_layers_required": float(shell_layers_required),
                "interior_layers_capacity": float(interior_capacity),
            },
        )

    for idx, candidate in enumerate(candidates):
        if idx in matched:
            continue
        families.append(
            PanelFamily(
                family_id=f"fam_{family_index:03d}",
                candidate_ids=[candidate.candidate_id],
                representative_candidate_id=candidate.candidate_id,
                face_ids=sorted(candidate.source_faces),
                total_area_mm2=float(candidate.area_mm2),
                layer_count=1,
                shell_layers_required=1,
                interior_layers_capacity=0,
                estimated_gap_mm=0.0,
                is_opposite_pair=False,
                axis_role="unpaired_shell",
                notes={},
            )
        )
        family_index += 1

    families.sort(key=lambda family: family.family_id)
    return families


def _select_families(
    *,
    families: List[PanelFamily],
    part_budget: int,
    config: Step1Config,
    candidate_map: Dict[str, CandidateRegion],
    material_thickness_mm: float,
    audit: AuditTrail,
) -> Tuple[List[PanelFamily], Dict[str, object]]:
    if config.selection_mode != "cavity_primary_nonoverlap_v1":
        raise ValueError(
            f"Unsupported selection_mode: {config.selection_mode}. "
            "Supported mode: cavity_primary_nonoverlap_v1"
        )
    if config.cavity_axis_policy != "single_primary":
        raise ValueError(
            f"Unsupported cavity_axis_policy: {config.cavity_axis_policy}. "
            "Supported policy: single_primary"
        )
    if config.overlap_enforcement != "hard_prevent":
        raise ValueError(
            f"Unsupported overlap_enforcement: {config.overlap_enforcement}. "
            "Supported policy: hard_prevent"
        )

    for family in families:
        family.cavity_id = None
        family.axis_role = "primary" if family.is_opposite_pair else "unpaired_shell"
        family.selected_shell_layers = 0
        family.selected_interior_layers = 0
        family.selected_layer_count = 0

    budget_left = max(1, int(part_budget))
    all_faces = {face for family in families for face in family.face_ids}
    uncovered_faces = set(all_faces)
    budget_initial = budget_left
    family_by_id = {family.family_id: family for family in families}

    weight_cov, weight_area = _normalize_weights(
        float(config.base_selection_weight_coverage),
        float(config.base_selection_weight_area),
    )
    max_area_total = max((float(f.total_area_mm2) for f in families), default=0.0)
    max_gap = max((float(f.estimated_gap_mm) for f in families), default=0.0)

    conflict_edges = _build_family_conflict_edges(
        families=families,
        candidate_map=candidate_map,
        config=config,
        material_thickness_mm=material_thickness_mm,
    )
    cavity_groups = _assign_cavity_roles(
        families=families,
        conflict_edges=conflict_edges,
        audit=audit,
    )

    shell_selected_ids: List[str] = []
    interior_increment_ids: List[str] = []
    blocked_shell_conflicts = 0
    blocked_interior_conflicts = 0

    # Pass 1: reserve shell layers with coverage-first weighting.
    while budget_left > 0:
        candidates = [
            family
            for family in families
            if family.selected_shell_layers == 0
            and family.shell_layers_required > 0
            and budget_left >= int(family.shell_layers_required)
        ]
        if not candidates:
            break

        face_gains = {
            family.family_id: len(set(family.face_ids).intersection(uncovered_faces))
            for family in candidates
        }
        max_face_gain = max(face_gains.values(), default=0)

        best: Optional[PanelFamily] = None
        best_score = float("-inf")
        best_unique_gain = 0
        for family in candidates:
            unique_gain = int(face_gains[family.family_id])
            if not _can_allocate_representative_layers(
                family=family,
                add_rep_layers=1,
                family_by_id=family_by_id,
                conflict_edges=conflict_edges,
            ):
                blocked_shell_conflicts += 1
                continue
            unique_norm = _safe_ratio(unique_gain, max_face_gain)
            area_norm = _safe_ratio(float(family.total_area_mm2), max_area_total)
            base_score = weight_cov * unique_norm + weight_area * area_norm
            if unique_gain <= 0:
                base_score *= 0.35
            shell_cost = max(1, int(family.shell_layers_required))
            score = base_score / float(shell_cost)
            tie = (
                score > best_score
                or (
                    math.isclose(score, best_score)
                    and best is not None
                    and family.family_id < best.family_id
                )
                or (math.isclose(score, best_score) and best is None)
            )
            if tie:
                best = family
                best_score = score
                best_unique_gain = unique_gain

        if best is None:
            break

        best.selected_shell_layers = int(best.shell_layers_required)
        best.selected_layer_count = int(best.selected_shell_layers)
        budget_left -= int(best.shell_layers_required)
        uncovered_faces -= set(best.face_ids)
        shell_selected_ids.append(best.family_id)

        audit.append_decision(
            phase_index=3,
            decision_type="shell_reservation",
            entity_ids=[best.family_id],
            alternatives=[
                {
                    "name": "reserve_shell_layers",
                    "cost": float(1.0 / max(best_score, 1e-6)),
                },
                {"name": "skip_shell", "cost": float(best_unique_gain)},
            ],
            selected="reserve_shell_layers",
            reason_codes=["coverage_first", "required_boundary_shell"],
            numeric_evidence={
                "family_score": float(best_score),
                "unique_face_gain": float(best_unique_gain),
                "budget_remaining": float(budget_left),
                "shell_layers_required": float(best.shell_layers_required),
            },
        )

        min_cov = config.stack_min_pass1_coverage_ratio
        if min_cov is not None:
            shell_cov = _safe_ratio(
                len(all_faces) - len(uncovered_faces), len(all_faces)
            )
            if shell_cov >= float(min_cov):
                break

    shell_coverage_ratio = _safe_ratio(
        len(all_faces) - len(uncovered_faces), len(all_faces)
    )

    # Pass 2: allocate interior layers to primary cavity owners only.
    while budget_left > 0:
        best: Optional[PanelFamily] = None
        best_score = float("-inf")
        best_extra_index = 0

        for family in families:
            if family.selected_shell_layers <= 0:
                continue
            if not family.is_opposite_pair:
                continue
            if family.axis_role != "primary":
                continue
            if family.selected_interior_layers >= max(
                0, int(family.interior_layers_capacity)
            ):
                continue
            if not _can_allocate_representative_layers(
                family=family,
                add_rep_layers=1,
                family_by_id=family_by_id,
                conflict_edges=conflict_edges,
            ):
                blocked_interior_conflicts += 1
                audit.append_decision(
                    phase_index=3,
                    decision_type="interior_increment_blocked_overlap",
                    entity_ids=[family.family_id],
                    alternatives=[
                        {"name": "block_increment", "cost": 0.0},
                        {"name": "allow_increment", "cost": 1.0},
                    ],
                    selected="block_increment",
                    reason_codes=["hard_prevent_overlap"],
                    numeric_evidence={
                        "selected_interior_layers": float(
                            family.selected_interior_layers
                        ),
                        "interior_layers_capacity": float(
                            family.interior_layers_capacity
                        ),
                    },
                )
                continue

            extra_index = max(1, int(family.selected_interior_layers) + 1)
            decay = _stack_decay_value(extra_index, config.stack_decay_schedule)
            area_norm = _safe_ratio(float(family.total_area_mm2), max_area_total)
            gap_norm = _safe_ratio(float(family.estimated_gap_mm), max_gap)
            score = (0.85 * area_norm + 0.15 * gap_norm) * decay
            tie = (
                score > best_score
                or (
                    math.isclose(score, best_score)
                    and best is not None
                    and family.family_id < best.family_id
                )
                or (math.isclose(score, best_score) and best is None)
            )
            if tie:
                best = family
                best_score = score
                best_extra_index = extra_index

        if best is None:
            break

        best.selected_interior_layers += 1
        best.selected_layer_count = int(best.selected_shell_layers) + int(
            best.selected_interior_layers
        )
        budget_left -= 1
        interior_increment_ids.append(best.family_id)

        audit.append_decision(
            phase_index=3,
            decision_type="family_selection_interior_increment",
            entity_ids=[best.family_id],
            alternatives=[
                {
                    "name": "add_interior_layer",
                    "cost": float(1.0 / max(best_score, 1e-6)),
                },
                {"name": "skip_interior_layer", "cost": float(best_score)},
            ],
            selected="add_interior_layer",
            reason_codes=["primary_axis_owner", "max_marginal_stack_score"],
            numeric_evidence={
                "family_score": float(best_score),
                "extra_layer_index": float(best_extra_index),
                "selected_shell_layers": float(best.selected_shell_layers),
                "selected_interior_layers": float(best.selected_interior_layers),
                "budget_remaining": float(budget_left),
            },
        )

    selected = [family for family in families if family.selected_layer_count > 0]
    selected.sort(key=lambda family: family.family_id)

    thin_gap_single_panel_count = int(
        sum(
            1
            for family in families
            if family.is_opposite_pair and int(family.shell_layers_required) == 1
        )
    )
    budget_spent_shell = int(sum(f.selected_shell_layers for f in selected))
    budget_spent_interior = int(sum(f.selected_interior_layers for f in selected))
    primary_family_ids = sorted(
        [
            family.family_id
            for family in families
            if family.is_opposite_pair and family.axis_role == "primary"
        ]
    )
    cavity_payload = [
        {
            "cavity_id": cavity_id,
            "family_ids": list(payload["family_ids"]),
            "primary_family_id": payload["primary_family_id"],
            "conflict_edge_count": int(payload["conflict_edge_count"]),
        }
        for cavity_id, payload in sorted(cavity_groups.items())
    ]

    selection_debug: Dict[str, object] = {
        "selection_mode": "cavity_primary_nonoverlap_v1",
        "budget_initial": int(budget_initial),
        "budget_remaining": int(budget_left),
        "budget_spent_shell": budget_spent_shell,
        "budget_spent_interior": budget_spent_interior,
        "shell_selected_family_ids": shell_selected_ids,
        "interior_increment_family_ids": interior_increment_ids,
        "shell_face_coverage_ratio": float(shell_coverage_ratio),
        "final_face_coverage_ratio": float(
            _safe_ratio(
                len({face for family in selected for face in family.face_ids}),
                len(all_faces),
            )
        ),
        "selected_family_count": int(len(selected)),
        "selected_panel_layers_total": int(
            sum(family.selected_layer_count for family in selected)
        ),
        "thin_gap_single_panel_count": thin_gap_single_panel_count,
        "blocked_shell_conflicts": int(blocked_shell_conflicts),
        "blocked_interior_conflicts": int(blocked_interior_conflicts),
        "cavity_count": int(len(cavity_payload)),
        "primary_family_ids": primary_family_ids,
        "cavities": cavity_payload,
        # Backward-compatible keys for existing dashboards/tests.
        "budget_spent_pass1": budget_spent_shell,
        "budget_spent_pass2": budget_spent_interior,
        "pass1_selected_family_ids": shell_selected_ids,
        "pass2_stack_increment_family_ids": interior_increment_ids,
        "pass1_face_coverage_ratio": float(shell_coverage_ratio),
    }
    return selected, selection_debug


def _build_family_conflict_edges(
    *,
    families: List[PanelFamily],
    candidate_map: Dict[str, CandidateRegion],
    config: Step1Config,
    material_thickness_mm: float,
) -> Dict[Tuple[str, str], Dict[str, float]]:
    conflicts: Dict[Tuple[str, str], Dict[str, float]] = {}
    opposite_families = [f for f in families if f.is_opposite_pair]
    cos_pair = math.cos(math.radians(config.pair_angle_tol_deg))
    clearance_mm = max(0.0, float(config.thin_gap_clearance_mm))

    for i, fam_a in enumerate(opposite_families):
        cand_a = candidate_map.get(fam_a.representative_candidate_id)
        if cand_a is None:
            continue
        poly_a = Polygon(cand_a.outline_2d, holes=cand_a.holes_2d)
        if poly_a.is_empty:
            continue
        n_a = _unit_vector(np.asarray(cand_a.normal, dtype=float))
        c_a = np.asarray(cand_a.center_3d, dtype=float)
        if n_a is None:
            continue

        for fam_b in opposite_families[i + 1 :]:
            cand_b = candidate_map.get(fam_b.representative_candidate_id)
            if cand_b is None:
                continue
            n_b = _unit_vector(np.asarray(cand_b.normal, dtype=float))
            if n_b is None:
                continue
            if float(np.dot(n_a, n_b)) > -cos_pair:
                continue

            projected_b = _reproject_candidate_polygon_to_reference(
                source=cand_b,
                reference=cand_a,
            )
            if projected_b is None or projected_b.is_empty:
                continue

            overlap_area = float(poly_a.intersection(projected_b).area)
            if overlap_area <= 1e-3:
                continue
            min_area = max(1e-6, min(float(poly_a.area), float(projected_b.area)))
            overlap_ratio = overlap_area / min_area
            if overlap_ratio < 0.05:
                continue

            c_b = np.asarray(cand_b.center_3d, dtype=float)
            separation_mm = abs(float(np.dot(c_b - c_a, n_a)))
            max_rep_layers = int(
                math.floor(
                    max(0.0, separation_mm - clearance_mm)
                    / max(material_thickness_mm, 1e-6)
                )
            )
            if max_rep_layers <= 0:
                continue

            key = tuple(sorted((fam_a.family_id, fam_b.family_id)))
            conflicts[key] = {
                "max_rep_layers": float(max_rep_layers),
                "separation_mm": float(separation_mm),
                "overlap_area_mm2": float(overlap_area),
                "overlap_ratio": float(overlap_ratio),
            }
    return conflicts


def _assign_cavity_roles(
    *,
    families: List[PanelFamily],
    conflict_edges: Dict[Tuple[str, str], Dict[str, float]],
    audit: AuditTrail,
) -> Dict[str, Dict[str, object]]:
    family_by_id = {family.family_id: family for family in families}
    adjacency: Dict[str, Set[str]] = {}
    for edge in conflict_edges:
        a_id, b_id = edge
        adjacency.setdefault(a_id, set()).add(b_id)
        adjacency.setdefault(b_id, set()).add(a_id)

    opposite_family_ids = sorted(
        [family.family_id for family in families if family.is_opposite_pair]
    )
    visited: Set[str] = set()
    cavity_groups: Dict[str, Dict[str, object]] = {}
    cavity_index = 0

    for family_id in opposite_family_ids:
        if family_id in visited:
            continue
        stack = [family_id]
        component: List[str] = []
        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            component.append(current)
            for neighbor in sorted(adjacency.get(current, set())):
                if neighbor not in visited:
                    stack.append(neighbor)

        component.sort()
        if not component:
            continue

        ranked = sorted(
            component,
            key=lambda fid: (
                -float(family_by_id[fid].estimated_gap_mm),
                -float(family_by_id[fid].total_area_mm2),
                -len(family_by_id[fid].face_ids),
                fid,
            ),
        )
        primary_id = ranked[0]
        cavity_id = f"cavity_{cavity_index:03d}"
        cavity_index += 1

        conflict_edge_count = 0
        for i in range(len(component)):
            for j in range(i + 1, len(component)):
                edge = tuple(sorted((component[i], component[j])))
                if edge in conflict_edges:
                    conflict_edge_count += 1

        for fid in component:
            family = family_by_id[fid]
            family.cavity_id = cavity_id
            family.axis_role = "primary" if fid == primary_id else "secondary"

        cavity_groups[cavity_id] = {
            "family_ids": component,
            "primary_family_id": primary_id,
            "conflict_edge_count": int(conflict_edge_count),
        }

        audit.append_decision(
            phase_index=3,
            decision_type="cavity_primary_axis_selection",
            entity_ids=component,
            alternatives=[
                {"name": fid, "cost": float(index)} for index, fid in enumerate(ranked)
            ],
            selected=primary_id,
            reason_codes=["single_primary_axis_policy"],
            numeric_evidence={
                "component_size": float(len(component)),
                "conflict_edge_count": float(conflict_edge_count),
            },
        )
    return cavity_groups


def _representative_layer_count(family: PanelFamily) -> int:
    if family.selected_shell_layers <= 0:
        return 0
    # Representative-side occupancy includes one shell plus interior layers.
    return 1 + max(0, int(family.selected_interior_layers))


def _can_allocate_representative_layers(
    *,
    family: PanelFamily,
    add_rep_layers: int,
    family_by_id: Dict[str, PanelFamily],
    conflict_edges: Dict[Tuple[str, str], Dict[str, float]],
) -> bool:
    target_rep_layers = _representative_layer_count(family) + max(
        0, int(add_rep_layers)
    )
    family_id = family.family_id
    for edge_key, payload in conflict_edges.items():
        if family_id not in edge_key:
            continue
        other_id = edge_key[1] if edge_key[0] == family_id else edge_key[0]
        other = family_by_id.get(other_id)
        if other is None:
            continue
        other_rep_layers = _representative_layer_count(other)
        max_rep_layers = max(1, int(payload.get("max_rep_layers", 1)))
        if target_rep_layers + other_rep_layers > max_rep_layers:
            return False
    return True


def _build_panel_instances(
    *,
    selected_families: List[PanelFamily],
    candidate_map: Dict[str, CandidateRegion],
    material_thickness_mm: float,
) -> List[PanelInstance]:
    panels: List[PanelInstance] = []
    panel_counter = 0

    def emit_panel(
        *,
        family: PanelFamily,
        candidate: CandidateRegion,
        layer_offset_from_face: int,
        role: str,
        shell_side: str,
        local_layer_index: int,
    ) -> None:
        nonlocal panel_counter
        normal = np.asarray(candidate.normal, dtype=float)
        center = np.asarray(candidate.center_3d, dtype=float)
        layer_origin = center - normal * material_thickness_mm * layer_offset_from_face
        panel = PanelInstance(
            panel_id=f"panel_{panel_counter:03d}",
            family_id=family.family_id,
            source_candidate_ids=list(family.candidate_ids),
            thickness_mm=float(material_thickness_mm),
            origin_3d=to_vec3(layer_origin),
            basis_u=candidate.basis_u,
            basis_v=candidate.basis_v,
            basis_n=candidate.normal,
            outline_2d=list(candidate.outline_2d),
            holes_2d=[list(hole) for hole in candidate.holes_2d],
            area_mm2=float(candidate.area_mm2),
            source_face_count=len(family.face_ids),
            metadata={
                "panel_role": role,
                "shell_side": shell_side,
                "layer_index_local": local_layer_index,
                "layer_index": panel_counter,
                "layer_count_selected": int(family.selected_layer_count),
                "layer_count_intent": int(family.layer_count),
                "selected_shell_layers": int(family.selected_shell_layers),
                "selected_interior_layers": int(family.selected_interior_layers),
                "shell_layers_required": int(family.shell_layers_required),
                "interior_layers_capacity": int(family.interior_layers_capacity),
                "is_opposite_pair": int(family.is_opposite_pair),
                "estimated_gap_mm": family.estimated_gap_mm,
                "cavity_id": family.cavity_id or "",
                "axis_role": family.axis_role,
            },
        )
        panels.append(panel)
        panel_counter += 1

    for family in selected_families:
        selected_shell_layers = max(0, int(family.selected_shell_layers))
        selected_interior_layers = max(0, int(family.selected_interior_layers))
        family.selected_layer_count = selected_shell_layers + selected_interior_layers
        if family.selected_layer_count <= 0:
            continue

        representative = candidate_map[family.representative_candidate_id]
        opposite_candidate: Optional[CandidateRegion] = None
        if family.is_opposite_pair:
            for candidate_id in family.candidate_ids:
                if candidate_id != family.representative_candidate_id:
                    opposite_candidate = candidate_map.get(candidate_id)
                    if opposite_candidate is not None:
                        break

        shell_emitted = 0
        if selected_shell_layers >= 1:
            emit_panel(
                family=family,
                candidate=representative,
                layer_offset_from_face=1,
                role="shell",
                shell_side="representative",
                local_layer_index=0,
            )
            shell_emitted += 1
        if (
            selected_shell_layers >= 2
            and opposite_candidate is not None
            and family.is_opposite_pair
        ):
            emit_panel(
                family=family,
                candidate=opposite_candidate,
                layer_offset_from_face=1,
                role="shell",
                shell_side="opposite",
                local_layer_index=1,
            )
            shell_emitted += 1

        for interior_index in range(selected_interior_layers):
            emit_panel(
                family=family,
                candidate=representative,
                layer_offset_from_face=1 + interior_index + 1,
                role="interior",
                shell_side="representative",
                local_layer_index=shell_emitted + interior_index,
            )
    return panels


def _validate_panels(
    *,
    panels: List[PanelInstance],
    config: Step1Config,
    max_sheet_size_mm: Tuple[float, float],
    audit: AuditTrail,
) -> List[Step1Violation]:
    violations: List[Step1Violation] = []
    sheet_w, sheet_h = float(max_sheet_size_mm[0]), float(max_sheet_size_mm[1])

    for panel in panels:
        polygon = Polygon(panel.outline_2d, holes=panel.holes_2d)
        if not polygon.is_valid:
            polygon = polygon.buffer(0)
        if polygon.is_empty:
            violations.append(
                Step1Violation(
                    code="panel_empty",
                    severity="error",
                    message="Panel polygon is empty after validity cleanup",
                    panel_id=panel.panel_id,
                )
            )
            continue
        if isinstance(polygon, MultiPolygon):
            polygon = max(polygon.geoms, key=lambda geom: geom.area)

        min_x, min_y, max_x, max_y = polygon.bounds
        width = float(max_x - min_x)
        height = float(max_y - min_y)

        if width > sheet_w + 1e-6:
            violations.append(
                Step1Violation(
                    code="sheet_width_exceeded",
                    severity="error",
                    message=f"Panel width {width:.2f}mm exceeds sheet width {sheet_w:.2f}mm",
                    panel_id=panel.panel_id,
                    value=width,
                    limit=sheet_w,
                )
            )
        if height > sheet_h + 1e-6:
            violations.append(
                Step1Violation(
                    code="sheet_height_exceeded",
                    severity="error",
                    message=f"Panel height {height:.2f}mm exceeds sheet height {sheet_h:.2f}mm",
                    panel_id=panel.panel_id,
                    value=height,
                    limit=sheet_h,
                )
            )

        short_dim = min(width, height)
        if short_dim < config.min_feature_mm:
            violations.append(
                Step1Violation(
                    code="minimum_feature_risk",
                    severity="warning",
                    message=(
                        f"Panel short dimension {short_dim:.2f}mm below "
                        f"minimum feature {config.min_feature_mm:.2f}mm"
                    ),
                    panel_id=panel.panel_id,
                    value=short_dim,
                    limit=config.min_feature_mm,
                )
            )

        area = float(polygon.area)
        if area < config.min_region_area_mm2:
            violations.append(
                Step1Violation(
                    code="panel_area_too_small",
                    severity="error",
                    message=(
                        f"Panel area {area:.2f}mm^2 below threshold "
                        f"{config.min_region_area_mm2:.2f}mm^2"
                    ),
                    panel_id=panel.panel_id,
                    value=area,
                    limit=config.min_region_area_mm2,
                )
            )

        clearance = float(getattr(polygon, "minimum_clearance", 0.0) or 0.0)
        if 0.0 < clearance < config.min_feature_mm:
            violations.append(
                Step1Violation(
                    code="clearance_risk",
                    severity="warning",
                    message=(
                        f"Panel minimum clearance {clearance:.2f}mm below "
                        f"feature floor {config.min_feature_mm:.2f}mm"
                    ),
                    panel_id=panel.panel_id,
                    value=clearance,
                    limit=config.min_feature_mm,
                )
            )

    overlaps = _detect_parallel_panel_overlaps(panels)
    for overlap in overlaps:
        violations.append(
            Step1Violation(
                code="parallel_panel_overlap",
                severity="error",
                message=(
                    f"Parallel panel overlap between {overlap['panel_a']} and "
                    f"{overlap['panel_b']} (penetration={overlap['penetration_mm']:.3f}mm)"
                ),
                panel_id=overlap["panel_a"],
                value=float(overlap["penetration_mm"]),
                limit=0.0,
            )
        )

    if config.trim_enabled:
        perp_overlaps = detect_perpendicular_panel_overlaps(panels, config)
        for ov in perp_overlaps:
            violations.append(
                Step1Violation(
                    code="perpendicular_panel_overlap",
                    severity="warning",
                    message=(
                        f"Perpendicular overlap: {ov['panel_a']} and {ov['panel_b']}, "
                        f"penetration {ov['penetration_mm']:.2f}mm"
                    ),
                    panel_id=ov["panel_a"],
                    value=ov["penetration_mm"],
                )
            )

    if violations:
        audit.append_decision(
            phase_index=6,
            decision_type="manufacturing_validation",
            entity_ids=[v.panel_id or "unknown" for v in violations],
            alternatives=[
                {"name": "accept_with_violations", "cost": float(len(violations))},
                {"name": "reject_run", "cost": float(len(violations) * 2)},
            ],
            selected="accept_with_violations",
            reason_codes=["step1_scope_no_autofix"],
            numeric_evidence={"violation_count": float(len(violations))},
        )

    return violations


def _build_design_payload(
    *,
    run_id: str,
    config: Step1Config,
    mesh_hash: str,
    scale_factor: float,
    mesh_bounds_mm: Sequence[float],
    material_thickness_mm: float,
    panel_families: List[PanelFamily],
    selected_families: List[PanelFamily],
    panels: List[PanelInstance],
    violations: List[Step1Violation],
    status: str,
    selection_debug: Optional[Dict[str, object]] = None,
    trim_decisions: Optional[List[TrimDecision]] = None,
) -> Dict[str, object]:
    selection_debug = selection_debug or {}
    return {
        "schema_version": "openscad_step1.design.v1",
        "run_id": run_id,
        "status": status,
        "units": "mm",
        "mesh": {
            "path": config.mesh_path,
            "sha256": mesh_hash,
            "scale_factor": scale_factor,
            "bounds_mm": [
                float(mesh_bounds_mm[0]),
                float(mesh_bounds_mm[1]),
                float(mesh_bounds_mm[2]),
            ],
        },
        "material": {
            "material_key": config.material_key,
            "thickness_mm": material_thickness_mm,
        },
        "config": {
            "part_budget_max": config.part_budget_max,
            "min_region_area_mm2": config.min_region_area_mm2,
            "coplanar_angle_tol_deg": config.coplanar_angle_tol_deg,
            "pair_angle_tol_deg": config.pair_angle_tol_deg,
            "selection_mode": config.selection_mode,
            "cavity_axis_policy": config.cavity_axis_policy,
            "shell_policy": config.shell_policy,
            "overlap_enforcement": config.overlap_enforcement,
            "thin_gap_clearance_mm": config.thin_gap_clearance_mm,
            "base_selection_weight_coverage": config.base_selection_weight_coverage,
            "base_selection_weight_area": config.base_selection_weight_area,
            "stack_decay_schedule": list(config.stack_decay_schedule),
            "stack_min_pass1_coverage_ratio": config.stack_min_pass1_coverage_ratio,
        },
        "families": [
            {
                "family_id": family.family_id,
                "candidate_ids": family.candidate_ids,
                "representative_candidate_id": family.representative_candidate_id,
                "face_ids": family.face_ids,
                "total_area_mm2": family.total_area_mm2,
                "layer_count": family.layer_count,
                "selected_layer_count": family.selected_layer_count,
                "shell_layers_required": family.shell_layers_required,
                "interior_layers_capacity": family.interior_layers_capacity,
                "selected_shell_layers": family.selected_shell_layers,
                "selected_interior_layers": family.selected_interior_layers,
                "estimated_gap_mm": family.estimated_gap_mm,
                "is_opposite_pair": family.is_opposite_pair,
                "cavity_id": family.cavity_id,
                "axis_role": family.axis_role,
                "notes": family.notes,
            }
            for family in panel_families
        ],
        "cavities": selection_debug.get("cavities", []),
        "selected_family_ids": [family.family_id for family in selected_families],
        "panels": [
            {
                "panel_id": panel.panel_id,
                "family_id": panel.family_id,
                "source_candidate_ids": panel.source_candidate_ids,
                "thickness_mm": panel.thickness_mm,
                "origin_3d": list(panel.origin_3d),
                "basis_u": list(panel.basis_u),
                "basis_v": list(panel.basis_v),
                "basis_n": list(panel.basis_n),
                "outline_2d": [list(point) for point in panel.outline_2d],
                "holes_2d": [
                    [list(point) for point in ring] for ring in panel.holes_2d
                ],
                "area_mm2": panel.area_mm2,
                "source_face_count": panel.source_face_count,
                "metadata": panel.metadata,
            }
            for panel in panels
        ],
        "violations": [
            {
                "code": violation.code,
                "severity": violation.severity,
                "message": violation.message,
                "panel_id": violation.panel_id,
                "value": violation.value,
                "limit": violation.limit,
            }
            for violation in violations
        ],
        "trim_decisions": [
            {
                "trimmed_panel_id": td.trimmed_panel_id,
                "receiving_panel_id": td.receiving_panel_id,
                "loss_trimmed_mm2": round(td.loss_trimmed_mm2, 2),
                "loss_receiving_mm2": round(td.loss_receiving_mm2, 2),
                "dihedral_angle_deg": round(td.dihedral_angle_deg, 1),
                "direction_reason": td.direction_reason,
            }
            for td in (trim_decisions or [])
        ],
    }


def _build_spatial_capsule(panels: List[PanelInstance]) -> Dict[str, object]:
    parts_payload: List[Dict[str, object]] = []
    aabbs: List[Tuple[np.ndarray, np.ndarray]] = []

    for panel in panels:
        poly = Polygon(panel.outline_2d, holes=panel.holes_2d)
        if poly.is_empty:
            continue
        if isinstance(poly, MultiPolygon):
            poly = max(poly.geoms, key=lambda geom: geom.area)

        min_x, min_y, max_x, max_y = poly.bounds
        local_corners = [
            np.array([min_x, min_y, 0.0]),
            np.array([max_x, min_y, 0.0]),
            np.array([max_x, max_y, 0.0]),
            np.array([min_x, max_y, 0.0]),
            np.array([min_x, min_y, panel.thickness_mm]),
            np.array([max_x, min_y, panel.thickness_mm]),
            np.array([max_x, max_y, panel.thickness_mm]),
            np.array([min_x, max_y, panel.thickness_mm]),
        ]
        world_corners = np.asarray(
            [_panel_local_to_world(panel, c) for c in local_corners]
        )
        aabb_min = world_corners.min(axis=0)
        aabb_max = world_corners.max(axis=0)
        aabbs.append((aabb_min, aabb_max))

        center_local = np.array(
            [
                (min_x + max_x) / 2.0,
                (min_y + max_y) / 2.0,
                panel.thickness_mm / 2.0,
            ]
        )
        center_world = _panel_local_to_world(panel, center_local)
        half_extents = [
            float((max_x - min_x) / 2.0),
            float((max_y - min_y) / 2.0),
            float(panel.thickness_mm / 2.0),
        ]

        parts_payload.append(
            {
                "part_id": panel.panel_id,
                "family_id": panel.family_id,
                "thickness_mm": float(panel.thickness_mm),
                "origin_3d": [float(v) for v in panel.origin_3d],
                "basis_u": [float(v) for v in panel.basis_u],
                "basis_v": [float(v) for v in panel.basis_v],
                "basis_n": [float(v) for v in panel.basis_n],
                "outline_2d": [[float(x), float(y)] for x, y in panel.outline_2d],
                "holes_2d": [
                    [[float(x), float(y)] for x, y in hole] for hole in panel.holes_2d
                ],
                "world_aabb": {
                    "min": [float(v) for v in aabb_min],
                    "max": [float(v) for v in aabb_max],
                },
                "obb": {
                    "center": [float(v) for v in center_world],
                    "axes": [
                        [float(v) for v in panel.basis_u],
                        [float(v) for v in panel.basis_v],
                        [float(v) for v in panel.basis_n],
                    ],
                    "half_extents": half_extents,
                },
            }
        )

    relations: List[Dict[str, object]] = []
    for i in range(len(parts_payload)):
        for j in range(i + 1, len(parts_payload)):
            a_min, a_max = aabbs[i]
            b_min, b_max = aabbs[j]
            overlap_x = min(a_max[0], b_max[0]) - max(a_min[0], b_min[0])
            overlap_y = min(a_max[1], b_max[1]) - max(a_min[1], b_min[1])
            overlap_z = min(a_max[2], b_max[2]) - max(a_min[2], b_min[2])

            if overlap_x > 0.0 and overlap_y > 0.0 and overlap_z > 0.0:
                rel_class = "overlapping"
                penetration = float(min(overlap_x, overlap_y, overlap_z))
                min_distance = 0.0
            else:
                dx = _interval_distance(a_min[0], a_max[0], b_min[0], b_max[0])
                dy = _interval_distance(a_min[1], a_max[1], b_min[1], b_max[1])
                dz = _interval_distance(a_min[2], a_max[2], b_min[2], b_max[2])
                min_distance = float(math.sqrt(dx * dx + dy * dy + dz * dz))
                touching = (
                    min_distance <= 1e-6
                    and overlap_x >= -1e-6
                    and overlap_y >= -1e-6
                    and overlap_z >= -1e-6
                )
                rel_class = "touching" if touching else "disjoint"
                penetration = 0.0

            relations.append(
                {
                    "part_a": parts_payload[i]["part_id"],
                    "part_b": parts_payload[j]["part_id"],
                    "class": rel_class,
                    "penetration_mm": penetration,
                    "min_distance_mm": min_distance,
                    "method": "world_aabb_approximation",
                }
            )

    return {
        "schema_version": "openscad_step1.spatial_capsule.v1",
        "units": "mm",
        "frame": {"handedness": "right", "up_axis": "Z"},
        "parts": parts_payload,
        "relations": relations,
    }


def _panel_local_to_world(panel: PanelInstance, local_xyz: np.ndarray) -> np.ndarray:
    u = np.asarray(panel.basis_u, dtype=float)
    v = np.asarray(panel.basis_v, dtype=float)
    n = np.asarray(panel.basis_n, dtype=float)
    o = np.asarray(panel.origin_3d, dtype=float)
    return o + u * local_xyz[0] + v * local_xyz[1] + n * local_xyz[2]


def _detect_parallel_panel_overlaps(
    panels: List[PanelInstance],
) -> List[Dict[str, float | str]]:
    overlaps: List[Dict[str, float | str]] = []
    panel_polys: Dict[str, Polygon] = {}
    for panel in panels:
        poly = Polygon(panel.outline_2d, holes=panel.holes_2d)
        if not poly.is_valid:
            poly = poly.buffer(0)
        if poly.is_empty:
            continue
        if isinstance(poly, MultiPolygon):
            poly = max(poly.geoms, key=lambda geom: geom.area)
        panel_polys[panel.panel_id] = poly

    for i in range(len(panels)):
        panel_a = panels[i]
        poly_a = panel_polys.get(panel_a.panel_id)
        if poly_a is None:
            continue
        n_a = _unit_vector(np.asarray(panel_a.basis_n, dtype=float))
        if n_a is None:
            continue
        u_a = np.asarray(panel_a.basis_u, dtype=float)
        v_a = np.asarray(panel_a.basis_v, dtype=float)
        o_a = np.asarray(panel_a.origin_3d, dtype=float)

        for j in range(i + 1, len(panels)):
            panel_b = panels[j]
            poly_b = panel_polys.get(panel_b.panel_id)
            if poly_b is None:
                continue
            n_b = _unit_vector(np.asarray(panel_b.basis_n, dtype=float))
            if n_b is None:
                continue

            dot = abs(float(np.dot(n_a, n_b)))
            if dot < 0.999:
                continue

            projected_b = _project_panel_polygon_to_reference(
                panel=panel_b,
                polygon=poly_b,
                reference_origin=o_a,
                reference_u=u_a,
                reference_v=v_a,
            )
            if projected_b is None or projected_b.is_empty:
                continue
            overlap_area = float(poly_a.intersection(projected_b).area)
            if overlap_area <= 1e-3:
                continue

            d_a0 = float(np.dot(o_a, n_a))
            d_a1 = d_a0 + float(panel_a.thickness_mm)
            lo_a, hi_a = (min(d_a0, d_a1), max(d_a0, d_a1))

            o_b = np.asarray(panel_b.origin_3d, dtype=float)
            d_b0 = float(np.dot(o_b, n_a))
            d_b1 = d_b0 + float(panel_b.thickness_mm) * float(np.dot(n_b, n_a))
            lo_b, hi_b = (min(d_b0, d_b1), max(d_b0, d_b1))

            penetration = min(hi_a, hi_b) - max(lo_a, lo_b)
            if penetration > 1e-5:
                overlaps.append(
                    {
                        "panel_a": panel_a.panel_id,
                        "panel_b": panel_b.panel_id,
                        "penetration_mm": float(penetration),
                        "overlap_area_mm2": float(overlap_area),
                    }
                )
    return overlaps


def _project_panel_polygon_to_reference(
    *,
    panel: PanelInstance,
    polygon: Polygon,
    reference_origin: np.ndarray,
    reference_u: np.ndarray,
    reference_v: np.ndarray,
) -> Optional[Polygon]:
    u = np.asarray(panel.basis_u, dtype=float)
    v = np.asarray(panel.basis_v, dtype=float)
    o = np.asarray(panel.origin_3d, dtype=float)

    outline_pts = []
    for x, y in list(polygon.exterior.coords)[:-1]:
        world = o + u * float(x) + v * float(y)
        outline_pts.append(
            (
                float(np.dot(world - reference_origin, reference_u)),
                float(np.dot(world - reference_origin, reference_v)),
            )
        )
    holes_pts: List[List[Tuple[float, float]]] = []
    for ring in polygon.interiors:
        hole = []
        for x, y in list(ring.coords)[:-1]:
            world = o + u * float(x) + v * float(y)
            hole.append(
                (
                    float(np.dot(world - reference_origin, reference_u)),
                    float(np.dot(world - reference_origin, reference_v)),
                )
            )
        if len(hole) >= 3:
            holes_pts.append(hole)

    if len(outline_pts) < 3:
        return None
    projected = Polygon(outline_pts, holes=holes_pts if holes_pts else None)
    if not projected.is_valid:
        projected = projected.buffer(0)
    if projected.is_empty:
        return None
    if isinstance(projected, MultiPolygon):
        projected = max(projected.geoms, key=lambda geom: geom.area)
    return projected


def _status_from_violations(violations: List[Step1Violation]) -> str:
    if any(violation.severity == "error" for violation in violations):
        return "error"
    if violations:
        return "warning"
    return "ok"


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return float(numerator) / float(denominator)


def _normalize_weights(weight_cov: float, weight_area: float) -> Tuple[float, float]:
    cov = max(0.0, float(weight_cov))
    area = max(0.0, float(weight_area))
    total = cov + area
    if total <= 1e-9:
        return 0.75, 0.25
    return cov / total, area / total


def _stack_decay_value(extra_layer_index: int, schedule: Sequence[float]) -> float:
    sanitized = [max(0.0, float(v)) for v in schedule if float(v) >= 0.0]
    if not sanitized:
        sanitized = [1.0, 0.65, 0.45, 0.30, 0.20]
    idx = max(1, int(extra_layer_index))
    if idx <= len(sanitized):
        return float(sanitized[idx - 1])
    tail = float(sanitized[-1])
    return float(tail * (0.85 ** (idx - len(sanitized))))


def _candidate_max_dimension(candidate: CandidateRegion) -> float:
    poly = Polygon(candidate.outline_2d)
    min_x, min_y, max_x, max_y = poly.bounds
    return float(max(max_x - min_x, max_y - min_y))


def _quantize_layers(
    *, raw_layers: float, stack_roundup_bias: float, max_stack_layers: int
) -> int:
    if raw_layers <= 1.0:
        return 1
    floor_layers = int(math.floor(raw_layers))
    fraction = raw_layers - floor_layers
    layers = floor_layers + (1 if fraction >= stack_roundup_bias else 0)
    layers = max(1, layers)
    return min(layers, max(1, int(max_stack_layers)))


def _unit_vector(vector: np.ndarray) -> Optional[np.ndarray]:
    norm = float(np.linalg.norm(vector))
    if norm < 1e-9:
        return None
    return np.asarray(vector, dtype=float) / norm


def _basis_from_normal(normal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    n = _unit_vector(normal)
    if n is None:
        raise ValueError("Normal cannot be zero.")
    reference = np.array([0.0, 0.0, 1.0])
    if abs(float(np.dot(reference, n))) > 0.9:
        reference = np.array([1.0, 0.0, 0.0])
    u = _unit_vector(np.cross(reference, n))
    if u is None:
        raise ValueError("Cannot construct basis u.")
    v = _unit_vector(np.cross(n, u))
    if v is None:
        raise ValueError("Cannot construct basis v.")
    return u, v


def _project_point(
    point_3d: np.ndarray,
    origin_3d: np.ndarray,
    basis_u: np.ndarray,
    basis_v: np.ndarray,
) -> Tuple[float, float]:
    delta = np.asarray(point_3d, dtype=float) - np.asarray(origin_3d, dtype=float)
    return float(np.dot(delta, basis_u)), float(np.dot(delta, basis_v))


def _largest_polygon(geom) -> Optional[Polygon]:
    if geom is None or geom.is_empty:
        return None
    polygons: List[Polygon] = []
    if isinstance(geom, Polygon):
        polygons = [geom]
    elif isinstance(geom, MultiPolygon):
        polygons = list(geom.geoms)
    elif isinstance(geom, GeometryCollection):
        polygons = [g for g in geom.geoms if isinstance(g, Polygon)]
    else:
        return None

    cleaned: List[Polygon] = []
    for poly in polygons:
        candidate = poly.buffer(0)
        if candidate.is_empty:
            continue
        if isinstance(candidate, Polygon):
            if candidate.area > 1e-6:
                cleaned.append(candidate)
        elif isinstance(candidate, MultiPolygon):
            cleaned.extend([p for p in candidate.geoms if p.area > 1e-6])

    if not cleaned:
        return None
    return max(
        cleaned, key=lambda polygon: (float(polygon.area), len(polygon.exterior.coords))
    )


def _reproject_candidate_polygon_to_reference(
    source: CandidateRegion, reference: CandidateRegion
) -> Optional[Polygon]:
    src_center = np.asarray(source.center_3d, dtype=float)
    src_u = np.asarray(source.basis_u, dtype=float)
    src_v = np.asarray(source.basis_v, dtype=float)
    ref_center = np.asarray(reference.center_3d, dtype=float)
    ref_u = np.asarray(reference.basis_u, dtype=float)
    ref_v = np.asarray(reference.basis_v, dtype=float)

    def map_ring(points: Sequence[Tuple[float, float]]) -> List[Tuple[float, float]]:
        mapped: List[Tuple[float, float]] = []
        for x, y in points:
            world_pt = src_center + src_u * float(x) + src_v * float(y)
            mapped.append(_project_point(world_pt, ref_center, ref_u, ref_v))
        return mapped

    ext = map_ring(source.outline_2d)
    holes = [map_ring(hole) for hole in source.holes_2d]
    projected = Polygon(ext, holes=holes)
    return _largest_polygon(projected)


def _interval_distance(a_min: float, a_max: float, b_min: float, b_max: float) -> float:
    if a_max < b_min:
        return float(b_min - a_max)
    if b_max < a_min:
        return float(a_min - b_max)
    return 0.0
