"""Panel trim resolution for perpendicular/angled panel intersections.

Detects volumetric overlaps at panel junctions, decides which panel to
shorten, and applies non-fragmenting half-plane cuts.  Joint geometry
(tabs/slots/dogbone) is deferred to a follow-up.

Adapted from the slab-subtraction trim system in ``step3_first_principles.py``.
"""

from __future__ import annotations

import copy
import math
from typing import Dict, List, Optional, Tuple

import numpy as np
from shapely.geometry import MultiPolygon, Point, Polygon, box
from shapely.ops import unary_union

from openscad_step1.audit import AuditTrail
from openscad_step1.contracts import (
    PanelInstance,
    Step1Config,
    TrimDecision,
)

# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

_HALF_PLANE_SIZE = 1e6  # large box extent for half-plane clipping


def _panel_outline_polygon(panel: PanelInstance) -> Polygon:
    """Build a Shapely Polygon from a panel's 2D outline + holes."""
    poly = Polygon(panel.outline_2d, holes=panel.holes_2d or [])
    if not poly.is_valid:
        poly = poly.buffer(0)
    return poly


def _clean_polygon(geom) -> Optional[Polygon]:
    """Return the largest connected Polygon from *geom*, or ``None``."""
    if geom is None or geom.is_empty:
        return None
    if isinstance(geom, Polygon):
        return geom if not geom.is_empty else None
    if isinstance(geom, MultiPolygon):
        largest = max(geom.geoms, key=lambda g: g.area)
        return largest if not largest.is_empty else None
    # GeometryCollection or similar — extract polygons
    polys = [g for g in getattr(geom, "geoms", []) if isinstance(g, Polygon) and not g.is_empty]
    if not polys:
        return None
    return max(polys, key=lambda g: g.area)


def _clip_polygon_half_plane(poly: Polygon, a: float, b: float, c: float) -> Optional[Polygon]:
    """Clip *poly* to the half-plane ``a*x + b*y >= c``.

    Returns the clipped Polygon (or None if empty).  Non-fragmenting by
    construction: the half-plane box is a single convex region.
    """
    norm = math.hypot(a, b)
    if norm < 1e-12:
        return poly  # degenerate — no clip
    an, bn = a / norm, b / norm
    cn = c / norm

    # Build a large quad representing the half-plane a*x + b*y >= c.
    # tangent direction: (-bn, an)
    tx, ty = -bn, an
    S = _HALF_PLANE_SIZE
    cx = an * (cn + S / 2.0)
    cy = bn * (cn + S / 2.0)
    corners = [
        (cx - tx * S - an * S / 2.0, cy - ty * S - bn * S / 2.0),
        (cx + tx * S - an * S / 2.0, cy + ty * S - bn * S / 2.0),
        (cx + tx * S + an * S / 2.0, cy + ty * S + bn * S / 2.0),
        (cx - tx * S + an * S / 2.0, cy - ty * S + bn * S / 2.0),
    ]
    hp_poly = Polygon(corners)
    result = poly.intersection(hp_poly)
    result = result.buffer(0)  # defensive cleanup against slivers
    return _clean_polygon(result)


# ---------------------------------------------------------------------------
# Slab detection
# ---------------------------------------------------------------------------

def _slab_projection_coeffs(
    src: PanelInstance, dst: PanelInstance,
) -> Optional[Tuple[float, float, float, float, float]]:
    """Shared math for slab intrusion and cut.

    Returns (a, b, d_src_origin, lo_dst, hi_dst) or None if panels are
    parallel in src's 2D frame.

    The key equation: a 2D point (u, v) in src's frame projects onto dst's
    normal axis as ``d_src_origin + a*u + b*v``.

    **Thickness compensation**: The 2D outline describes the inner face
    (z=0 in panel-local coordinates), but the panel extends to z=thickness
    along ``basis_n``.  When panels are not perpendicular, the outer face
    projects to ``d_src_origin + a*u + b*v + c_n * thickness`` where
    ``c_n = dot(n_src, n_dst)``.

    We widen the dst slab boundaries asymmetrically — only in the direction
    the outer face extends — so the intrusion strip covers all z-layers of
    the source panel without double-counting:
      lo_dst_eff = lo_dst - max(0, c_n * t)
      hi_dst_eff = hi_dst + max(0, -c_n * t)
    """
    n_dst = np.asarray(dst.basis_n, dtype=float)
    n_src = np.asarray(src.basis_n, dtype=float)
    u_src = np.asarray(src.basis_u, dtype=float)
    v_src = np.asarray(src.basis_v, dtype=float)
    o_src = np.asarray(src.origin_3d, dtype=float)
    o_dst = np.asarray(dst.origin_3d, dtype=float)

    a = float(np.dot(n_dst, u_src))
    b = float(np.dot(n_dst, v_src))
    if math.hypot(a, b) < 1e-9:
        return None  # dst normal is parallel to src normal

    # dst slab boundaries
    d_dst0 = float(np.dot(o_dst, n_dst))
    d_dst1 = d_dst0 + float(dst.thickness_mm)
    lo_dst = min(d_dst0, d_dst1)
    hi_dst = max(d_dst0, d_dst1)

    # Asymmetric thickness compensation
    c_n = float(np.dot(n_src, n_dst))
    t = float(src.thickness_mm)
    lo_dst -= max(0.0, c_n * t)
    hi_dst += max(0.0, -c_n * t)

    d_src_origin = float(np.dot(o_src, n_dst))
    return a, b, d_src_origin, lo_dst, hi_dst


def _panel_slab_intrusion(src: PanelInstance, dst: PanelInstance) -> List[Polygon]:
    """Return polygons in *src*'s 2D frame inside *dst*'s material slab.

    Convention: ``origin_3d`` = anti-normal face,
    ``origin_3d + basis_n * thickness`` = outward surface.
    """
    coeffs = _slab_projection_coeffs(src, dst)
    if coeffs is None:
        return []
    a, b, d_src_origin, lo_dst, hi_dst = coeffs

    c_lo = lo_dst - d_src_origin
    c_hi = hi_dst - d_src_origin

    src_poly = _panel_outline_polygon(src)
    if src_poly.is_empty:
        return []

    # Clip to strip: a*u + b*v >= c_lo  AND  a*u + b*v <= c_hi
    clipped = _clip_polygon_half_plane(src_poly, a, b, c_lo)
    if clipped is None or clipped.is_empty:
        return []
    clipped = _clip_polygon_half_plane(clipped, -a, -b, -c_hi)
    if clipped is None or clipped.is_empty:
        return []

    return [clipped]


def _panel_slab_cut(src: PanelInstance, dst: PanelInstance) -> Optional[Polygon]:
    """Return half-plane polygon in *src*'s 2D frame past *dst*'s slab entry face.

    Subtracting this from src's outline produces a non-fragmenting trim.
    The entry face is the dst slab boundary closest to src's bulk.
    """
    coeffs = _slab_projection_coeffs(src, dst)
    if coeffs is None:
        return None
    a, b, d_src_origin, lo_dst, hi_dst = coeffs

    # Determine which slab face is the "entry" face — the one closer to
    # the centroid of src's outline (in world coordinates, projected onto n_dst).
    src_poly = _panel_outline_polygon(src)
    if src_poly.is_empty:
        return None
    cx, cy = src_poly.centroid.x, src_poly.centroid.y
    d_centroid = d_src_origin + a * cx + b * cy

    if abs(d_centroid - lo_dst) <= abs(d_centroid - hi_dst):
        # Entry at lo_dst: cut everything past it (a*u + b*v >= c_entry)
        c_entry = lo_dst - d_src_origin
        cut = _clip_polygon_half_plane(
            box(-_HALF_PLANE_SIZE, -_HALF_PLANE_SIZE, _HALF_PLANE_SIZE, _HALF_PLANE_SIZE),
            a, b, c_entry,
        )
    else:
        # Entry at hi_dst: cut everything past it (-a*u - b*v >= -c_entry)
        c_entry = hi_dst - d_src_origin
        cut = _clip_polygon_half_plane(
            box(-_HALF_PLANE_SIZE, -_HALF_PLANE_SIZE, _HALF_PLANE_SIZE, _HALF_PLANE_SIZE),
            -a, -b, -c_entry,
        )

    return cut


def _panel_intrusion_overlaps_target(
    intrusion_polys: List[Polygon],
    src: PanelInstance,
    dst: PanelInstance,
) -> bool:
    """Cross-projection gate: does the intrusion actually land on dst's outline?

    Projects the intrusion centroid from src's 2D → 3D → dst's 2D, checks if
    it falls within/near dst's outline.  Prevents phantom trims between
    spatially separated panels whose slab planes happen to intersect.

    Uses centroid projection (not area overlap) because perpendicular
    panels collapse to a degenerate line when projecting one frame's 2D
    region into the other's 2D frame.
    """
    if not intrusion_polys:
        return False

    u_src = np.asarray(src.basis_u, dtype=float)
    v_src = np.asarray(src.basis_v, dtype=float)
    o_src = np.asarray(src.origin_3d, dtype=float)

    u_dst = np.asarray(dst.basis_u, dtype=float)
    v_dst = np.asarray(dst.basis_v, dtype=float)
    o_dst = np.asarray(dst.origin_3d, dtype=float)

    union_poly = unary_union(intrusion_polys)
    if union_poly.is_empty:
        return False
    centroid = union_poly.centroid
    cu, cv = centroid.x, centroid.y

    # src 2D → 3D
    world_pt = o_src + u_src * cu + v_src * cv

    # 3D → dst 2D
    delta = world_pt - o_dst
    du = float(np.dot(delta, u_dst))
    dv = float(np.dot(delta, v_dst))

    dst_poly = _panel_outline_polygon(dst)
    if dst_poly.is_empty:
        return False

    test_pt = Point(du, dv)
    return dst_poly.buffer(dst.thickness_mm * 0.5).contains(test_pt)


# ---------------------------------------------------------------------------
# Trim scoring
# ---------------------------------------------------------------------------

def _effective_loss(outline: Polygon, cut_polys: List[Polygon]) -> float:
    """Simulate subtraction + clean, return area lost.  Returns inf if no cuts."""
    if not cut_polys:
        return float("inf")
    original_area = outline.area
    if original_area < 1e-9:
        return float("inf")
    combined_cut = unary_union(cut_polys)
    result = outline.difference(combined_cut)
    cleaned = _clean_polygon(result)
    if cleaned is None:
        return original_area  # entire panel would be destroyed
    return original_area - cleaned.area


def _structural_trim_priority(
    panel_a: PanelInstance,
    panel_b: PanelInstance,
    loss_a: float,
    loss_b: float,
    config: Step1Config,
) -> Tuple[str, str]:
    """Tiebreaker when effective losses are within 15%.

    Returns (trimmed_panel_id, direction_reason).
    """
    area_a = _panel_outline_polygon(panel_a).area
    area_b = _panel_outline_polygon(panel_b).area

    # 1. Trim smaller-area panel (less structurally important)
    if abs(area_a - area_b) > 0.01 * max(area_a, area_b):
        if area_a < area_b:
            return panel_a.panel_id, "area_tiebreak"
        return panel_b.panel_id, "area_tiebreak"

    # 2. Trim horizontal panel over vertical (furniture convention)
    if config.trim_structural_tiebreak:
        n_a = np.asarray(panel_a.basis_n, dtype=float)
        n_b = np.asarray(panel_b.basis_n, dtype=float)
        vert_a = abs(float(n_a[2]))  # 0=vertical panel, 1=horizontal
        vert_b = abs(float(n_b[2]))
        if abs(vert_a - vert_b) > config.trim_vertical_bias:
            if vert_a > vert_b:
                return panel_a.panel_id, "vertical_bias"
            return panel_b.panel_id, "vertical_bias"

    # 3. Deterministic fallback: higher panel_id gets trimmed
    if panel_a.panel_id > panel_b.panel_id:
        return panel_a.panel_id, "id_tiebreak"
    return panel_b.panel_id, "id_tiebreak"


# ---------------------------------------------------------------------------
# Main trim function
# ---------------------------------------------------------------------------

def resolve_panel_trims(
    *,
    panels: List[PanelInstance],
    config: Step1Config,
    audit: AuditTrail,
) -> Tuple[List[PanelInstance], List[TrimDecision], Dict]:
    """Phase 5: detect perpendicular overlaps and apply non-fragmenting trims.

    Returns (trimmed_panels, trim_decisions, debug_info).
    Uses collect-then-subtract for order-independence.
    """
    if not config.trim_enabled:
        return list(panels), [], {}

    # Deep copy panels so we don't mutate originals during collection
    panels = [copy.deepcopy(p) for p in panels]
    original_polys = {p.panel_id: _panel_outline_polygon(p) for p in panels}
    panel_map = {p.panel_id: p for p in panels}

    # Phase 1 — Collect (read original geometry only)
    # pending_cuts: panel_id -> list of (cut_polygon, estimated_loss, TrimDecision)
    pending_cuts: Dict[str, List[Tuple[Polygon, float, TrimDecision]]] = {}
    decisions: List[TrimDecision] = []
    debug_pairs: List[Dict] = []

    for i in range(len(panels)):
        for j in range(i + 1, len(panels)):
            pa, pb = panels[i], panels[j]
            n_a = np.asarray(pa.basis_n, dtype=float)
            n_b = np.asarray(pb.basis_n, dtype=float)

            # Skip parallel panels
            dot = abs(float(np.dot(n_a, n_b)))
            if dot > config.trim_parallel_dot_threshold:
                continue

            dihedral_deg = float(math.degrees(math.acos(min(1.0, dot))))

            # Compute slab intrusion both directions
            intrusion_a_in_b = _panel_slab_intrusion(pa, pb)
            intrusion_b_in_a = _panel_slab_intrusion(pb, pa)

            area_a_in_b = sum(p.area for p in intrusion_a_in_b) if intrusion_a_in_b else 0.0
            area_b_in_a = sum(p.area for p in intrusion_b_in_a) if intrusion_b_in_a else 0.0

            # Skip if both intrusions below threshold
            if (area_a_in_b < config.trim_min_intrusion_area_mm2
                    and area_b_in_a < config.trim_min_intrusion_area_mm2):
                continue

            # Cross-projection gate
            gate_a = (
                area_a_in_b >= config.trim_min_intrusion_area_mm2
                and _panel_intrusion_overlaps_target(intrusion_a_in_b, pa, pb)
            )
            gate_b = (
                area_b_in_a >= config.trim_min_intrusion_area_mm2
                and _panel_intrusion_overlaps_target(intrusion_b_in_a, pb, pa)
            )
            if not gate_a and not gate_b:
                continue

            # Compute half-plane cuts only for directions that passed the gate
            cut_a = _panel_slab_cut(pa, pb) if gate_a else None
            cut_b = _panel_slab_cut(pb, pa) if gate_b else None

            # Score with effective loss
            poly_a = original_polys[pa.panel_id]
            poly_b = original_polys[pb.panel_id]

            loss_a = _effective_loss(poly_a, [cut_a]) if cut_a else float("inf")
            loss_b = _effective_loss(poly_b, [cut_b]) if cut_b else float("inf")

            if loss_a == float("inf") and loss_b == float("inf"):
                continue

            # Decide direction
            if loss_a == float("inf"):
                trimmed_panel_id = pb.panel_id
                direction_reason = "effective_loss"
            elif loss_b == float("inf"):
                trimmed_panel_id = pa.panel_id
                direction_reason = "effective_loss"
            else:
                ratio = min(loss_a, loss_b) / max(loss_a, loss_b) if max(loss_a, loss_b) > 1e-9 else 1.0
                if ratio < 0.85:
                    # Clear winner — trim the one with less loss
                    trimmed_panel_id = pa.panel_id if loss_a <= loss_b else pb.panel_id
                    direction_reason = "effective_loss"
                else:
                    # Within 15% — use structural tiebreaker
                    trimmed_panel_id, direction_reason = _structural_trim_priority(
                        pa, pb, loss_a, loss_b, config,
                    )

            receiving_panel_id = pb.panel_id if trimmed_panel_id == pa.panel_id else pa.panel_id
            cut_poly = cut_a if trimmed_panel_id == pa.panel_id else cut_b
            est_loss = loss_a if trimmed_panel_id == pa.panel_id else loss_b

            if cut_poly is None:
                continue

            td = TrimDecision(
                trimmed_panel_id=trimmed_panel_id,
                receiving_panel_id=receiving_panel_id,
                loss_trimmed_mm2=est_loss,
                loss_receiving_mm2=loss_b if trimmed_panel_id == pa.panel_id else loss_a,
                dihedral_angle_deg=dihedral_deg,
                direction_reason=direction_reason,
            )
            decisions.append(td)

            if trimmed_panel_id not in pending_cuts:
                pending_cuts[trimmed_panel_id] = []
            pending_cuts[trimmed_panel_id].append((cut_poly, est_loss, td))

            debug_pairs.append({
                "panel_a": pa.panel_id,
                "panel_b": pb.panel_id,
                "trimmed": trimmed_panel_id,
                "loss_a_mm2": loss_a if loss_a != float("inf") else None,
                "loss_b_mm2": loss_b if loss_b != float("inf") else None,
                "dihedral_deg": dihedral_deg,
                "direction_reason": direction_reason,
            })

            # Audit decision
            audit.append_decision(
                phase_index=5,
                decision_type="trim_direction",
                entity_ids=[trimmed_panel_id, receiving_panel_id],
                alternatives=[
                    {"name": f"trim_{pa.panel_id}", "cost": loss_a if loss_a != float("inf") else -1.0},
                    {"name": f"trim_{pb.panel_id}", "cost": loss_b if loss_b != float("inf") else -1.0},
                ],
                selected=f"trim_{trimmed_panel_id}",
                reason_codes=[direction_reason],
                numeric_evidence={
                    "loss_a_mm2": loss_a if loss_a != float("inf") else -1.0,
                    "loss_b_mm2": loss_b if loss_b != float("inf") else -1.0,
                    "dihedral_deg": dihedral_deg,
                },
            )

    # Phase 2 — Apply (deterministic order)
    for panel_id, cuts in sorted(pending_cuts.items()):
        panel = panel_map[panel_id]
        original_poly = original_polys[panel_id]
        original_area = original_poly.area

        # Sort cuts by estimated loss ascending (small edge trims first)
        cuts_sorted = sorted(cuts, key=lambda c: c[1])

        current_outline = original_poly
        for cut_poly, _est_loss, _td in cuts_sorted:
            new_outline = current_outline.difference(cut_poly)
            cleaned = _clean_polygon(new_outline)
            if cleaned is None:
                continue  # would destroy panel entirely

            cumulative_loss = (original_area - cleaned.area) / original_area
            if cumulative_loss > config.trim_max_loss_fraction:
                continue  # budget exceeded

            current_outline = cleaned

        # Update panel geometry
        if current_outline is not None and not current_outline.is_empty:
            exterior_coords = list(current_outline.exterior.coords)
            panel.outline_2d = [(float(x), float(y)) for x, y in exterior_coords]
            panel.holes_2d = [
                [(float(x), float(y)) for x, y in ring.coords]
                for ring in current_outline.interiors
            ]
            panel.area_mm2 = float(current_outline.area)

    debug = {
        "trim_pairs_evaluated": len(debug_pairs),
        "trim_pairs_applied": len(decisions),
        "trim_pair_details": debug_pairs,
    }
    return panels, decisions, debug


# ---------------------------------------------------------------------------
# Perpendicular overlap detection (for validation)
# ---------------------------------------------------------------------------

def detect_perpendicular_panel_overlaps(
    panels: List[PanelInstance],
    config: Step1Config,
) -> List[Dict]:
    """Detect remaining perpendicular panel overlaps after trim.

    Returns list of dicts with panel_a, panel_b, penetration_mm.
    Uses the same slab intrusion math as the trim system.

    An overlap is only reported when BOTH panels mutually intrude into
    each other's slab (and both pass the cross-projection gate).
    After trimming, the trimmed panel no longer intrudes into the
    receiving panel's slab, so correctly-trimmed junctions are not flagged.
    """
    overlaps: List[Dict] = []

    for i in range(len(panels)):
        for j in range(i + 1, len(panels)):
            pa, pb = panels[i], panels[j]
            n_a = np.asarray(pa.basis_n, dtype=float)
            n_b = np.asarray(pb.basis_n, dtype=float)

            dot = abs(float(np.dot(n_a, n_b)))
            if dot > config.trim_parallel_dot_threshold:
                continue  # parallel — handled by existing detector

            # Check slab intrusion both directions
            intrusion_a = _panel_slab_intrusion(pa, pb)
            intrusion_b = _panel_slab_intrusion(pb, pa)

            area_a = sum(p.area for p in intrusion_a) if intrusion_a else 0.0
            area_b = sum(p.area for p in intrusion_b) if intrusion_b else 0.0

            # Require BOTH directions to have significant intrusion
            if (area_a < config.trim_min_intrusion_area_mm2
                    or area_b < config.trim_min_intrusion_area_mm2):
                continue

            # Cross-projection gate — both directions must pass
            gate_a = _panel_intrusion_overlaps_target(intrusion_a, pa, pb)
            gate_b = _panel_intrusion_overlaps_target(intrusion_b, pb, pa)
            if not gate_a or not gate_b:
                continue

            # Use the thickness as a proxy for penetration depth
            penetration = float(min(pa.thickness_mm, pb.thickness_mm))

            overlaps.append({
                "panel_a": pa.panel_id,
                "panel_b": pb.panel_id,
                "penetration_mm": penetration,
                "intrusion_area_a_mm2": area_a,
                "intrusion_area_b_mm2": area_b,
            })

    return overlaps
