"""Static HTML reports for hybrid fabrication benchmark evaluations."""

from __future__ import annotations

import ast
import hashlib
import html
import json
import os
from collections.abc import Iterable, Mapping
from pathlib import Path
from urllib.parse import quote, urlparse

DEFAULT_TITLE = "Hybrid Benchmark Visual Report"
REPORT_SCHEMA_VERSION = "fabrication.hybrid_visual_report.v0"

STRATEGY_COLORS = {
    "planar_skin": "#2f80ed",
    "contour_stack": "#f2994a",
    "waffle_ribs": "#8f7a1f",
    "voxel_blocks": "#27ae60",
}

COUNT_LABELS = {
    "meshes": "Meshes",
    "errors": "Errors",
    "mixed_strategy_meshes": "Mixed strategy meshes",
    "meshes_with_boundary_joints": "Meshes with boundary joints",
}


def write_visual_report(
    path: str | Path,
    evaluation_or_rows: Mapping[str, object] | Iterable[Mapping[str, object]],
    *,
    title: str = DEFAULT_TITLE,
) -> Path:
    """Write a self-contained HTML report and return the output path."""

    output_path = Path(path)
    html_report = render_visual_report(
        evaluation_or_rows,
        base_dir=output_path.parent,
        title=title,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html_report, encoding="utf-8")
    return output_path


def write_evaluation_report(
    evaluation: Mapping[str, object], output_path: Path
) -> Path:
    """Compatibility wrapper used by the benchmark evaluator."""

    return write_visual_report(output_path, evaluation)


def render_visual_report(
    evaluation_or_rows: Mapping[str, object] | Iterable[Mapping[str, object]],
    *,
    base_dir: str | Path | None = None,
    title: str = DEFAULT_TITLE,
) -> str:
    """Render a deterministic, standalone HTML report for evaluation rows."""

    payload = _coerce_evaluation(evaluation_or_rows)
    rows = _coerce_rows(payload.get("rows", []))
    counts = _evaluation_counts(payload, rows)
    status = _evaluation_status(payload, rows)
    flag_counts = _flag_counts(payload, rows)
    strategy_totals = _strategy_totals(payload, rows)
    strategy_part_totals = _strategy_part_totals(payload, rows)
    strategy_volume_totals = _strategy_volume_totals(payload, rows)
    all_strategies = _all_strategy_ids(payload, rows, strategy_totals)
    all_strategies = sorted(
        set(all_strategies) | set(strategy_part_totals) | set(strategy_volume_totals)
    )
    base_path = Path(base_dir) if base_dir is not None else None

    page_title = _escape(title)
    return "\n".join(
        [
            "<!doctype html>",
            '<html lang="en">',
            "<head>",
            '  <meta charset="utf-8">',
            '  <meta name="viewport" content="width=device-width, initial-scale=1">',
            f"  <title>{page_title}</title>",
            _style_block(),
            "</head>",
            "<body>",
            '  <main class="report-shell">',
            '    <section class="hero">',
            f'      <p class="eyebrow">{_escape(REPORT_SCHEMA_VERSION)}</p>',
            f"      <h1>{page_title}</h1>",
            f'      <div class="hero-status">{_status_badge(status)}</div>',
            "    </section>",
            _render_summary_section(payload, counts, status),
            _render_strategy_section(strategy_totals, all_strategies),
            _render_burden_section(strategy_part_totals, strategy_volume_totals),
            _render_flag_section(flag_counts),
            _render_rows_section(rows, base_path),
            "  </main>",
            "</body>",
            "</html>",
            "",
        ]
    )


def _coerce_evaluation(
    evaluation_or_rows: Mapping[str, object] | Iterable[Mapping[str, object]],
) -> dict[str, object]:
    if isinstance(evaluation_or_rows, Mapping):
        payload = dict(evaluation_or_rows)
        if "rows" in payload:
            payload["rows"] = _coerce_rows(payload["rows"])
            return payload
        if _looks_like_row(payload):
            return {"rows": [payload]}
        payload["rows"] = []
        return payload

    return {"rows": _coerce_rows(evaluation_or_rows)}


def _coerce_rows(value: object) -> list[dict[str, object]]:
    if value is None:
        return []
    if isinstance(value, (str, bytes)):
        raise TypeError("visual report rows must be an iterable of mappings")
    if not isinstance(value, Iterable):
        raise TypeError("visual report rows must be an iterable of mappings")

    rows: list[dict[str, object]] = []
    for row in value:
        if not isinstance(row, Mapping):
            raise TypeError("each visual report row must be a mapping")
        rows.append(dict(row))
    return rows


def _looks_like_row(value: Mapping[str, object]) -> bool:
    row_keys = {
        "mesh",
        "mesh_path",
        "status",
        "strategy_mix",
        "hybrid_plan",
        "source_ranking",
    }
    return any(key in value for key in row_keys)


def _evaluation_counts(
    payload: Mapping[str, object], rows: list[dict[str, object]]
) -> dict[str, object]:
    counts = dict(_mapping_value(payload.get("counts")))
    counts.setdefault("meshes", len(rows))
    counts.setdefault("errors", sum(1 for row in rows if _status_text(row) == "error"))
    counts.setdefault(
        "mixed_strategy_meshes",
        sum(
            1 for row in rows if len(_strategy_mix_items(row.get("strategy_mix"))) >= 2
        ),
    )
    counts.setdefault(
        "meshes_with_boundary_joints",
        sum(1 for row in rows if _int_value(row.get("joints")) > 0),
    )
    return counts


def _evaluation_status(
    payload: Mapping[str, object], rows: list[dict[str, object]]
) -> str:
    status = str(payload.get("status", "")).strip().lower()
    if status:
        return status

    if rows and all(_status_text(row) == "error" for row in rows):
        return "error"
    has_errors = any(_status_text(row) == "error" for row in rows)
    has_flags = any(_string_list(row.get("flags")) for row in rows)
    return "warning" if has_errors or has_flags else "ok"


def _flag_counts(
    payload: Mapping[str, object], rows: list[dict[str, object]]
) -> dict[str, int]:
    raw_counts = _mapping_value(payload.get("flag_counts"))
    if raw_counts:
        return {
            str(flag): _int_value(count)
            for flag, count in sorted(raw_counts.items(), key=lambda item: str(item[0]))
            if _int_value(count) > 0
        }

    counts: dict[str, int] = {}
    for row in rows:
        for flag in _string_list(row.get("flags")):
            counts[flag] = counts.get(flag, 0) + 1
    return dict(sorted(counts.items()))


def _strategy_totals(
    payload: Mapping[str, object], rows: list[dict[str, object]]
) -> dict[str, int]:
    raw_totals = _mapping_value(payload.get("strategy_region_use"))
    if raw_totals:
        return _positive_int_mapping(raw_totals)

    totals: dict[str, int] = {}
    for row in rows:
        for strategy_id, count in _strategy_mix_items(row.get("strategy_mix")):
            totals[strategy_id] = totals.get(strategy_id, 0) + count
    return dict(sorted(totals.items()))


def _strategy_part_totals(
    payload: Mapping[str, object], rows: list[dict[str, object]]
) -> dict[str, int]:
    raw_totals = _mapping_value(payload.get("strategy_part_use"))
    if raw_totals:
        return _positive_int_mapping(raw_totals)

    totals: dict[str, int] = {}
    for row in rows:
        mapping = _mapping_value(row.get("strategy_part_burden"))
        for strategy_id, count in _positive_int_mapping(mapping).items():
            totals[strategy_id] = totals.get(strategy_id, 0) + count
    return dict(sorted(totals.items()))


def _strategy_volume_totals(
    payload: Mapping[str, object], rows: list[dict[str, object]]
) -> dict[str, float]:
    raw_totals = _mapping_value(payload.get("strategy_volume_use_mm3"))
    if raw_totals:
        return _positive_float_mapping(raw_totals)

    totals: dict[str, float] = {}
    for row in rows:
        mapping = _mapping_value(row.get("strategy_volume_burden_mm3"))
        for strategy_id, volume in _positive_float_mapping(mapping).items():
            totals[strategy_id] = totals.get(strategy_id, 0.0) + volume
    return dict(sorted(totals.items()))


def _all_strategy_ids(
    payload: Mapping[str, object],
    rows: list[dict[str, object]],
    strategy_totals: Mapping[str, int],
) -> list[str]:
    strategies = set(strategy_totals)
    configured = payload.get("strategies")
    if isinstance(configured, str):
        if configured != "config_default":
            strategies.add(configured)
    elif isinstance(configured, Iterable):
        for strategy_id in configured:
            strategies.add(str(strategy_id))

    for row in rows:
        for strategy_id, _count in _strategy_mix_items(row.get("strategy_mix")):
            strategies.add(strategy_id)
        winner = str(row.get("source_winner", "")).strip()
        if winner:
            strategies.add(winner)

    return sorted(strategies)


def _render_summary_section(
    payload: Mapping[str, object], counts: Mapping[str, object], status: str
) -> str:
    strategies = payload.get("strategies", "")
    if isinstance(strategies, (list, tuple)):
        strategy_text = ", ".join(str(strategy_id) for strategy_id in strategies)
    else:
        strategy_text = str(strategies) if strategies else "not specified"

    rows = [
        ("Status", _status_badge(status)),
        ("Elapsed seconds", _escape(_format_float(payload.get("elapsed_s"), digits=3))),
        ("Strategies", _escape(strategy_text)),
    ]
    schema_version = payload.get("schema_version")
    if schema_version:
        rows.append(("Evaluation schema", _escape(schema_version)))
    for key, label in COUNT_LABELS.items():
        rows.append((label, _escape(_format_plain(counts.get(key, 0)))))

    body = "\n".join(
        [
            "          <tr>"
            f'<th scope="row">{_escape(label)}</th>'
            f"<td>{value}</td>"
            "</tr>"
            for label, value in rows
        ]
    )
    return "\n".join(
        [
            '    <section class="card">',
            "      <h2>Summary</h2>",
            '      <table class="summary-table">',
            "        <tbody>",
            body,
            "        </tbody>",
            "      </table>",
            "    </section>",
        ]
    )


def _render_strategy_section(
    strategy_totals: Mapping[str, int], all_strategies: list[str]
) -> str:
    legend = _render_strategy_legend(all_strategies)
    if not strategy_totals:
        body = '<p class="empty">No strategy assignments were recorded.</p>'
    else:
        max_count = max(strategy_totals.values())
        rows = []
        for strategy_id, count in sorted(
            strategy_totals.items(), key=lambda item: (-item[1], item[0])
        ):
            width = 100.0 * (float(count) / float(max_count or 1))
            rows.append(
                '        <div class="usage-row">'
                f'<span class="swatch" style="background:{_strategy_color(strategy_id)}"></span>'
                f'<span class="usage-label">{_escape(strategy_id)}</span>'
                '<span class="usage-track">'
                f'<span class="usage-fill" style="width:{_format_percent(width)};'
                f'background:{_strategy_color(strategy_id)}"></span>'
                "</span>"
                f'<span class="usage-count">{count}</span>'
                "</div>"
            )
        body = "\n".join(['      <div class="strategy-usage">', *rows, "      </div>"])

    return "\n".join(
        [
            '    <section class="card">',
            "      <h2>Strategy Mix</h2>",
            legend,
            body,
            "    </section>",
        ]
    )


def _render_burden_section(
    part_totals: Mapping[str, int], volume_totals: Mapping[str, float]
) -> str:
    if not part_totals and not volume_totals:
        return ""

    return "\n".join(
        [
            '    <section class="card">',
            "      <h2>Strategy Burden</h2>",
            '      <p class="section-note">Counts selected composed parts and estimated material volume by source strategy.</p>',
            '      <div class="burden-grid">',
            '        <div class="burden-panel">',
            "          <h3>Selected Parts</h3>",
            _render_int_usage(part_totals, empty_text="No selected parts recorded."),
            "        </div>",
            '        <div class="burden-panel">',
            "          <h3>Estimated Volume</h3>",
            _render_float_usage(
                volume_totals,
                empty_text="No material volume recorded.",
                formatter=_format_volume_mm3,
            ),
            "        </div>",
            "      </div>",
            "    </section>",
        ]
    )


def _render_int_usage(
    totals: Mapping[str, int], *, empty_text: str = "No values recorded."
) -> str:
    if not totals:
        return f'<p class="empty">{_escape(empty_text)}</p>'
    max_count = max(totals.values())
    rows = []
    for strategy_id, count in sorted(
        totals.items(), key=lambda item: (-item[1], item[0])
    ):
        width = 100.0 * (float(count) / float(max_count or 1))
        rows.append(
            _usage_row(
                strategy_id=strategy_id,
                width=width,
                value_text=str(count),
            )
        )
    return "\n".join(
        ['          <div class="strategy-usage">', *rows, "          </div>"]
    )


def _render_float_usage(
    totals: Mapping[str, float],
    *,
    empty_text: str = "No values recorded.",
    formatter,
) -> str:
    if not totals:
        return f'<p class="empty">{_escape(empty_text)}</p>'
    max_value = max(totals.values())
    rows = []
    for strategy_id, value in sorted(
        totals.items(), key=lambda item: (-float(item[1]), item[0])
    ):
        width = 100.0 * (float(value) / float(max_value or 1.0))
        rows.append(
            _usage_row(
                strategy_id=strategy_id,
                width=width,
                value_text=formatter(value),
            )
        )
    return "\n".join(
        ['          <div class="strategy-usage">', *rows, "          </div>"]
    )


def _usage_row(*, strategy_id: str, width: float, value_text: str) -> str:
    return (
        '        <div class="usage-row">'
        f'<span class="swatch" style="background:{_strategy_color(strategy_id)}"></span>'
        f'<span class="usage-label">{_escape(strategy_id)}</span>'
        '<span class="usage-track">'
        f'<span class="usage-fill" style="width:{_format_percent(width)};'
        f'background:{_strategy_color(strategy_id)}"></span>'
        "</span>"
        f'<span class="usage-count">{_escape(value_text)}</span>'
        "</div>"
    )


def _render_strategy_legend(strategy_ids: list[str]) -> str:
    if not strategy_ids:
        return ""
    chips = []
    for strategy_id in strategy_ids:
        chips.append(
            '        <span class="legend-chip">'
            f'<span class="swatch" style="background:{_strategy_color(strategy_id)}"></span>'
            f"{_escape(strategy_id)}"
            "</span>"
        )
    return "\n".join(['      <div class="legend">', *chips, "      </div>"])


def _render_flag_section(flag_counts: Mapping[str, int]) -> str:
    if not flag_counts:
        body = '<p class="empty">No quality flags were recorded.</p>'
    else:
        badges = []
        for flag, count in sorted(flag_counts.items()):
            badges.append(
                '        <span class="flag-chip">'
                f"{_escape(flag)} <strong>{count}</strong>"
                "</span>"
            )
        body = "\n".join(['      <div class="flag-cloud">', *badges, "      </div>"])

    return "\n".join(
        [
            '    <section class="card">',
            "      <h2>Flags</h2>",
            body,
            "    </section>",
        ]
    )


def _render_rows_section(rows: list[dict[str, object]], base_dir: Path | None) -> str:
    if not rows:
        table = '<p class="empty">No mesh rows were supplied.</p>'
    else:
        table_rows = [
            _render_detail_row(index, row, base_dir) for index, row in enumerate(rows)
        ]
        table = "\n".join(
            [
                '      <div class="table-wrap">',
                '        <table class="detail-table">',
                "          <thead>",
                "            <tr>",
                "              <th>Mesh</th>",
                "              <th>Status</th>",
                "              <th>Score</th>",
                "              <th>Regions</th>",
                "              <th>Parts</th>",
                "              <th>Joints</th>",
                "              <th>Reuse/Shared</th>",
                "              <th>Preview</th>",
                "              <th>Strategy mix</th>",
                "              <th>Winner</th>",
                "              <th>Flags</th>",
                "              <th>Warnings</th>",
                "              <th>Elapsed</th>",
                "              <th>Artifacts</th>",
                "            </tr>",
                "          </thead>",
                "          <tbody>",
                *table_rows,
                "          </tbody>",
                "        </table>",
                "      </div>",
            ]
        )

    return "\n".join(
        [
            '    <section class="card wide">',
            "      <h2>Mesh Details</h2>",
            table,
            "    </section>",
        ]
    )


def _render_detail_row(
    index: int, row: Mapping[str, object], base_dir: Path | None
) -> str:
    mesh_name = str(row.get("mesh") or "")
    if not mesh_name:
        mesh_path = str(row.get("mesh_path") or "")
        mesh_name = Path(mesh_path).name if mesh_path else f"mesh_{index + 1:03d}"

    artifacts = " ".join(
        [
            _artifact_link(row.get("hybrid_plan"), "hybrid_plan", base_dir),
            _artifact_link(row.get("source_ranking"), "source_ranking", base_dir),
        ]
    )
    return "\n".join(
        [
            "            <tr>",
            f'              <th scope="row">{_escape(mesh_name)}</th>',
            f"              <td>{_status_badge(_status_text(row))}</td>",
            f"              <td>{_escape(_format_float(row.get('overall_score'), digits=3))}</td>",
            f"              <td>{_escape(_format_plain(row.get('regions', 0)))}</td>",
            f"              <td>{_escape(_format_plain(row.get('parts', 0)))}</td>",
            f"              <td>{_escape(_format_plain(row.get('joints', 0)))}</td>",
            f"              <td>{_escape(_reuse_text(row))}</td>",
            f"              <td>{_render_region_preview(row.get('hybrid_plan'), base_dir)}</td>",
            f"              <td>{_render_strategy_mix(row.get('strategy_mix'))}</td>",
            f"              <td>{_escape(str(row.get('source_winner') or ''))}</td>",
            f"              <td>{_render_row_flags(row.get('flags'))}</td>",
            f"              <td>{_render_warnings(row.get('warnings'), row.get('warning_count'))}</td>",
            f"              <td>{_escape(_format_float(row.get('elapsed_s'), digits=3))}s</td>",
            f'              <td class="artifact-links">{artifacts}</td>',
            "            </tr>",
        ]
    )


def _render_region_preview(value: object, base_dir: Path | None) -> str:
    plan_path = _artifact_path(value, base_dir)
    if plan_path is None or not plan_path.exists():
        return '<span class="muted">preview unavailable</span>'

    try:
        payload = json.loads(plan_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return '<span class="muted">preview unavailable</span>'

    regions = payload.get("regions", [])
    assignments = payload.get("assignments", [])
    if not isinstance(regions, list) or not regions:
        return '<span class="muted">preview unavailable</span>'
    if not isinstance(assignments, list):
        assignments = []

    strategy_by_region = {
        str(assignment.get("region_id")): str(assignment.get("strategy_id"))
        for assignment in assignments
        if isinstance(assignment, Mapping)
    }
    boxes = []
    for region in regions:
        if not isinstance(region, Mapping):
            continue
        aabb_min = _float_triplet(region.get("aabb_min"))
        aabb_max = _float_triplet(region.get("aabb_max"))
        if aabb_min is None or aabb_max is None:
            continue
        region_id = str(region.get("region_id") or "")
        boxes.append(
            (region_id, aabb_min, aabb_max, strategy_by_region.get(region_id, ""))
        )

    if not boxes:
        return '<span class="muted">preview unavailable</span>'

    axis_x, axis_y = _preview_axes(boxes)
    min_x = min(minimum[axis_x] for _id, minimum, _maximum, _strategy in boxes)
    max_x = max(maximum[axis_x] for _id, _minimum, maximum, _strategy in boxes)
    min_y = min(minimum[axis_y] for _id, minimum, _maximum, _strategy in boxes)
    max_y = max(maximum[axis_y] for _id, _minimum, maximum, _strategy in boxes)
    span_x = max(max_x - min_x, 1e-9)
    span_y = max(max_y - min_y, 1e-9)

    width = 180.0
    height = 96.0
    pad = 8.0
    rects = []
    labels = []
    for region_id, aabb_min, aabb_max, strategy_id in boxes:
        x = pad + (aabb_min[axis_x] - min_x) / span_x * (width - 2 * pad)
        right = pad + (aabb_max[axis_x] - min_x) / span_x * (width - 2 * pad)
        y_top = pad + (max_y - aabb_max[axis_y]) / span_y * (height - 2 * pad)
        y_bottom = pad + (max_y - aabb_min[axis_y]) / span_y * (height - 2 * pad)
        rect_width = max(right - x, 1.0)
        rect_height = max(y_bottom - y_top, 1.0)
        color = _strategy_color(strategy_id or "unassigned")
        label = f"{region_id}: {strategy_id or 'unassigned'}"
        labels.append(label)
        rects.append(
            "<rect "
            f'x="{_format_svg_number(x)}" y="{_format_svg_number(y_top)}" '
            f'width="{_format_svg_number(rect_width)}" '
            f'height="{_format_svg_number(rect_height)}" '
            f'fill="{color}" opacity="0.74">'
            f"<title>{_escape(label)}</title>"
            "</rect>"
        )

    axis_label = f"{'xyz'[axis_x]}/{'xyz'[axis_y]}"
    label_text = "; ".join(labels)
    return "\n".join(
        [
            f'<svg class="region-preview" viewBox="0 0 180 96" role="img" aria-label="{_attr(label_text)}">',
            '  <rect x="0.5" y="0.5" width="179" height="95" rx="8" fill="#f7f9fb" stroke="#d7e0e7" />',
            *[f"  {rect}" for rect in rects],
            f'  <text x="8" y="90" class="preview-axis">{_escape(axis_label)}</text>',
            "</svg>",
        ]
    )


def _artifact_path(value: object, base_dir: Path | None) -> Path | None:
    path_text = str(value or "").strip()
    if not path_text:
        return None
    path = Path(path_text)
    if path.is_absolute() or path.exists():
        return path
    if base_dir is not None:
        candidate = base_dir / path
        if candidate.exists():
            return candidate
    return path


def _float_triplet(value: object) -> tuple[float, float, float] | None:
    if not isinstance(value, Iterable) or isinstance(value, (str, bytes)):
        return None
    values = list(value)
    if len(values) != 3:
        return None
    try:
        return (float(values[0]), float(values[1]), float(values[2]))
    except (TypeError, ValueError):
        return None


def _preview_axes(
    boxes: list[
        tuple[str, tuple[float, float, float], tuple[float, float, float], str]
    ],
) -> tuple[int, int]:
    mins = [min(box[1][axis] for box in boxes) for axis in range(3)]
    maxs = [max(box[2][axis] for box in boxes) for axis in range(3)]
    spans = [(maxs[axis] - mins[axis], axis) for axis in range(3)]
    spans.sort(key=lambda item: (-item[0], item[1]))
    return spans[0][1], spans[1][1]


def _render_strategy_mix(value: object) -> str:
    items = _strategy_mix_items(value)
    if not items:
        return '<span class="muted">No assignments</span>'

    chips = []
    for strategy_id, count in items:
        chips.append(
            '          <span class="mix-chip">'
            f'<span class="swatch" style="background:{_strategy_color(strategy_id)}"></span>'
            f"{_escape(strategy_id)} <strong>{count}</strong>"
            "</span>"
        )
    return "\n".join(
        [
            _render_mix_svg(items),
            '        <div class="mix-labels">',
            *chips,
            "        </div>",
        ]
    )


def _render_mix_svg(items: list[tuple[str, int]]) -> str:
    total = sum(count for _strategy_id, count in items)
    if total <= 0:
        return '<span class="muted">No assignments</span>'

    width = 160.0
    x_pos = 0.0
    rects = []
    for index, (strategy_id, count) in enumerate(items):
        segment_width = (
            width - x_pos
            if index == len(items) - 1
            else width * (float(count) / float(total))
        )
        rects.append(
            f'<rect x="{_format_svg_number(x_pos)}" y="1" '
            f'width="{_format_svg_number(segment_width)}" height="14" '
            f'fill="{_strategy_color(strategy_id)}" />'
        )
        x_pos += segment_width

    label = ", ".join(f"{strategy_id}: {count}" for strategy_id, count in items)
    return "\n".join(
        [
            f'        <svg class="mix-bar" viewBox="0 0 160 16" role="img" aria-label="{_attr(label)}">',
            '          <rect x="0" y="1" width="160" height="14" rx="7" fill="#e7edf2" />',
            *[f"          {rect}" for rect in rects],
            "        </svg>",
        ]
    )


def _render_row_flags(value: object) -> str:
    flags = _string_list(value)
    if not flags:
        return '<span class="muted">None</span>'
    return " ".join(
        f'<span class="flag-chip row-flag">{_escape(flag)}</span>' for flag in flags
    )


def _render_warnings(value: object, count_value: object) -> str:
    warnings = _string_list(value)
    if warnings:
        return "<br>".join(_escape(warning) for warning in warnings)
    warning_count = _int_value(count_value)
    if warning_count > 0:
        return _escape(f"{warning_count} warning(s)")
    return '<span class="muted">None</span>'


def _reuse_text(row: Mapping[str, object]) -> str:
    reuse_count = _int_value(row.get("source_part_reuse_count", 0))
    shared_count = _int_value(row.get("source_part_shared_count", 0))
    return f"{reuse_count}/{shared_count}"


def _artifact_link(value: object, label: str, base_dir: Path | None) -> str:
    path_text = str(value or "").strip()
    if not path_text:
        return f'<span class="muted">{_escape(label)} unavailable</span>'
    href = _href_for_path(path_text, base_dir)
    return (
        f'<a href="{_attr(href)}" title="{_attr(path_text)}">' f"{_escape(label)}</a>"
    )


def _href_for_path(path_text: str, base_dir: Path | None) -> str:
    parsed = urlparse(path_text)
    if parsed.scheme:
        return path_text

    path = Path(path_text)
    if path.is_absolute():
        if base_dir is not None:
            try:
                return _quote_href(os.path.relpath(path, base_dir))
            except ValueError:
                pass
        return path.as_uri()
    return _quote_href(path_text)


def _strategy_mix_items(value: object) -> list[tuple[str, int]]:
    mapping = _mapping_value(value)
    return [
        (str(strategy_id), count)
        for strategy_id, count in _positive_int_mapping(mapping).items()
    ]


def _positive_int_mapping(value: Mapping[object, object]) -> dict[str, int]:
    result: dict[str, int] = {}
    for key, raw_count in value.items():
        count = _int_value(raw_count)
        if count > 0:
            result[str(key)] = count
    return dict(sorted(result.items()))


def _positive_float_mapping(value: Mapping[object, object]) -> dict[str, float]:
    result: dict[str, float] = {}
    for key, raw_value in value.items():
        try:
            amount = float(str(raw_value))
        except (TypeError, ValueError):
            continue
        if amount > 0.0:
            result[str(key)] = amount
    return dict(sorted(result.items()))


def _mapping_value(value: object) -> dict[object, object]:
    value = _literal_value(value)
    if isinstance(value, Mapping):
        return dict(value)
    return {}


def _string_list(value: object) -> list[str]:
    value = _literal_value(value)
    if value is None:
        return []
    if isinstance(value, str):
        stripped = value.strip()
        return [stripped] if stripped else []
    if isinstance(value, Iterable):
        return [str(item) for item in value if str(item)]
    return [str(value)]


def _literal_value(value: object) -> object:
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    if not stripped or stripped[0] not in "[{(":
        return value
    try:
        return ast.literal_eval(stripped)
    except (SyntaxError, ValueError):
        return value


def _status_text(row: Mapping[str, object]) -> str:
    return str(row.get("status") or "unknown").strip().lower() or "unknown"


def _status_badge(status: str) -> str:
    normalized = str(status or "unknown").strip().lower() or "unknown"
    css_class = normalized if normalized in {"ok", "warning", "error"} else "unknown"
    return (
        f'<span class="status status-{css_class}">'
        f"{_escape(normalized.upper())}</span>"
    )


def _strategy_color(strategy_id: str) -> str:
    if strategy_id in STRATEGY_COLORS:
        return STRATEGY_COLORS[strategy_id]

    digest = hashlib.sha256(strategy_id.encode("utf-8")).hexdigest()
    hue = int(digest[:6], 16) % 360
    return f"hsl({hue} 64% 42%)"


def _int_value(value: object) -> int:
    try:
        return int(float(str(value)))
    except (TypeError, ValueError):
        return 0


def _format_plain(value: object) -> str:
    if isinstance(value, float):
        return _format_float(value, digits=3)
    return str(value)


def _format_float(value: object, *, digits: int) -> str:
    try:
        return f"{float(str(value)):.{digits}f}"
    except (TypeError, ValueError):
        return "0." + ("0" * digits)


def _format_volume_mm3(value: object) -> str:
    try:
        amount = float(str(value))
    except (TypeError, ValueError):
        amount = 0.0
    if amount >= 1_000_000_000:
        return f"{amount / 1_000_000_000:.2f}B mm3"
    if amount >= 1_000_000:
        return f"{amount / 1_000_000:.2f}M mm3"
    if amount >= 1_000:
        return f"{amount / 1_000:.2f}k mm3"
    return f"{amount:.1f} mm3"


def _format_percent(value: float) -> str:
    return f"{value:.3f}".rstrip("0").rstrip(".") + "%"


def _format_svg_number(value: float) -> str:
    return f"{value:.3f}".rstrip("0").rstrip(".") or "0"


def _quote_href(value: str) -> str:
    return quote(value.replace(os.sep, "/"), safe="/._-~:")


def _escape(value: object) -> str:
    return html.escape(str(value), quote=False)


def _attr(value: object) -> str:
    return html.escape(str(value), quote=True)


def _style_block() -> str:
    return """  <style>
    :root {
      --ink: #17212b;
      --muted: #607080;
      --line: #d7e0e7;
      --paper: #fbf8f1;
      --card: #ffffff;
      --ok: #157f4f;
      --warning: #9a5b00;
      --error: #b42318;
      --unknown: #56616d;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      color: var(--ink);
      background:
        radial-gradient(circle at 16% 12%, rgba(47, 128, 237, 0.13), transparent 26rem),
        linear-gradient(135deg, #fbf8f1 0%, #edf4f7 100%);
      font-family: "Avenir Next", "Segoe UI", sans-serif;
      line-height: 1.45;
    }
    a { color: #175a96; font-weight: 700; text-decoration: none; }
    a:hover { text-decoration: underline; }
    .report-shell {
      max-width: 1220px;
      margin: 0 auto;
      padding: 32px 20px 56px;
    }
    .hero {
      border: 1px solid rgba(23, 33, 43, 0.14);
      background: rgba(255, 255, 255, 0.72);
      border-radius: 24px;
      padding: 28px;
      box-shadow: 0 24px 70px rgba(23, 33, 43, 0.12);
    }
    .eyebrow {
      margin: 0 0 10px;
      color: var(--muted);
      font-size: 0.78rem;
      font-weight: 800;
      letter-spacing: 0.08em;
      text-transform: uppercase;
    }
    h1 {
      margin: 0;
      font-family: Georgia, "Times New Roman", serif;
      font-size: clamp(2.2rem, 6vw, 4.9rem);
      line-height: 0.95;
      letter-spacing: -0.055em;
    }
    h2 { margin: 0 0 18px; font-size: 1.05rem; letter-spacing: -0.01em; }
    h3 { margin: 0 0 12px; font-size: 0.86rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.06em; }
    .hero-status { margin-top: 22px; }
    .card {
      margin-top: 20px;
      border: 1px solid var(--line);
      background: rgba(255, 255, 255, 0.86);
      border-radius: 20px;
      padding: 22px;
      box-shadow: 0 12px 34px rgba(23, 33, 43, 0.08);
    }
    .wide { overflow: hidden; }
    .summary-table, .detail-table { width: 100%; border-collapse: collapse; }
    .summary-table th {
      width: 260px;
      color: var(--muted);
      font-weight: 750;
      text-align: left;
    }
    .summary-table th, .summary-table td {
      border-top: 1px solid var(--line);
      padding: 10px 0;
      vertical-align: top;
    }
    .summary-table tr:first-child th, .summary-table tr:first-child td {
      border-top: 0;
    }
    .status {
      display: inline-flex;
      align-items: center;
      border-radius: 999px;
      padding: 0.26rem 0.62rem;
      color: #fff;
      font-size: 0.75rem;
      font-weight: 850;
      letter-spacing: 0.045em;
    }
    .status-ok { background: var(--ok); }
    .status-warning { background: var(--warning); }
    .status-error { background: var(--error); }
    .status-unknown { background: var(--unknown); }
    .legend, .flag-cloud, .mix-labels {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      align-items: center;
    }
    .legend { margin-bottom: 18px; }
    .legend-chip, .mix-chip, .flag-chip {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      border: 1px solid var(--line);
      border-radius: 999px;
      background: #fff;
      padding: 0.24rem 0.55rem;
      font-size: 0.78rem;
      white-space: nowrap;
    }
    .flag-chip { background: #fff5df; border-color: #ead199; color: #704100; }
    .row-flag { margin: 2px 3px 2px 0; }
    .swatch {
      display: inline-block;
      width: 0.72rem;
      height: 0.72rem;
      border-radius: 999px;
      box-shadow: inset 0 0 0 1px rgba(0, 0, 0, 0.16);
      flex: 0 0 auto;
    }
    .strategy-usage {
      display: grid;
      gap: 10px;
    }
    .section-note {
      margin: -8px 0 18px;
      color: var(--muted);
      font-size: 0.9rem;
    }
    .burden-grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 20px;
    }
    .burden-panel {
      border: 1px solid var(--line);
      border-radius: 16px;
      background: rgba(251, 248, 241, 0.62);
      padding: 16px;
    }
    .usage-row {
      display: grid;
      grid-template-columns: auto minmax(120px, 190px) minmax(140px, 1fr) auto;
      gap: 10px;
      align-items: center;
    }
    .usage-track {
      display: block;
      height: 13px;
      border-radius: 999px;
      overflow: hidden;
      background: #e7edf2;
    }
    .usage-fill {
      display: block;
      height: 100%;
      border-radius: inherit;
    }
    .usage-count { color: var(--muted); font-variant-numeric: tabular-nums; }
    .empty, .muted { color: var(--muted); }
    .table-wrap {
      overflow-x: auto;
      margin: 0 -22px -22px;
      padding: 0 22px 22px;
    }
    .detail-table {
      min-width: 1080px;
      font-size: 0.88rem;
    }
    .detail-table th, .detail-table td {
      border-top: 1px solid var(--line);
      padding: 12px 10px;
      text-align: left;
      vertical-align: top;
    }
    .detail-table thead th {
      border-top: 0;
      color: var(--muted);
      font-size: 0.72rem;
      letter-spacing: 0.05em;
      text-transform: uppercase;
    }
    .mix-bar {
      display: block;
      width: 160px;
      height: 16px;
      margin-bottom: 7px;
      border-radius: 999px;
      overflow: hidden;
    }
    .region-preview {
      display: block;
      width: 180px;
      height: 96px;
    }
    .preview-axis {
      fill: var(--muted);
      font-size: 9px;
      font-weight: 700;
      letter-spacing: 0.05em;
      text-transform: uppercase;
    }
    .artifact-links {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
    }
    @media (max-width: 720px) {
      .report-shell { padding: 18px 12px 42px; }
      .hero, .card { border-radius: 16px; padding: 18px; }
      .summary-table th, .summary-table td {
        display: block;
        width: 100%;
      }
      .summary-table td { border-top: 0; padding-top: 0; }
      .usage-row {
        grid-template-columns: auto 1fr auto;
      }
      .usage-track {
        grid-column: 1 / -1;
      }
      .burden-grid {
        grid-template-columns: 1fr;
      }
    }
  </style>"""
