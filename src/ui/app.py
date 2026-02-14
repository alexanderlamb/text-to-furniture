"""
Main NiceGUI application — full-screen 3D viewer with floating overlays.

Layout:
- Full-viewport 3D scene as the main canvas
- Top-left overlay: mesh library (last 5, search, upload/generate)
- Right overlay: decomposition params A/B as tabs, results, run buttons
- Settings drawer: provider + API keys (right side, toggled from header)

Run with:
    python scripts/run_ui.py
"""

import json
import os
import logging
from pathlib import Path

from nicegui import app, run as nicegui_run, ui

from ui.state import AppState
from ui.scene_helpers import (
    render_mesh, render_slabs, render_manufacturing_parts,
    normalize_mesh_for_viewer,
)
from ui.components import (
    build_parameter_panel,
    build_component_table,
    build_svg_preview,
    build_manufacturing_results,
    build_progress_log,
    build_debug_report,
)
from ui.workers import run_decomposition, run_mesh_generation, create_progress_file

logger = logging.getLogger(__name__)

# Directories
BASE_DIR = Path(__file__).resolve().parent.parent.parent
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "output"
SETTINGS_PATH = BASE_DIR / ".ui_settings.json"

MESH_EXTENSIONS = {".glb", ".stl", ".obj"}

# CSS for floating overlay panels
OVERLAY_CSS = (
    "position:fixed; z-index:100; backdrop-filter:blur(12px); "
    "background:rgba(255,255,255,0.92); border-radius:12px; "
    "box-shadow:0 4px 24px rgba(0,0,0,0.15); overflow-y:auto;"
)
MESH_OVERLAY_STYLE = OVERLAY_CSS + " top:72px; left:16px; width:300px; max-height:calc(100vh - 88px);"
PARAMS_OVERLAY_STYLE = OVERLAY_CSS + " top:72px; right:16px; width:340px; max-height:calc(100vh - 88px);"


# ═══════════════════════════════════════════════════════════════════════════
# Settings persistence
# ═══════════════════════════════════════════════════════════════════════════

def _load_settings() -> dict:
    defaults = {"provider": "tripo", "tripo_api_key": "", "meshy_api_key": ""}
    if SETTINGS_PATH.is_file():
        try:
            with open(SETTINGS_PATH) as f:
                saved = json.load(f)
            defaults.update(saved)
        except (json.JSONDecodeError, OSError):
            pass
    return defaults


def _save_settings(settings: dict) -> None:
    with open(SETTINGS_PATH, "w") as f:
        json.dump(settings, f, indent=2)


# ═══════════════════════════════════════════════════════════════════════════
# Mesh library helpers
# ═══════════════════════════════════════════════════════════════════════════

def _scan_mesh_library(query: str = "") -> list[dict]:
    """Scan uploads/ for mesh files, filtered by query, newest-first, max 5."""
    meshes = []
    if not UPLOAD_DIR.is_dir():
        return meshes
    for p in UPLOAD_DIR.iterdir():
        if p.suffix.lower() in MESH_EXTENSIONS and p.is_file():
            if p.stem.endswith("_normalized"):
                continue
            if query and query.lower() not in p.name.lower():
                continue
            meshes.append({
                "name": p.name,
                "path": str(p),
                "size": f"{p.stat().st_size / 1024:.0f} KB",
                "mtime": p.stat().st_mtime,
            })
    meshes.sort(key=lambda m: m["mtime"], reverse=True)
    return meshes[:5]


# ═══════════════════════════════════════════════════════════════════════════
# Mesh loading helper
# ═══════════════════════════════════════════════════════════════════════════

def _set_mesh(state: AppState, mesh_path: str, mesh_name: str) -> None:
    """Normalize a mesh for the 3D viewer and update state."""
    state.mesh_path = mesh_path
    state.mesh_filename = mesh_name
    try:
        norm_path = normalize_mesh_for_viewer(mesh_path, str(UPLOAD_DIR))
        norm_name = os.path.basename(norm_path)
        state.mesh_serving_url = f"/meshes/{norm_name}"
    except Exception as exc:
        logger.warning("Mesh normalization failed, serving original: %s", exc)
        state.mesh_serving_url = f"/meshes/{mesh_name}"


# ═══════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    UPLOAD_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)
    _patch_nicegui_run_setup()

    app.add_static_files("/meshes", str(UPLOAD_DIR))
    app.add_static_files("/output", str(OUTPUT_DIR))

    @ui.page("/")
    def index():
        _build_page()

    ui.run(title="Text-to-Furniture", port=8080, reload=False)


def _patch_nicegui_run_setup() -> None:
    """Patch NiceGUI startup so missing process semaphores don't crash UI."""
    if getattr(nicegui_run.setup, "_ttf_safe_patch", False):
        return

    original_setup = nicegui_run.setup

    def _safe_setup() -> None:
        try:
            original_setup()
        except (NotImplementedError, PermissionError, OSError) as exc:
            logger.warning(
                "Process pool unavailable; continuing without cpu_bound workers: %s",
                exc,
            )
            nicegui_run.process_pool = None

    _safe_setup._ttf_safe_patch = True  # type: ignore[attr-defined]
    nicegui_run.setup = _safe_setup


# ═══════════════════════════════════════════════════════════════════════════
# Page builder
# ═══════════════════════════════════════════════════════════════════════════

def _build_page() -> None:
    state = AppState()
    settings = _load_settings()

    # Auto-load most recent mesh on startup
    existing = _scan_mesh_library()
    if existing:
        newest = existing[0]
        _set_mesh(state, newest["path"], newest["name"])

    # Shared mutable for search query
    search = {"query": ""}

    # ── Thin header bar ─────────────────────────────────────────────────
    with ui.header().classes(
        "bg-blue-800 text-white items-center h-12 px-4"
    ).style("min-height:48px"):
        ui.label("Text-to-Furniture").classes("text-lg font-bold")
        ui.space()
        ui.button(
            icon="settings", on_click=lambda: settings_drawer.toggle()
        ).props("flat round text-color=white size=sm")

    # ── Settings drawer ─────────────────────────────────────────────────
    with ui.right_drawer(value=False).classes(
        "bg-gray-50 p-4"
    ).style("z-index:200") as settings_drawer:
        ui.label("Settings").classes("text-xl font-bold mb-4")
        ui.separator()

        ui.label("Default Provider").classes("text-sm font-medium mt-2")
        ui.select(
            {"tripo": "Tripo3D", "meshy": "Meshy"},
            value=settings["provider"],
            label="Provider",
            on_change=lambda e: settings.update(provider=e.value),
        ).classes("w-full")

        ui.separator().classes("my-3")
        ui.label("API Keys").classes("text-sm font-medium")

        ui.input(
            label="Tripo3D API Key",
            password=True,
            password_toggle_button=True,
            value=settings.get("tripo_api_key", ""),
            on_change=lambda e: settings.update(tripo_api_key=e.value),
        ).classes("w-full")

        ui.input(
            label="Meshy API Key",
            password=True,
            password_toggle_button=True,
            value=settings.get("meshy_api_key", ""),
            on_change=lambda e: settings.update(meshy_api_key=e.value),
        ).classes("w-full")

        ui.label(
            "Saved to .ui_settings.json (gitignored)."
        ).classes("text-xs text-gray-400 mt-1")

        ui.button(
            "Save Settings",
            on_click=lambda: _handle_save_settings(settings),
            icon="save",
        ).classes("w-full mt-4").props("color=primary")

    # ── Full-screen 3D scene ────────────────────────────────────────────
    with ui.element("div").classes("w-full").style(
        "position:fixed; top:48px; left:0; right:0; bottom:0;"
    ) as scene_container:
        _build_main_scene(state)

    # ── Mesh library overlay (top-left) ─────────────────────────────────
    with ui.element("div").style(MESH_OVERLAY_STYLE).classes("p-3"):
        ui.label("Meshes").classes("text-sm font-bold mb-1")

        # Search bar — lambda captures library_list_container by name (late-binding)
        ui.input(
            placeholder="Search meshes…",
            on_change=lambda e: _handle_search(
                e.value, search, state, scene_container,
                library_list_container,
            ),
        ).props('dense outlined clearable').classes("w-full mb-2")

        # Mesh list container (created inside the overlay, after search bar)
        library_list_container = ui.element("div").classes("w-full")
        with library_list_container:
            _build_mesh_list(state, scene_container, library_list_container, search)

        ui.separator().classes("my-2")

        # Upload
        ui.upload(
            label="Upload .glb / .stl / .obj",
            auto_upload=True,
            on_upload=lambda e: _handle_upload(
                e, state, scene_container, library_list_container, search,
            ),
        ).props(
            'accept=".glb,.stl,.obj,.GLB,.STL,.OBJ" dense'
        ).classes("w-full")

        ui.separator().classes("my-2")

        # Generate
        prompt_input = ui.input(
            placeholder="Describe furniture…",
        ).props("dense outlined").classes("w-full")
        gen_progress_container = ui.element("div").classes("w-full")
        with ui.row().classes("w-full items-center gap-1 mt-1"):
            gen_spinner = ui.spinner(size="xs", color="primary")
            gen_spinner.visible = False
            ui.button(
                "Generate",
                on_click=lambda: _handle_generate(
                    state, settings, prompt_input.value,
                    scene_container, library_list_container,
                    search, gen_spinner, gen_progress_container,
                ),
                icon="auto_awesome",
            ).props("dense size=sm color=primary")

    # ── Params / Results overlay (right) ────────────────────────────────
    with ui.element("div").style(PARAMS_OVERLAY_STYLE).classes("p-3"):
        params_container = ui.element("div")
        with params_container:
            _build_params_overlay(state, scene_container, params_container)


# ═══════════════════════════════════════════════════════════════════════════
# Sub-builders
# ═══════════════════════════════════════════════════════════════════════════

def _build_main_scene(state: AppState) -> None:
    """Full-viewport 3D scene — original mesh + slab overlays."""
    if state.mesh_serving_url:
        ext = "glb"  # normalized meshes are always GLB
        with ui.scene().classes("w-full h-full") as scene:
            render_mesh(scene, state.mesh_serving_url, ext)

            # Render slabs/parts from any completed runs
            # Resolve the normalized mesh file path for manufacturing rendering
            viewer_mesh_file = ""
            if state.mesh_serving_url:
                norm_name = state.mesh_serving_url.split("/")[-1]
                viewer_mesh_file = str(UPLOAD_DIR / norm_name)

            for slot in state.slots:
                if slot.mfg_summary and slot.mfg_summary.get("components"):
                    render_manufacturing_parts(
                        scene, slot.mfg_summary, viewer_mesh_file,
                    )
                elif slot.result and slot.result.design.components:
                    render_slabs(scene, slot.result.design)
    else:
        with ui.scene().classes("w-full h-full"):
            pass  # Empty scene — dark background as placeholder


def _build_mesh_list(
    state: AppState,
    scene_container,
    library_list_container,
    search: dict,
) -> None:
    """Render the mesh file list (max 5, filtered by search)."""
    meshes = _scan_mesh_library(search.get("query", ""))

    if not meshes:
        ui.label("No meshes found.").classes("text-gray-400 italic text-xs")
        return

    for m in meshes:
        is_active = state.mesh_path == m["path"]
        with ui.row().classes(
            "w-full items-center gap-2 p-1.5 rounded cursor-pointer "
            + ("bg-blue-50 border border-blue-200" if is_active else "hover:bg-gray-100")
        ):
            ui.icon("view_in_ar", size="xs").classes(
                "text-blue-600" if is_active else "text-gray-400"
            )
            with ui.column().classes("flex-grow gap-0 min-w-0"):
                ui.label(m["name"]).classes(
                    "text-xs font-medium truncate"
                    + (" text-blue-700" if is_active else "")
                )
                ui.label(m["size"]).classes("text-[10px] text-gray-400")
            if is_active:
                ui.badge("active", color="blue").props("outline dense")
            else:
                ui.button(
                    "Load",
                    on_click=lambda mp=m["path"], mn=m["name"]: _handle_load_mesh(
                        mp, mn, state, scene_container,
                        library_list_container, search,
                    ),
                ).props("flat dense size=xs")


def _build_params_overlay(
    state: AppState,
    scene_container,
    params_container,
) -> None:
    """Decomposition parameters A/B as tabs, with run buttons and results."""
    with ui.tabs().classes("w-full") as tabs:
        tab_a = ui.tab("Run A")
        tab_b = ui.tab("Run B")

    with ui.tab_panels(tabs, value=tab_a).classes("w-full"):
        for idx, tab in enumerate([tab_a, tab_b]):
            label = "A" if idx == 0 else "B"
            other_label = "B" if idx == 0 else "A"
            slot = state.slots[idx]
            other_slot = state.slots[1 - idx]

            with ui.tab_panel(tab):
                def make_refresh(sc=scene_container, pc=params_container, st=state):
                    def refresh():
                        try:
                            sc.clear()
                            with sc:
                                _build_main_scene(st)
                            pc.clear()
                            with pc:
                                _build_params_overlay(st, sc, pc)
                        except Exception:
                            pass  # Client disconnected during run
                    return refresh

                refresh = make_refresh()

                build_parameter_panel(
                    slot, other_slot, label, other_label,
                    on_change=lambda: None,
                )

                # Run button
                can_run = state.mesh_path is not None and not slot.running
                with ui.row().classes("w-full justify-center my-2"):
                    ui.button(
                        f"Run ({label})",
                        on_click=lambda si=idx, r=refresh: _handle_run(
                            state, si, r
                        ),
                        icon="play_arrow",
                    ).props(
                        "color=primary" + ("" if can_run else " disabled")
                    ).classes("w-full")

                    if slot.running:
                        ui.spinner(size="sm")

                # Live progress log while running
                if slot.running and slot.progress_file:
                    build_progress_log(slot)

                # Error display
                if slot.error and not slot.running:
                    ui.separator().classes("my-1")
                    with ui.column().classes("w-full"):
                        ui.label("Pipeline Failed").classes(
                            "text-sm font-bold text-red-600"
                        )
                        with ui.expansion(
                            "Error Details", icon="error"
                        ).classes("w-full"):
                            ui.label(slot.error).classes(
                                "text-xs text-red-500 whitespace-pre-wrap"
                            ).style("font-family: monospace; word-break: break-all;")
                        # Show progress log with what happened before the error
                        if slot.progress_file:
                            with ui.expansion(
                                "Pipeline Log", icon="receipt_long"
                            ).classes("w-full"):
                                build_progress_log(slot)

                # Manufacturing results
                if slot.mfg_summary:
                    build_manufacturing_results(slot.mfg_summary)
                    if slot.result:
                        with ui.expansion(
                            "Components", icon="table_chart"
                        ).classes("w-full"):
                            build_component_table(slot)
                        build_svg_preview(slot)
                    build_debug_report(slot)


# ═══════════════════════════════════════════════════════════════════════════
# Event handlers
# ═══════════════════════════════════════════════════════════════════════════

def _handle_save_settings(settings: dict) -> None:
    _save_settings(settings)
    ui.notify("Settings saved.", type="positive")


def _handle_search(
    query: str,
    search: dict,
    state: AppState,
    scene_container,
    library_list_container,
) -> None:
    search["query"] = query or ""
    library_list_container.clear()
    with library_list_container:
        _build_mesh_list(state, scene_container, library_list_container, search)


def _handle_load_mesh(
    mesh_path: str,
    mesh_name: str,
    state: AppState,
    scene_container,
    library_list_container,
    search: dict,
) -> None:
    _set_mesh(state, mesh_path, mesh_name)

    scene_container.clear()
    with scene_container:
        _build_main_scene(state)

    library_list_container.clear()
    with library_list_container:
        _build_mesh_list(state, scene_container, library_list_container, search)

    ui.notify(f"Loaded {mesh_name}", type="positive")


async def _handle_upload(
    event,
    state: AppState,
    scene_container,
    library_list_container,
    search: dict,
) -> None:
    name = event.name
    content = event.content.read()

    dest = UPLOAD_DIR / name
    with open(dest, "wb") as f:
        f.write(content)

    ext = name.rsplit(".", 1)[-1].lower()

    if ext == "obj":
        import trimesh
        mesh = trimesh.load(str(dest))
        glb_path = dest.with_suffix(".glb")
        mesh.export(str(glb_path), file_type="glb")
        name = glb_path.name
        dest = glb_path

    _set_mesh(state, str(dest), name)

    scene_container.clear()
    with scene_container:
        _build_main_scene(state)

    library_list_container.clear()
    with library_list_container:
        _build_mesh_list(state, scene_container, library_list_container, search)

    ui.notify(f"Uploaded {name}", type="positive")


async def _handle_generate(
    state: AppState,
    settings: dict,
    prompt: str,
    scene_container,
    library_list_container,
    search: dict,
    spinner,
    gen_progress_container=None,
) -> None:
    if not prompt.strip():
        ui.notify("Enter a text prompt.", type="warning")
        return

    provider_name = settings.get("provider", "tripo")
    if provider_name == "tripo":
        resolved_key = (settings.get("tripo_api_key") or "").strip()
        if not resolved_key:
            resolved_key = os.environ.get("TRIPO_API_KEY", "")
    else:
        resolved_key = (settings.get("meshy_api_key") or "").strip()
        if not resolved_key:
            resolved_key = os.environ.get("MESHY_API_KEY", "")

    if not resolved_key:
        ui.notify("API key required — open Settings.", type="warning")
        return

    spinner.visible = True

    # Create progress file for generation tracking
    gen_progress_file = create_progress_file()

    # Show live progress in the overlay
    gen_timer = None
    if gen_progress_container is not None:
        gen_progress_container.clear()
        with gen_progress_container:
            # Create a temporary RunSlot-like object for progress display
            import time as _time
            from ui.workers import read_progress as _read_progress

            start_t = _time.time()
            step_lbl = ui.label("Connecting to API...").classes(
                "text-xs text-blue-600"
            )
            elapsed_lbl = ui.label("").classes("text-[10px] text-gray-400")

            def _poll_gen():
                entries = _read_progress(gen_progress_file)
                elapsed = _time.time() - start_t
                elapsed_lbl.text = f"Elapsed: {elapsed:.0f}s"
                if entries:
                    latest = entries[-1]
                    lvl = latest.get("level", "info")
                    txt = latest.get("detail", "")
                    if lvl == "error":
                        step_lbl.text = f"Error: {txt[:100]}"
                        step_lbl.classes(
                            replace="text-xs text-red-600"
                        )
                    else:
                        step_lbl.text = txt[:100] if txt else "Working..."
                        step_lbl.classes(
                            replace="text-xs text-blue-600"
                        )

            gen_timer = ui.timer(0.5, _poll_gen)

    def after_generate():
        spinner.visible = False
        if gen_timer:
            gen_timer.deactivate()
        if gen_progress_container is not None:
            gen_progress_container.clear()
        if state.mesh_path:
            _set_mesh(state, state.mesh_path, state.mesh_filename)

        scene_container.clear()
        with scene_container:
            _build_main_scene(state)

        library_list_container.clear()
        with library_list_container:
            _build_mesh_list(
                state, scene_container, library_list_container, search
            )

    await run_mesh_generation(
        state,
        provider_name,
        resolved_key,
        prompt.strip(),
        str(UPLOAD_DIR),
        after_generate,
        gen_progress_file,
    )


async def _handle_run(state: AppState, slot_idx: int, refresh_fn) -> None:
    if state.mesh_path is None:
        ui.notify("Load a mesh first.", type="warning")
        return

    slot = state.slots[slot_idx]
    if slot.running:
        ui.notify("Already running.", type="info")
        return

    await run_decomposition(
        state,
        slot_idx,
        output_dir=str(OUTPUT_DIR),
        notify=refresh_fn,
    )
