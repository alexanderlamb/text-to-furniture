"""Tests for mesh cleanup preprocessing."""

from pathlib import Path

import numpy as np
import trimesh

from mesh_cleanup import MeshCleanupConfig, clean_mesh_geometry, simplify_mesh_geometry
from mesh_decomposer import DecompositionConfig, decompose


def _make_noisy_plane(
    nx: int = 25,
    ny: int = 25,
    spacing_mm: float = 5.0,
    noise_mm: float = 0.6,
    tilt_deg: float = 0.0,
    z_offset_mm: float = 0.0,
) -> trimesh.Trimesh:
    """Build a triangulated noisy plane mesh."""
    xs = np.arange(nx, dtype=float) * spacing_mm
    ys = np.arange(ny, dtype=float) * spacing_mm
    grid_x, grid_y = np.meshgrid(xs, ys, indexing="xy")
    rng = np.random.default_rng(42)
    z = rng.normal(0.0, noise_mm, size=grid_x.shape)

    vertices = np.column_stack(
        [grid_x.reshape(-1), grid_y.reshape(-1), z.reshape(-1)]
    )
    faces = []
    for j in range(ny - 1):
        for i in range(nx - 1):
            v00 = j * nx + i
            v10 = v00 + 1
            v01 = v00 + nx
            v11 = v01 + 1
            faces.append([v00, v10, v11])
            faces.append([v00, v11, v01])
    mesh = trimesh.Trimesh(vertices=vertices, faces=np.array(faces), process=False)

    if abs(tilt_deg) > 1e-9:
        angle = np.radians(tilt_deg)
        rot = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, np.cos(angle), -np.sin(angle)],
                [0.0, np.sin(angle), np.cos(angle)],
            ]
        )
        mesh.vertices = mesh.vertices @ rot.T

    if abs(z_offset_mm) > 1e-9:
        mesh.vertices[:, 2] += z_offset_mm

    return mesh


def _fit_plane_rmse(points: np.ndarray) -> float:
    centroid = np.mean(points, axis=0)
    centred = points - centroid
    _, _, vh = np.linalg.svd(centred, full_matrices=False)
    normal = vh[-1]
    normal /= np.linalg.norm(normal)
    d = float(np.dot(normal, centroid))
    distances = np.abs(points @ normal - d)
    return float(np.sqrt(np.mean(distances**2)))


def _abs_angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    a_n = a / np.linalg.norm(a)
    b_n = b / np.linalg.norm(b)
    dot = abs(float(np.dot(a_n, b_n)))
    dot = float(np.clip(dot, -1.0, 1.0))
    return float(np.degrees(np.arccos(dot)))


class TestMeshCleanup:
    """Test flattening, parallel alignment, and simplification behavior."""

    def test_flattens_near_planar_region(self):
        mesh = _make_noisy_plane(nx=30, ny=30, spacing_mm=4.0, noise_mm=0.35)
        before_rmse = _fit_plane_rmse(mesh.vertices)

        cleaned = clean_mesh_geometry(
            mesh,
            MeshCleanupConfig(
                enabled=True,
                planar_angle_threshold_deg=20.0,
                planar_distance_threshold_mm=2.5,
                parallel_angle_threshold_deg=10.0,
                boundary_simplify_tolerance_mm=0.0,
                min_region_area_mm2=200.0,
                max_iterations=2,
            ),
        )
        after_rmse = _fit_plane_rmse(cleaned.vertices)

        assert after_rmse < before_rmse
        assert after_rmse < before_rmse * 0.7

    def test_aligns_near_parallel_faces(self):
        mesh_a = _make_noisy_plane(
            nx=20, ny=20, spacing_mm=6.0, noise_mm=0.3, tilt_deg=1.5, z_offset_mm=0.0
        )
        mesh_b = _make_noisy_plane(
            nx=20, ny=20, spacing_mm=6.0, noise_mm=0.3, tilt_deg=5.0, z_offset_mm=80.0
        )
        combined = trimesh.util.concatenate([mesh_a, mesh_b])

        before_parts = combined.split(only_watertight=False)
        assert len(before_parts) == 2
        before_angle = _abs_angle_deg(
            trimesh.points.plane_fit(before_parts[0].vertices)[1],
            trimesh.points.plane_fit(before_parts[1].vertices)[1],
        )

        cleaned = clean_mesh_geometry(
            combined,
            MeshCleanupConfig(
                enabled=True,
                planar_angle_threshold_deg=15.0,
                planar_distance_threshold_mm=2.0,
                parallel_angle_threshold_deg=8.0,
                boundary_simplify_tolerance_mm=0.0,
                min_region_area_mm2=200.0,
                max_iterations=2,
            ),
        )
        after_parts = cleaned.split(only_watertight=False)
        assert len(after_parts) == 2
        after_angle = _abs_angle_deg(
            trimesh.points.plane_fit(after_parts[0].vertices)[1],
            trimesh.points.plane_fit(after_parts[1].vertices)[1],
        )

        assert after_angle < before_angle
        assert after_angle < 0.5

    def test_simplifies_dense_planar_mesh(self):
        mesh = _make_noisy_plane(nx=60, ny=60, spacing_mm=0.5, noise_mm=0.02)
        before_faces = len(mesh.faces)

        cleaned = clean_mesh_geometry(
            mesh,
            MeshCleanupConfig(
                enabled=True,
                planar_angle_threshold_deg=10.0,
                planar_distance_threshold_mm=0.5,
                parallel_angle_threshold_deg=8.0,
                boundary_simplify_tolerance_mm=2.0,
                min_region_area_mm2=100.0,
                max_iterations=2,
            ),
        )
        after_faces = len(cleaned.faces)

        assert after_faces < before_faces
        assert after_faces > 0

    def test_disabled_cleanup_leaves_mesh_unchanged(self):
        mesh = _make_noisy_plane(nx=15, ny=15, spacing_mm=5.0, noise_mm=0.5)
        before_vertices = mesh.vertices.copy()
        before_faces = mesh.faces.copy()

        cleaned = clean_mesh_geometry(mesh, MeshCleanupConfig(enabled=False))

        assert np.array_equal(cleaned.faces, before_faces)
        assert np.allclose(cleaned.vertices, before_vertices)

    def test_simplify_mesh_geometry_reduces_faces(self):
        mesh = _make_noisy_plane(nx=80, ny=80, spacing_mm=1.0, noise_mm=0.15)
        before_faces = len(mesh.faces)

        simplified, stats = simplify_mesh_geometry(
            mesh,
            MeshCleanupConfig(
                simplify_enabled=True,
                simplify_target_reduction=0.5,
                simplify_min_faces=1000,
                simplify_max_normal_change_deg=90.0,
                simplify_max_bbox_drift_ratio=1.0,
            ),
        )

        assert stats["enabled"] is True
        assert stats["before_faces"] == before_faces
        assert len(simplified.faces) < before_faces
        assert stats["after_faces"] == len(simplified.faces)

    def test_simplify_mesh_geometry_skips_small_mesh(self, box_mesh):
        simplified, stats = simplify_mesh_geometry(
            box_mesh,
            MeshCleanupConfig(
                simplify_enabled=True,
                simplify_target_reduction=0.5,
                simplify_min_faces=3000,
            ),
        )

        assert stats["mode"] == "skip_small"
        assert len(simplified.faces) == len(box_mesh.faces)

    def test_simplify_mesh_geometry_reverts_when_guard_exceeded(self):
        mesh = _make_noisy_plane(nx=70, ny=70, spacing_mm=1.0, noise_mm=0.25)
        before_faces = len(mesh.faces)

        simplified, stats = simplify_mesh_geometry(
            mesh,
            MeshCleanupConfig(
                simplify_enabled=True,
                simplify_target_reduction=0.8,
                simplify_min_faces=1000,
                simplify_max_normal_change_deg=0.0,
                simplify_max_bbox_drift_ratio=1.0,
            ),
        )

        assert stats["guard_reverted"] is True
        assert len(simplified.faces) == before_faces


class TestCleanupIntegration:
    """Integration coverage for both decomposition paths."""

    def test_voxel_decompose_with_cleanup(self, box_mesh, tmp_path: Path):
        mesh_path = tmp_path / "box.stl"
        box_mesh.export(mesh_path)

        design = decompose(
            str(mesh_path),
            DecompositionConfig(
                max_slabs=8,
                coverage_target=0.6,
                optimize_iterations=0,
                mesh_cleanup=MeshCleanupConfig(
                    enabled=True,
                    boundary_simplify_tolerance_mm=0.5,
                    min_region_area_mm2=100.0,
                ),
            ),
            optimize=False,
        )

        assert len(design.components) > 0
        is_valid, errors = design.validate()
        assert is_valid, f"Design should validate after cleanup, got {errors}"

    def test_decompose_debug_contains_simplify_fields(self, box_mesh, tmp_path: Path):
        mesh_path = tmp_path / "box_debug.stl"
        box_mesh.export(mesh_path)

        design = decompose(
            str(mesh_path),
            DecompositionConfig(
                max_slabs=4,
                coverage_target=0.5,
                optimize_iterations=0,
                mesh_cleanup=MeshCleanupConfig(
                    enabled=False,
                    simplify_enabled=True,
                    simplify_target_reduction=0.5,
                    simplify_min_faces=1,
                    simplify_max_normal_change_deg=90.0,
                    simplify_max_bbox_drift_ratio=1.0,
                ),
            ),
            optimize=False,
        )

        debug = design.metadata.get("decomposition_debug", {})
        assert "simplify_enabled" in debug
        assert "simplify_mode" in debug
        assert "simplify_before_faces" in debug
        assert "simplify_after_faces" in debug
        assert "simplify_guard_reverted" in debug
