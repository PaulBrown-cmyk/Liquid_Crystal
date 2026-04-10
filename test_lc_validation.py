"""Regression checks for the liquid-crystal FEM validation baselines."""

from __future__ import annotations

import unittest
import tempfile
from pathlib import Path

import numpy as np

from grid1 import CylinderGrid
from lc_fem import LiquidCrystalFEMSolver


REPO_ROOT = Path(__file__).resolve().parent
GRID_FILE = REPO_ROOT / "straight_cylinder_grid_with_grid.txt"


class LiquidCrystalValidationTests(unittest.TestCase):
    """Keep the validated LC baselines from drifting silently."""

    @classmethod
    def setUpClass(cls) -> None:
        if not GRID_FILE.exists():
            raise unittest.SkipTest(f"Missing reference grid file: {GRID_FILE}")

        cls._tmpdir = tempfile.TemporaryDirectory()
        mesh_path = Path(cls._tmpdir.name) / "structured_solve_mesh.npz"
        cylinder = CylinderGrid(
            diameter_um=5,
            length_um=20,
            num_boundary_points_per_z=20,
            num_z_levels=20,
            num_inner_points=20,
            min_distance_um=1.0,
        )
        cylinder.save_structured_solve_mesh(
            str(mesh_path),
            num_radial_layers=5,
            num_theta_points=24,
            num_axial_layers=12,
            radial_cluster_power=2.0,
            axial_cluster_power=2.0,
        )
        cls.mesh_file = str(mesh_path)

        # Build both material variants once so the benchmark methods can reuse
        # the same mesh without paying the load/build cost repeatedly.
        cls.one_constant_solver = LiquidCrystalFEMSolver(
            coordinates_file=str(GRID_FILE),
            mesh_file=cls.mesh_file,
            material_preset="5CB_room_temperature",
            anchoring_preset="planar_side_homeotropic_caps",
            solver_method="trust-krylov",
            random_seed=0,
        )
        cls.anisotropic_solver = LiquidCrystalFEMSolver(
            coordinates_file=str(GRID_FILE),
            mesh_file=cls.mesh_file,
            material_preset="5CB_anisotropic_room_temperature",
            anchoring_preset="planar_side_homeotropic_caps",
            solver_method="trust-krylov",
            random_seed=0,
        )

    @classmethod
    def tearDownClass(cls) -> None:
        if hasattr(cls, "_tmpdir"):
            cls._tmpdir.cleanup()

    def _result_map(self, results):
        return {result.name: result for result in results}

    def test_reference_cases_are_well_behaved(self) -> None:
        """Uniform and incompatible boundary cases should match expectations."""
        results = self._result_map(self.one_constant_solver.validate_reference_cases())

        self.assertLess(abs(results["free_uniform_z"].energy_density), 1.0e-12)
        self.assertLess(abs(results["planar_side_homeotropic_caps_uniform_z"].energy_density), 1.0e-12)
        self.assertGreater(results["homeotropic_all_uniform_z"].energy_density, 1.0)

    def test_structured_mesh_tags_are_present(self) -> None:
        """The solve mesh should distinguish sidewall and cap faces explicitly."""
        kinds = {face.kind for face in self.one_constant_solver.boundary_faces}
        self.assertIn("sidewall", kinds)
        self.assertIn("top_cap", kinds)
        self.assertIn("bottom_cap", kinds)

    def test_refinement_suggestion_exists(self) -> None:
        """The structured solve mesh should yield a follow-up refinement prescription."""
        suggestion = self.one_constant_solver.suggest_refined_mesh_parameters()
        self.assertIsNotNone(suggestion)
        self.assertGreaterEqual(suggestion.num_radial_layers, 3)
        self.assertGreaterEqual(suggestion.num_axial_layers, 3)

    def test_wall_dominated_refinement_prefers_radial_resolution(self) -> None:
        """A hot wall should bias the refinement suggestion toward radial and theta resolution."""
        radii = np.linalg.norm(self.one_constant_solver.vertices[:, :2], axis=1)
        radii_norm = radii / max(float(np.max(radii)), 1.0e-30)
        temperature_field = 298.15 + 8.0 * np.square(radii_norm)
        uniform_z = np.tile(np.array([0.0, 0.0, 1.0]), (self.one_constant_solver.n_nodes, 1))

        suggestion = self.one_constant_solver.suggest_refined_mesh_parameters(
            directors=uniform_z,
            temperature_field=temperature_field,
        )

        self.assertIsNotNone(suggestion)
        self.assertGreater(suggestion.num_radial_layers, 5)
        self.assertGreater(suggestion.num_theta_points, 24)
        self.assertGreater(suggestion.wall_hot_fraction, suggestion.cap_hot_fraction)

    def test_uniform_directors_remain_near_zero(self) -> None:
        """Uniform director fields should remain essentially energy-free."""
        results = self._result_map(self.one_constant_solver.benchmark_uniform_directors())
        for name in ("uniform_x", "uniform_y", "uniform_z"):
            self.assertLess(abs(results[name].energy_density), 1.0e-12, msg=name)

        anisotropic_results = self._result_map(self.anisotropic_solver.benchmark_uniform_directors())
        for name in ("uniform_x", "uniform_y", "uniform_z"):
            self.assertLess(abs(anisotropic_results[name].energy_density), 1.0e-12, msg=name)

    def test_helical_twist_matches_analytic_scale(self) -> None:
        """The twist benchmark should stay close to the analytic Frank result."""
        one_constant = self.one_constant_solver.benchmark_helical_twist()[0]
        anisotropic = self.anisotropic_solver.benchmark_helical_twist()[0]

        self.assertIsNotNone(one_constant.reference_energy_density)
        self.assertIsNotNone(anisotropic.reference_energy_density)
        self.assertLess(abs(one_constant.density_delta), 0.10 * one_constant.reference_energy_density)
        self.assertLess(abs(anisotropic.density_delta), 0.20 * anisotropic.reference_energy_density)
        self.assertLess(anisotropic.energy_density, one_constant.energy_density)

    def test_boundary_alignment_ordering_is_physical(self) -> None:
        """Matching anchoring should cost less than mismatched anchoring."""
        one_constant = self._result_map(self.one_constant_solver.benchmark_cylinder_alignment_cases())
        anisotropic = self._result_map(self.anisotropic_solver.benchmark_cylinder_alignment_cases())

        self.assertLess(one_constant["radial_homeotropic_side"].energy_density, one_constant["radial_planar_side"].energy_density)
        self.assertLess(one_constant["azimuthal_planar_side"].energy_density, one_constant["azimuthal_homeotropic_side"].energy_density)
        self.assertLess(anisotropic["radial_homeotropic_side"].energy_density, anisotropic["radial_planar_side"].energy_density)
        self.assertLess(anisotropic["azimuthal_planar_side"].energy_density, anisotropic["azimuthal_homeotropic_side"].energy_density)


if __name__ == "__main__":
    unittest.main()
