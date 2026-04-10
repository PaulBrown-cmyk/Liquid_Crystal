"""Regression checks for the Q-tensor Landau-de Gennes backend."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from grid1 import CylinderGrid
from lc_ldg import LandauDeGennesQTensorSolver


REPO_ROOT = Path(__file__).resolve().parent
GRID_FILE = REPO_ROOT / "straight_cylinder_grid_with_grid.txt"


class LandauDeGennesValidationTests(unittest.TestCase):
    """Keep the first-pass Q-tensor backend honest around the 5CB transition."""

    @classmethod
    def setUpClass(cls) -> None:
        if not GRID_FILE.exists():
            raise unittest.SkipTest(f"Missing reference grid file: {GRID_FILE}")

        cls._tmpdir = tempfile.TemporaryDirectory()
        mesh_path = Path(cls._tmpdir.name) / "qtensor_structured_mesh.npz"
        cylinder = CylinderGrid(
            diameter_um=5,
            length_um=20,
            num_boundary_points_per_z=16,
            num_z_levels=12,
            num_inner_points=16,
            min_distance_um=1.0,
        )
        cylinder.save_structured_solve_mesh(
            str(mesh_path),
            num_radial_layers=4,
            num_theta_points=16,
            num_axial_layers=8,
            radial_cluster_power=2.0,
            axial_cluster_power=2.0,
        )
        cls.mesh_file = str(mesh_path)

    def _make_solver(self) -> LandauDeGennesQTensorSolver:
        return LandauDeGennesQTensorSolver(
            coordinates_file=str(GRID_FILE),
            mesh_file=self.mesh_file,
            qtensor_preset="5CB_ldg_room_temperature",
            anchoring_preset="free",
            max_iterations=200,
            random_seed=0,
        )

    @classmethod
    def tearDownClass(cls) -> None:
        if hasattr(cls, "_tmpdir"):
            cls._tmpdir.cleanup()

    def test_low_temperature_is_nematic(self) -> None:
        """Below the 5CB transition, the scalar order should remain appreciable."""
        result = self._make_solver().relax_static(temperature_K=298.15)
        self.assertTrue(result.success)
        self.assertGreater(result.mean_scalar_order, 0.15)
        self.assertLess(result.isotropic_fraction, 0.8)

    def test_high_temperature_is_near_isotropic(self) -> None:
        """Above the 5CB transition, the solver should soften toward isotropy."""
        result = self._make_solver().relax_static(temperature_K=310.0)
        self.assertTrue(result.success)
        self.assertLess(result.mean_scalar_order, 0.08)
        self.assertGreater(result.isotropic_fraction, 0.6)

    def test_order_drops_with_temperature(self) -> None:
        """The mean scalar order should decrease when the temperature rises."""
        low = self._make_solver().relax_static(temperature_K=298.15)
        high = self._make_solver().relax_static(temperature_K=310.0)
        self.assertGreater(low.mean_scalar_order, high.mean_scalar_order)


if __name__ == "__main__":
    unittest.main()
