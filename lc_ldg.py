"""Landau-de Gennes Q-tensor solver for the liquid-crystal cylinder.

This is a first-pass upgrade for defect-capable nematic physics. It reuses the
body-fitted cylinder mesh from the director solver, but represents the state as
a symmetric traceless Q-tensor so the scalar order can soften toward the
isotropic phase near the nematic-isotropic transition.
"""

from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass
from typing import List, Sequence, Tuple

os.environ["MPLCONFIGDIR"] = os.path.join("/tmp", "codex_mplconfig")
os.environ["XDG_CACHE_HOME"] = os.path.join("/tmp", "codex_cache")
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)
os.makedirs(os.environ["XDG_CACHE_HOME"], exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

from lc_fem import BOLTZMANN_CONSTANT, BoundaryFace, LiquidCrystalFEMSolver

Q_TENSOR_MATERIAL_PRESETS = {
    # Values compiled from standard 5CB Landau-de Gennes fits used in the
    # literature. The intent here is to provide a physically meaningful
    # first-pass Q-tensor backend, not a fully re-fit parameter set.
    "5CB_ldg_room_temperature": {
        "A0": 0.044e6,  # J / (m^3 K)
        "B": 0.816e6,  # J / m^3
        "C": 0.45e6,  # J / m^3
        "L1": 6.0e-12,  # J / m
        "L2": 18.0e-12,  # J / m
        "T_star": 307.0,  # K
        "T_NI": 308.5,  # K
    }
}

Q_COMPONENT_BASIS = np.array(
    [
        [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, -1.0]],
        [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        [[0.0, 0.0, 1.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]],
        [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]],
    ],
    dtype=float,
)
Q_COMPONENT_GRAM = np.array([[float(np.tensordot(a, b, axes=2)) for b in Q_COMPONENT_BASIS] for a in Q_COMPONENT_BASIS], dtype=float)

# Four-point symmetric quadrature on a tetrahedron. This integrates cubic
# polynomials exactly and is a substantial improvement over centroid-only
# evaluation for the nonlinear Landau-de Gennes bulk term.
TETRA_QUAD_BARYCENTRIC = np.array(
    [
        [0.5854101966249685, 0.1381966011250105, 0.1381966011250105, 0.1381966011250105],
        [0.1381966011250105, 0.5854101966249685, 0.1381966011250105, 0.1381966011250105],
        [0.1381966011250105, 0.1381966011250105, 0.5854101966249685, 0.1381966011250105],
        [0.1381966011250105, 0.1381966011250105, 0.1381966011250105, 0.5854101966249685],
    ],
    dtype=float,
)
TETRA_QUAD_WEIGHTS = np.full(4, 0.25, dtype=float)


@dataclass
class QTensorStaticResult:
    """Result from a single Q-tensor relaxation."""

    energy_j: float
    energy_density_j_m3: float
    energy_kbt: float
    mean_scalar_order: float
    min_scalar_order: float
    max_scalar_order: float
    isotropic_fraction: float
    temperature_K: float
    q_components: np.ndarray
    scalar_order: np.ndarray
    principal_directors: np.ndarray
    iterations: int
    success: bool


@dataclass
class QTensorSweepResult:
    """Result from a temperature sweep in Q-tensor mode."""

    temperatures_K: List[float]
    energy_j: List[float]
    energy_density_j_m3: List[float]
    energy_kbt: List[float]
    mean_scalar_order: List[float]
    min_scalar_order: List[float]
    max_scalar_order: List[float]
    isotropic_fraction: List[float]
    final_state: QTensorStaticResult


class LandauDeGennesQTensorSolver:
    """Q-tensor Landau-de Gennes solver on the same cylinder mesh as the director solver."""

    def __init__(
        self,
        coordinates_file: str,
        mesh_file: str | None = None,
        temperature_profile: dict | None = None,
        temperature_K: float | None = None,
        qtensor_preset: str = "5CB_ldg_room_temperature",
        anchoring_preset: str = "planar_side_homeotropic_caps",
        W_side: float | None = None,
        W_caps: float | None = None,
        max_iterations: int = 200,
        tolerance: float = 1.0e-8,
        random_seed: int | None = None,
    ):
        if qtensor_preset not in Q_TENSOR_MATERIAL_PRESETS:
            valid = ", ".join(sorted(Q_TENSOR_MATERIAL_PRESETS))
            raise ValueError(f"Unknown qtensor_preset={qtensor_preset!r}. Valid presets: {valid}")

        self.material_preset = qtensor_preset
        self.material = Q_TENSOR_MATERIAL_PRESETS[qtensor_preset]
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.random_seed = random_seed
        self.rng = np.random.default_rng(random_seed)

        # Reuse the validated cylinder geometry, mesh connectivity, and
        # boundary tagging from the director solver. The Q-tensor backend only
        # changes the field representation and free-energy model.
        self.geometry = LiquidCrystalFEMSolver(
            coordinates_file=coordinates_file,
            mesh_file=mesh_file,
            material_preset="5CB_room_temperature",
            temperature_profile=temperature_profile,
            temperature_K=temperature_K,
            anchoring_preset=anchoring_preset,
            W_side=W_side,
            W_caps=W_caps,
            solver_method="lbfgs",
            random_seed=random_seed,
        )

        self.vertices = self.geometry.vertices
        self.n_nodes = self.geometry.n_nodes
        self.tetrahedra = self.geometry.tetrahedra
        self.tetra_volumes = self.geometry.tetra_volumes
        self.tetra_gradN = self.geometry.tetra_gradN
        self.boundary_faces = self.geometry.boundary_faces
        self.domain_volume = self.geometry.domain_volume
        self.side_mode = self.geometry.side_mode
        self.cap_mode = self.geometry.cap_mode
        self.W_side = self.geometry.W_side
        self.W_caps = self.geometry.W_caps
        self.temperature_profile = self.geometry.temperature_profile
        self.current_temperature_K = float(self.geometry.temperature_K)
        self._spacing = float(self.geometry._spacing)
        self._energy_scale = max(self.material["L1"] * self.domain_volume / max(self._spacing**2, 1.0e-30), 1.0e-18)

        self.q_components = self._initial_q_components(self.current_temperature_K)

    @staticmethod
    def _q_matrix_from_components(components: np.ndarray) -> np.ndarray:
        """Convert the 5 independent Q components into a full symmetric traceless matrix."""
        qxx, qxy, qxz, qyy, qyz = components
        qzz = -(qxx + qyy)
        return np.array(
            [
                [qxx, qxy, qxz],
                [qxy, qyy, qyz],
                [qxz, qyz, qzz],
            ],
            dtype=float,
        )

    @staticmethod
    def _components_from_q_matrix(q_matrix: np.ndarray) -> np.ndarray:
        """Project a full symmetric traceless Q matrix back to the 5 independent components."""
        # The state vector uses the direct independent entries:
        # Qxx, Qxy, Qxz, Qyy, Qyz, with Qzz reconstructed by tracelessness.
        # Returning the matrix entries themselves keeps the anchoring target
        # in the same basis as the solver variables.
        return np.array(
            [
                q_matrix[0, 0],
                q_matrix[0, 1],
                q_matrix[0, 2],
                q_matrix[1, 1],
                q_matrix[1, 2],
            ],
            dtype=float,
        )

    def _equilibrium_scalar_order(self, temperature_K: float) -> float:
        """Return the temperature-dependent uniaxial order parameter for 5CB.

        We use the standard Landau-de Gennes positive root below the nematic-
        isotropic transition and collapse to zero above it.
        """
        if temperature_K >= self.material["T_NI"]:
            return 0.0

        a = self.material["A0"] * (temperature_K - self.material["T_star"])
        b = self.material["B"]
        c = self.material["C"]
        discriminant = b * b - 24.0 * a * c
        if discriminant <= 0.0:
            return 0.0
        return max(0.0, (b + np.sqrt(discriminant)) / (4.0 * c))

    def _preferred_surface_q(self, face: BoundaryFace, temperature_K: float) -> np.ndarray | None:
        """Return the preferred surface Q-tensor for a boundary face.

        Homeotropic anchoring prefers a uniaxial state aligned with the
        outward normal. Planar anchoring prefers a degenerate tangential state
        defined only by the normal, which is the more physical Q-tensor form
        for an apolar nematic surface energy.
        """
        normal = np.asarray(face.normal, dtype=float)
        normal = normal / max(float(np.linalg.norm(normal)), 1.0e-30)
        identity = np.eye(3, dtype=float)
        scalar_order = self._equilibrium_scalar_order(temperature_K)

        if face.kind == "sidewall":
            if self.side_mode == "free":
                return None
            if self.side_mode == "homeotropic":
                return scalar_order * (np.outer(normal, normal) - identity / 3.0)
            # Planar sidewall anchoring: use a degenerate tangential target.
            return scalar_order * ((identity - np.outer(normal, normal)) / 2.0 - identity / 3.0)

        if face.kind in {"top_cap", "bottom_cap"}:
            if self.cap_mode == "free":
                return None
            if self.cap_mode == "homeotropic":
                cap_normal = np.array([0.0, 0.0, 1.0 if face.kind == "top_cap" else -1.0], dtype=float)
                return scalar_order * (np.outer(cap_normal, cap_normal) - identity / 3.0)
            return scalar_order * ((identity - np.outer(normal, normal)) / 2.0 - identity / 3.0)

        return None

    def _initial_q_components(self, temperature_K: float) -> np.ndarray:
        """Start from a uniform z-aligned nematic state."""
        scalar_order = self._equilibrium_scalar_order(temperature_K)
        director = np.tile(np.array([0.0, 0.0, 1.0], dtype=float), (self.n_nodes, 1))
        return self._directors_to_q_components(director, scalar_order)

    def _directors_to_q_components(self, directors: np.ndarray, scalar_order: float | np.ndarray) -> np.ndarray:
        """Convert directors into the 5-component Q parameterization."""
        directors = np.asarray(directors, dtype=float)
        scalar_order = np.asarray(scalar_order, dtype=float)
        if scalar_order.ndim == 0:
            scalar_order = np.full(directors.shape[0], float(scalar_order), dtype=float)

        q = np.zeros((directors.shape[0], 5), dtype=float)
        nx, ny, nz = directors[:, 0], directors[:, 1], directors[:, 2]
        q[:, 0] = scalar_order * (nx * nx - 1.0 / 3.0)
        q[:, 1] = scalar_order * (nx * ny)
        q[:, 2] = scalar_order * (nx * nz)
        q[:, 3] = scalar_order * (ny * ny - 1.0 / 3.0)
        q[:, 4] = scalar_order * (ny * nz)
        return q

    def set_temperature(self, temperature_K: float) -> None:
        """Update the ambient temperature used by the bulk and surface terms."""
        self.current_temperature_K = float(temperature_K)

    def compute_energy_and_gradient(self, q_components: np.ndarray, temperature_K: float | None = None) -> Tuple[float, np.ndarray]:
        """Compute the Landau-de Gennes free energy and its gradient.

        The field is represented by 5 independent components per node:
        Qxx, Qxy, Qxz, Qyy, Qyz. The remaining diagonal component is fixed by
        tracelessness: Qzz = -(Qxx + Qyy).
        """
        q_components = np.asarray(q_components, dtype=float).reshape(self.n_nodes, 5)
        grad = np.zeros_like(q_components)
        total_energy = 0.0

        if temperature_K is None:
            if self.temperature_profile.enabled:
                temperature_field = self.geometry._temperature_field(time=0.0)
            else:
                temperature_field = np.full(self.n_nodes, self.current_temperature_K, dtype=float)
        else:
            temperature_field = np.full(self.n_nodes, float(temperature_K), dtype=float)

        for tet_idx, tet in enumerate(self.tetrahedra):
            nodes = tet
            g = self.tetra_gradN[tet_idx]
            volume = float(self.tetra_volumes[tet_idx])
            q_nodes = q_components[nodes]
            bulk_density = 0.0
            bulk_grad_components = np.zeros((4, 5), dtype=float)
            for qp_bary, qp_weight in zip(TETRA_QUAD_BARYCENTRIC, TETRA_QUAD_WEIGHTS):
                q_qp = np.tensordot(qp_bary, q_nodes, axes=(0, 0))
                q_matrix = self._q_matrix_from_components(q_qp)

                local_temperature = float(np.dot(qp_bary, temperature_field[nodes]))
                a = self.material["A0"] * (local_temperature - self.material["T_star"])
                b = self.material["B"]
                c = self.material["C"]

                q2 = q_matrix @ q_matrix
                tr_q2 = float(np.trace(q_matrix @ q_matrix))
                tr_q3 = float(np.trace(q2 @ q_matrix))
                bulk_density += qp_weight * (
                    0.5 * a * tr_q2 - (b / 3.0) * tr_q3 + 0.25 * c * (tr_q2**2)
                )

                bulk_grad_matrix = a * q_matrix - b * q2 + c * tr_q2 * q_matrix
                bulk_grad_components_qp = np.array(
                    [float(np.tensordot(bulk_grad_matrix, basis, axes=2)) for basis in Q_COMPONENT_BASIS],
                    dtype=float,
                )
                for local_node in range(4):
                    bulk_grad_components[local_node] += qp_weight * qp_bary[local_node] * bulk_grad_components_qp

            total_energy += bulk_density * volume
            for local_node in range(4):
                grad[nodes[local_node]] += volume * bulk_grad_components[local_node]

            # Exact FE gradient for the standard L1/L2 Q-tensor elastic terms.
            comp_grad = np.zeros((5, 3), dtype=float)
            for comp in range(5):
                comp_grad[comp] = q_nodes[:, comp] @ g

            div_q = np.zeros(3, dtype=float)
            for a_idx, basis in enumerate(Q_COMPONENT_BASIS):
                div_q += basis[:, 0] * comp_grad[a_idx, 0] + basis[:, 1] * comp_grad[a_idx, 1] + basis[:, 2] * comp_grad[a_idx, 2]

            elastic_density = 0.5 * self.material["L1"] * float(np.sum(Q_COMPONENT_GRAM * np.einsum("ai,bi->ab", comp_grad, comp_grad))) + 0.5 * self.material["L2"] * float(np.dot(div_q, div_q))
            total_energy += elastic_density * volume

            elastic_grad_components = np.zeros((5, 3), dtype=float)
            for a_idx, basis in enumerate(Q_COMPONENT_BASIS):
                elastic_grad_components[a_idx] = self.material["L1"] * sum(
                    Q_COMPONENT_GRAM[a_idx, b_idx] * comp_grad[b_idx] for b_idx in range(5)
                )
                elastic_grad_components[a_idx] += self.material["L2"] * np.array(
                    [
                        div_q[0] * basis[0, 0] + div_q[1] * basis[1, 0] + div_q[2] * basis[2, 0],
                        div_q[0] * basis[0, 1] + div_q[1] * basis[1, 1] + div_q[2] * basis[2, 1],
                        div_q[0] * basis[0, 2] + div_q[1] * basis[1, 2] + div_q[2] * basis[2, 2],
                    ],
                    dtype=float,
                )

            for local_node in range(4):
                for comp in range(5):
                    grad[nodes[local_node], comp] += volume * float(np.dot(g[local_node], elastic_grad_components[comp]))

        for face in self.boundary_faces:
            if face.kind == "sidewall":
                W = self.W_side
            else:
                W = self.W_caps
            if W <= 0.0:
                continue

            temp = float(np.mean(temperature_field[face.nodes]))
            target_q_matrix = self._preferred_surface_q(face, temp)
            if target_q_matrix is None:
                continue

            target_q = self._components_from_q_matrix(target_q_matrix)
            q_face = np.mean(q_components[face.nodes], axis=0)
            diff = q_face - target_q
            scale = float(self.geometry._anchoring_temperature_scale(temp, kind=face.kind))
            face_energy = 0.5 * W * scale * face.area * float(np.dot(diff, diff))
            total_energy += face_energy
            grad_increment = (W * scale * face.area / 3.0) * diff
            for node in face.nodes:
                grad[node] += grad_increment

        return total_energy, grad

    def _objective(self, q_flat: np.ndarray) -> Tuple[float, np.ndarray]:
        energy, grad = self.compute_energy_and_gradient(q_flat)
        return energy / self._energy_scale, grad.reshape(-1) / self._energy_scale

    def _q_to_directors_and_scalar_order(self, q_components: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        q_components = np.asarray(q_components, dtype=float).reshape(self.n_nodes, 5)
        scalar_order = np.zeros(self.n_nodes, dtype=float)
        directors = np.zeros((self.n_nodes, 3), dtype=float)
        for idx, comp in enumerate(q_components):
            q_matrix = self._q_matrix_from_components(comp)
            eigvals, eigvecs = np.linalg.eigh(q_matrix)
            max_idx = int(np.argmax(eigvals))
            scalar_order[idx] = 1.5 * float(eigvals[max_idx])
            director = eigvecs[:, max_idx]
            if director[2] < 0.0:
                director = -director
            directors[idx] = director / max(float(np.linalg.norm(director)), 1.0e-30)
        return directors, scalar_order

    def relax_static(self, temperature_K: float | None = None, q0: np.ndarray | None = None) -> QTensorStaticResult:
        """Minimize the Q-tensor free energy at a single temperature."""
        if temperature_K is not None:
            self.set_temperature(temperature_K)

        if q0 is None:
            q0 = self.q_components

        result = minimize(
            fun=lambda x: self._objective(x)[0],
            x0=np.asarray(q0, dtype=float).reshape(-1),
            jac=lambda x: self._objective(x)[1],
            method="L-BFGS-B",
            options={"maxiter": self.max_iterations, "ftol": self.tolerance},
        )

        self.q_components = result.x.reshape(self.n_nodes, 5)
        energy, _ = self.compute_energy_and_gradient(self.q_components, temperature_K=self.current_temperature_K)
        energy_density = energy / max(self.domain_volume, 1.0e-30)
        directors, scalar_order = self._q_to_directors_and_scalar_order(self.q_components)
        isotropic_fraction = float(np.mean(scalar_order < 0.1))
        return QTensorStaticResult(
            energy_j=energy,
            energy_density_j_m3=energy_density,
            energy_kbt=energy / (BOLTZMANN_CONSTANT * max(self.current_temperature_K, 1.0e-30)),
            mean_scalar_order=float(np.mean(scalar_order)),
            min_scalar_order=float(np.min(scalar_order)),
            max_scalar_order=float(np.max(scalar_order)),
            isotropic_fraction=isotropic_fraction,
            temperature_K=self.current_temperature_K,
            q_components=self.q_components.copy(),
            scalar_order=scalar_order,
            principal_directors=directors,
            iterations=int(result.nit),
            success=bool(result.success),
        )

    def sweep_temperatures(self, temperatures_K: Sequence[float]) -> QTensorSweepResult:
        """Relax the Q-tensor state across a temperature sequence."""
        temperatures = [float(t) for t in temperatures_K]
        if not temperatures:
            raise ValueError("At least one temperature is required for a Q-tensor sweep.")

        energy_j: List[float] = []
        energy_density_j_m3: List[float] = []
        energy_kbt: List[float] = []
        mean_scalar_order: List[float] = []
        min_scalar_order: List[float] = []
        max_scalar_order: List[float] = []
        isotropic_fraction: List[float] = []

        q_state = self.q_components.copy()
        final_state = None
        for temperature in temperatures:
            final_state = self.relax_static(temperature_K=temperature, q0=q_state)
            q_state = final_state.q_components.copy()
            energy_j.append(final_state.energy_j)
            energy_density_j_m3.append(final_state.energy_density_j_m3)
            energy_kbt.append(final_state.energy_kbt)
            mean_scalar_order.append(final_state.mean_scalar_order)
            min_scalar_order.append(final_state.min_scalar_order)
            max_scalar_order.append(final_state.max_scalar_order)
            isotropic_fraction.append(final_state.isotropic_fraction)

        assert final_state is not None
        return QTensorSweepResult(
            temperatures_K=temperatures,
            energy_j=energy_j,
            energy_density_j_m3=energy_density_j_m3,
            energy_kbt=energy_kbt,
            mean_scalar_order=mean_scalar_order,
            min_scalar_order=min_scalar_order,
            max_scalar_order=max_scalar_order,
            isotropic_fraction=isotropic_fraction,
            final_state=final_state,
        )

    def save_qtensor_summary(self, result: QTensorStaticResult, filename: str) -> None:
        """Write a compact human-readable summary for the Q-tensor state."""
        with open(filename, "w", encoding="utf-8") as fh:
            fh.write("Liquid-crystal Q-tensor summary\n")
            fh.write(f"material_preset: {self.material_preset}\n")
            fh.write(f"temperature_K: {result.temperature_K:.6f}\n")
            fh.write(f"energy_j: {result.energy_j:.6e}\n")
            fh.write(f"energy_density_j_m3: {result.energy_density_j_m3:.6e}\n")
            fh.write(f"energy_kbt: {result.energy_kbt:.6e}\n")
            fh.write(f"mean_scalar_order: {result.mean_scalar_order:.6e}\n")
            fh.write(f"min_scalar_order: {result.min_scalar_order:.6e}\n")
            fh.write(f"max_scalar_order: {result.max_scalar_order:.6e}\n")
            fh.write(f"isotropic_fraction: {result.isotropic_fraction:.6e}\n")
            fh.write(f"success: {result.success}\n")
            fh.write(f"iterations: {result.iterations}\n")

    def save_qtensor_field(self, result: QTensorStaticResult, filename: str) -> None:
        """Write the nodal Q-tensor state to CSV for inspection."""
        directors = result.principal_directors
        with open(filename, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(
                [
                    "x_um",
                    "y_um",
                    "z_um",
                    "Qxx",
                    "Qxy",
                    "Qxz",
                    "Qyy",
                    "Qyz",
                    "scalar_order",
                    "director_x",
                    "director_y",
                    "director_z",
                ]
            )
            for vertex, comp, s, director in zip(self.vertices, result.q_components, result.scalar_order, directors):
                writer.writerow(
                    [
                        f"{vertex[0] * 1e6:.8e}",
                        f"{vertex[1] * 1e6:.8e}",
                        f"{vertex[2] * 1e6:.8e}",
                        f"{comp[0]:.8e}",
                        f"{comp[1]:.8e}",
                        f"{comp[2]:.8e}",
                        f"{comp[3]:.8e}",
                        f"{comp[4]:.8e}",
                        f"{s:.8e}",
                        f"{director[0]:.8e}",
                        f"{director[1]:.8e}",
                        f"{director[2]:.8e}",
                    ]
                )

    def save_qtensor_director_plot(self, result: QTensorStaticResult, filename: str, max_vectors: int = 250) -> None:
        """Visualize the principal director field with a scalar-order color map."""
        directors = result.principal_directors
        scalar_order = result.scalar_order
        n = len(directors)
        stride = max(1, n // max_vectors)
        indices = np.arange(0, n, stride)

        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection="3d")
        colors = plt.cm.viridis(np.clip(scalar_order[indices], 0.0, np.max(scalar_order) if np.max(scalar_order) > 0 else 1.0))
        x_um = self.vertices[indices, 0] * 1e6
        y_um = self.vertices[indices, 1] * 1e6
        z_um = self.vertices[indices, 2] * 1e6
        ax.quiver(
            x_um,
            y_um,
            z_um,
            directors[indices, 0],
            directors[indices, 1],
            directors[indices, 2],
            length=0.35,
            normalize=True,
            color=colors,
            linewidth=1.0,
        )
        ax.scatter(x_um, y_um, z_um, c=scalar_order[indices], cmap="viridis", s=4, alpha=0.5)
        ax.set_xlabel("x (um)")
        ax.set_ylabel("y (um)")
        ax.set_zlabel("z (um)")
        ax.set_title("Q-tensor principal director field")
        mappable = plt.cm.ScalarMappable(cmap="viridis")
        mappable.set_array(scalar_order)
        fig.colorbar(mappable, ax=ax, shrink=0.6, pad=0.1, label="Scalar order")
        fig.tight_layout()
        fig.savefig(filename, dpi=300)
        plt.close(fig)

    def save_qtensor_order_map(self, result: QTensorStaticResult, filename: str, projection: str = "rho_z") -> None:
        """Save a projected scalar-order heatmap."""
        centroids = np.zeros((self.n_nodes, 3), dtype=float)
        centroids[:, :] = self.vertices
        if projection == "rho_z":
            x_values = np.linalg.norm(centroids[:, :2], axis=1) * 1e6
            y_values = centroids[:, 2] * 1e6
            xlabel, ylabel = "rho (um)", "z (um)"
        elif projection == "x_z":
            x_values = centroids[:, 0] * 1e6
            y_values = centroids[:, 2] * 1e6
            xlabel, ylabel = "x (um)", "z (um)"
        elif projection == "x_y":
            x_values = centroids[:, 0] * 1e6
            y_values = centroids[:, 1] * 1e6
            xlabel, ylabel = "x (um)", "y (um)"
        elif projection == "theta_z":
            theta = np.degrees(np.mod(np.arctan2(centroids[:, 1], centroids[:, 0]), 2.0 * np.pi))
            x_values = theta
            y_values = centroids[:, 2] * 1e6
            xlabel, ylabel = "theta (deg)", "z (um)"
        else:
            raise ValueError(f"Unsupported projection={projection!r}")

        fig, ax = plt.subplots(figsize=(10, 6))
        scatter = ax.scatter(x_values, y_values, c=result.scalar_order, cmap="viridis", s=16, alpha=0.9)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title("Q-tensor scalar-order map")
        fig.colorbar(scatter, ax=ax, label="Scalar order")
        fig.tight_layout()
        fig.savefig(filename, dpi=300)
        plt.close(fig)

    def save_qtensor_bundle(self, result: QTensorStaticResult, output_prefix: str) -> None:
        """Write a compact diagnostic bundle for a single Q-tensor relaxation."""
        self.save_qtensor_summary(result, f"{output_prefix}_qtensor_summary.txt")
        self.save_qtensor_field(result, f"{output_prefix}_qtensor_field.csv")
        self.save_qtensor_director_plot(result, f"{output_prefix}_qtensor_director_field.jpg")
        self.save_qtensor_order_map(result, f"{output_prefix}_qtensor_order_map.jpg", projection="rho_z")

    def save_qtensor_sweep_summary(self, result: QTensorSweepResult, filename: str) -> None:
        """Write a concise summary for the temperature sweep."""
        with open(filename, "w", encoding="utf-8") as fh:
            fh.write("Liquid-crystal Q-tensor temperature sweep summary\n")
            fh.write(f"material_preset: {self.material_preset}\n")
            fh.write(f"temperature_window_K: {result.temperatures_K[0]:.6f} -> {result.temperatures_K[-1]:.6f}\n")
            midpoint = len(result.temperatures_K) // 2
            fh.write(f"midpoint_temperature_K: {result.temperatures_K[midpoint]:.6f}\n")
            fh.write(f"midpoint_mean_scalar_order: {result.mean_scalar_order[midpoint]:.6e}\n")
            fh.write(f"endpoint_mean_scalar_order: {result.mean_scalar_order[0]:.6e} -> {result.mean_scalar_order[-1]:.6e}\n")
            fh.write(f"endpoint_isotropic_fraction: {result.isotropic_fraction[0]:.6e} -> {result.isotropic_fraction[-1]:.6e}\n")
            fh.write(f"endpoint_energy_density_J_m3: {result.energy_density_j_m3[0]:.6e} -> {result.energy_density_j_m3[-1]:.6e}\n")

    def save_qtensor_sweep_data(self, result: QTensorSweepResult, filename: str) -> None:
        """Write the temperature sweep observables to CSV."""
        with open(filename, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(
                [
                    "temperature_K",
                    "energy_j",
                    "energy_density_j_m3",
                    "energy_kbt",
                    "mean_scalar_order",
                    "min_scalar_order",
                    "max_scalar_order",
                    "isotropic_fraction",
                ]
            )
            for values in zip(
                result.temperatures_K,
                result.energy_j,
                result.energy_density_j_m3,
                result.energy_kbt,
                result.mean_scalar_order,
                result.min_scalar_order,
                result.max_scalar_order,
                result.isotropic_fraction,
            ):
                writer.writerow([f"{value:.8e}" for value in values])

    def save_qtensor_sweep_plot(self, result: QTensorSweepResult, filename: str) -> None:
        """Plot the sweep response across temperature."""
        fig, axes = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
        axes[0].plot(result.temperatures_K, result.mean_scalar_order, marker="o", label="mean scalar order")
        axes[0].plot(result.temperatures_K, result.min_scalar_order, marker="s", label="min scalar order")
        axes[0].plot(result.temperatures_K, result.max_scalar_order, marker="^", label="max scalar order")
        axes[0].set_ylabel("Scalar order")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(result.temperatures_K, result.energy_density_j_m3, marker="o", color="tab:red", label="energy density")
        axes[1].plot(result.temperatures_K, result.isotropic_fraction, marker="s", color="tab:gray", label="isotropic fraction")
        axes[1].set_xlabel("Temperature (K)")
        axes[1].set_ylabel("Energy density / isotropic fraction")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(filename, dpi=300)
        plt.close(fig)

    def run_temperature_sweep(self, temperatures_K: Sequence[float], output_prefix: str) -> QTensorSweepResult:
        """Run a temperature sweep and write the associated diagnostics."""
        q_state = self.q_components.copy()
        energy_j: List[float] = []
        energy_density_j_m3: List[float] = []
        energy_kbt: List[float] = []
        mean_scalar_order: List[float] = []
        min_scalar_order: List[float] = []
        max_scalar_order: List[float] = []
        isotropic_fraction: List[float] = []
        final_state: QTensorStaticResult | None = None

        for temperature in temperatures_K:
            final_state = self.relax_static(temperature_K=float(temperature), q0=q_state)
            q_state = final_state.q_components.copy()
            energy_j.append(final_state.energy_j)
            energy_density_j_m3.append(final_state.energy_density_j_m3)
            energy_kbt.append(final_state.energy_kbt)
            mean_scalar_order.append(final_state.mean_scalar_order)
            min_scalar_order.append(final_state.min_scalar_order)
            max_scalar_order.append(final_state.max_scalar_order)
            isotropic_fraction.append(final_state.isotropic_fraction)

        if final_state is None:
            raise ValueError("Temperature sweep received no temperatures.")

        result = QTensorSweepResult(
            temperatures_K=list(map(float, temperatures_K)),
            energy_j=energy_j,
            energy_density_j_m3=energy_density_j_m3,
            energy_kbt=energy_kbt,
            mean_scalar_order=mean_scalar_order,
            min_scalar_order=min_scalar_order,
            max_scalar_order=max_scalar_order,
            isotropic_fraction=isotropic_fraction,
            final_state=final_state,
        )
        self.save_qtensor_sweep_data(result, f"{output_prefix}_qtensor_temperature_sweep.csv")
        self.save_qtensor_sweep_plot(result, f"{output_prefix}_qtensor_temperature_sweep.jpg")
        self.save_qtensor_sweep_summary(result, f"{output_prefix}_qtensor_temperature_sweep_summary.txt")
        self.save_qtensor_bundle(final_state, f"{output_prefix}_qtensor_temperature_sweep_final")
        return result


def run_qtensor_static(
    coordinates_file: str,
    mesh_file: str | None = None,
    output_prefix: str = "ldg_qtensor",
    temperature_K: float | None = None,
    temperature_profile: dict | None = None,
    qtensor_preset: str = "5CB_ldg_room_temperature",
    anchoring_preset: str = "planar_side_homeotropic_caps",
    W_side: float | None = None,
    W_caps: float | None = None,
    max_iterations: int = 200,
    tolerance: float = 1.0e-8,
    random_seed: int | None = None,
) -> tuple[LandauDeGennesQTensorSolver, QTensorStaticResult]:
    """Convenience wrapper for the static Q-tensor solver."""
    solver = LandauDeGennesQTensorSolver(
        coordinates_file=coordinates_file,
        mesh_file=mesh_file,
        temperature_profile=temperature_profile,
        temperature_K=temperature_K,
        qtensor_preset=qtensor_preset,
        anchoring_preset=anchoring_preset,
        W_side=W_side,
        W_caps=W_caps,
        max_iterations=max_iterations,
        tolerance=tolerance,
        random_seed=random_seed,
    )
    result = solver.relax_static(temperature_K=temperature_K)
    solver.save_qtensor_bundle(result, output_prefix)
    return solver, result
