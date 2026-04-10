import os
from dataclasses import dataclass
from typing import List, Tuple

# Keep Matplotlib's cache in a writable location when running headless.
os.environ.setdefault("MPLCONFIGDIR", os.path.join("/tmp", "codex_mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", os.path.join("/tmp", "codex_cache"))

import matplotlib

# The FEM solver is usually run headless from the command line, so use a
# non-interactive backend to avoid cache and display issues on CI or remote
# shells.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull, Delaunay, cKDTree


@dataclass
class BoundaryFace:
    """Triangular boundary face extracted from the mesh hull."""

    nodes: np.ndarray
    area: float
    normal: np.ndarray
    kind: str


class LiquidCrystalFEMSolver:
    """Finite-element continuum solver for a director field on a cylinder.

    The solver uses:
    - a tetrahedral Delaunay mesh built from the supplied point cloud
    - a one-constant Frank-Oseen bulk energy with optional cholesteric pitch
    - face-based Rapini-Papoular anchoring on the actual boundary triangles

    The relaxation scheme uses L-BFGS in spherical-angle space so the unit
    director constraint is enforced by construction. That gives us a more
    robust nonlinear solve than the earlier projected-gradient prototype while
    keeping the implementation readable.
    """

    def __init__(
        self,
        coordinates_file: str,
        K: float = 6.0e-12,
        q0: float = 2.0 * np.pi / (0.5e-6),
        W_side: float | None = None,
        W_caps: float | None = None,
        side_mode: str | None = None,
        cap_mode: str | None = None,
        step_size: float = 0.05,
        max_iterations: int = 200,
        tolerance: float = 1.0e-8,
        anchoring_preset: str = "planar_side_homeotropic_caps",
    ):
        self._anchoring_presets = {
            # Tangential sidewall plus normal end caps is a common finite-cylinder
            # configuration and gives the solver a physically meaningful default.
            "planar_side_homeotropic_caps": {
                "side_mode": "planar",
                "cap_mode": "homeotropic",
                "W_side": 1.0e-5,
                "W_caps": 1.0e-5,
            },
            "planar_all": {
                "side_mode": "planar",
                "cap_mode": "planar",
                "W_side": 1.0e-5,
                "W_caps": 1.0e-5,
            },
            "homeotropic_all": {
                "side_mode": "homeotropic",
                "cap_mode": "homeotropic",
                "W_side": 1.0e-5,
                "W_caps": 1.0e-5,
            },
            "free": {
                "side_mode": "free",
                "cap_mode": "free",
                "W_side": 0.0,
                "W_caps": 0.0,
            },
        }

        self.coordinates_file = coordinates_file
        self.K = K
        self.q0 = q0
        self.step_size = step_size
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.anchoring_preset = anchoring_preset

        self._configure_anchoring(
            anchoring_preset=anchoring_preset,
            W_side=W_side,
            W_caps=W_caps,
            side_mode=side_mode,
            cap_mode=cap_mode,
        )

        self.vertices = self._load_vertices(coordinates_file)
        self.n_nodes = len(self.vertices)
        self._spacing = self._estimate_spacing(self.vertices)
        self._volume_tol = max((self._spacing ** 3) * 1.0e-8, 1.0e-30)
        # The raw energy and gradient are expressed in SI units and are
        # therefore numerically tiny; scale the optimization problem by a
        # characteristic elastic energy so the solver sees a healthy magnitude.
        self._energy_scale = max(self.K * self._spacing, 1.0e-30)
        self._gradient_scale = self._energy_scale
        self.directors = self._initialize_directors()

        self._build_mesh()

    def _load_vertices(self, filename: str) -> np.ndarray:
        """Load the grid written by grid1.py and convert microns to meters."""
        data = np.loadtxt(filename)
        if data.ndim == 1:
            data = data[np.newaxis, :]
        vertices = data[:, :3].astype(float)
        vertices *= 1e-6
        return vertices

    def _configure_anchoring(
        self,
        anchoring_preset: str,
        W_side: float | None,
        W_caps: float | None,
        side_mode: str | None,
        cap_mode: str | None,
    ) -> None:
        """Resolve the surface anchoring setup from a named preset.

        The named presets make the cylinder boundary configuration explicit, but
        callers can still override any patch-level value for special cases.
        """
        if anchoring_preset not in self._anchoring_presets:
            valid = ", ".join(sorted(self._anchoring_presets))
            raise ValueError(f"Unknown anchoring_preset={anchoring_preset!r}. Valid presets: {valid}")

        preset = self._anchoring_presets[anchoring_preset]
        self.side_mode = preset["side_mode"] if side_mode is None else side_mode
        self.cap_mode = preset["cap_mode"] if cap_mode is None else cap_mode
        self.W_side = preset["W_side"] if W_side is None else W_side
        self.W_caps = preset["W_caps"] if W_caps is None else W_caps

    def _initialize_directors(self) -> np.ndarray:
        """Start from a random unit director at every node."""
        theta = np.random.uniform(0.0, 2.0 * np.pi, self.n_nodes)
        phi = np.random.uniform(0.0, np.pi, self.n_nodes)
        directors = np.column_stack(
            (
                np.cos(theta) * np.sin(phi),
                np.sin(theta) * np.sin(phi),
                np.cos(phi),
            )
        )
        return self.normalize_directors(directors)

    @staticmethod
    def _directors_to_angles(directors: np.ndarray) -> np.ndarray:
        """Convert unit directors to spherical angles for constrained optimization."""
        theta = np.mod(np.arctan2(directors[:, 1], directors[:, 0]), 2.0 * np.pi)
        phi = np.arccos(np.clip(directors[:, 2], -1.0, 1.0))
        return np.concatenate([theta, phi])

    @staticmethod
    def _angles_to_directors(theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
        """Convert spherical angles back to unit director vectors."""
        sin_phi = np.sin(phi)
        return np.column_stack(
            (
                np.cos(theta) * sin_phi,
                np.sin(theta) * sin_phi,
                np.cos(phi),
            )
        )

    @staticmethod
    def _angles_gradient(director_grad: np.ndarray, theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
        """Apply the chain rule from director gradients to angle gradients."""
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)
        sin_phi = np.sin(phi)
        cos_phi = np.cos(phi)

        dndtheta = np.column_stack(
            (
                -sin_theta * sin_phi,
                cos_theta * sin_phi,
                np.zeros_like(theta),
            )
        )
        dndphi = np.column_stack(
            (
                cos_theta * cos_phi,
                sin_theta * cos_phi,
                -sin_phi,
            )
        )

        grad_theta = np.sum(director_grad * dndtheta, axis=1)
        grad_phi = np.sum(director_grad * dndphi, axis=1)
        return np.concatenate([grad_theta, grad_phi])

    @staticmethod
    def normalize_directors(directors: np.ndarray) -> np.ndarray:
        """Normalize each nodal director to unit length."""
        norms = np.linalg.norm(directors, axis=1, keepdims=True)
        norms = np.where(norms == 0.0, 1.0, norms)
        return directors / norms

    def _build_mesh(self) -> None:
        """Assemble the tetrahedral mesh and precompute geometry terms."""
        self.tetra = Delaunay(self.vertices)
        valid_tetrahedra = []
        valid_volumes = []
        valid_gradients = []

        for tet in self.tetra.simplices:
            coords = self.vertices[tet]
            try:
                volume, grad_n = self._tetra_geometry(coords)
            except np.linalg.LinAlgError:
                # Delaunay can emit nearly coplanar cells on boundary-rich point
                # clouds. Those elements carry no reliable volume information and
                # should not participate in the continuum assembly.
                continue

            if volume <= self._volume_tol:
                continue

            valid_tetrahedra.append(tet)
            valid_volumes.append(volume)
            valid_gradients.append(grad_n)

        self.tetrahedra = np.asarray(valid_tetrahedra, dtype=int)
        self.tetra_volumes = np.asarray(valid_volumes, dtype=float)
        self.tetra_gradN = np.asarray(valid_gradients, dtype=float)

        self.boundary_faces = self._extract_boundary_faces()

    @staticmethod
    def _tetra_geometry(coords: np.ndarray) -> Tuple[float, np.ndarray]:
        """Return tetra volume and gradients of the linear shape functions.

        For a linear tetrahedron the shape functions are affine, so their
        gradients are constant inside the element. We compute them from the
        inverse of the 4x4 nodal coordinate matrix.
        """
        m = np.column_stack((np.ones(4), coords))
        inv_m = np.linalg.inv(m)

        # Columns of inv_m correspond to the coefficients of each shape function.
        grad_n = inv_m[1:, :].T  # shape: (4, 3)

        volume = abs(np.linalg.det(coords[1:] - coords[0])) / 6.0
        return volume, grad_n

    @staticmethod
    def _estimate_spacing(vertices: np.ndarray) -> float:
        """Estimate the characteristic mesh spacing from nearest neighbors."""
        if len(vertices) < 2:
            return 1.0

        tree = cKDTree(vertices)
        distances, _ = tree.query(vertices, k=min(2, len(vertices)))
        nearest = distances[:, 1] if distances.ndim == 2 else distances[1:]
        nearest = nearest[np.isfinite(nearest) & (nearest > 0.0)]
        if nearest.size == 0:
            return 1.0
        return float(np.median(nearest))

    def _extract_boundary_faces(self) -> List[BoundaryFace]:
        """Extract boundary faces from the convex hull and classify them.

        The cylinder axis is aligned with z. Faces with normals mostly aligned
        with ±z are treated as end caps; the rest are sidewall faces.
        """
        hull = ConvexHull(self.vertices)
        domain_centroid = self.vertices.mean(axis=0)
        faces: List[BoundaryFace] = []

        for face in hull.simplices:
            pts = self.vertices[face]
            edge1 = pts[1] - pts[0]
            edge2 = pts[2] - pts[0]
            normal = np.cross(edge1, edge2)
            norm = np.linalg.norm(normal)
            if norm == 0.0:
                continue

            area = 0.5 * norm
            normal = normal / norm
            centroid = pts.mean(axis=0)

            # Flip the normal so it points outward from the volume.
            if np.dot(normal, centroid - domain_centroid) < 0.0:
                normal = -normal

            kind = "cap" if abs(normal[2]) >= 0.6 else "sidewall"
            faces.append(BoundaryFace(nodes=face, area=area, normal=normal, kind=kind))

        return faces

    def compute_energy_and_gradient(self, directors: np.ndarray) -> Tuple[float, np.ndarray]:
        """Compute the continuum free energy and its gradient.

        The bulk term uses a one-constant Frank-Oseen form with a cholesteric
        pitch penalty. This is a continuum discretization on tetrahedral linear
        elements, so the gradients are assembled element-by-element instead of
        being inferred from array storage order.
        """
        directors = self.normalize_directors(directors)
        grad = np.zeros_like(directors)
        total_energy = 0.0

        for tet_idx, tet in enumerate(self.tetrahedra):
            nodes = tet
            n = directors[nodes]  # shape: (4, 3)
            g = self.tetra_gradN[tet_idx]  # shape: (4, 3)
            volume = self.tetra_volumes[tet_idx]

            # Gradients of the director components in the element.
            grad_n = n.T @ g  # 3x3, rows correspond to x/y/z director components
            div_n = np.trace(grad_n)
            curl_n = np.array(
                [
                    grad_n[2, 1] - grad_n[1, 2],
                    grad_n[0, 2] - grad_n[2, 0],
                    grad_n[1, 0] - grad_n[0, 1],
                ]
            )
            n_avg = n.mean(axis=0)

            # One-constant cholesteric Frank-Oseen density:
            # 0.5*K*(div(n)^2 + |curl(n)|^2 + 2 q0 n·curl(n) + q0^2)
            density = 0.5 * self.K * (div_n**2 + float(np.dot(curl_n, curl_n)) + 2.0 * self.q0 * np.dot(n_avg, curl_n) + self.q0**2)
            total_energy += density * volume

            # Element contribution to the gradient.
            for local_node in range(4):
                g_local = g[local_node]

                # Derivative of divergence wrt a nodal director vector.
                div_grad = np.array([g_local[0], g_local[1], g_local[2]])

                # Linear map from a nodal director vector to the curl.
                curl_jac = np.array(
                    [
                        [0.0, -g_local[2], g_local[1]],
                        [g_local[2], 0.0, -g_local[0]],
                        [-g_local[1], g_local[0], 0.0],
                    ]
                )

                # For the quadratic term, use the exact assembled gradient from
                # the element's linear basis functions.
                # Assemble the actual projected energy gradient.
                grad[ nodes[local_node] ] += self.K * volume * (
                    div_n * div_grad
                    + curl_jac.T @ curl_n
                    + self.q0 * (0.25 * curl_n + curl_jac.T @ n_avg)
                )

        # Boundary anchoring on the real boundary triangles.
        surface_energy = 0.0
        for face in self.boundary_faces:
            if face.kind == "sidewall":
                mode = self.side_mode
                W = self.W_side
            else:
                mode = self.cap_mode
                W = self.W_caps

            if mode == "free" or W == 0.0:
                continue

            node_vectors = directors[face.nodes]
            dot_values = node_vectors @ face.normal
            face_mass = (face.area / 12.0) * (np.ones((3, 3)) + np.eye(3))
            weighted_dot = face_mass @ dot_values

            if mode == "planar":
                # Tangential anchoring: penalize the normal component.
                face_energy = 0.5 * W * float(dot_values @ weighted_dot)
                dE_dn_face = W * weighted_dot[:, np.newaxis] * face.normal
            elif mode == "homeotropic":
                # Homeotropic anchoring: penalize misalignment with the normal.
                face_energy = 0.5 * W * (face.area - float(dot_values @ weighted_dot))
                dE_dn_face = -W * weighted_dot[:, np.newaxis] * face.normal
            else:
                raise ValueError(f"Unknown anchoring mode: {mode}")

            surface_energy += face_energy

            for local_idx, node in enumerate(face.nodes):
                grad[node] += dE_dn_face[local_idx]

        total_energy += surface_energy
        return total_energy, grad

    def project_gradient(self, directors: np.ndarray, grad: np.ndarray) -> np.ndarray:
        """Project the gradient onto the tangent plane of the unit sphere."""
        radial = np.sum(grad * directors, axis=1, keepdims=True)
        return grad - radial * directors

    def relax(self) -> List[float]:
        """Relax the director field using an L-BFGS solve in angle space."""
        theta_phi0 = self._directors_to_angles(self.directors)
        initial_energy, _ = self.compute_energy_and_gradient(self.directors)
        energies: List[float] = [initial_energy]
        n = self.n_nodes
        bounds = [(0.0, 2.0 * np.pi)] * n + [(0.0, np.pi)] * n

        def objective(theta_phi: np.ndarray) -> tuple[float, np.ndarray]:
            theta = theta_phi[:n]
            phi = theta_phi[n:]
            directors = self._angles_to_directors(theta, phi)
            energy, director_grad = self.compute_energy_and_gradient(directors)
            angle_grad = self._angles_gradient(director_grad, theta, phi)
            return energy / self._energy_scale, angle_grad / self._energy_scale

        def callback(theta_phi: np.ndarray) -> None:
            theta = theta_phi[:n]
            phi = theta_phi[n:]
            directors = self._angles_to_directors(theta, phi)
            energy, _ = self.compute_energy_and_gradient(directors)
            energies.append(energy)

        result = minimize(
            objective,
            theta_phi0,
            method="L-BFGS-B",
            jac=True,
            bounds=bounds,
            callback=callback,
            options={
                "maxiter": self.max_iterations,
                "ftol": self.tolerance,
                "gtol": self.tolerance,
                "maxls": 50,
            },
        )

        final_theta = result.x[:n]
        final_phi = result.x[n:]
        self.directors = self._angles_to_directors(final_theta, final_phi)

        final_energy = float(result.fun * self._energy_scale)
        if not energies or abs(energies[-1] - final_energy) > 0.0:
            energies.append(final_energy)

        return energies

    def save_director_field(self, filename: str) -> None:
        with open(filename, "w") as f:
            f.write("# x (m)     y (m)     z (m)     n_x     n_y     n_z\n")
            for i, point in enumerate(self.vertices):
                n = self.directors[i]
                f.write(
                    f"{point[0]:.6e}     {point[1]:.6e}     {point[2]:.6e}     "
                    f"{n[0]:.6e}     {n[1]:.6e}     {n[2]:.6e}\n"
                )

    def load_director_field(self, filename: str) -> None:
        data = np.loadtxt(filename)
        if data.ndim == 1:
            data = data[np.newaxis, :]
        if data.shape[1] < 6:
            raise ValueError(f"{filename} does not contain x, y, z, n_x, n_y, n_z columns")

        self.vertices = data[:, :3]
        self.n_nodes = len(self.vertices)
        self._spacing = self._estimate_spacing(self.vertices)
        self._volume_tol = max((self._spacing ** 3) * 1.0e-8, 1.0e-30)
        self._energy_scale = max(self.K * self._spacing, 1.0e-30)
        self._gradient_scale = self._energy_scale
        self.directors = self.normalize_directors(data[:, 3:6])
        self._build_mesh()

    def plot_director_field(self, ax, title: str = "Finite Element Director Field") -> None:
        ax.clear()
        dirs = self.normalize_directors(self.directors)
        ax.quiver(
            self.vertices[:, 0] * 1e6,
            self.vertices[:, 1] * 1e6,
            self.vertices[:, 2] * 1e6,
            dirs[:, 0],
            dirs[:, 1],
            dirs[:, 2],
            length=0.1,
            normalize=True,
            pivot="middle",
        )
        ax.set_title(title)
        ax.set_xlabel("X (µm)")
        ax.set_ylabel("Y (µm)")
        ax.set_zlabel("Z (µm)")
        ax.set_xlim(np.min(self.vertices[:, 0]) * 1e6, np.max(self.vertices[:, 0]) * 1e6)
        ax.set_ylim(np.min(self.vertices[:, 1]) * 1e6, np.max(self.vertices[:, 1]) * 1e6)
        ax.set_zlim(np.min(self.vertices[:, 2]) * 1e6, np.max(self.vertices[:, 2]) * 1e6)
        plt.draw()

    def plot_energy(self, energies: List[float], filename: str = "energy_per_iteration.jpg") -> None:
        if not energies:
            return
        plt.figure(figsize=(10, 6))
        plt.plot(energies, label="Free Energy")
        plt.xlabel("Iteration")
        plt.ylabel("F (J)")
        plt.title("Free Energy Per Iteration")
        plt.legend()
        plt.grid(True)
        plt.savefig(filename, dpi=600)

    def plot_angle_histogram(self, filename: str = "histogram.jpg") -> None:
        dirs = self.normalize_directors(self.directors)
        cos_theta = np.clip(dirs[:, 2], -1.0, 1.0)
        angles = np.degrees(np.arccos(cos_theta))
        plt.figure(figsize=(8, 6))
        plt.hist(angles, bins=30, edgecolor="k", alpha=0.7)
        plt.xlabel("Angle (degrees)")
        plt.ylabel("Frequency")
        plt.title("Director Angle Distribution Relative to the z-axis")
        plt.savefig(filename, dpi=600)


def run_fem_solver(
    coordinates_file: str,
    checkpoint_file: str | None = None,
    run_time: int = 500,
    output_prefix: str = "fem",
    anchoring_preset: str = "planar_side_homeotropic_caps",
) -> Tuple[LiquidCrystalFEMSolver, List[float]]:
    """Convenience wrapper for the main program."""
    # Apply Rapini-Papoular anchoring on the full closed cylinder boundary by
    # default. The preset can be swapped for other patch-wise boundary setups.
    solver = LiquidCrystalFEMSolver(
        coordinates_file,
        max_iterations=run_time,
        anchoring_preset=anchoring_preset,
    )
    if checkpoint_file:
        solver.load_director_field(checkpoint_file)

    energies = solver.relax()
    solver.save_director_field(f"{output_prefix}_optimized_director_field.txt")
    solver.plot_energy(energies, filename=f"{output_prefix}_energy.jpg")
    solver.plot_angle_histogram(filename=f"{output_prefix}_histogram.jpg")
    return solver, energies
