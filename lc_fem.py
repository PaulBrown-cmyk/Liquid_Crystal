import os
import csv
from dataclasses import dataclass
from typing import List, Tuple

# Keep Matplotlib's cache in a writable location when running headless.
os.environ["MPLCONFIGDIR"] = os.path.join("/tmp", "codex_mplconfig")
os.environ["XDG_CACHE_HOME"] = os.path.join("/tmp", "codex_cache")
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)
os.makedirs(os.environ["XDG_CACHE_HOME"], exist_ok=True)

import matplotlib

# The FEM solver is usually run headless from the command line, so use a
# non-interactive backend to avoid cache and display issues on CI or remote
# shells.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import imageio.v2 as imageio
from scipy.optimize import minimize
from scipy.spatial import ConvexHull, Delaunay, cKDTree

BOLTZMANN_CONSTANT = 1.380649e-23

MATERIAL_PRESETS = {
    # 5CB is a standard room-temperature nematic material. The literature
    # values below are commonly cited near 25 C: K11 ~ 5.9 pN, K22 ~ 4.5 pN,
    # K33 ~ 9.9 pN, with a one-constant equivalent near 7.9 pN.
    "5CB_room_temperature": {
        "model": "one_constant_frank_oseen",
        "temperature_K": 298.15,
        "temperature_C": 25.0,
        "K1": 5.9e-12,
        "K2": 4.5e-12,
        "K3": 9.9e-12,
        "K_eff": 7.9e-12,
        "q0": 0.0,
    },
    "generic_nematic": {
        "model": "one_constant_frank_oseen",
        "temperature_K": 298.15,
        "temperature_C": 25.0,
        "K1": 6.0e-12,
        "K2": 6.0e-12,
        "K3": 6.0e-12,
        "K_eff": 6.0e-12,
        "q0": 0.0,
    },
    # This keeps the extension point explicit for future chiral materials while
    # still sharing the same one-constant validation path today.
    "generic_cholesteric": {
        "model": "one_constant_frank_oseen",
        "temperature_K": 298.15,
        "temperature_C": 25.0,
        "K1": 6.0e-12,
        "K2": 6.0e-12,
        "K3": 6.0e-12,
        "K_eff": 6.0e-12,
        "q0": 2.0 * np.pi / (0.5e-6),
    },
    # Full anisotropic Oseen-Frank backend. Keep this opt-in so the validated
    # default path remains the simple one-constant model for now.
    "5CB_anisotropic_room_temperature": {
        "model": "anisotropic_oseen_frank",
        "temperature_K": 298.15,
        "temperature_C": 25.0,
        "K1": 5.9e-12,
        "K2": 4.5e-12,
        "K3": 9.9e-12,
        "K_eff": 7.9e-12,
        "q0": 0.0,
    },
}


@dataclass
class BoundaryFace:
    """Triangular boundary face extracted from the mesh hull."""

    nodes: np.ndarray
    area: float
    normal: np.ndarray
    kind: str


@dataclass
class DynamicsResult:
    """Time history returned by the overdamped LC dynamics solver."""

    times: List[float]
    energies: List[float]
    order_parameters: List[float]
    snapshots: List[np.ndarray]
    snapshot_times: List[float]
    angle_histogram_edges: np.ndarray
    angle_histogram_counts: np.ndarray


@dataclass
class MaterialState:
    """Material parameters for a liquid-crystal free-energy model."""

    name: str
    model: str
    temperature_K: float
    temperature_C: float
    K1: float
    K2: float
    K3: float
    K_eff: float
    q0: float


@dataclass
class ReferenceCheckResult:
    """Energy check for a simple analytically interpretable field."""

    name: str
    excess_energy: float
    energy_density: float
    energy_kbt: float
    expected_behavior: str


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
        material_preset: str = "5CB_room_temperature",
        K: float | None = None,
        q0: float | None = None,
        temperature_K: float | None = None,
        W_side: float | None = None,
        W_caps: float | None = None,
        side_mode: str | None = None,
        cap_mode: str | None = None,
        step_size: float = 0.05,
        max_iterations: int = 200,
        tolerance: float = 1.0e-8,
        anchoring_preset: str = "planar_side_homeotropic_caps",
        solver_method: str = "trust-krylov",
        random_seed: int | None = None,
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
        self._configure_material(material_preset=material_preset, K=K, q0=q0, temperature_K=temperature_K)
        self.step_size = step_size
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.anchoring_preset = anchoring_preset
        self.solver_method = solver_method
        self.random_seed = random_seed
        self.rng = np.random.default_rng(random_seed)

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
        self._hessp_step = 1.0e-6
        self.directors = self._initialize_directors()

        self._build_mesh()
        self.domain_volume = float(ConvexHull(self.vertices).volume)
        self._characteristic_length = self._estimate_characteristic_length(self.vertices)
        self._elastic_density_scale = self._compute_elastic_density_scale()
        self.reference_energy = self._reference_energy()
        self.thermal_energy = BOLTZMANN_CONSTANT * self.temperature_K

    def _load_vertices(self, filename: str) -> np.ndarray:
        """Load the grid written by grid1.py and convert microns to meters."""
        data = np.loadtxt(filename)
        if data.ndim == 1:
            data = data[np.newaxis, :]
        vertices = data[:, :3].astype(float)
        vertices *= 1e-6
        return vertices

    def _configure_material(
        self,
        material_preset: str,
        K: float | None,
        q0: float | None,
        temperature_K: float | None,
    ) -> MaterialState:
        """Load physical material constants from a named preset.

        The default preset uses room-temperature 5CB values. Explicit keyword
        overrides are still supported for experiments, but the solver should
        generally start from a real material preset rather than ad hoc numbers.
        """
        if material_preset not in MATERIAL_PRESETS:
            valid = ", ".join(sorted(MATERIAL_PRESETS))
            raise ValueError(f"Unknown material_preset={material_preset!r}. Valid presets: {valid}")

        preset = MATERIAL_PRESETS[material_preset]
        self.material_preset = material_preset
        self.elastic_constants = {
            "K1": preset["K1"],
            "K2": preset["K2"],
            "K3": preset["K3"],
            "K_eff": preset["K_eff"],
        }
        self.temperature_K = preset["temperature_K"] if temperature_K is None else temperature_K
        self.temperature_C = self.temperature_K - 273.15
        self.K = preset["K_eff"] if K is None else K
        self.q0 = preset["q0"] if q0 is None else q0
        self.thermal_energy = BOLTZMANN_CONSTANT * self.temperature_K
        state = MaterialState(
            name=material_preset,
            model=str(preset.get("model", "one_constant_frank_oseen")),
            temperature_K=self.temperature_K,
            temperature_C=self.temperature_C,
            K1=self.elastic_constants["K1"],
            K2=self.elastic_constants["K2"],
            K3=self.elastic_constants["K3"],
            K_eff=self.K,
            q0=self.q0,
        )
        self.material_state = state
        return state

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
        theta = self.rng.uniform(0.0, 2.0 * np.pi, self.n_nodes)
        phi = self.rng.uniform(0.0, np.pi, self.n_nodes)
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

    def _angle_objective(self, theta_phi: np.ndarray) -> Tuple[float, np.ndarray]:
        """Return scaled energy and gradient in angle coordinates."""
        n = self.n_nodes
        theta = theta_phi[:n]
        phi = theta_phi[n:]
        directors = self._angles_to_directors(theta, phi)
        energy, director_grad = self.compute_energy_and_gradient(directors)
        angle_grad = self._angles_gradient(director_grad, theta, phi)
        return energy / self._energy_scale, angle_grad / self._energy_scale

    def _angle_hessp(self, theta_phi: np.ndarray, direction: np.ndarray) -> np.ndarray:
        """Approximate a Hessian-vector product for Newton-Krylov methods.

        We use a centered finite-difference directional derivative of the
        scaled angle-space gradient. This keeps the implementation sparse in
        the optimization sense: the solver only ever sees Hessian-vector
        products, not a dense Hessian matrix.
        """
        step = self._hessp_step
        forward_energy, forward_grad = self._angle_objective(theta_phi + step * direction)
        backward_energy, backward_grad = self._angle_objective(theta_phi - step * direction)
        _ = forward_energy, backward_energy
        return (forward_grad - backward_grad) / (2.0 * step)

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
        self.domain_volume = float(ConvexHull(self.vertices).volume)

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

    @staticmethod
    def _estimate_characteristic_length(vertices: np.ndarray) -> float:
        """Estimate the smallest geometric length scale of the domain."""
        spans = np.ptp(vertices, axis=0)
        positive_spans = spans[spans > 0.0]
        if positive_spans.size == 0:
            return 1.0
        return float(np.min(positive_spans))

    def _compute_elastic_density_scale(self) -> float:
        """Return a rough K/L^2 scale for quick physical validation."""
        length = max(self._characteristic_length, 1.0e-30)
        k_max = max(self.elastic_constants["K1"], self.elastic_constants["K2"], self.elastic_constants["K3"])
        return float(k_max / (length**2))

    def _reference_energy(self) -> float:
        """Return the baseline constant energy for the current material model."""
        if self.material_state.model == "anisotropic_oseen_frank":
            twist_prefactor = self.elastic_constants["K2"]
        else:
            twist_prefactor = self.K
        return 0.5 * twist_prefactor * (self.q0**2) * self.domain_volume

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
            if self.material_state.model == "anisotropic_oseen_frank":
                density, local_grad = self._anisotropic_element_energy_and_gradient(
                    volume=volume,
                    grad_n=grad_n,
                    div_n=div_n,
                    curl_n=curl_n,
                    n_avg=n_avg,
                    g=g,
                )
                total_energy += density
                for local_node in range(4):
                    grad[nodes[local_node]] += local_grad[local_node]
            else:
                density, local_grad = self._one_constant_element_energy_and_gradient(
                    volume=volume,
                    grad_n=grad_n,
                    div_n=div_n,
                    curl_n=curl_n,
                    n_avg=n_avg,
                    g=g,
                )
                total_energy += density
                for local_node in range(4):
                    grad[nodes[local_node]] += local_grad[local_node]

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

    def _one_constant_element_energy_and_gradient(
        self,
        volume: float,
        grad_n: np.ndarray,
        div_n: float,
        curl_n: np.ndarray,
        n_avg: np.ndarray,
        g: np.ndarray,
    ) -> Tuple[float, np.ndarray]:
        """Return the one-constant Frank-Oseen element contribution."""
        density = 0.5 * self.K * (
            div_n**2 + float(np.dot(curl_n, curl_n)) + 2.0 * self.q0 * np.dot(n_avg, curl_n) + self.q0**2
        )

        local_grad = np.zeros((4, 3), dtype=float)
        for local_node in range(4):
            g_local = g[local_node]
            div_grad = np.array([g_local[0], g_local[1], g_local[2]])
            curl_jac = np.array(
                [
                    [0.0, -g_local[2], g_local[1]],
                    [g_local[2], 0.0, -g_local[0]],
                    [-g_local[1], g_local[0], 0.0],
                ]
            )
            local_grad[local_node] = self.K * volume * (
                div_n * div_grad
                + curl_jac.T @ curl_n
                + self.q0 * (0.25 * curl_n + curl_jac.T @ n_avg)
            )

        return density * volume, local_grad

    def _anisotropic_element_energy_and_gradient(
        self,
        volume: float,
        grad_n: np.ndarray,
        div_n: float,
        curl_n: np.ndarray,
        n_avg: np.ndarray,
        g: np.ndarray,
    ) -> Tuple[float, np.ndarray]:
        """Return the full anisotropic Oseen-Frank element contribution."""
        K1 = self.elastic_constants["K1"]
        K2 = self.elastic_constants["K2"]
        K3 = self.elastic_constants["K3"]

        twist = float(np.dot(n_avg, curl_n) + self.q0)
        bend = np.cross(n_avg, curl_n)
        density = 0.5 * (K1 * div_n**2 + K2 * twist**2 + K3 * float(np.dot(bend, bend)))

        local_grad = np.zeros((4, 3), dtype=float)
        for local_node in range(4):
            g_local = g[local_node]
            div_grad = np.array([g_local[0], g_local[1], g_local[2]])
            curl_jac = np.array(
                [
                    [0.0, -g_local[2], g_local[1]],
                    [g_local[2], 0.0, -g_local[0]],
                    [-g_local[1], g_local[0], 0.0],
                ]
            )
            twist_grad = 0.25 * curl_n + curl_jac.T @ n_avg
            bend_grad = 0.25 * np.cross(curl_n, bend) + curl_jac.T @ np.cross(bend, n_avg)
            local_grad[local_node] = volume * (K1 * div_n * div_grad + K2 * twist * twist_grad + K3 * bend_grad)

        return density * volume, local_grad

    def project_gradient(self, directors: np.ndarray, grad: np.ndarray) -> np.ndarray:
        """Project the gradient onto the tangent plane of the unit sphere."""
        radial = np.sum(grad * directors, axis=1, keepdims=True)
        return grad - radial * directors

    def relax(self) -> List[float]:
        """Relax the director field in angle space.

        The default path uses a Newton-Krylov style solve with Hessian-vector
        products so we can keep the optimization lightweight without forming a
        dense Hessian. L-BFGS remains available as a fallback for comparison.
        """
        theta_phi0 = self._directors_to_angles(self.directors)
        initial_energy, _ = self.compute_energy_and_gradient(self.directors)
        energies: List[float] = [self._excess_free_energy(initial_energy)]
        n = self.n_nodes
        method = self.solver_method.lower()
        if method not in {"lbfgs", "l-bfgs-b", "newton-cg", "trust-krylov"}:
            raise ValueError(
                f"Unknown solver_method={self.solver_method!r}. "
                "Expected lbfgs, newton-cg, or trust-krylov."
            )

        def callback(theta_phi: np.ndarray) -> None:
            energy, _ = self._angle_objective(theta_phi)
            energy = energy * self._energy_scale
            energies.append(self._excess_free_energy(energy))

        if method in {"lbfgs", "l-bfgs-b"}:
            result = minimize(
                self._angle_objective,
                theta_phi0,
                method="L-BFGS-B",
                bounds=[(None, None)] * (2 * n),
                jac=True,
                callback=callback,
                options={
                    "maxiter": self.max_iterations,
                    "ftol": self.tolerance,
                    "gtol": self.tolerance,
                    "maxls": 50,
                },
            )
        else:
            solver_options = {
                "maxiter": self.max_iterations,
                "disp": False,
            }
            if method == "newton-cg":
                # Newton-CG uses `xtol` rather than `gtol`.
                solver_options["xtol"] = self.tolerance
            else:
                solver_options["gtol"] = self.tolerance

            result = minimize(
                self._angle_objective,
                theta_phi0,
                method=method,
                hessp=self._angle_hessp,
                jac=True,
                callback=callback,
                options=solver_options,
            )

        final_theta = result.x[:n]
        final_phi = result.x[n:]
        self.directors = self._angles_to_directors(final_theta, final_phi)

        final_energy = self._excess_free_energy(float(result.fun * self._energy_scale))
        if not energies or abs(energies[-1] - final_energy) > 0.0:
            energies.append(final_energy)

        return energies

    def _compute_order_parameter(self, directors: np.ndarray) -> float:
        """Return a simple scalar order metric along the cylinder axis.

        This is not a full nematic tensor diagnostic, but it is a useful
        observable for watching the field evolve in time.
        """
        dirs = self.normalize_directors(directors)
        return float(0.5 * (3.0 * np.mean(dirs[:, 2] ** 2) - 1.0))

    def _excess_free_energy(self, raw_energy: float) -> float:
        """Return the reported free energy with the constant baseline removed."""
        return float(raw_energy - self.reference_energy)

    def _energy_density(self, excess_energy: float) -> float:
        """Return the excess energy density in J/m^3."""
        if self.domain_volume <= 0.0:
            return float("nan")
        return float(excess_energy / self.domain_volume)

    def _energy_in_kbt(self, excess_energy: float) -> float:
        """Return the excess free energy in thermal units."""
        if self.thermal_energy <= 0.0:
            return float("nan")
        return float(excess_energy / self.thermal_energy)

    def energy_summary(self, excess_energy: float) -> str:
        """Return a compact physical interpretation of a free-energy value."""
        density = self._energy_density(excess_energy)
        scale_ratio = density / self._elastic_density_scale if self._elastic_density_scale > 0.0 else float("nan")
        return (
            f"{density:.3e} J/m^3 "
            f"({excess_energy:.6e} J, "
            f"{self._energy_in_kbt(excess_energy):.3e} kBT, "
            f"{scale_ratio:.3e} x K/L^2)"
        )

    @staticmethod
    def _director_angles_degrees(directors: np.ndarray) -> np.ndarray:
        """Return polar angles relative to the cylinder axis in degrees."""
        dirs = LiquidCrystalFEMSolver.normalize_directors(directors)
        cos_theta = np.clip(dirs[:, 2], -1.0, 1.0)
        return np.degrees(np.arccos(cos_theta))

    def _angle_histogram(self, directors: np.ndarray, bins: int) -> Tuple[np.ndarray, np.ndarray]:
        """Compute a director-angle histogram for a snapshot."""
        angles = self._director_angles_degrees(directors)
        return np.histogram(angles, bins=bins, range=(0.0, 180.0))

    def simulate_dynamics(
        self,
        total_time: float = 10.0,
        time_step: float = 0.05,
        mobility: float = 1.0,
        thermal_noise_strength: float = 0.01,
        snapshot_interval: int = 10,
        histogram_bins: int = 30,
        max_steps: int | None = None,
        random_seed: int | None = None,
    ) -> DynamicsResult:
        """Evolve the director field with overdamped gradient-flow dynamics.

        The update is a model-A style relaxational dynamics on the director
        manifold:

            d n / d t = -M P_n (δF/δn)

        where `P_n` projects the force into the tangent plane of the unit
        sphere. Time here is a model time, not a calibrated experimental time.
        A small thermal_noise_strength adds stochastic tangential forcing so we
        can observe fluctuation-driven motion around the relaxational drift.
        """
        directors = self.directors.copy()
        rng = np.random.default_rng(random_seed) if random_seed is not None else self.rng
        total_steps = max_steps if max_steps is not None else int(np.ceil(total_time / time_step))
        total_steps = max(total_steps, 1)

        times: List[float] = []
        energies: List[float] = []
        order_parameters: List[float] = []
        snapshots: List[np.ndarray] = []
        snapshot_times: List[float] = []
        histogram_counts: List[np.ndarray] = []
        histogram_edges: np.ndarray | None = None

        current_time = 0.0
        step = 0
        while step <= total_steps:
            energy, grad = self.compute_energy_and_gradient(directors)
            times.append(current_time)
            energies.append(self._excess_free_energy(float(energy)))
            order_parameters.append(self._compute_order_parameter(directors))

            if step % max(snapshot_interval, 1) == 0:
                snapshot = directors.copy()
                snapshots.append(snapshot)
                snapshot_times.append(current_time)
                counts, edges = self._angle_histogram(snapshot, bins=histogram_bins)
                histogram_counts.append(counts.astype(float))
                histogram_edges = edges

            if step == total_steps:
                break

            projected = self.project_gradient(directors, grad) / self._energy_scale

            if thermal_noise_strength > 0.0:
                # Tangential Gaussian noise approximates a stochastic LC bath.
                noise = self.project_gradient(directors, rng.normal(size=directors.shape))
                noise = noise / max(np.linalg.norm(noise) / np.sqrt(self.n_nodes), 1.0e-30)
                stochastic_term = thermal_noise_strength * np.sqrt(time_step) * noise
                trial = self.normalize_directors(
                    directors - mobility * time_step * projected + stochastic_term
                )
                trial_energy, _ = self.compute_energy_and_gradient(trial)
                if not np.isfinite(trial_energy):
                    break
                directors = trial
                current_time += time_step
            else:
                trial_step = time_step
                accepted = False

                # Backtracking keeps the explicit dynamics stable on stiff meshes.
                while trial_step >= 1.0e-8:
                    trial = self.normalize_directors(directors - mobility * trial_step * projected)
                    trial_energy, _ = self.compute_energy_and_gradient(trial)
                    if trial_energy <= energy:
                        directors = trial
                        current_time += trial_step
                        accepted = True
                        break
                    trial_step *= 0.5

                if not accepted:
                    break

            step += 1

        self.directors = directors
        if histogram_edges is None:
            histogram_edges = np.linspace(0.0, 180.0, histogram_bins + 1)
        if histogram_counts:
            histogram_array = np.vstack(histogram_counts)
        else:
            histogram_array = np.empty((0, histogram_bins), dtype=float)
        return DynamicsResult(
            times,
            energies,
            order_parameters,
            snapshots,
            snapshot_times,
            histogram_edges,
            histogram_array,
        )

    def save_dynamics_trace(self, result: DynamicsResult, filename: str) -> None:
        """Save a compact CSV trace of the dynamic run."""
        with open(filename, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["time", "excess_energy_j", "energy_density_j_m3", "energy_kbt", "order_parameter"])
            for t, e, s in zip(result.times, result.energies, result.order_parameters):
                writer.writerow(
                    [
                        f"{t:.8e}",
                        f"{e:.8e}",
                        f"{self._energy_density(e):.8e}",
                        f"{self._energy_in_kbt(e):.8e}",
                        f"{s:.8e}",
                    ]
                )

    def save_dynamics_histogram_data(self, result: DynamicsResult, filename: str) -> None:
        """Save per-snapshot director-angle histogram counts."""
        if result.angle_histogram_counts.size == 0:
            return

        edges = result.angle_histogram_edges
        with open(filename, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(
                [
                    "snapshot_index",
                    "snapshot_time",
                    "angle_bin_start_deg",
                    "angle_bin_end_deg",
                    "count",
                ]
            )
            for snapshot_index, (snapshot_time, counts) in enumerate(
                zip(result.snapshot_times, result.angle_histogram_counts)
            ):
                for bin_index, count in enumerate(counts):
                    writer.writerow(
                        [
                            snapshot_index,
                            f"{snapshot_time:.8e}",
                            f"{edges[bin_index]:.8e}",
                            f"{edges[bin_index + 1]:.8e}",
                            f"{count:.8e}",
                        ]
                    )

    def plot_dynamics_trace(self, result: DynamicsResult, filename: str = "dynamics_trace.jpg") -> None:
        """Plot excess energy density and the scalar order parameter over time."""
        if not result.times:
            return

        fig, ax1 = plt.subplots(figsize=(10, 6))
        densities = [self._energy_density(energy) for energy in result.energies]
        ax1.plot(result.times, densities, color="tab:blue", label="Excess Energy Density")
        ax1.set_xlabel("Model Time")
        ax1.set_ylabel("Excess f (J/m^3)", color="tab:blue")
        ax1.tick_params(axis="y", labelcolor="tab:blue")

        ax2 = ax1.twinx()
        ax2.plot(result.times, result.order_parameters, color="tab:orange", label="Order Parameter")
        ax2.set_ylabel("Szz", color="tab:orange")
        ax2.tick_params(axis="y", labelcolor="tab:orange")

        fig.suptitle("Liquid Crystal Dynamics Trace")
        fig.tight_layout()
        fig.savefig(filename, dpi=300)
        plt.close(fig)

    def plot_dynamics_histogram_heatmap(
        self, result: DynamicsResult, filename: str = "dynamics_histograms.jpg"
    ) -> None:
        """Plot a time-resolved director-angle histogram heatmap."""
        if result.angle_histogram_counts.size == 0:
            return

        angle_edges = result.angle_histogram_edges
        times = np.asarray(result.snapshot_times, dtype=float)
        counts = result.angle_histogram_counts
        if times.size == 1:
            time_min = times[0] - 0.5
            time_max = times[0] + 0.5
        else:
            time_min = times[0]
            time_max = times[-1]
        fig, ax = plt.subplots(figsize=(10, 6))
        image = ax.imshow(
            counts,
            aspect="auto",
            origin="lower",
            interpolation="nearest",
            extent=[angle_edges[0], angle_edges[-1], time_min, time_max],
            cmap="viridis",
        )
        ax.set_xlabel("Director angle relative to z-axis (degrees)")
        ax.set_ylabel("Snapshot time")
        ax.set_title("Director Angle Histograms by Snapshot")
        colorbar = fig.colorbar(image, ax=ax)
        colorbar.set_label("Count")
        fig.tight_layout()
        fig.savefig(filename, dpi=300)
        plt.close(fig)

    def save_dynamics_movie(self, result: DynamicsResult, filename: str = "dynamics.gif") -> None:
        """Write a simple GIF of the director evolution."""
        if not result.snapshots:
            return

        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection="3d")
        frames = []

        for snapshot, t in zip(result.snapshots, result.snapshot_times):
            old_directors = self.directors
            self.directors = snapshot
            self.plot_director_field(ax, title=f"Relaxation Dynamics t={t:.3f}")
            fig.canvas.draw()
            width, height = fig.canvas.get_width_height()
            frame = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(height, width, 4)
            frames.append(frame.copy())
            self.directors = old_directors

        imageio.mimsave(filename, frames, duration=0.25)
        plt.close(fig)

    def save_dynamics_bundle(self, result: DynamicsResult, output_prefix: str) -> None:
        """Write the standard dynamics outputs plus histogram diagnostics."""
        self.save_dynamics_trace(result, filename=f"{output_prefix}_trace.csv")
        self.plot_dynamics_trace(result, filename=f"{output_prefix}_trace.jpg")
        self.save_dynamics_histogram_data(result, filename=f"{output_prefix}_histograms.csv")
        self.plot_dynamics_histogram_heatmap(result, filename=f"{output_prefix}_histograms.jpg")
        self.save_dynamics_movie(result, filename=f"{output_prefix}_movie.gif")

    def validate_reference_cases(self) -> List[ReferenceCheckResult]:
        """Evaluate a few simple director fields that should be easy to interpret.

        These checks are useful when validating the implementation: the free
        and planar-side/homeotropic-cap cases should be very close to zero for
        a uniform z-directed field, while an incompatible anchoring choice
        should produce a positive excess energy density.
        """
        cases: List[ReferenceCheckResult] = []
        uniform_z = np.tile(np.array([0.0, 0.0, 1.0]), (self.n_nodes, 1))
        original_directors = self.directors.copy()
        original_side_mode = self.side_mode
        original_cap_mode = self.cap_mode
        original_W_side = self.W_side
        original_W_caps = self.W_caps

        try:
            for name, side_mode, cap_mode, W_side, W_caps, expected in [
                (
                    "free_uniform_z",
                    "free",
                    "free",
                    0.0,
                    0.0,
                    "bulk energy should be ~0 for a uniform director with free boundaries",
                ),
                (
                    "planar_side_homeotropic_caps_uniform_z",
                    "planar",
                    "homeotropic",
                    self.W_side,
                    self.W_caps,
                    "bulk and surface energy should be ~0 for the validated default boundary choice",
                ),
                (
                    "homeotropic_all_uniform_z",
                    "homeotropic",
                    "homeotropic",
                    self.W_side,
                    self.W_caps,
                    "energy should be positive because the sidewall is incompatible with uniform z alignment",
                ),
            ]:
                self.side_mode = side_mode
                self.cap_mode = cap_mode
                self.W_side = W_side
                self.W_caps = W_caps
                self.directors = uniform_z.copy()
                energy, _ = self.compute_energy_and_gradient(self.directors)
                excess = self._excess_free_energy(float(energy))
                cases.append(
                    ReferenceCheckResult(
                        name=name,
                        excess_energy=excess,
                        energy_density=self._energy_density(excess),
                        energy_kbt=self._energy_in_kbt(excess),
                        expected_behavior=expected,
                    )
                )
        finally:
            self.directors = original_directors
            self.side_mode = original_side_mode
            self.cap_mode = original_cap_mode
            self.W_side = original_W_side
            self.W_caps = original_W_caps

        return cases

    def benchmark_uniform_directors(self) -> List[ReferenceCheckResult]:
        """Benchmark the bulk model on uniform director states.

        Uniform fields with free boundaries should be at or very near zero
        excess energy for the default 5CB and anisotropic 5CB presets. That
        makes this a good smoke test for the material backend itself.
        """
        original_directors = self.directors.copy()
        original_side_mode = self.side_mode
        original_cap_mode = self.cap_mode
        original_W_side = self.W_side
        original_W_caps = self.W_caps

        fields = {
            "uniform_x": np.tile(np.array([1.0, 0.0, 0.0]), (self.n_nodes, 1)),
            "uniform_y": np.tile(np.array([0.0, 1.0, 0.0]), (self.n_nodes, 1)),
            "uniform_z": np.tile(np.array([0.0, 0.0, 1.0]), (self.n_nodes, 1)),
        }
        results: List[ReferenceCheckResult] = []

        try:
            self.side_mode = "free"
            self.cap_mode = "free"
            self.W_side = 0.0
            self.W_caps = 0.0

            for name, field in fields.items():
                self.directors = field.copy()
                energy, _ = self.compute_energy_and_gradient(self.directors)
                excess = self._excess_free_energy(float(energy))
                results.append(
                    ReferenceCheckResult(
                        name=name,
                        excess_energy=excess,
                        energy_density=self._energy_density(excess),
                        energy_kbt=self._energy_in_kbt(excess),
                        expected_behavior="uniform director with free boundaries should be near zero excess energy",
                    )
                )
        finally:
            self.directors = original_directors
            self.side_mode = original_side_mode
            self.cap_mode = original_cap_mode
            self.W_side = original_W_side
            self.W_caps = original_W_caps

        return results

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
        self._characteristic_length = self._estimate_characteristic_length(self.vertices)
        self._elastic_density_scale = self._compute_elastic_density_scale()
        self.reference_energy = self._reference_energy()

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
        densities = [self._energy_density(energy) for energy in energies]
        plt.plot(densities, label="Excess Energy Density")
        plt.xlabel("Iteration")
        plt.ylabel("Excess f (J/m^3)")
        plt.title("Excess Energy Density Per Iteration")
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
    material_preset: str = "5CB_room_temperature",
    anchoring_preset: str = "planar_side_homeotropic_caps",
    sidewall_anchoring_strength: float | None = None,
    cap_anchoring_strength: float | None = None,
    solver_method: str = "trust-krylov",
    random_seed: int | None = None,
) -> Tuple[LiquidCrystalFEMSolver, List[float]]:
    """Convenience wrapper for the main program."""
    # Apply Rapini-Papoular anchoring on the full closed cylinder boundary by
    # default. The preset can be swapped for other patch-wise boundary setups.
    solver = LiquidCrystalFEMSolver(
        coordinates_file,
        max_iterations=run_time,
        material_preset=material_preset,
        anchoring_preset=anchoring_preset,
        W_side=sidewall_anchoring_strength,
        W_caps=cap_anchoring_strength,
        solver_method=solver_method,
        random_seed=random_seed,
    )
    if checkpoint_file:
        solver.load_director_field(checkpoint_file)

    energies = solver.relax()
    solver.save_director_field(f"{output_prefix}_optimized_director_field.txt")
    solver.plot_energy(energies, filename=f"{output_prefix}_energy.jpg")
    solver.plot_angle_histogram(filename=f"{output_prefix}_histogram.jpg")
    return solver, energies


def run_fem_dynamics(
    coordinates_file: str,
    checkpoint_file: str | None = None,
    total_time: float = 10.0,
    time_step: float = 0.05,
    mobility: float = 1.0,
    thermal_noise_strength: float = 0.01,
    snapshot_interval: int = 10,
    histogram_bins: int = 30,
    output_prefix: str = "fem_dynamics",
    material_preset: str = "5CB_room_temperature",
    anchoring_preset: str = "planar_side_homeotropic_caps",
    sidewall_anchoring_strength: float | None = None,
    cap_anchoring_strength: float | None = None,
    solver_method: str = "trust-krylov",
    random_seed: int | None = None,
) -> Tuple[LiquidCrystalFEMSolver, DynamicsResult]:
    """Run overdamped LC dynamics and write trace/visualization outputs."""
    solver = LiquidCrystalFEMSolver(
        coordinates_file,
        max_iterations=1,
        material_preset=material_preset,
        anchoring_preset=anchoring_preset,
        W_side=sidewall_anchoring_strength,
        W_caps=cap_anchoring_strength,
        solver_method=solver_method,
        random_seed=random_seed,
    )
    if checkpoint_file:
        solver.load_director_field(checkpoint_file)

    result = solver.simulate_dynamics(
        total_time=total_time,
        time_step=time_step,
        mobility=mobility,
        thermal_noise_strength=thermal_noise_strength,
        snapshot_interval=snapshot_interval,
        histogram_bins=histogram_bins,
        random_seed=random_seed,
    )
    solver.save_director_field(f"{output_prefix}_final_director_field.txt")
    solver.save_dynamics_bundle(result, output_prefix=output_prefix)
    return solver, result
