import copy
import json
import os
import csv
from dataclasses import dataclass
from typing import List, Sequence, Tuple

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
from grid1 import CylinderGrid

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
    principal_order_parameters: List[float]
    mean_angle_degrees: List[float]
    p90_angle_degrees: List[float]
    mean_temperature_K: List[float]
    min_temperature_K: List[float]
    max_temperature_K: List[float]
    snapshots: List[np.ndarray]
    temperature_snapshots: List[np.ndarray]
    snapshot_times: List[float]
    angle_histogram_edges: np.ndarray
    angle_histogram_counts: np.ndarray


@dataclass
class TemperatureSweepResult:
    """Equilibrium observables collected across a temperature sweep."""

    temperatures_K: List[float]
    excess_energies: List[float]
    energy_densities: List[float]
    energy_kbt: List[float]
    order_parameters: List[float]
    principal_order_parameters: List[float]
    mean_angle_degrees: List[float]
    p90_angle_degrees: List[float]
    bulk_softening_scales: List[float]
    sidewall_anchoring_scales: List[float]
    cap_anchoring_scales: List[float]
    effective_K1_pN: List[float]
    effective_K2_pN: List[float]
    effective_K3_pN: List[float]
    effective_Keff_pN: List[float]
    mean_temperature_K: List[float]
    min_temperature_K: List[float]
    max_temperature_K: List[float]
    temperature_snapshots: List[np.ndarray]
    endpoint_order_shift: float
    snapshots: List[np.ndarray]


@dataclass
class TemperatureHysteresisResult:
    """Paired heating/cooling observables for a hysteresis scan."""

    heating: TemperatureSweepResult
    cooling: TemperatureSweepResult
    temperatures_K: List[float]
    energy_density_gap: List[float]
    order_parameter_gap: List[float]
    principal_order_gap: List[float]
    midpoint_temperature_K: float
    midpoint_energy_density_gap: float
    midpoint_order_parameter_gap: float
    midpoint_principal_order_gap: float
    midpoint_loop_width: float
    max_energy_density_gap: float
    max_order_parameter_gap: float
    max_principal_order_gap: float


@dataclass
class RefinementIndicatorResult:
    """Per-element refinement hints for the structured solve mesh."""

    centroids_m: np.ndarray
    scores: np.ndarray
    director_scores: np.ndarray
    temperature_scores: np.ndarray


@dataclass
class MeshRefinementSuggestion:
    """Suggested follow-up mesh parameters for the next structured solve."""

    diameter_um: float
    length_um: float
    num_radial_layers: int
    num_theta_points: int
    num_axial_layers: int
    radial_cluster_power: float
    axial_cluster_power: float
    wall_hot_fraction: float
    cap_hot_fraction: float
    max_refinement_score: float
    mean_refinement_score: float


@dataclass
class MeshRecipe:
    """Current structured mesh recipe used as the refinement baseline."""

    num_radial_layers: int
    num_theta_points: int
    num_axial_layers: int
    radial_cluster_power: float
    axial_cluster_power: float


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
class TemperatureProfile:
    """Spatial and temporal temperature law used by the FEM solver.

    The profile is intentionally simple enough to validate:
    - uniform: constant temperature everywhere
    - linear_z / linear_r: static axial or radial gradients
    - oscillatory_z: a small axial oscillation around the base temperature

    The `law` controls how strongly the temperature reshapes the material
    response. The default `linear` law behaves like a small softening law.
    The `ldg_proxy` law mimics a Landau-de Gennes style order-parameter
    softening with a temperature-dependent reduction factor. It remains a
    phenomenological softening law rather than a full Q-tensor model.
    """

    enabled: bool
    mode: str
    law: str
    base_temperature_K: float
    reference_temperature_K: float
    transition_temperature_K: float
    temperature_gradient_z_K_per_m: float
    temperature_gradient_r_K_per_m: float
    oscillation_amplitude_K: float
    oscillation_frequency_hz: float
    phase_rad: float
    order_floor: float
    elastic_alpha_per_K: float
    anchoring_alpha_per_K: float
    sidewall_anchoring_alpha_per_K: float
    cap_anchoring_alpha_per_K: float
    mobility_gamma_per_K: float
    noise_multiplier: float


@dataclass
class ReferenceCheckResult:
    """Energy check for a simple analytically interpretable field."""

    name: str
    excess_energy: float
    energy_density: float
    energy_kbt: float
    expected_behavior: str
    reference_energy_density: float | None = None
    density_delta: float | None = None


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
        mesh_file: str | None = None,
        material_preset: str = "5CB_room_temperature",
        K: float | None = None,
        q0: float | None = None,
        temperature_K: float | None = None,
        temperature_profile: dict | None = None,
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
        self.mesh_file = mesh_file
        self._configure_material(material_preset=material_preset, K=K, q0=q0, temperature_K=temperature_K)
        self._configure_temperature(temperature_profile=temperature_profile, temperature_K=temperature_K)
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

        self._mesh_loaded_from_file = False
        self._load_geometry(coordinates_file, mesh_file=mesh_file)
        # The raw energy and gradient are expressed in SI units and are
        # therefore numerically tiny; scale the optimization problem by a
        # characteristic elastic energy so the solver sees a healthy magnitude.
        self._hessp_step = 1.0e-6
        self.directors = self._initialize_directors()
        self._build_mesh()
        self.domain_volume = float(ConvexHull(self.vertices).volume)
        self._characteristic_length = self._estimate_characteristic_length(self.vertices)
        self._elastic_density_scale = self._compute_elastic_density_scale()
        self.reference_energy = self._reference_energy()
        self.thermal_energy = BOLTZMANN_CONSTANT * self.temperature_K

    def _load_geometry(self, coordinates_file: str, mesh_file: str | None = None) -> None:
        """Load either a structured solve mesh or the legacy point cloud."""
        if mesh_file and os.path.exists(mesh_file):
            mesh_data = np.load(mesh_file, allow_pickle=False)
            if "nodes_m" in mesh_data:
                vertices = np.asarray(mesh_data["nodes_m"], dtype=float)
            elif "nodes_um" in mesh_data:
                vertices = np.asarray(mesh_data["nodes_um"], dtype=float) * 1.0e-6
            else:
                raise ValueError(f"Mesh file {mesh_file!r} does not contain nodes_m or nodes_um.")

            self.vertices = vertices
            self.n_nodes = len(self.vertices)
            self._spacing = self._estimate_spacing(self.vertices)
            self._volume_tol = max((self._spacing ** 3) * 1.0e-8, 1.0e-30)
            self._energy_scale = max(self.K * self._spacing, 1.0e-30)
            self._gradient_scale = self._energy_scale
            self._mesh_tetrahedra = np.asarray(mesh_data["tetrahedra"], dtype=int) if "tetrahedra" in mesh_data else None
            self._mesh_boundary_faces = (
                np.asarray(mesh_data["boundary_faces"], dtype=int) if "boundary_faces" in mesh_data else None
            )
            if "boundary_face_kind" in mesh_data:
                self._mesh_boundary_face_kind = np.asarray(mesh_data["boundary_face_kind"]).astype(str)
            else:
                self._mesh_boundary_face_kind = None
            self._mesh_metadata = {}
            if "metadata_json" in mesh_data:
                try:
                    metadata_raw = mesh_data["metadata_json"].item() if hasattr(mesh_data["metadata_json"], "item") else mesh_data["metadata_json"]
                    self._mesh_metadata = json.loads(str(metadata_raw))
                except Exception:
                    self._mesh_metadata = {}
            self._mesh_loaded_from_file = True
            return

        data = np.loadtxt(coordinates_file)
        if data.ndim == 1:
            data = data[np.newaxis, :]
        vertices = data[:, :3].astype(float)
        vertices *= 1e-6
        self.vertices = vertices
        self.n_nodes = len(self.vertices)
        self._spacing = self._estimate_spacing(self.vertices)
        self._volume_tol = max((self._spacing ** 3) * 1.0e-8, 1.0e-30)
        self._energy_scale = max(self.K * self._spacing, 1.0e-30)
        self._gradient_scale = self._energy_scale
        self._mesh_tetrahedra = None
        self._mesh_boundary_faces = None
        self._mesh_boundary_face_kind = None
        self._mesh_metadata = {}

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

    def _configure_temperature(self, temperature_profile: dict | None, temperature_K: float | None) -> None:
        """Resolve the thermal profile used to soften couplings and drive noise."""
        profile = {
            "enabled": False,
            "mode": "uniform",
            "law": "linear",
            "base_temperature_K": self.temperature_K if temperature_K is None else temperature_K,
            "reference_temperature_K": self.temperature_K if temperature_K is None else temperature_K,
            "transition_temperature_K": max(self.temperature_K + 8.0, 300.0),
            "temperature_gradient_z_K_per_m": 0.0,
            "temperature_gradient_r_K_per_m": 0.0,
            "oscillation_amplitude_K": 0.0,
            "oscillation_frequency_hz": 0.0,
            "phase_rad": 0.0,
            "order_floor": 0.2,
            "elastic_alpha_per_K": 0.015,
            "anchoring_alpha_per_K": 0.02,
            "sidewall_anchoring_alpha_per_K": 0.025,
            "cap_anchoring_alpha_per_K": 0.01,
            "mobility_gamma_per_K": 0.01,
            "noise_multiplier": 1.0,
        }
        if temperature_profile:
            profile.update(temperature_profile)

        gradient_z = profile.get("gradient_z_K_per_m", profile.get("temperature_gradient_z_K_per_m", 0.0))
        gradient_r = profile.get("gradient_r_K_per_m", profile.get("temperature_gradient_r_K_per_m", 0.0))

        self.temperature_profile = TemperatureProfile(
            enabled=bool(profile["enabled"]),
            mode=str(profile["mode"]),
            law=str(profile["law"]),
            base_temperature_K=float(profile["base_temperature_K"]),
            reference_temperature_K=float(profile["reference_temperature_K"]),
            transition_temperature_K=float(profile["transition_temperature_K"]),
            temperature_gradient_z_K_per_m=float(gradient_z),
            temperature_gradient_r_K_per_m=float(gradient_r),
            oscillation_amplitude_K=float(profile["oscillation_amplitude_K"]),
            oscillation_frequency_hz=float(profile["oscillation_frequency_hz"]),
            phase_rad=float(profile["phase_rad"]),
            order_floor=float(profile["order_floor"]),
            elastic_alpha_per_K=float(profile["elastic_alpha_per_K"]),
            anchoring_alpha_per_K=float(profile["anchoring_alpha_per_K"]),
            sidewall_anchoring_alpha_per_K=float(profile["sidewall_anchoring_alpha_per_K"]),
            cap_anchoring_alpha_per_K=float(profile["cap_anchoring_alpha_per_K"]),
            mobility_gamma_per_K=float(profile["mobility_gamma_per_K"]),
            noise_multiplier=float(profile["noise_multiplier"]),
        )

        if self.temperature_profile.enabled:
            self.temperature_K = self.temperature_profile.base_temperature_K
            self.temperature_C = self.temperature_K - 273.15
            self.thermal_energy = BOLTZMANN_CONSTANT * self.temperature_K
            self.material_state.temperature_K = self.temperature_K
            self.material_state.temperature_C = self.temperature_C

    def _temperature_field(self, time: float = 0.0) -> np.ndarray:
        """Return the nodal temperature field for the current thermal profile."""
        if not self.temperature_profile.enabled:
            return np.full(self.n_nodes, self.temperature_profile.base_temperature_K, dtype=float)

        vertices = self.vertices
        profile = self.temperature_profile
        temperatures = np.full(self.n_nodes, profile.base_temperature_K, dtype=float)

        if profile.mode in {"linear_z", "linear_z_plus_radial", "oscillatory_z"}:
            z_ref = float(np.mean(vertices[:, 2]))
            temperatures += profile.temperature_gradient_z_K_per_m * (vertices[:, 2] - z_ref)

        if profile.mode in {"linear_r", "linear_z_plus_radial"}:
            xy = vertices[:, :2]
            radii = np.linalg.norm(xy, axis=1)
            r_ref = float(np.mean(radii))
            temperatures += profile.temperature_gradient_r_K_per_m * (radii - r_ref)

        if profile.mode == "oscillatory_z":
            temperatures += profile.oscillation_amplitude_K * np.sin(
                2.0 * np.pi * profile.oscillation_frequency_hz * time + profile.phase_rad
            )

        return temperatures

    def _temperature_order_factor(self, temperature: np.ndarray | float) -> np.ndarray:
        """Return a smooth order-factor used to soften temperature couplings."""
        profile = self.temperature_profile
        temperatures = np.asarray(temperature, dtype=float)

        if profile.law == "ldg_proxy":
            tc = max(profile.transition_temperature_K, profile.reference_temperature_K + 1.0e-9)
            denominator = max(tc - profile.reference_temperature_K, 1.0e-9)
            factor = np.sqrt(np.maximum(tc - temperatures, 0.0) / denominator)
        else:
            factor = 1.0 - profile.elastic_alpha_per_K * (temperatures - profile.reference_temperature_K)

        return np.clip(factor, profile.order_floor, 2.0)

    def _elastic_temperature_scale(self, temperature: np.ndarray | float) -> np.ndarray:
        """Return the multiplicative scale applied to bulk elastic constants."""
        order_factor = self._temperature_order_factor(temperature)
        return np.asarray(order_factor, dtype=float) ** 2

    def _anchoring_temperature_scale(self, temperature: np.ndarray | float, kind: str = "sidewall") -> np.ndarray:
        """Return the multiplicative scale applied to Rapini-Papoular anchoring."""
        order_factor = self._temperature_order_factor(temperature)
        profile = self.temperature_profile
        if kind == "sidewall":
            alpha = profile.sidewall_anchoring_alpha_per_K
        elif kind in {"cap", "top_cap", "bottom_cap"}:
            alpha = profile.cap_anchoring_alpha_per_K
        else:
            alpha = profile.anchoring_alpha_per_K
        # Allow anchoring to soften a bit faster than the bulk elastic response.
        return np.clip(order_factor ** (1.0 + alpha), profile.order_floor, 2.0)

    def _mobility_temperature_scale(self, temperature: np.ndarray | float) -> np.ndarray:
        """Return a temperature-dependent mobility factor."""
        profile = self.temperature_profile
        temperatures = np.asarray(temperature, dtype=float)
        relative = temperatures - profile.reference_temperature_K
        order_factor = self._temperature_order_factor(temperatures)
        return np.exp(profile.mobility_gamma_per_K * relative) / np.maximum(order_factor, profile.order_floor)

    def _thermal_noise_scale(self, temperature: np.ndarray | float) -> np.ndarray:
        """Return the fluctuation amplitude implied by the local temperature."""
        profile = self.temperature_profile
        temperatures = np.asarray(temperature, dtype=float)
        order_factor = self._temperature_order_factor(temperatures)
        physical_scale = np.sqrt(2.0 * BOLTZMANN_CONSTANT * temperatures / max(self._energy_scale, 1.0e-30))
        return profile.noise_multiplier * physical_scale / np.maximum(order_factor, profile.order_floor)

    def _effective_elastic_constants(self, temperature: np.ndarray | float) -> Tuple[float, float, float, float]:
        """Return mean temperature-softened elastic constants in pN for plotting."""
        scale = float(np.mean(self._elastic_temperature_scale(temperature)))
        return (
            self.elastic_constants["K1"] * scale * 1.0e12,
            self.elastic_constants["K2"] * scale * 1.0e12,
            self.elastic_constants["K3"] * scale * 1.0e12,
            self.K * scale * 1.0e12,
        )

    def _temperature_scale_summary(self, temperature_field: np.ndarray) -> Tuple[float, float, float]:
        """Return average bulk and boundary temperature scales for diagnostics."""
        bulk_scale = float(np.mean(self._elastic_temperature_scale(temperature_field)))

        sidewall_scales: List[float] = []
        cap_scales: List[float] = []
        for face in self.boundary_faces:
            local_temperature = float(np.mean(temperature_field[face.nodes]))
            anchoring_scale = float(self._anchoring_temperature_scale(local_temperature, kind=face.kind))
            if face.kind == "sidewall":
                sidewall_scales.append(anchoring_scale)
            else:
                cap_scales.append(anchoring_scale)

        sidewall_scale = float(np.mean(sidewall_scales)) if sidewall_scales else float("nan")
        cap_scale = float(np.mean(cap_scales)) if cap_scales else float("nan")
        return bulk_scale, sidewall_scale, cap_scale

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
        if self._mesh_tetrahedra is not None:
            tetrahedra = np.asarray(self._mesh_tetrahedra, dtype=int)
            valid_tetrahedra = []
            valid_volumes = []
            valid_gradients = []

            for tet in tetrahedra:
                coords = self.vertices[tet]
                try:
                    volume, grad_n = self._tetra_geometry(coords)
                except np.linalg.LinAlgError:
                    continue

                if volume <= self._volume_tol:
                    continue

                valid_tetrahedra.append(tet)
                valid_volumes.append(volume)
                valid_gradients.append(grad_n)

            self.tetrahedra = np.asarray(valid_tetrahedra, dtype=int)
            self.tetra_volumes = np.asarray(valid_volumes, dtype=float)
            self.tetra_gradN = np.asarray(valid_gradients, dtype=float)
            self.boundary_faces = self._load_boundary_faces_from_mesh()
            self.domain_volume = float(np.sum(self.tetra_volumes))
            return

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

    def _load_boundary_faces_from_mesh(self) -> List[BoundaryFace]:
        """Return boundary faces tagged by the structured mesh generator."""
        if self._mesh_boundary_faces is None:
            return []

        domain_centroid = self.vertices.mean(axis=0)
        faces: List[BoundaryFace] = []
        kinds = self._mesh_boundary_face_kind
        if kinds is None:
            kinds = np.array(["sidewall"] * len(self._mesh_boundary_faces), dtype="<U16")

        for face_nodes, kind in zip(self._mesh_boundary_faces, kinds):
            pts = self.vertices[face_nodes]
            edge1 = pts[1] - pts[0]
            edge2 = pts[2] - pts[0]
            normal = np.cross(edge1, edge2)
            norm = np.linalg.norm(normal)
            if norm == 0.0:
                continue
            area = 0.5 * norm
            normal = normal / norm
            centroid = pts.mean(axis=0)
            if np.dot(normal, centroid - domain_centroid) < 0.0:
                normal = -normal
            faces.append(BoundaryFace(nodes=np.asarray(face_nodes, dtype=int), area=area, normal=normal, kind=str(kind)))

        return faces

    def _extract_boundary_faces(self) -> List[BoundaryFace]:
        """Extract boundary faces from the convex hull and classify them."""
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

            if normal[2] > 0.6:
                kind = "top_cap"
            elif normal[2] < -0.6:
                kind = "bottom_cap"
            else:
                kind = "sidewall"
            faces.append(BoundaryFace(nodes=face, area=area, normal=normal, kind=kind))

        return faces

    def compute_energy_and_gradient(self, directors: np.ndarray, time: float = 0.0) -> Tuple[float, np.ndarray]:
        """Compute the continuum free energy and its gradient.

        The bulk term uses a one-constant Frank-Oseen form with a cholesteric
        pitch penalty. This is a continuum discretization on tetrahedral linear
        elements, so the gradients are assembled element-by-element instead of
        being inferred from array storage order.
        """
        directors = self.normalize_directors(directors)
        grad = np.zeros_like(directors)
        total_energy = 0.0
        temperature_field = self._temperature_field(time)

        for tet_idx, tet in enumerate(self.tetrahedra):
            nodes = tet
            n = directors[nodes]  # shape: (4, 3)
            g = self.tetra_gradN[tet_idx]  # shape: (4, 3)
            volume = self.tetra_volumes[tet_idx]
            local_temperature = float(np.mean(temperature_field[nodes]))
            elastic_scale = float(self._elastic_temperature_scale(local_temperature))

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
                    K1=self.elastic_constants["K1"] * elastic_scale,
                    K2=self.elastic_constants["K2"] * elastic_scale,
                    K3=self.elastic_constants["K3"] * elastic_scale,
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
                    K_local=self.K * elastic_scale,
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
            local_temperature = float(np.mean(temperature_field[face.nodes]))
            anchoring_scale = float(self._anchoring_temperature_scale(local_temperature, kind=face.kind))
            W_local = W * anchoring_scale

            if mode == "planar":
                # Tangential anchoring: penalize the normal component.
                face_energy = 0.5 * W_local * float(dot_values @ weighted_dot)
                dE_dn_face = W_local * weighted_dot[:, np.newaxis] * face.normal
            elif mode == "homeotropic":
                # Homeotropic anchoring: penalize misalignment with the normal.
                face_energy = 0.5 * W_local * (face.area - float(dot_values @ weighted_dot))
                dE_dn_face = -W_local * weighted_dot[:, np.newaxis] * face.normal
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
        K_local: float,
    ) -> Tuple[float, np.ndarray]:
        """Return the one-constant Frank-Oseen element contribution."""
        density = 0.5 * K_local * (
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
            local_grad[local_node] = K_local * volume * (
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
        K1: float,
        K2: float,
        K3: float,
    ) -> Tuple[float, np.ndarray]:
        """Return the full anisotropic Oseen-Frank element contribution."""
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

    def _compute_nematic_order_metrics(self, directors: np.ndarray) -> Tuple[float, float, float]:
        """Return principal nematic order and angle-spread diagnostics.

        The principal order parameter is the largest eigenvalue of the
        spatially averaged Q tensor. The angle statistics are computed relative
        to the cylinder axis and give a quick sense of how sharply aligned the
        field is at each time step.
        """
        dirs = self.normalize_directors(directors)
        q_tensor = np.mean(
            1.5 * dirs[:, :, np.newaxis] * dirs[:, np.newaxis, :] - 0.5 * np.eye(3),
            axis=0,
        )
        eigvals = np.linalg.eigvalsh(q_tensor)
        principal_order = float(np.max(eigvals))
        angles = self._director_angles_degrees(dirs)
        mean_angle = float(np.mean(angles))
        p90_angle = float(np.percentile(angles, 90.0))
        return principal_order, mean_angle, p90_angle

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

    def _helical_twist_field(self, twist_wavenumber: float) -> np.ndarray:
        """Return a simple helical director field n=(cos qz, sin qz, 0)."""
        z = self.vertices[:, 2]
        return self.normalize_directors(
            np.column_stack(
                (
                    np.cos(twist_wavenumber * z),
                    np.sin(twist_wavenumber * z),
                    np.zeros_like(z),
                )
            )
        )

    def _radial_field(self) -> np.ndarray:
        """Return the outward radial field in the cylinder cross-section."""
        xy = self.vertices[:, :2]
        radii = np.linalg.norm(xy, axis=1, keepdims=True)
        directions = np.zeros_like(self.vertices)
        nonzero = radii[:, 0] > 1.0e-30
        directions[nonzero, 0] = xy[nonzero, 0] / radii[nonzero, 0]
        directions[nonzero, 1] = xy[nonzero, 1] / radii[nonzero, 0]
        directions[~nonzero, 0] = 1.0
        return self.normalize_directors(directions)

    def _azimuthal_field(self) -> np.ndarray:
        """Return the tangential azimuthal field in the cylinder cross-section."""
        xy = self.vertices[:, :2]
        radii = np.linalg.norm(xy, axis=1, keepdims=True)
        directions = np.zeros_like(self.vertices)
        nonzero = radii[:, 0] > 1.0e-30
        directions[nonzero, 0] = -xy[nonzero, 1] / radii[nonzero, 0]
        directions[nonzero, 1] = xy[nonzero, 0] / radii[nonzero, 0]
        directions[~nonzero, 1] = 1.0
        return self.normalize_directors(directions)

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
        principal_order_parameters: List[float] = []
        mean_angle_degrees: List[float] = []
        p90_angle_degrees: List[float] = []
        mean_temperature_K: List[float] = []
        min_temperature_K: List[float] = []
        max_temperature_K: List[float] = []
        snapshots: List[np.ndarray] = []
        temperature_snapshots: List[np.ndarray] = []
        snapshot_times: List[float] = []
        histogram_counts: List[np.ndarray] = []
        histogram_edges: np.ndarray | None = None

        current_time = 0.0
        step = 0
        while step <= total_steps:
            temperature_field = self._temperature_field(current_time)
            energy, grad = self.compute_energy_and_gradient(directors, time=current_time)
            times.append(current_time)
            energies.append(self._excess_free_energy(float(energy)))
            order_parameters.append(self._compute_order_parameter(directors))
            principal_order, mean_angle, p90_angle = self._compute_nematic_order_metrics(directors)
            principal_order_parameters.append(principal_order)
            mean_angle_degrees.append(mean_angle)
            p90_angle_degrees.append(p90_angle)
            mean_temperature_K.append(float(np.mean(temperature_field)))
            min_temperature_K.append(float(np.min(temperature_field)))
            max_temperature_K.append(float(np.max(temperature_field)))

            if step % max(snapshot_interval, 1) == 0:
                snapshot = directors.copy()
                snapshots.append(snapshot)
                temperature_snapshots.append(temperature_field.copy())
                snapshot_times.append(current_time)
                counts, edges = self._angle_histogram(snapshot, bins=histogram_bins)
                histogram_counts.append(counts.astype(float))
                histogram_edges = edges

            if step == total_steps:
                break

            projected = self.project_gradient(directors, grad) / self._energy_scale
            mobility_field = mobility * self._mobility_temperature_scale(temperature_field)

            if thermal_noise_strength > 0.0:
                # Tangential Gaussian noise approximates a stochastic LC bath.
                noise = self.project_gradient(directors, rng.normal(size=directors.shape))
                noise = noise / max(np.linalg.norm(noise) / np.sqrt(self.n_nodes), 1.0e-30)
                stochastic_scale = thermal_noise_strength * self._thermal_noise_scale(temperature_field)
                stochastic_term = np.sqrt(time_step) * stochastic_scale[:, np.newaxis] * noise
                trial = self.normalize_directors(
                    directors - time_step * mobility_field[:, np.newaxis] * projected + stochastic_term
                )
                trial_energy, _ = self.compute_energy_and_gradient(trial, time=current_time + time_step)
                if not np.isfinite(trial_energy):
                    break
                directors = trial
                current_time += time_step
            else:
                trial_step = time_step
                accepted = False

                # Backtracking keeps the explicit dynamics stable on stiff meshes.
                while trial_step >= 1.0e-8:
                    trial = self.normalize_directors(
                        directors - trial_step * mobility_field[:, np.newaxis] * projected
                    )
                    trial_energy, _ = self.compute_energy_and_gradient(trial, time=current_time + trial_step)
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
            principal_order_parameters,
            mean_angle_degrees,
            p90_angle_degrees,
            mean_temperature_K,
            min_temperature_K,
            max_temperature_K,
            snapshots,
            temperature_snapshots,
            snapshot_times,
            histogram_edges,
            histogram_array,
        )

    def simulate_temperature_sweep(
        self,
        temperatures_K: Sequence[float],
    ) -> TemperatureSweepResult:
        """Relax the field at a sequence of temperatures and record equilibria.

        This is the main validation/diagnostic sweep for temperature effects:
        we warm or cool the same cylinder, relax to equilibrium at each
        temperature, and record the resulting order parameter, free energy, and
        coupling scales. The field is allowed to continue from one temperature
        to the next so we can observe equilibrium shifts and possible
        softening-driven transitions.
        """
        temperatures = [float(t) for t in temperatures_K]
        if not temperatures:
            return TemperatureSweepResult(
                [],
                [],
                [],
                [],
                [],
                [],
                [],
                [],
                [],
                [],
                [],
                [],
                [],
                [],
                [],
                [],
                [],
                [],
                0.0,
                [],
            )

        original_temperature_K = self.temperature_K
        original_temperature_C = self.temperature_C
        original_thermal_energy = self.thermal_energy
        original_profile = copy.deepcopy(self.temperature_profile)
        original_material_temperature_K = self.material_state.temperature_K
        original_material_temperature_C = self.material_state.temperature_C

        excess_energies: List[float] = []
        energy_densities: List[float] = []
        energy_kbt: List[float] = []
        order_parameters: List[float] = []
        principal_order_parameters: List[float] = []
        mean_angle_degrees: List[float] = []
        p90_angle_degrees: List[float] = []
        bulk_softening_scales: List[float] = []
        sidewall_anchoring_scales: List[float] = []
        cap_anchoring_scales: List[float] = []
        effective_K1_pN: List[float] = []
        effective_K2_pN: List[float] = []
        effective_K3_pN: List[float] = []
        effective_Keff_pN: List[float] = []
        mean_temperature_K: List[float] = []
        min_temperature_K: List[float] = []
        max_temperature_K: List[float] = []
        snapshots: List[np.ndarray] = []
        temperature_snapshots: List[np.ndarray] = []

        try:
            for temperature_K in temperatures:
                self.temperature_profile.base_temperature_K = temperature_K
                self.temperature_K = temperature_K
                self.temperature_C = temperature_K - 273.15
                self.thermal_energy = BOLTZMANN_CONSTANT * self.temperature_K
                self.material_state.temperature_K = self.temperature_K
                self.material_state.temperature_C = self.temperature_C

                energies = self.relax()
                excess_energy = float(energies[-1])
                temperature_field = self._temperature_field(time=0.0)
                principal_order, mean_angle, p90_angle = self._compute_nematic_order_metrics(self.directors)
                bulk_scale, sidewall_scale, cap_scale = self._temperature_scale_summary(temperature_field)
                k1_eff, k2_eff, k3_eff, keff_eff = self._effective_elastic_constants(temperature_field)

                excess_energies.append(excess_energy)
                energy_densities.append(self._energy_density(excess_energy))
                energy_kbt.append(self._energy_in_kbt(excess_energy))
                order_parameters.append(self._compute_order_parameter(self.directors))
                principal_order_parameters.append(principal_order)
                mean_angle_degrees.append(mean_angle)
                p90_angle_degrees.append(p90_angle)
                bulk_softening_scales.append(bulk_scale)
                sidewall_anchoring_scales.append(sidewall_scale)
                cap_anchoring_scales.append(cap_scale)
                effective_K1_pN.append(k1_eff)
                effective_K2_pN.append(k2_eff)
                effective_K3_pN.append(k3_eff)
                effective_Keff_pN.append(keff_eff)
                mean_temperature_K.append(float(np.mean(temperature_field)))
                min_temperature_K.append(float(np.min(temperature_field)))
                max_temperature_K.append(float(np.max(temperature_field)))
                temperature_snapshots.append(temperature_field.copy())
                snapshots.append(self.directors.copy())
        finally:
            # Restore the temperature metadata in case the caller reuses this
            # solver for something else after the sweep. The director field is
            # intentionally left at the last relaxed state so the final
            # equilibrium can be visualized or saved.
            self.temperature_profile = original_profile
            self.temperature_K = original_temperature_K
            self.temperature_C = original_temperature_C
            self.thermal_energy = original_thermal_energy
            self.material_state.temperature_K = original_material_temperature_K
            self.material_state.temperature_C = original_material_temperature_C

        endpoint_order_shift = float(
            np.hypot(
                order_parameters[-1] - order_parameters[0],
                principal_order_parameters[-1] - principal_order_parameters[0],
            )
        )

        return TemperatureSweepResult(
            temperatures,
            excess_energies,
            energy_densities,
            energy_kbt,
            order_parameters,
            principal_order_parameters,
            mean_angle_degrees,
            p90_angle_degrees,
            bulk_softening_scales,
            sidewall_anchoring_scales,
            cap_anchoring_scales,
            effective_K1_pN,
            effective_K2_pN,
            effective_K3_pN,
            effective_Keff_pN,
            mean_temperature_K,
            min_temperature_K,
            max_temperature_K,
            temperature_snapshots,
            endpoint_order_shift,
            snapshots,
        )

    @staticmethod
    def _reverse_temperature_sweep_result(result: TemperatureSweepResult) -> TemperatureSweepResult:
        """Return a sweep result reordered from high temperature back to low."""
        return TemperatureSweepResult(
            temperatures_K=list(reversed(result.temperatures_K)),
            excess_energies=list(reversed(result.excess_energies)),
            energy_densities=list(reversed(result.energy_densities)),
            energy_kbt=list(reversed(result.energy_kbt)),
            order_parameters=list(reversed(result.order_parameters)),
            principal_order_parameters=list(reversed(result.principal_order_parameters)),
            mean_angle_degrees=list(reversed(result.mean_angle_degrees)),
            p90_angle_degrees=list(reversed(result.p90_angle_degrees)),
            bulk_softening_scales=list(reversed(result.bulk_softening_scales)),
            sidewall_anchoring_scales=list(reversed(result.sidewall_anchoring_scales)),
            cap_anchoring_scales=list(reversed(result.cap_anchoring_scales)),
            effective_K1_pN=list(reversed(result.effective_K1_pN)),
            effective_K2_pN=list(reversed(result.effective_K2_pN)),
            effective_K3_pN=list(reversed(result.effective_K3_pN)),
            effective_Keff_pN=list(reversed(result.effective_Keff_pN)),
            mean_temperature_K=list(reversed(result.mean_temperature_K)),
            min_temperature_K=list(reversed(result.min_temperature_K)),
            max_temperature_K=list(reversed(result.max_temperature_K)),
            temperature_snapshots=list(reversed(result.temperature_snapshots)),
            endpoint_order_shift=result.endpoint_order_shift,
            snapshots=list(reversed(result.snapshots)),
        )

    def simulate_temperature_hysteresis(
        self,
        temperatures_K: Sequence[float],
    ) -> TemperatureHysteresisResult:
        """Run a heating sweep followed by a cooling sweep on the same sample."""
        temperatures = [float(t) for t in temperatures_K]
        if not temperatures:
            empty_sweep = TemperatureSweepResult(
                [],
                [],
                [],
                [],
                [],
                [],
                [],
                [],
                [],
                [],
                [],
                [],
                [],
                [],
                [],
                [],
                [],
                [],
                [],
                [],
                0.0,
                [],
            )
            return TemperatureHysteresisResult(
                empty_sweep,
                empty_sweep,
                [],
                [],
                [],
                [],
                float("nan"),
                float("nan"),
                float("nan"),
                float("nan"),
                float("nan"),
                float("nan"),
                float("nan"),
                float("nan"),
            )

        heating = self.simulate_temperature_sweep(temperatures)
        cooling_raw = self.simulate_temperature_sweep(list(reversed(temperatures)))
        cooling = self._reverse_temperature_sweep_result(cooling_raw)

        energy_density_gap = [cooling.energy_densities[i] - heating.energy_densities[i] for i in range(len(temperatures))]
        order_parameter_gap = [cooling.order_parameters[i] - heating.order_parameters[i] for i in range(len(temperatures))]
        principal_order_gap = [
            cooling.principal_order_parameters[i] - heating.principal_order_parameters[i]
            for i in range(len(temperatures))
        ]

        return TemperatureHysteresisResult(
            heating=heating,
            cooling=cooling,
            temperatures_K=heating.temperatures_K,
            energy_density_gap=energy_density_gap,
            order_parameter_gap=order_parameter_gap,
            principal_order_gap=principal_order_gap,
            midpoint_temperature_K=float(temperatures[len(temperatures) // 2]),
            midpoint_energy_density_gap=float(energy_density_gap[len(temperatures) // 2]),
            midpoint_order_parameter_gap=float(order_parameter_gap[len(temperatures) // 2]),
            midpoint_principal_order_gap=float(principal_order_gap[len(temperatures) // 2]),
            midpoint_loop_width=float(
                np.hypot(
                    order_parameter_gap[len(temperatures) // 2],
                    principal_order_gap[len(temperatures) // 2],
                )
            ),
            max_energy_density_gap=float(np.max(np.abs(energy_density_gap))),
            max_order_parameter_gap=float(np.max(np.abs(order_parameter_gap))),
            max_principal_order_gap=float(np.max(np.abs(principal_order_gap))),
        )

    def save_dynamics_trace(self, result: DynamicsResult, filename: str) -> None:
        """Save a compact CSV trace of the dynamic run."""
        with open(filename, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(
                [
                    "time",
                    "excess_energy_j",
                    "energy_density_j_m3",
                    "energy_kbt",
                    "order_parameter",
                    "principal_order_parameter",
                    "mean_angle_deg",
                    "p90_angle_deg",
                    "mean_temperature_K",
                    "min_temperature_K",
                    "max_temperature_K",
                ]
            )
            for idx, (t, e, s, p, mean_angle, p90_angle) in enumerate(
                zip(
                    result.times,
                    result.energies,
                    result.order_parameters,
                    result.principal_order_parameters,
                    result.mean_angle_degrees,
                    result.p90_angle_degrees,
                )
            ):
                writer.writerow(
                    [
                        f"{t:.8e}",
                        f"{e:.8e}",
                        f"{self._energy_density(e):.8e}",
                        f"{self._energy_in_kbt(e):.8e}",
                        f"{s:.8e}",
                        f"{p:.8e}",
                        f"{mean_angle:.8e}",
                        f"{p90_angle:.8e}",
                        f"{result.mean_temperature_K[idx]:.8e}",
                        f"{result.min_temperature_K[idx]:.8e}",
                        f"{result.max_temperature_K[idx]:.8e}",
                    ]
                )

    def save_dynamics_alignment_data(self, result: DynamicsResult, filename: str) -> None:
        """Save nematic alignment statistics for each simulation step."""
        with open(filename, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(
                [
                    "time",
                    "principal_order_parameter",
                    "mean_angle_deg",
                    "p90_angle_deg",
                    "order_parameter_szz",
                    "mean_temperature_K",
                    "min_temperature_K",
                    "max_temperature_K",
                ]
            )
            for idx, (t, principal, mean_angle, p90_angle, szz) in enumerate(
                zip(
                    result.times,
                    result.principal_order_parameters,
                    result.mean_angle_degrees,
                    result.p90_angle_degrees,
                    result.order_parameters,
                )
            ):
                writer.writerow(
                    [
                        f"{t:.8e}",
                        f"{principal:.8e}",
                        f"{mean_angle:.8e}",
                        f"{p90_angle:.8e}",
                        f"{szz:.8e}",
                        f"{result.mean_temperature_K[idx]:.8e}",
                        f"{result.min_temperature_K[idx]:.8e}",
                        f"{result.max_temperature_K[idx]:.8e}",
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

    def plot_dynamics_alignment_trace(
        self, result: DynamicsResult, filename: str = "dynamics_alignment.jpg"
    ) -> None:
        """Plot principal nematic order and angle spread over time."""
        if not result.times:
            return

        fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

        axes[0].plot(result.times, result.principal_order_parameters, color="tab:green", label="Principal Order")
        axes[0].plot(result.times, result.order_parameters, color="tab:orange", linestyle="--", label="Szz")
        axes[0].set_ylabel("Order")
        axes[0].legend(loc="best")
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(result.times, result.mean_angle_degrees, color="tab:blue", label="Mean Angle")
        axes[1].plot(result.times, result.p90_angle_degrees, color="tab:red", linestyle="--", label="90th Percentile")
        axes[1].set_ylabel("Angle (deg)")
        axes[1].legend(loc="best")
        axes[1].grid(True, alpha=0.3)

        axes[2].plot(result.times, result.mean_temperature_K, color="tab:purple", label="Mean Temperature")
        axes[2].plot(result.times, result.min_temperature_K, color="tab:cyan", linestyle="--", label="Min Temperature")
        axes[2].plot(result.times, result.max_temperature_K, color="tab:brown", linestyle=":", label="Max Temperature")
        axes[2].set_ylabel("Temperature (K)")
        axes[2].set_xlabel("Model Time")
        axes[2].legend(loc="best")
        axes[2].grid(True, alpha=0.3)

        fig.suptitle("Liquid Crystal Alignment and Thermal Diagnostics")
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
            self.plot_director_field(ax, title=f"Relaxation Dynamics t={t:.3f}", add_colorbar=False)
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
        self.save_dynamics_alignment_data(result, filename=f"{output_prefix}_alignment.csv")
        self.plot_dynamics_alignment_trace(result, filename=f"{output_prefix}_alignment.jpg")
        self.save_dynamics_histogram_data(result, filename=f"{output_prefix}_histograms.csv")
        self.plot_dynamics_histogram_heatmap(result, filename=f"{output_prefix}_histograms.jpg")
        if result.snapshots:
            self.save_director_orientation_map(
                result.snapshots[-1],
                filename=f"{output_prefix}_director_map.jpg",
                title="Final Director Orientation Map",
                projection="rho_z",
            )
            self.save_director_orientation_map(
                result.snapshots[-1],
                filename=f"{output_prefix}_director_xz_map.jpg",
                title="Final Director Orientation Map (x-z projection)",
                projection="x_z",
            )
            self.save_director_orientation_map(
                result.snapshots[-1],
                filename=f"{output_prefix}_director_xy_map.jpg",
                title="Final Director Orientation Map (x-y projection)",
                projection="x_y",
            )
            self.save_director_orientation_map(
                result.snapshots[-1],
                filename=f"{output_prefix}_director_theta_z_map.jpg",
                title="Final Director Orientation Map (theta-z projection)",
                projection="theta_z",
            )
            self.save_director_alignment_map(
                result.snapshots[-1],
                filename=f"{output_prefix}_alignment_map.jpg",
                title="Final Director Alignment Map",
                projection="rho_z",
            )
            self.save_director_alignment_map(
                result.snapshots[-1],
                filename=f"{output_prefix}_alignment_xz_map.jpg",
                title="Final Director Alignment Map (x-z projection)",
                projection="x_z",
            )
            self.save_director_alignment_map(
                result.snapshots[-1],
                filename=f"{output_prefix}_alignment_xy_map.jpg",
                title="Final Director Alignment Map (x-y projection)",
                projection="x_y",
            )
            self.save_director_alignment_map(
                result.snapshots[-1],
                filename=f"{output_prefix}_alignment_theta_z_map.jpg",
                title="Final Director Alignment Map (theta-z projection)",
                projection="theta_z",
            )
        if result.temperature_snapshots:
            self.save_temperature_field_map(
                result.temperature_snapshots[-1],
                filename=f"{output_prefix}_temperature_map.jpg",
                title="Final Temperature Map",
                projection="rho_z",
            )
            self.save_temperature_field_map(
                result.temperature_snapshots[-1],
                filename=f"{output_prefix}_temperature_xz_map.jpg",
                title="Final Temperature Map (x-z projection)",
                projection="x_z",
            )
            self.save_temperature_field_map(
                result.temperature_snapshots[-1],
                filename=f"{output_prefix}_temperature_xy_map.jpg",
                title="Final Temperature Map (x-y projection)",
                projection="x_y",
            )
            self.save_temperature_field_map(
                result.temperature_snapshots[-1],
                filename=f"{output_prefix}_temperature_theta_z_map.jpg",
                title="Final Temperature Map (theta-z projection)",
                projection="theta_z",
            )
        if result.snapshots:
            temperature_field = result.temperature_snapshots[-1] if result.temperature_snapshots else None
            self.save_sidewall_anchoring_data(
                result.snapshots[-1],
                temperature_field=temperature_field,
                filename=f"{output_prefix}_sidewall_anchoring.csv",
            )
            self.save_sidewall_anchoring_map(
                result.snapshots[-1],
                temperature_field=temperature_field,
                filename=f"{output_prefix}_sidewall_anchoring_theta_z_map.jpg",
                title=f"Final Sidewall Anchoring Energy Density ({self.side_mode})",
            )
            self.save_refinement_indicator_data(
                result.snapshots[-1],
                temperature_field=temperature_field,
                filename=f"{output_prefix}_refinement_hint.csv",
            )
            self.save_refinement_indicator_map(
                result.snapshots[-1],
                temperature_field=temperature_field,
                filename=f"{output_prefix}_refinement_hint_map.jpg",
                title="Final Mesh Refinement Hint Map",
                projection="rho_z",
            )
            self.save_refined_mesh_suggestion(
                result.snapshots[-1],
                temperature_field=temperature_field,
                filename=f"{output_prefix}_refined_mesh_suggestion.json",
            )
            self.save_refined_solve_mesh(
                result.snapshots[-1],
                temperature_field=temperature_field,
                filename=f"{output_prefix}_refined_solve_mesh.npz",
            )
        self.save_dynamics_movie(result, filename=f"{output_prefix}_movie.gif")

    def save_temperature_sweep_data(self, result: TemperatureSweepResult, filename: str) -> None:
        """Save the equilibrium observables from a temperature sweep."""
        with open(filename, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(
                [
                    "temperature_K",
                    "excess_energy_j",
                    "energy_density_j_m3",
                    "energy_kbt",
                    "order_parameter",
                    "principal_order_parameter",
                    "mean_angle_deg",
                    "p90_angle_deg",
                    "bulk_softening_scale",
                    "sidewall_anchoring_scale",
                    "cap_anchoring_scale",
                    "effective_K1_pN",
                    "effective_K2_pN",
                    "effective_K3_pN",
                    "effective_Keff_pN",
                    "mean_temperature_K",
                    "min_temperature_K",
                    "max_temperature_K",
                ]
            )
            for idx, temperature_K in enumerate(result.temperatures_K):
                writer.writerow(
                    [
                        f"{temperature_K:.8e}",
                        f"{result.excess_energies[idx]:.8e}",
                        f"{result.energy_densities[idx]:.8e}",
                        f"{result.energy_kbt[idx]:.8e}",
                        f"{result.order_parameters[idx]:.8e}",
                        f"{result.principal_order_parameters[idx]:.8e}",
                        f"{result.mean_angle_degrees[idx]:.8e}",
                        f"{result.p90_angle_degrees[idx]:.8e}",
                        f"{result.bulk_softening_scales[idx]:.8e}",
                        f"{result.sidewall_anchoring_scales[idx]:.8e}",
                        f"{result.cap_anchoring_scales[idx]:.8e}",
                        f"{result.effective_K1_pN[idx]:.8e}",
                        f"{result.effective_K2_pN[idx]:.8e}",
                        f"{result.effective_K3_pN[idx]:.8e}",
                        f"{result.effective_Keff_pN[idx]:.8e}",
                        f"{result.mean_temperature_K[idx]:.8e}",
                        f"{result.min_temperature_K[idx]:.8e}",
                        f"{result.max_temperature_K[idx]:.8e}",
                    ]
                )

    def plot_temperature_sweep(self, result: TemperatureSweepResult, filename: str = "temperature_sweep.jpg") -> None:
        """Plot equilibrium observables as a function of temperature."""
        if not result.temperatures_K:
            return

        temperatures = np.asarray(result.temperatures_K, dtype=float)
        fig, axes = plt.subplots(4, 1, figsize=(10, 14), sharex=True)

        axes[0].plot(temperatures, result.energy_densities, color="tab:blue", marker="o", label="Excess Energy Density")
        axes[0].set_ylabel("Excess f (J/m^3)")
        axes[0].legend(loc="best")
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(temperatures, result.principal_order_parameters, color="tab:green", marker="o", label="Principal Order")
        axes[1].plot(temperatures, result.order_parameters, color="tab:orange", linestyle="--", marker="s", label="Szz")
        axes[1].set_ylabel("Order")
        axes[1].legend(loc="best")
        axes[1].grid(True, alpha=0.3)

        axes[2].plot(temperatures, result.mean_angle_degrees, color="tab:purple", marker="o", label="Mean Angle")
        axes[2].plot(temperatures, result.p90_angle_degrees, color="tab:red", linestyle="--", marker="s", label="90th Percentile")
        axes[2].set_ylabel("Angle (deg)")
        axes[2].legend(loc="best")
        axes[2].grid(True, alpha=0.3)

        axes[3].plot(temperatures, result.bulk_softening_scales, color="tab:blue", marker="o", label="Bulk Softening")
        axes[3].plot(temperatures, result.sidewall_anchoring_scales, color="tab:green", linestyle="--", marker="s", label="Sidewall Anchoring")
        axes[3].plot(temperatures, result.cap_anchoring_scales, color="tab:orange", linestyle=":", marker="^", label="Cap Anchoring")
        axes[3].plot(temperatures, result.effective_K1_pN, color="tab:purple", linestyle="-.", marker="d", label="K1 (pN)")
        axes[3].plot(temperatures, result.effective_K2_pN, color="tab:red", marker="o", label="K2 (pN)")
        axes[3].plot(temperatures, result.effective_K3_pN, color="tab:brown", linestyle="--", marker="s", label="K3 (pN)")
        axes[3].plot(temperatures, result.effective_Keff_pN, color="tab:gray", linestyle=":", marker="^", label="K_eff (pN)")
        axes[3].set_ylabel("Scale / K_i")
        axes[3].set_xlabel("Sweep base temperature (K)")
        axes[3].legend(loc="best", ncol=2)
        axes[3].grid(True, alpha=0.3)
        fig.suptitle("Liquid Crystal Temperature Sweep Equilibria")
        fig.tight_layout()
        fig.savefig(filename, dpi=300)
        plt.close(fig)

    def save_temperature_sweep_bundle(self, result: TemperatureSweepResult, output_prefix: str) -> None:
        """Write the standard sweep outputs for the temperature diagnostics."""
        self.save_temperature_sweep_data(result, filename=f"{output_prefix}_temperature_sweep.csv")
        self.plot_temperature_sweep(result, filename=f"{output_prefix}_temperature_sweep.jpg")
        self.save_temperature_sweep_summary(result, filename=f"{output_prefix}_temperature_sweep_summary.txt")
        if result.snapshots:
            old_directors = self.directors.copy()
            self.directors = result.snapshots[-1]
            self.save_director_field(f"{output_prefix}_temperature_sweep_final_director_field.txt")
            self.save_director_orientation_map(
                result.snapshots[-1],
                filename=f"{output_prefix}_temperature_sweep_director_map.jpg",
                title="Temperature Sweep Final Director Orientation Map",
                projection="rho_z",
            )
            self.save_director_orientation_map(
                result.snapshots[-1],
                filename=f"{output_prefix}_temperature_sweep_director_xz_map.jpg",
                title="Temperature Sweep Final Director Orientation Map (x-z projection)",
                projection="x_z",
            )
            self.save_director_orientation_map(
                result.snapshots[-1],
                filename=f"{output_prefix}_temperature_sweep_director_xy_map.jpg",
                title="Temperature Sweep Final Director Orientation Map (x-y projection)",
                projection="x_y",
            )
            self.save_director_orientation_map(
                result.snapshots[-1],
                filename=f"{output_prefix}_temperature_sweep_director_theta_z_map.jpg",
                title="Temperature Sweep Final Director Orientation Map (theta-z projection)",
                projection="theta_z",
            )
            self.save_director_alignment_map(
                result.snapshots[-1],
                filename=f"{output_prefix}_temperature_sweep_alignment_map.jpg",
                title="Temperature Sweep Final Director Alignment Map",
                projection="rho_z",
            )
            self.save_director_alignment_map(
                result.snapshots[-1],
                filename=f"{output_prefix}_temperature_sweep_alignment_xz_map.jpg",
                title="Temperature Sweep Final Director Alignment Map (x-z projection)",
                projection="x_z",
            )
            self.save_director_alignment_map(
                result.snapshots[-1],
                filename=f"{output_prefix}_temperature_sweep_alignment_xy_map.jpg",
                title="Temperature Sweep Final Director Alignment Map (x-y projection)",
                projection="x_y",
            )
            self.save_director_alignment_map(
                result.snapshots[-1],
                filename=f"{output_prefix}_temperature_sweep_alignment_theta_z_map.jpg",
                title="Temperature Sweep Final Director Alignment Map (theta-z projection)",
                projection="theta_z",
            )
            if result.temperature_snapshots:
                self.save_temperature_field_map(
                    result.temperature_snapshots[-1],
                    filename=f"{output_prefix}_temperature_sweep_temperature_map.jpg",
                    title="Temperature Sweep Final Temperature Map",
                    projection="rho_z",
                )
                self.save_temperature_field_map(
                    result.temperature_snapshots[-1],
                    filename=f"{output_prefix}_temperature_sweep_temperature_xz_map.jpg",
                    title="Temperature Sweep Final Temperature Map (x-z projection)",
                    projection="x_z",
                )
                self.save_temperature_field_map(
                    result.temperature_snapshots[-1],
                    filename=f"{output_prefix}_temperature_sweep_temperature_xy_map.jpg",
                    title="Temperature Sweep Final Temperature Map (x-y projection)",
                    projection="x_y",
                )
                self.save_temperature_field_map(
                    result.temperature_snapshots[-1],
                    filename=f"{output_prefix}_temperature_sweep_temperature_theta_z_map.jpg",
                    title="Temperature Sweep Final Temperature Map (theta-z projection)",
                    projection="theta_z",
                )
            temperature_field = result.temperature_snapshots[-1] if result.temperature_snapshots else None
            self.save_sidewall_anchoring_data(
                result.snapshots[-1],
                temperature_field=temperature_field,
                filename=f"{output_prefix}_temperature_sweep_sidewall_anchoring.csv",
            )
            self.save_sidewall_anchoring_map(
                result.snapshots[-1],
                temperature_field=temperature_field,
                filename=f"{output_prefix}_temperature_sweep_sidewall_anchoring_theta_z_map.jpg",
                title=f"Temperature Sweep Final Sidewall Anchoring Energy Density ({self.side_mode})",
            )
            self.save_refinement_indicator_data(
                result.snapshots[-1],
                temperature_field=temperature_field,
                filename=f"{output_prefix}_temperature_sweep_refinement_hint.csv",
            )
            self.save_refinement_indicator_map(
                result.snapshots[-1],
                temperature_field=temperature_field,
                filename=f"{output_prefix}_temperature_sweep_refinement_hint_map.jpg",
                title="Temperature Sweep Final Mesh Refinement Hint Map",
                projection="rho_z",
            )
            self.save_refined_mesh_suggestion(
                result.snapshots[-1],
                temperature_field=temperature_field,
                filename=f"{output_prefix}_temperature_sweep_refined_mesh_suggestion.json",
            )
            self.save_refined_solve_mesh(
                result.snapshots[-1],
                temperature_field=temperature_field,
                filename=f"{output_prefix}_temperature_sweep_refined_solve_mesh.npz",
            )
            self.directors = old_directors

    def save_temperature_sweep_summary(self, result: TemperatureSweepResult, filename: str) -> None:
        """Write a compact human-readable summary of the temperature sweep."""
        if not result.temperatures_K:
            return

        midpoint = len(result.temperatures_K) // 2
        temperature_field = result.temperature_snapshots[-1] if result.temperature_snapshots else None
        # The refinement suggestion is a follow-up mesh recipe, not an in-place
        # remesh. Writing it into the summary file keeps the next-step geometry
        # easy to find without digging through the JSON artifact.
        refinement = self.suggest_refined_mesh_parameters(
            directors=result.snapshots[-1] if result.snapshots else None,
            temperature_field=temperature_field,
        )
        sidewall_mean, sidewall_std, sidewall_min, sidewall_max, sidewall_total = self._sidewall_anchoring_summary(
            directors=result.snapshots[-1] if result.snapshots else None,
            temperature_field=temperature_field,
        )
        with open(filename, "w", encoding="utf-8") as fh:
            fh.write("Liquid-crystal temperature sweep summary\n")
            fh.write(f"material_preset: {self.material_preset}\n")
            fh.write(f"anchoring_preset: {self.anchoring_preset}\n")
            fh.write(f"headline_sidewall_anchoring_mode: {self.side_mode}\n")
            fh.write(f"temperature_window_K: {result.temperatures_K[0]:.6f} -> {result.temperatures_K[-1]:.6f}\n")
            fh.write(f"headline_midpoint_temperature_K: {result.temperatures_K[midpoint]:.6f}\n")
            fh.write(f"headline_midpoint_order_parameter: {result.order_parameters[midpoint]:.6e}\n")
            fh.write(f"headline_midpoint_principal_order: {result.principal_order_parameters[midpoint]:.6e}\n")
            fh.write(f"headline_endpoint_order_shift: {result.endpoint_order_shift:.6e}\n")
            fh.write(f"headline_endpoint_Keff_pN: {result.effective_Keff_pN[0]:.6f} -> {result.effective_Keff_pN[-1]:.6f}\n")
            fh.write(f"headline_endpoint_energy_density_J_m3: {result.energy_densities[0]:.6e} -> {result.energy_densities[-1]:.6e}\n")
            fh.write(f"headline_sidewall_anchoring_mean_J_m2: {sidewall_mean:.6e}\n")
            fh.write(f"headline_sidewall_anchoring_std_J_m2: {sidewall_std:.6e}\n")
            fh.write(f"headline_sidewall_anchoring_minmax_J_m2: {sidewall_min:.6e} -> {sidewall_max:.6e}\n")
            fh.write(f"headline_sidewall_anchoring_total_J: {sidewall_total:.6e}\n")
            if refinement is not None:
                current = self.current_mesh_recipe()
                if current is not None:
                    fh.write(
                        "headline_refined_mesh_delta: "
                        f"dr{refinement.num_radial_layers - current.num_radial_layers:+d} "
                        f"dθ{refinement.num_theta_points - current.num_theta_points:+d} "
                        f"dz{refinement.num_axial_layers - current.num_axial_layers:+d} "
                        f"kr{refinement.radial_cluster_power - current.radial_cluster_power:+.2f} "
                        f"ka{refinement.axial_cluster_power - current.axial_cluster_power:+.2f}\n"
                    )
                fh.write(
                    "headline_refined_mesh_recipe: "
                    f"radial={refinement.num_radial_layers}, "
                    f"theta={refinement.num_theta_points}, "
                    f"axial={refinement.num_axial_layers}, "
                    f"cluster(r,a)={refinement.radial_cluster_power:.2f}/{refinement.axial_cluster_power:.2f}\n"
                )
                fh.write(f"headline_refined_mesh_wall_hot_fraction: {refinement.wall_hot_fraction:.6f}\n")
                fh.write(f"headline_refined_mesh_cap_hot_fraction: {refinement.cap_hot_fraction:.6f}\n")
                fh.write(f"headline_refined_mesh_max_score: {refinement.max_refinement_score:.6e}\n")
                fh.write(f"headline_refined_mesh_mean_score: {refinement.mean_refinement_score:.6e}\n")

    def save_temperature_hysteresis_data(self, result: TemperatureHysteresisResult, filename: str) -> None:
        """Save paired heating/cooling observables for a hysteresis scan."""
        with open(filename, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(
                [
                    "phase",
                    "temperature_K",
                    "excess_energy_j",
                    "energy_density_j_m3",
                    "energy_kbt",
                    "order_parameter",
                    "principal_order_parameter",
                    "mean_angle_deg",
                    "p90_angle_deg",
                    "bulk_softening_scale",
                    "sidewall_anchoring_scale",
                    "cap_anchoring_scale",
                    "mean_temperature_K",
                    "min_temperature_K",
                    "max_temperature_K",
                    "paired_energy_density_gap_j_m3",
                    "paired_order_parameter_gap",
                    "paired_principal_order_gap",
                ]
            )

            for phase, sweep in (("heating", result.heating), ("cooling", result.cooling)):
                for idx, temperature_K in enumerate(sweep.temperatures_K):
                    writer.writerow(
                        [
                            phase,
                            f"{temperature_K:.8e}",
                            f"{sweep.excess_energies[idx]:.8e}",
                            f"{sweep.energy_densities[idx]:.8e}",
                            f"{sweep.energy_kbt[idx]:.8e}",
                            f"{sweep.order_parameters[idx]:.8e}",
                            f"{sweep.principal_order_parameters[idx]:.8e}",
                            f"{sweep.mean_angle_degrees[idx]:.8e}",
                            f"{sweep.p90_angle_degrees[idx]:.8e}",
                            f"{sweep.bulk_softening_scales[idx]:.8e}",
                            f"{sweep.sidewall_anchoring_scales[idx]:.8e}",
                            f"{sweep.cap_anchoring_scales[idx]:.8e}",
                            f"{sweep.mean_temperature_K[idx]:.8e}",
                            f"{sweep.min_temperature_K[idx]:.8e}",
                            f"{sweep.max_temperature_K[idx]:.8e}",
                            f"{result.energy_density_gap[idx]:.8e}",
                            f"{result.order_parameter_gap[idx]:.8e}",
                            f"{result.principal_order_gap[idx]:.8e}",
                        ]
                    )

    def plot_temperature_hysteresis(
        self, result: TemperatureHysteresisResult, filename: str = "temperature_hysteresis.jpg"
    ) -> None:
        """Plot paired heating/cooling curves and their hysteresis gap."""
        if not result.temperatures_K:
            return

        temperatures = np.asarray(result.temperatures_K, dtype=float)
        fig, axes = plt.subplots(3, 1, figsize=(10, 13), sharex=True)

        axes[0].plot(
            temperatures,
            result.heating.energy_densities,
            color="tab:red",
            marker="o",
            label="Heating",
        )
        axes[0].plot(
            temperatures,
            result.cooling.energy_densities,
            color="tab:blue",
            marker="s",
            linestyle="--",
            label="Cooling",
        )
        axes[0].set_ylabel("Excess f (J/m^3)")
        axes[0].legend(loc="best")
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(
            temperatures,
            result.heating.principal_order_parameters,
            color="tab:red",
            marker="o",
            label="Heating principal order",
        )
        axes[1].plot(
            temperatures,
            result.cooling.principal_order_parameters,
            color="tab:blue",
            marker="s",
            linestyle="--",
            label="Cooling principal order",
        )
        axes[1].plot(
            temperatures,
            result.heating.order_parameters,
            color="tab:orange",
            linestyle=":",
            label="Heating Szz",
        )
        axes[1].plot(
            temperatures,
            result.cooling.order_parameters,
            color="tab:green",
            linestyle="-.",
            label="Cooling Szz",
        )
        axes[1].set_ylabel("Order")
        axes[1].legend(loc="best")
        axes[1].grid(True, alpha=0.3)

        axes[2].plot(temperatures, result.order_parameter_gap, color="tab:purple", marker="o", label="Principal order gap")
        axes[2].plot(
            temperatures,
            result.energy_density_gap,
            color="tab:brown",
            marker="s",
            linestyle="--",
            label="Energy density gap",
        )
        axes[2].set_ylabel("Cooling - Heating")
        axes[2].set_xlabel("Temperature (K)")
        axes[2].legend(loc="best")
        axes[2].grid(True, alpha=0.3)

        fig.suptitle("Liquid Crystal Heating/Cooling Hysteresis")
        fig.tight_layout()
        fig.savefig(filename, dpi=300)
        plt.close(fig)

    def save_temperature_hysteresis_bundle(self, result: TemperatureHysteresisResult, output_prefix: str) -> None:
        """Write the standard outputs for a paired heating/cooling sweep."""
        self.save_temperature_hysteresis_data(result, filename=f"{output_prefix}_temperature_hysteresis.csv")
        self.plot_temperature_hysteresis(result, filename=f"{output_prefix}_temperature_hysteresis.jpg")
        self.save_temperature_hysteresis_summary(result, filename=f"{output_prefix}_temperature_hysteresis_summary.txt")
        old_directors = self.directors.copy()
        if result.cooling.snapshots:
            self.directors = result.cooling.snapshots[-1]
        self.save_director_field(f"{output_prefix}_temperature_hysteresis_final_director_field.txt")
        self.directors = old_directors

    def save_temperature_hysteresis_summary(self, result: TemperatureHysteresisResult, filename: str) -> None:
        """Write a compact human-readable summary of the hysteresis loop."""
        if not result.temperatures_K:
            return

        with open(filename, "w", encoding="utf-8") as fh:
            fh.write("Liquid-crystal temperature hysteresis summary\n")
            fh.write(f"material_preset: {self.material_preset}\n")
            fh.write(f"anchoring_preset: {self.anchoring_preset}\n")
            fh.write(f"temperature_window_K: {result.temperatures_K[0]:.6f} -> {result.temperatures_K[-1]:.6f}\n")
            fh.write(f"headline_midpoint_loop_width: {result.midpoint_loop_width:.6e}\n")
            fh.write(f"midpoint_temperature_K: {result.midpoint_temperature_K:.6f}\n")
            fh.write(f"midpoint_energy_density_gap_J_m3: {result.midpoint_energy_density_gap:.6e}\n")
            fh.write(f"midpoint_order_parameter_gap: {result.midpoint_order_parameter_gap:.6e}\n")
            fh.write(f"midpoint_principal_order_gap: {result.midpoint_principal_order_gap:.6e}\n")
            fh.write(f"midpoint_loop_width: {result.midpoint_loop_width:.6e}\n")
            fh.write(f"max_energy_density_gap_J_m3: {result.max_energy_density_gap:.6e}\n")
            fh.write(f"max_order_parameter_gap: {result.max_order_parameter_gap:.6e}\n")
            fh.write(f"max_principal_order_gap: {result.max_principal_order_gap:.6e}\n")
            fh.write(f"final_cooling_energy_density_J_m3: {result.cooling.energy_densities[-1]:.6e}\n")
            fh.write(f"final_cooling_order_parameter: {result.cooling.order_parameters[-1]:.6e}\n")
            fh.write(f"final_cooling_principal_order: {result.cooling.principal_order_parameters[-1]:.6e}\n")

    def collect_validation_results(self) -> List[ReferenceCheckResult]:
        """Run all reference and benchmark checks used to validate the solver."""
        results: List[ReferenceCheckResult] = []
        results.extend(self.validate_reference_cases())
        results.extend(self.benchmark_uniform_directors())
        results.extend(self.benchmark_helical_twist())
        results.extend(self.benchmark_cylinder_alignment_cases())
        return results

    def save_validation_report(self, results: List[ReferenceCheckResult], output_prefix: str) -> None:
        """Write a compact validation bundle for quick regression checks.

        The CSV keeps the raw numeric values for comparisons, while the text
        file is meant to be readable in a terminal or CI log.
        """
        csv_filename = f"{output_prefix}_validation.csv"
        txt_filename = f"{output_prefix}_validation.txt"

        with open(csv_filename, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(
                [
                    "name",
                    "excess_energy_j",
                    "energy_density_j_m3",
                    "energy_kbt",
                    "expected_behavior",
                    "reference_energy_density_j_m3",
                    "density_delta_j_m3",
                ]
            )
            for result in results:
                writer.writerow(
                    [
                        result.name,
                        f"{result.excess_energy:.16e}",
                        f"{result.energy_density:.16e}",
                        f"{result.energy_kbt:.16e}",
                        result.expected_behavior,
                        "" if result.reference_energy_density is None else f"{result.reference_energy_density:.16e}",
                        "" if result.density_delta is None else f"{result.density_delta:.16e}",
                    ]
                )

        with open(txt_filename, "w", encoding="utf-8") as fh:
            fh.write("Liquid-crystal FEM validation report\n")
            fh.write(f"material_preset: {self.material_preset}\n")
            fh.write(f"anchoring_preset: {self.anchoring_preset}\n")
            fh.write(f"solver_method: {self.solver_method}\n")
            fh.write(f"domain_volume_m3: {self.domain_volume:.16e}\n")
            fh.write("\n")
            for result in results:
                fh.write(f"{result.name}\n")
                fh.write(f"  excess_energy_j: {result.excess_energy:.16e}\n")
                fh.write(f"  energy_density_j_m3: {result.energy_density:.16e}\n")
                fh.write(f"  energy_kbt: {result.energy_kbt:.16e}\n")
                fh.write(f"  expected_behavior: {result.expected_behavior}\n")
                if result.reference_energy_density is not None:
                    fh.write(f"  reference_energy_density_j_m3: {result.reference_energy_density:.16e}\n")
                if result.density_delta is not None:
                    fh.write(f"  density_delta_j_m3: {result.density_delta:.16e}\n")
                fh.write("\n")

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

    def benchmark_helical_twist(self, twist_wavenumber: float | None = None) -> List[ReferenceCheckResult]:
        """Benchmark a simple helical twist field against the material model.

        The exact continuum result for n=(cos qz, sin qz, 0) is known for the
        one-constant and anisotropic Oseen-Frank forms, so this is a compact
        way to compare the two backends on a nontrivial distortion.
        """
        if twist_wavenumber is None:
            length = max(float(np.ptp(self.vertices[:, 2])), 1.0e-30)
            twist_wavenumber = 2.0 * np.pi / length

        original_directors = self.directors.copy()
        original_side_mode = self.side_mode
        original_cap_mode = self.cap_mode
        original_W_side = self.W_side
        original_W_caps = self.W_caps

        try:
            self.side_mode = "free"
            self.cap_mode = "free"
            self.W_side = 0.0
            self.W_caps = 0.0
            self.directors = self._helical_twist_field(twist_wavenumber)
            energy, _ = self.compute_energy_and_gradient(self.directors)
            excess = self._excess_free_energy(float(energy))
            if self.material_state.model == "anisotropic_oseen_frank":
                reference_density = 0.5 * self.elastic_constants["K2"] * (twist_wavenumber - self.q0) ** 2
            else:
                reference_density = 0.5 * self.K * (twist_wavenumber - self.q0) ** 2

            return [
                ReferenceCheckResult(
                    name="helical_twist",
                    excess_energy=excess,
                    energy_density=self._energy_density(excess),
                    energy_kbt=self._energy_in_kbt(excess),
                    expected_behavior="helical twist should produce a positive bulk distortion energy with the expected twist scaling",
                    reference_energy_density=reference_density,
                    density_delta=self._energy_density(excess) - reference_density,
                )
            ]
        finally:
            self.directors = original_directors
            self.side_mode = original_side_mode
            self.cap_mode = original_cap_mode
            self.W_side = original_W_side
            self.W_caps = original_W_caps


    def benchmark_cylinder_alignment_cases(self) -> List[ReferenceCheckResult]:
        """Benchmark a few simple cylinder-specific alignment families.

        These are useful for checking whether the anchoring choices are doing
        the expected thing:
        - a radial field should favor homeotropic sidewall anchoring
        - an azimuthal field should favor planar sidewall anchoring
        - both fields should be disfavored by the opposite anchoring choice
        """
        original_directors = self.directors.copy()
        original_side_mode = self.side_mode
        original_cap_mode = self.cap_mode
        original_W_side = self.W_side
        original_W_caps = self.W_caps

        cases = [
            (
                "radial_homeotropic_side",
                self._radial_field(),
                "homeotropic",
                "free",
                self.W_side,
                0.0,
                "radial field should be low-energy under homeotropic sidewall anchoring",
            ),
            (
                "radial_planar_side",
                self._radial_field(),
                "planar",
                "free",
                self.W_side,
                0.0,
                "radial field should be penalized by planar sidewall anchoring",
            ),
            (
                "azimuthal_planar_side",
                self._azimuthal_field(),
                "planar",
                "free",
                self.W_side,
                0.0,
                "azimuthal field should be low-energy under planar sidewall anchoring",
            ),
            (
                "azimuthal_homeotropic_side",
                self._azimuthal_field(),
                "homeotropic",
                "free",
                self.W_side,
                0.0,
                "azimuthal field should be penalized by homeotropic sidewall anchoring",
            ),
        ]

        results: List[ReferenceCheckResult] = []

        try:
            for name, field, side_mode, cap_mode, W_side, W_caps, expected in cases:
                self.directors = field.copy()
                self.side_mode = side_mode
                self.cap_mode = cap_mode
                self.W_side = W_side
                self.W_caps = W_caps
                energy, _ = self.compute_energy_and_gradient(self.directors)
                excess = self._excess_free_energy(float(energy))
                results.append(
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

        loaded_vertices = data[:, :3]
        if loaded_vertices.shape == self.vertices.shape and np.allclose(loaded_vertices, self.vertices, atol=1.0e-12):
            self.directors = self.normalize_directors(data[:, 3:6])
            return

        self.vertices = loaded_vertices
        self.n_nodes = len(self.vertices)
        self._spacing = self._estimate_spacing(self.vertices)
        self._volume_tol = max((self._spacing ** 3) * 1.0e-8, 1.0e-30)
        self._energy_scale = max(self.K * self._spacing, 1.0e-30)
        self._gradient_scale = self._energy_scale
        self.directors = self.normalize_directors(data[:, 3:6])
        self._mesh_tetrahedra = None
        self._mesh_boundary_faces = None
        self._mesh_boundary_face_kind = None
        self._mesh_metadata = {}
        self._build_mesh()
        self._characteristic_length = self._estimate_characteristic_length(self.vertices)
        self._elastic_density_scale = self._compute_elastic_density_scale()
        self.reference_energy = self._reference_energy()

    def plot_director_field(
        self,
        ax,
        title: str = "Finite Element Director Field",
        add_colorbar: bool = True,
    ) -> None:
        ax.clear()
        dirs = self.normalize_directors(self.directors)
        angles = self._director_angles_degrees(dirs)
        norm = plt.Normalize(vmin=0.0, vmax=90.0)
        cmap = plt.cm.viridis
        colors = cmap(norm(angles))
        xs = self.vertices[:, 0] * 1e6
        ys = self.vertices[:, 1] * 1e6
        zs = self.vertices[:, 2] * 1e6
        ax.scatter(xs, ys, zs, c=angles, cmap=cmap, norm=norm, s=16, alpha=0.85, depthshade=False)
        stride = max(len(self.vertices) // 250, 1)
        for point, direction, color in zip(self.vertices[::stride], dirs[::stride], colors[::stride]):
            ax.quiver(
                point[0] * 1e6,
                point[1] * 1e6,
                point[2] * 1e6,
                direction[0],
                direction[1],
                direction[2],
                color=color,
                length=0.08,
                normalize=True,
                pivot="middle",
                linewidth=0.8,
            )
        if add_colorbar:
            scalar_map = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
            scalar_map.set_array([])
            ax.figure.colorbar(scalar_map, ax=ax, pad=0.08, shrink=0.78, label="Director angle to z-axis (deg)")
        ax.set_title(title)
        ax.set_xlabel("X (µm)")
        ax.set_ylabel("Y (µm)")
        ax.set_zlabel("Z (µm)")
        ax.set_xlim(np.min(self.vertices[:, 0]) * 1e6, np.max(self.vertices[:, 0]) * 1e6)
        ax.set_ylim(np.min(self.vertices[:, 1]) * 1e6, np.max(self.vertices[:, 1]) * 1e6)
        ax.set_zlim(np.min(self.vertices[:, 2]) * 1e6, np.max(self.vertices[:, 2]) * 1e6)
        try:
            ax.set_box_aspect(
                (
                    np.ptp(self.vertices[:, 0]),
                    np.ptp(self.vertices[:, 1]),
                    np.ptp(self.vertices[:, 2]),
                )
            )
        except Exception:
            pass
        plt.draw()

    def _director_angles_degrees(self, directors: np.ndarray | None = None) -> np.ndarray:
        """Return director tilt angles to the cylinder axis in degrees."""
        dirs = self.normalize_directors(self.directors if directors is None else directors)
        cos_theta = np.clip(np.abs(dirs[:, 2]), 0.0, 1.0)
        return np.degrees(np.arccos(cos_theta))

    def _projected_coordinates(self, projection: str = "rho_z") -> Tuple[np.ndarray, np.ndarray, str, str]:
        """Return 2D coordinates and axis labels for a projection heatmap."""
        projection_key = projection.replace("-", "_").lower()
        if projection_key in {"rho_z", "rhoz", "radial_z", "radial"}:
            x_values = np.linalg.norm(self.vertices[:, :2], axis=1) * 1e6
            y_values = self.vertices[:, 2] * 1e6
            return x_values, y_values, "radial distance ρ (µm)", "z (µm)"
        if projection_key in {"x_z", "xz", "xslice"}:
            x_values = self.vertices[:, 0] * 1e6
            y_values = self.vertices[:, 2] * 1e6
            return x_values, y_values, "x (µm)", "z (µm)"
        if projection_key in {"x_y", "xy", "basal", "basal_plane"}:
            x_values = self.vertices[:, 0] * 1e6
            y_values = self.vertices[:, 1] * 1e6
            return x_values, y_values, "x (µm)", "y (µm)"
        if projection_key in {"theta_z", "thetaz", "surface", "unwrapped"}:
            theta = np.degrees(np.arctan2(self.vertices[:, 1], self.vertices[:, 0]))
            theta = np.mod(theta, 360.0)
            z_values = self.vertices[:, 2] * 1e6
            return theta, z_values, "azimuthal angle θ (deg)", "z (µm)"
        raise ValueError(f"Unsupported projection '{projection}'")

    @staticmethod
    def _projected_points(points_m: np.ndarray, projection: str = "rho_z") -> Tuple[np.ndarray, np.ndarray, str, str]:
        """Project arbitrary 3D points to a 2D plotting frame."""
        projection_key = projection.replace("-", "_").lower()
        points = np.asarray(points_m, dtype=float)
        if projection_key in {"rho_z", "rhoz", "radial_z", "radial"}:
            x_values = np.linalg.norm(points[:, :2], axis=1) * 1e6
            y_values = points[:, 2] * 1e6
            return x_values, y_values, "radial distance ρ (µm)", "z (µm)"
        if projection_key in {"x_z", "xz", "xslice"}:
            x_values = points[:, 0] * 1e6
            y_values = points[:, 2] * 1e6
            return x_values, y_values, "x (µm)", "z (µm)"
        if projection_key in {"x_y", "xy", "basal", "basal_plane"}:
            x_values = points[:, 0] * 1e6
            y_values = points[:, 1] * 1e6
            return x_values, y_values, "x (µm)", "y (µm)"
        if projection_key in {"theta_z", "thetaz", "surface", "unwrapped"}:
            theta = np.degrees(np.arctan2(points[:, 1], points[:, 0]))
            theta = np.mod(theta, 360.0)
            z_values = points[:, 2] * 1e6
            return theta, z_values, "azimuthal angle θ (deg)", "z (µm)"
        raise ValueError(f"Unsupported projection '{projection}'")

    def plot_projected_scalar_map(
        self,
        ax,
        scalars: Sequence[float],
        title: str,
        cbar_label: str,
        cmap: str = "viridis",
        projection: str = "rho_z",
        grid_bins: int = 120,
    ) -> None:
        """Plot a projected heatmap for a scalar field on the cylinder."""
        ax.clear()
        x_values, y_values, xlabel, ylabel = self._projected_coordinates(projection=projection)
        values = np.asarray(scalars, dtype=float)
        if values.size == 0:
            return

        x_edges = np.linspace(float(np.min(x_values)), float(np.max(x_values)), max(int(grid_bins), 2) + 1)
        y_edges = np.linspace(float(np.min(y_values)), float(np.max(y_values)), max(int(grid_bins), 2) + 1)
        weighted_sum, _, _ = np.histogram2d(x_values, y_values, bins=(x_edges, y_edges), weights=values)
        counts, _, _ = np.histogram2d(x_values, y_values, bins=(x_edges, y_edges))
        with np.errstate(divide="ignore", invalid="ignore"):
            averaged = np.divide(
                weighted_sum,
                counts,
                out=np.full_like(weighted_sum, np.nan, dtype=float),
                where=counts > 0,
            )
        heatmap = np.ma.masked_invalid(averaged.T)
        image = ax.imshow(
            heatmap,
            origin="lower",
            aspect="auto",
            extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],
            cmap=cmap,
        )
        ax.scatter(x_values, y_values, c="white", s=3, alpha=0.12, linewidths=0.0)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.figure.colorbar(image, ax=ax, pad=0.03, label=cbar_label)

    def plot_point_scalar_map(
        self,
        ax,
        x_values: Sequence[float],
        y_values: Sequence[float],
        scalars: Sequence[float],
        title: str,
        cbar_label: str,
        xlabel: str,
        ylabel: str,
        cmap: str = "viridis",
        grid_bins: int = 120,
    ) -> None:
        """Plot a projected heatmap from arbitrary point samples."""
        ax.clear()
        x_values = np.asarray(x_values, dtype=float)
        y_values = np.asarray(y_values, dtype=float)
        values = np.asarray(scalars, dtype=float)
        if values.size == 0:
            return

        x_edges = np.linspace(float(np.min(x_values)), float(np.max(x_values)), max(int(grid_bins), 2) + 1)
        y_edges = np.linspace(float(np.min(y_values)), float(np.max(y_values)), max(int(grid_bins), 2) + 1)
        weighted_sum, _, _ = np.histogram2d(x_values, y_values, bins=(x_edges, y_edges), weights=values)
        counts, _, _ = np.histogram2d(x_values, y_values, bins=(x_edges, y_edges))
        with np.errstate(divide="ignore", invalid="ignore"):
            averaged = np.divide(
                weighted_sum,
                counts,
                out=np.full_like(weighted_sum, np.nan, dtype=float),
                where=counts > 0,
            )
        heatmap = np.ma.masked_invalid(averaged.T)
        image = ax.imshow(
            heatmap,
            origin="lower",
            aspect="auto",
            extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],
            cmap=cmap,
        )
        ax.scatter(x_values, y_values, c="white", s=3, alpha=0.12, linewidths=0.0)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.figure.colorbar(image, ax=ax, pad=0.03, label=cbar_label)

    def save_director_orientation_map(
        self,
        directors: np.ndarray | None,
        filename: str,
        title: str = "Director Orientation Map",
        projection: str = "rho_z",
    ) -> None:
        """Save a 2D projected heatmap of the director tilt angle."""
        old_directors = self.directors
        try:
            if directors is not None:
                self.directors = directors
            fig, ax = plt.subplots(figsize=(10, 6))
            angles = self._director_angles_degrees()
            self.plot_projected_scalar_map(
                ax,
                angles,
                title=title,
                cbar_label="Director angle to z-axis (deg)",
                cmap="viridis",
                projection=projection,
            )
            fig.tight_layout()
            fig.savefig(filename, dpi=300)
            plt.close(fig)
        finally:
            self.directors = old_directors

    def save_temperature_field_map(
        self,
        temperature_field: np.ndarray | None,
        filename: str,
        title: str = "Temperature Field Map",
        projection: str = "rho_z",
    ) -> None:
        """Save a 2D projected heatmap of the local temperature field."""
        if temperature_field is None:
            return
        fig, ax = plt.subplots(figsize=(10, 6))
        self.plot_projected_scalar_map(
            ax,
            temperature_field,
            title=title,
            cbar_label="Temperature (K)",
            cmap="inferno",
            projection=projection,
        )
        fig.tight_layout()
        fig.savefig(filename, dpi=300)
        plt.close(fig)

    def _sidewall_anchoring_face_samples(
        self,
        directors: np.ndarray | None = None,
        temperature_field: np.ndarray | None = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return sidewall face samples for the Rapini-Papoular energy density."""
        dirs = self.normalize_directors(self.directors if directors is None else directors)
        local_temperature_field = (
            self._temperature_field(time=0.0) if temperature_field is None else np.asarray(temperature_field, dtype=float)
        )

        theta_degrees: List[float] = []
        z_um: List[float] = []
        density_j_m2: List[float] = []
        face_area_m2: List[float] = []

        for face in self.boundary_faces:
            if face.kind != "sidewall":
                continue

            node_vectors = dirs[face.nodes]
            dot_values = node_vectors @ face.normal
            face_mass = (face.area / 12.0) * (np.ones((3, 3)) + np.eye(3))
            weighted_dot = face_mass @ dot_values
            local_temperature = float(np.mean(local_temperature_field[face.nodes]))
            anchoring_scale = float(self._anchoring_temperature_scale(local_temperature, kind="sidewall"))
            W_local = self.W_side * anchoring_scale

            if self.side_mode == "free" or W_local == 0.0:
                face_energy = 0.0
            elif self.side_mode == "planar":
                face_energy = 0.5 * W_local * float(dot_values @ weighted_dot)
            elif self.side_mode == "homeotropic":
                face_energy = 0.5 * W_local * (face.area - float(dot_values @ weighted_dot))
            else:
                raise ValueError(f"Unknown anchoring mode: {self.side_mode}")

            centroid = np.mean(self.vertices[face.nodes], axis=0)
            theta = float(np.degrees(np.arctan2(centroid[1], centroid[0])) % 360.0)
            theta_degrees.append(theta)
            z_um.append(float(centroid[2] * 1e6))
            face_area_m2.append(float(face.area))
            density_j_m2.append(float(face_energy / face.area) if face.area > 0.0 else 0.0)

        return (
            np.asarray(theta_degrees, dtype=float),
            np.asarray(z_um, dtype=float),
            np.asarray(density_j_m2, dtype=float),
            np.asarray(face_area_m2, dtype=float),
        )

    def save_sidewall_anchoring_data(
        self,
        directors: np.ndarray | None,
        temperature_field: np.ndarray | None,
        filename: str,
    ) -> None:
        """Save per-face sidewall anchoring energies for the unwrapped wall."""
        theta_degrees, z_um, density_j_m2, face_area_m2 = self._sidewall_anchoring_face_samples(
            directors=directors,
            temperature_field=temperature_field,
        )
        if theta_degrees.size == 0:
            return

        with open(filename, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["theta_deg", "z_um", "surface_energy_density_j_m2", "face_area_m2"])
            for theta, z, density, area in zip(theta_degrees, z_um, density_j_m2, face_area_m2):
                writer.writerow([f"{theta:.8e}", f"{z:.8e}", f"{density:.8e}", f"{area:.8e}"])

    def save_sidewall_anchoring_map(
        self,
        directors: np.ndarray | None,
        temperature_field: np.ndarray | None,
        filename: str,
        title: str = "Sidewall Anchoring Map",
    ) -> None:
        """Save the sidewall Rapini-Papoular energy density on the theta-z wall."""
        theta_degrees, z_um, density_j_m2, _ = self._sidewall_anchoring_face_samples(
            directors=directors,
            temperature_field=temperature_field,
        )
        if theta_degrees.size == 0:
            return

        fig, ax = plt.subplots(figsize=(10, 6))
        self.plot_point_scalar_map(
            ax,
            theta_degrees,
            z_um,
            density_j_m2,
            title=title,
            cbar_label="Rapini-Papoular sidewall energy density (J/m^2)",
            xlabel="azimuthal angle θ (deg)",
            ylabel="z (µm)",
            cmap="magma",
        )
        fig.tight_layout()
        fig.savefig(filename, dpi=300)
        plt.close(fig)

    def _sidewall_anchoring_summary(
        self,
        directors: np.ndarray | None = None,
        temperature_field: np.ndarray | None = None,
    ) -> Tuple[float, float, float, float, float]:
        """Return compact summary statistics for the sidewall anchoring density."""
        _, _, density_j_m2, face_area_m2 = self._sidewall_anchoring_face_samples(
            directors=directors,
            temperature_field=temperature_field,
        )
        if density_j_m2.size == 0:
            return (float("nan"),) * 5
        mean_density = float(np.mean(density_j_m2))
        std_density = float(np.std(density_j_m2))
        min_density = float(np.min(density_j_m2))
        max_density = float(np.max(density_j_m2))
        total_energy = float(np.sum(density_j_m2 * face_area_m2))
        return mean_density, std_density, min_density, max_density, total_energy

    def compute_refinement_indicator(
        self,
        directors: np.ndarray | None = None,
        temperature_field: np.ndarray | None = None,
    ) -> RefinementIndicatorResult:
        """Return a simple per-element refinement score.

        The score is intentionally lightweight rather than fully adaptive: it
        highlights regions with strong director gradients or strong local
        temperature variation so we can later feed those hotspots into a real
        remeshing pass.
        """
        dirs = self.normalize_directors(self.directors if directors is None else directors)
        temp_field = self._temperature_field(time=0.0) if temperature_field is None else np.asarray(temperature_field, dtype=float)

        centroids: List[np.ndarray] = []
        director_scores: List[float] = []
        temperature_scores: List[float] = []

        char_length = max(self._characteristic_length, 1.0e-30)
        temp_span = max(float(np.ptp(temp_field)), 1.0e-9)

        for tet_idx, tet in enumerate(self.tetrahedra):
            nodes = tet
            g = self.tetra_gradN[tet_idx]
            n = dirs[nodes]
            grad_n = n.T @ g
            temp_grad = temp_field[nodes] @ g
            director_score = float(np.linalg.norm(grad_n, ord="fro") * char_length)
            temperature_score = float(np.linalg.norm(temp_grad) * char_length / temp_span)
            director_scores.append(director_score)
            temperature_scores.append(temperature_score)
            centroids.append(np.mean(self.vertices[nodes], axis=0))

        director_arr = np.asarray(director_scores, dtype=float)
        temperature_arr = np.asarray(temperature_scores, dtype=float)
        scores = director_arr + temperature_arr

        return RefinementIndicatorResult(
            centroids_m=np.asarray(centroids, dtype=float),
            scores=scores,
            director_scores=director_arr,
            temperature_scores=temperature_arr,
        )

    def save_refinement_indicator_data(
        self,
        directors: np.ndarray | None,
        temperature_field: np.ndarray | None,
        filename: str,
    ) -> None:
        """Save the per-element refinement hints to CSV."""
        indicator = self.compute_refinement_indicator(directors=directors, temperature_field=temperature_field)
        with open(filename, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(
                [
                    "centroid_x_um",
                    "centroid_y_um",
                    "centroid_z_um",
                    "refinement_score",
                    "director_score",
                    "temperature_score",
                ]
            )
            for centroid, score, director_score, temperature_score in zip(
                indicator.centroids_m,
                indicator.scores,
                indicator.director_scores,
                indicator.temperature_scores,
            ):
                writer.writerow(
                    [
                        f"{centroid[0] * 1e6:.8e}",
                        f"{centroid[1] * 1e6:.8e}",
                        f"{centroid[2] * 1e6:.8e}",
                        f"{score:.8e}",
                        f"{director_score:.8e}",
                        f"{temperature_score:.8e}",
                    ]
                )

    def save_refinement_indicator_map(
        self,
        directors: np.ndarray | None,
        temperature_field: np.ndarray | None,
        filename: str,
        title: str = "Mesh Refinement Hint Map",
        projection: str = "rho_z",
    ) -> None:
        """Save a projected map of the refinement score."""
        indicator = self.compute_refinement_indicator(directors=directors, temperature_field=temperature_field)
        if indicator.scores.size == 0:
            return
        fig, ax = plt.subplots(figsize=(10, 6))
        x_values, y_values, xlabel, ylabel = self._projected_points(indicator.centroids_m, projection=projection)
        self.plot_point_scalar_map(
            ax,
            x_values,
            y_values,
            indicator.scores,
            title=title,
            cbar_label="Refinement hint (dimensionless)",
            xlabel=xlabel,
            ylabel=ylabel,
            cmap="cividis",
        )
        fig.tight_layout()
        fig.savefig(filename, dpi=300)
        plt.close(fig)

    def suggest_refined_mesh_parameters(
        self,
        directors: np.ndarray | None = None,
        temperature_field: np.ndarray | None = None,
    ) -> MeshRefinementSuggestion | None:
        """Convert the refinement hints into a follow-up structured mesh recipe."""
        if self._mesh_metadata is None and self._mesh_tetrahedra is None:
            return None

        indicator = self.compute_refinement_indicator(directors=directors, temperature_field=temperature_field)
        if indicator.scores.size == 0:
            return None

        metadata = dict(self._mesh_metadata or {})
        # Keep the physical cylinder dimensions fixed and only recommend a
        # denser discretization where the current solution says it is needed.
        diameter_um = float(metadata.get("diameter_um", 2.0 * np.max(np.linalg.norm(self.vertices[:, :2], axis=1)) * 1e6))
        length_um = float(metadata.get("length_um", np.ptp(self.vertices[:, 2]) * 1e6))
        num_radial_layers = int(metadata.get("num_radial_layers", 5))
        num_theta_points = int(metadata.get("num_theta_points", 24))
        num_axial_layers = int(metadata.get("num_axial_layers", 12))
        radial_cluster_power = float(metadata.get("radial_cluster_power", 2.0))
        axial_cluster_power = float(metadata.get("axial_cluster_power", 2.0))

        scores = np.asarray(indicator.scores, dtype=float)
        centroids = np.asarray(indicator.centroids_m, dtype=float)
        if centroids.size == 0:
            return None

        radius = np.linalg.norm(centroids[:, :2], axis=1)
        z = centroids[:, 2]
        radius_norm = radius / max(float(np.max(radius)), 1.0e-30)
        z_min = float(np.min(z))
        z_span = max(float(np.ptp(z)), 1.0e-30)
        z_norm = (z - z_min) / z_span

        score_threshold = float(np.percentile(scores, 80.0))
        active = scores >= score_threshold
        if not np.any(active):
            active = scores >= float(np.mean(scores))
        if not np.any(active):
            active = np.ones_like(scores, dtype=bool)

        # High refinement scores near the wall favor more radial layers, while
        # scores near the caps favor more axial layers. This is a lightweight
        # proxy for a real adaptive remesher.
        wall_hot_fraction = float(np.mean(radius_norm[active] >= 0.8))
        cap_hot_fraction = float(np.mean((z_norm[active] <= 0.2) | (z_norm[active] >= 0.8)))
        mean_score = float(np.mean(scores))
        max_score = float(np.max(scores))
        boundary_dominance = wall_hot_fraction - cap_hot_fraction

        if wall_hot_fraction >= 0.5:
            # Wall-dominated hotspots usually mean the boundary layer and
            # azimuthal texture are under-resolved, so bias the next mesh more
            # strongly toward radial and theta resolution.
            num_radial_layers += 3
            num_theta_points += 6
            radial_cluster_power = min(radial_cluster_power + 1.0, 6.0)
        if cap_hot_fraction >= 0.5:
            # Cap-dominated hotspots are more naturally handled by adding
            # axial layers and tightening the clustering near the ends.
            num_axial_layers += 3
            axial_cluster_power = min(axial_cluster_power + 1.0, 6.0)

        if boundary_dominance >= 0.25:
            num_theta_points += 4
        elif boundary_dominance <= -0.25:
            num_axial_layers += 1

        if max_score > 5.0 * max(mean_score, 1.0e-12):
            # A very sharp hotspot gets one more layer in both directions to
            # keep the next pass conservative instead of overconfident.
            num_radial_layers += 1
            num_axial_layers += 1

        return MeshRefinementSuggestion(
            diameter_um=diameter_um,
            length_um=length_um,
            num_radial_layers=max(num_radial_layers, 3),
            num_theta_points=max(num_theta_points, 12),
            num_axial_layers=max(num_axial_layers, 3),
            radial_cluster_power=radial_cluster_power,
            axial_cluster_power=axial_cluster_power,
            wall_hot_fraction=wall_hot_fraction,
            cap_hot_fraction=cap_hot_fraction,
            max_refinement_score=max_score,
            mean_refinement_score=mean_score,
        )

    def current_mesh_recipe(self) -> MeshRecipe | None:
        """Return the structured-mesh recipe currently loaded by the solver."""
        if self._mesh_metadata is None and self._mesh_tetrahedra is None:
            return None
        metadata = dict(self._mesh_metadata or {})
        return MeshRecipe(
            num_radial_layers=int(metadata.get("num_radial_layers", 5)),
            num_theta_points=int(metadata.get("num_theta_points", 24)),
            num_axial_layers=int(metadata.get("num_axial_layers", 12)),
            radial_cluster_power=float(metadata.get("radial_cluster_power", 2.0)),
            axial_cluster_power=float(metadata.get("axial_cluster_power", 2.0)),
        )

    def save_refined_solve_mesh(
        self,
        directors: np.ndarray | None,
        temperature_field: np.ndarray | None,
        filename: str,
    ) -> None:
        """Generate a follow-up structured mesh with locally increased resolution."""
        suggestion = self.suggest_refined_mesh_parameters(directors=directors, temperature_field=temperature_field)
        if suggestion is None:
            return

        cylinder = CylinderGrid(
            diameter_um=suggestion.diameter_um,
            length_um=suggestion.length_um,
            num_boundary_points_per_z=suggestion.num_theta_points,
            num_z_levels=suggestion.num_axial_layers,
            num_inner_points=20,
            min_distance_um=max(suggestion.diameter_um / max(suggestion.num_radial_layers * 2.0, 1.0), 0.1),
        )
        cylinder.save_structured_solve_mesh(
            filename,
            num_radial_layers=suggestion.num_radial_layers,
            num_theta_points=suggestion.num_theta_points,
            num_axial_layers=suggestion.num_axial_layers,
            radial_cluster_power=suggestion.radial_cluster_power,
            axial_cluster_power=suggestion.axial_cluster_power,
        )

    def save_refined_mesh_suggestion(
        self,
        directors: np.ndarray | None,
        temperature_field: np.ndarray | None,
        filename: str,
    ) -> None:
        """Write the refinement prescription in a compact JSON file."""
        suggestion = self.suggest_refined_mesh_parameters(directors=directors, temperature_field=temperature_field)
        if suggestion is None:
            return
        with open(filename, "w", encoding="utf-8") as fh:
            json.dump(
                {
                    "diameter_um": suggestion.diameter_um,
                    "length_um": suggestion.length_um,
                    "num_radial_layers": suggestion.num_radial_layers,
                    "num_theta_points": suggestion.num_theta_points,
                    "num_axial_layers": suggestion.num_axial_layers,
                    "radial_cluster_power": suggestion.radial_cluster_power,
                    "axial_cluster_power": suggestion.axial_cluster_power,
                    "wall_hot_fraction": suggestion.wall_hot_fraction,
                    "cap_hot_fraction": suggestion.cap_hot_fraction,
                    "max_refinement_score": suggestion.max_refinement_score,
                    "mean_refinement_score": suggestion.mean_refinement_score,
                },
                fh,
                indent=2,
                sort_keys=True,
            )

    def save_director_alignment_map(
        self,
        directors: np.ndarray | None,
        filename: str,
        title: str = "Director Alignment Map",
        projection: str = "rho_z",
    ) -> None:
        """Save a 2D projected heatmap of the local nematic alignment."""
        if directors is None:
            return
        old_directors = self.directors
        try:
            self.directors = directors
            fig, ax = plt.subplots(figsize=(10, 6))
            dirs = self.normalize_directors(self.directors)
            alignment = 0.5 * (3.0 * np.square(np.abs(dirs[:, 2])) - 1.0)
            self.plot_projected_scalar_map(
                ax,
                alignment,
                title=title,
                cbar_label="Local alignment Szz-like score",
                cmap="plasma",
                projection=projection,
            )
            fig.tight_layout()
            fig.savefig(filename, dpi=300)
            plt.close(fig)
        finally:
            self.directors = old_directors

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
    mesh_file: str | None = None,
    checkpoint_file: str | None = None,
    run_time: int = 500,
    output_prefix: str = "fem",
    material_preset: str = "5CB_room_temperature",
    anchoring_preset: str = "planar_side_homeotropic_caps",
    sidewall_anchoring_strength: float | None = None,
    cap_anchoring_strength: float | None = None,
    temperature_profile: dict | None = None,
    solver_method: str = "trust-krylov",
    random_seed: int | None = None,
) -> Tuple[LiquidCrystalFEMSolver, List[float]]:
    """Convenience wrapper for the main program."""
    # Apply Rapini-Papoular anchoring on the full closed cylinder boundary by
    # default. The preset can be swapped for other patch-wise boundary setups.
    solver = LiquidCrystalFEMSolver(
        coordinates_file,
        mesh_file=mesh_file,
        max_iterations=run_time,
        material_preset=material_preset,
        anchoring_preset=anchoring_preset,
        W_side=sidewall_anchoring_strength,
        W_caps=cap_anchoring_strength,
        temperature_profile=temperature_profile,
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
    mesh_file: str | None = None,
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
    temperature_profile: dict | None = None,
    solver_method: str = "trust-krylov",
    random_seed: int | None = None,
) -> Tuple[LiquidCrystalFEMSolver, DynamicsResult]:
    """Run overdamped LC dynamics and write trace/visualization outputs."""
    solver = LiquidCrystalFEMSolver(
        coordinates_file,
        mesh_file=mesh_file,
        max_iterations=1,
        material_preset=material_preset,
        anchoring_preset=anchoring_preset,
        W_side=sidewall_anchoring_strength,
        W_caps=cap_anchoring_strength,
        temperature_profile=temperature_profile,
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


def run_fem_temperature_sweep(
    coordinates_file: str,
    mesh_file: str | None = None,
    checkpoint_file: str | None = None,
    temperatures_K: Sequence[float] | None = None,
    output_prefix: str = "temperature_sweep",
    material_preset: str = "5CB_room_temperature",
    anchoring_preset: str = "planar_side_homeotropic_caps",
    sidewall_anchoring_strength: float | None = None,
    cap_anchoring_strength: float | None = None,
    temperature_profile: dict | None = None,
    solver_method: str = "trust-krylov",
    random_seed: int | None = None,
    run_time: int = 500,
) -> Tuple[LiquidCrystalFEMSolver, TemperatureSweepResult]:
    """Relax the solver across a sequence of temperatures and save diagnostics."""
    solver = LiquidCrystalFEMSolver(
        coordinates_file,
        mesh_file=mesh_file,
        max_iterations=run_time,
        material_preset=material_preset,
        anchoring_preset=anchoring_preset,
        W_side=sidewall_anchoring_strength,
        W_caps=cap_anchoring_strength,
        temperature_profile=temperature_profile,
        solver_method=solver_method,
        random_seed=random_seed,
    )
    if checkpoint_file:
        solver.load_director_field(checkpoint_file)

    if temperatures_K is None:
        temperatures = np.linspace(295.0, 310.0, 9)
    else:
        temperatures = [float(value) for value in temperatures_K]

    result = solver.simulate_temperature_sweep(temperatures)
    solver.save_temperature_sweep_bundle(result, output_prefix=output_prefix)
    return solver, result


def run_fem_temperature_hysteresis(
    coordinates_file: str,
    mesh_file: str | None = None,
    checkpoint_file: str | None = None,
    temperatures_K: Sequence[float] | None = None,
    output_prefix: str = "temperature_hysteresis",
    material_preset: str = "5CB_room_temperature",
    anchoring_preset: str = "planar_side_homeotropic_caps",
    sidewall_anchoring_strength: float | None = None,
    cap_anchoring_strength: float | None = None,
    temperature_profile: dict | None = None,
    solver_method: str = "lbfgs",
    random_seed: int | None = None,
    run_time: int = 500,
) -> Tuple[LiquidCrystalFEMSolver, TemperatureHysteresisResult]:
    """Run paired heating/cooling sweeps and save the hysteresis diagnostics."""
    solver = LiquidCrystalFEMSolver(
        coordinates_file,
        mesh_file=mesh_file,
        max_iterations=run_time,
        material_preset=material_preset,
        anchoring_preset=anchoring_preset,
        W_side=sidewall_anchoring_strength,
        W_caps=cap_anchoring_strength,
        temperature_profile=temperature_profile,
        solver_method=solver_method,
        random_seed=random_seed,
    )
    if checkpoint_file:
        solver.load_director_field(checkpoint_file)

    if temperatures_K is None:
        temperatures = np.linspace(295.0, 310.0, 9)
    else:
        temperatures = [float(value) for value in temperatures_K]

    result = solver.simulate_temperature_hysteresis(temperatures)
    solver.save_temperature_hysteresis_bundle(result, output_prefix=output_prefix)
    return solver, result
