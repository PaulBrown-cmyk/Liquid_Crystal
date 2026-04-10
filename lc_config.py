"""YAML-backed configuration helpers for the liquid crystal solver."""

from __future__ import annotations

import copy
import os
from typing import Any, Dict

import yaml

DEFAULT_CONFIG_PATH = "lc_fem_config.yaml"

DEFAULT_CONFIG: Dict[str, Any] = {
    "coordinates_file": "straight_cylinder_grid_with_grid.txt",
    "mesh_file": "straight_cylinder_solve_mesh.npz",
    "checkpoint_file": None,
    "simulation_mode": "dynamics",
    "run_time": 500,
    "output_prefix": "cholesteric_fem",
    "material_preset": "5CB_room_temperature",
    "anchoring_preset": "planar_side_homeotropic_caps",
    "sidewall_anchoring_strength": 1.0e-5,
    "cap_anchoring_strength": 1.0e-5,
    "solver_method": "trust-krylov",
    "random_seed": None,
    "temperature_sweep": {
        "start_temperature_K": 295.0,
        "stop_temperature_K": 310.0,
        "num_points": 9,
        "max_iterations": 40,
        "solver_method": "lbfgs",
    },
    "temperature": {
        "enabled": False,
        "mode": "uniform",
        "law": "linear",
        "profile_name": None,
        "profile_file": "temperature_profiles.yaml",
        "base_temperature_K": 298.15,
        "reference_temperature_K": 298.15,
        "transition_temperature_K": 307.0,
        "gradient_z_K_per_m": 0.0,
        "gradient_r_K_per_m": 0.0,
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
    },
    "dynamics": {
        "total_time": 10.0,
        "time_step": 0.05,
        "mobility": 1.0,
        "thermal_noise_strength": 0.01,
        "snapshot_interval": 10,
        "histogram_bins": 30,
        "random_seed": None,
    },
    "qtensor": {
        "material_preset": "5CB_ldg_room_temperature",
        "temperature_K": 298.15,
        "max_iterations": 200,
        "tolerance": 1.0e-8,
    },
    "grid": {
        "diameter_um": 5,
        "length_um": 20,
        "num_boundary_points_per_z": 20,
        "num_z_levels": 20,
        "num_inner_points": 20,
        "min_distance_um": 1.0,
    },
    "mesh": {
        "num_radial_layers": 5,
        "num_theta_points": 24,
        "num_axial_layers": 12,
        "radial_cluster_power": 2.0,
        "axial_cluster_power": 2.0,
    },
}


def _deep_update(base: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_solver_config(config_path: str | None = None) -> Dict[str, Any]:
    """Load the solver configuration from YAML, falling back to defaults."""
    config = copy.deepcopy(DEFAULT_CONFIG)
    path = config_path or DEFAULT_CONFIG_PATH

    if path and os.path.exists(path):
        with open(path, "r", encoding="utf-8") as fh:
            loaded = yaml.safe_load(fh) or {}
        if not isinstance(loaded, dict):
            raise ValueError(f"Configuration file {path} must contain a YAML mapping.")
        _deep_update(config, loaded)

    return config


def load_temperature_profile(profile_name: str | None, profile_path: str | None = None) -> Dict[str, Any]:
    """Load a named thermal preset from a YAML file.

    The file is expected to map preset names to flat temperature dictionaries.
    If no profile is requested, return an empty mapping so callers can skip
    merging entirely.
    """
    if not profile_name:
        return {}

    path = profile_path or DEFAULT_CONFIG["temperature"].get("profile_file", "temperature_profiles.yaml")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Temperature profile file {path!r} does not exist.")

    with open(path, "r", encoding="utf-8") as fh:
        loaded = yaml.safe_load(fh) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Temperature profile file {path} must contain a YAML mapping.")

    if profile_name not in loaded:
        valid = ", ".join(sorted(map(str, loaded.keys())))
        raise ValueError(f"Unknown temperature profile {profile_name!r}. Valid profiles: {valid}")

    profile = loaded[profile_name]
    if not isinstance(profile, dict):
        raise ValueError(f"Temperature profile {profile_name!r} in {path} must be a YAML mapping.")

    return copy.deepcopy(profile)


def save_default_config(config_path: str = DEFAULT_CONFIG_PATH) -> None:
    """Write the default YAML configuration to disk."""
    with open(config_path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(DEFAULT_CONFIG, fh, sort_keys=False)
