"""YAML-backed configuration helpers for the liquid crystal solver."""

from __future__ import annotations

import copy
import os
from typing import Any, Dict

import yaml

DEFAULT_CONFIG_PATH = "lc_fem_config.yaml"

DEFAULT_CONFIG: Dict[str, Any] = {
    "coordinates_file": "straight_cylinder_grid_with_grid.txt",
    "checkpoint_file": None,
    "run_time": 500,
    "output_prefix": "cholesteric_fem",
    "anchoring_preset": "planar_side_homeotropic_caps",
    "solver_method": "trust-krylov",
    "grid": {
        "diameter_um": 5,
        "length_um": 20,
        "num_boundary_points_per_z": 20,
        "num_z_levels": 20,
        "num_inner_points": 20,
        "min_distance_um": 1.0,
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


def save_default_config(config_path: str = DEFAULT_CONFIG_PATH) -> None:
    """Write the default YAML configuration to disk."""
    with open(config_path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(DEFAULT_CONFIG, fh, sort_keys=False)
