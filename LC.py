"""FEM command-line entry point for the liquid crystal cylinder solver."""

from __future__ import annotations

import argparse
import logging
import os
import re

# Keep Matplotlib usable in headless shells and CI.
os.environ["MPLCONFIGDIR"] = os.path.join("/tmp", "codex_mplconfig")
os.environ["XDG_CACHE_HOME"] = os.path.join("/tmp", "codex_cache")
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)
os.makedirs(os.environ["XDG_CACHE_HOME"], exist_ok=True)
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from grid1 import CylinderGrid
from lc_config import DEFAULT_CONFIG_PATH, load_solver_config
from lc_fem import run_fem_dynamics, run_fem_solver

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def ensure_default_cylinder_grid(filename: str, grid_config: dict | None = None) -> str:
    """Create the canonical straight-cylinder point cloud if it is missing."""
    if os.path.exists(filename):
        return filename

    grid_config = grid_config or {}
    cylinder = CylinderGrid(
        diameter_um=grid_config.get("diameter_um", 5),
        length_um=grid_config.get("length_um", 20),
        num_boundary_points_per_z=grid_config.get("num_boundary_points_per_z", 20),
        num_z_levels=grid_config.get("num_z_levels", 20),
        num_inner_points=grid_config.get("num_inner_points", 20),
        min_distance_um=grid_config.get("min_distance_um", 1.0),
    )
    cylinder.generate_straight_cylinder_with_grid()
    cylinder.save_grid_to_file(filename)
    logger.info("Generated default cylinder grid at %s", filename)
    return filename


def main(argv=None):
    """Command-line entry point for the FEM solver."""
    parser = argparse.ArgumentParser(description="Run the liquid-crystal FEM solver on a cylindrical grid.")
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH, help="YAML configuration file with solver defaults.")
    parser.add_argument("--coordinates-file", default=None, help="Input point-cloud file produced by grid1.py.")
    parser.add_argument("--checkpoint-file", default=None, help="Optional director checkpoint to resume from.")
    parser.add_argument(
        "--simulation-mode",
        default=None,
        choices=["static", "dynamics"],
        help="Choose static minimization or overdamped LC dynamics.",
    )
    parser.add_argument("--run-time", type=int, default=None, help="Maximum number of FEM optimization iterations.")
    parser.add_argument("--output-prefix", default=None, help="Prefix for output files.")
    parser.add_argument("--random-seed", type=int, default=None, help="Seed for reproducible solver randomness.")
    parser.add_argument(
        "--material-preset",
        default=None,
        choices=[
            "5CB_room_temperature",
            "5CB_anisotropic_room_temperature",
            "generic_nematic",
            "generic_cholesteric",
        ],
        help="Named liquid-crystal material preset for elastic constants.",
    )
    parser.add_argument(
        "--anchoring-preset",
        default=None,
        choices=["planar_side_homeotropic_caps", "planar_all", "homeotropic_all", "free"],
        help="Named boundary-anchoring preset to use on the cylinder hull.",
    )
    parser.add_argument(
        "--sidewall-anchoring-strength",
        type=float,
        default=None,
        help="Rapini-Papoular strength applied to sidewall faces.",
    )
    parser.add_argument(
        "--cap-anchoring-strength",
        type=float,
        default=None,
        help="Rapini-Papoular strength applied to end-cap faces.",
    )
    parser.add_argument(
        "--solver-method",
        default=None,
        choices=["trust-krylov", "newton-cg", "lbfgs"],
        help="Nonlinear optimizer used by the FEM solver.",
    )
    parser.add_argument("--total-time", type=float, default=None, help="Total model time for dynamics runs.")
    parser.add_argument("--time-step", type=float, default=None, help="Time step for overdamped dynamics.")
    parser.add_argument("--mobility", type=float, default=None, help="Mobility factor for dynamics updates.")
    parser.add_argument(
        "--thermal-noise-strength",
        type=float,
        default=None,
        help="Dimensionless stochastic forcing amplitude for dynamics runs.",
    )
    parser.add_argument(
        "--snapshot-interval",
        type=int,
        default=None,
        help="Write a dynamics snapshot every N steps.",
    )
    parser.add_argument(
        "--histogram-bins",
        type=int,
        default=None,
        help="Number of bins used for snapshot director-angle histograms.",
    )
    args = parser.parse_args(argv)

    config = load_solver_config(args.config)
    coordinates_file = args.coordinates_file or config["coordinates_file"]
    checkpoint_file = args.checkpoint_file if args.checkpoint_file is not None else config.get("checkpoint_file")
    simulation_mode = args.simulation_mode or config["simulation_mode"]
    run_time = args.run_time if args.run_time is not None else config["run_time"]
    output_prefix = args.output_prefix or config["output_prefix"]
    random_seed = args.random_seed if args.random_seed is not None else config.get("random_seed")
    material_preset = args.material_preset or config.get("material_preset", "5CB_room_temperature")
    anchoring_preset = args.anchoring_preset or config["anchoring_preset"]
    solver_method = args.solver_method or config["solver_method"]
    dynamics_config = config.get("dynamics", {})
    total_time = args.total_time if args.total_time is not None else dynamics_config.get("total_time", 10.0)
    time_step = args.time_step if args.time_step is not None else dynamics_config.get("time_step", 0.05)
    mobility = args.mobility if args.mobility is not None else dynamics_config.get("mobility", 1.0)
    thermal_noise_strength = (
        args.thermal_noise_strength
        if args.thermal_noise_strength is not None
        else dynamics_config.get("thermal_noise_strength", 0.01)
    )
    snapshot_interval = (
        args.snapshot_interval if args.snapshot_interval is not None else dynamics_config.get("snapshot_interval", 10)
    )
    histogram_bins = args.histogram_bins if args.histogram_bins is not None else dynamics_config.get("histogram_bins", 30)
    dynamics_seed = dynamics_config.get("random_seed", random_seed)
    sidewall_anchoring_strength = (
        args.sidewall_anchoring_strength
        if args.sidewall_anchoring_strength is not None
        else config.get("sidewall_anchoring_strength")
    )
    cap_anchoring_strength = (
        args.cap_anchoring_strength
        if args.cap_anchoring_strength is not None
        else config.get("cap_anchoring_strength")
    )

    coordinates_file = ensure_default_cylinder_grid(coordinates_file, grid_config=config.get("grid"))

    if checkpoint_file is None:
        checkpoint_files = [name for name in os.listdir() if name.startswith("checkpoint_iter_")]
        if checkpoint_files:
            checkpoint_files.sort(key=lambda name: int(re.findall(r"\d+", name)[0]))
            checkpoint_file = checkpoint_files[-1]
            logger.info("Found checkpoint %s; resuming the FEM solver from that state.", checkpoint_file)

    if simulation_mode == "dynamics":
        solver, result = run_fem_dynamics(
            coordinates_file=coordinates_file,
            checkpoint_file=checkpoint_file,
            total_time=total_time,
            time_step=time_step,
            mobility=mobility,
            thermal_noise_strength=thermal_noise_strength,
            snapshot_interval=snapshot_interval,
            histogram_bins=histogram_bins,
            output_prefix=output_prefix,
            material_preset=material_preset,
            anchoring_preset=anchoring_preset,
            sidewall_anchoring_strength=sidewall_anchoring_strength,
            cap_anchoring_strength=cap_anchoring_strength,
            solver_method=solver_method,
            random_seed=dynamics_seed,
        )
        logger.info("Final dynamic excess free energy: %s", solver.energy_summary(result.energies[-1]))
    else:
        solver, energies = run_fem_solver(
            coordinates_file=coordinates_file,
            checkpoint_file=checkpoint_file,
            run_time=run_time,
            output_prefix=output_prefix,
            material_preset=material_preset,
            anchoring_preset=anchoring_preset,
            sidewall_anchoring_strength=sidewall_anchoring_strength,
            cap_anchoring_strength=cap_anchoring_strength,
            solver_method=solver_method,
            random_seed=random_seed,
        )
        logger.info("Final continuum excess free energy: %s", solver.energy_summary(energies[-1]))

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")
    plot_title = "Dynamic FEM Director Field" if simulation_mode == "dynamics" else "Relaxed FEM Director Field"
    solver.plot_director_field(ax, title=plot_title)
    fig.savefig(f"{output_prefix}_director_field.jpg", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
