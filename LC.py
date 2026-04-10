"""FEM command-line entry point for the liquid crystal cylinder solver."""

from __future__ import annotations

import argparse
import logging
import os
import re

import numpy as np

# Keep Matplotlib usable in headless shells and CI.
os.environ["MPLCONFIGDIR"] = os.path.join("/tmp", "codex_mplconfig")
os.environ["XDG_CACHE_HOME"] = os.path.join("/tmp", "codex_cache")
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)
os.makedirs(os.environ["XDG_CACHE_HOME"], exist_ok=True)
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from grid1 import CylinderGrid
from lc_config import DEFAULT_CONFIG, DEFAULT_CONFIG_PATH, load_solver_config, load_temperature_profile
from lc_fem import (
    LiquidCrystalFEMSolver,
    run_fem_dynamics,
    run_fem_solver,
    run_fem_temperature_hysteresis,
    run_fem_temperature_sweep,
)
from lc_ldg import LandauDeGennesQTensorSolver, run_qtensor_static

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


def ensure_default_cylinder_solve_mesh(
    filename: str,
    grid_config: dict | None = None,
    mesh_config: dict | None = None,
) -> str:
    """Create the body-fitted solve mesh if it is missing."""
    if os.path.exists(filename):
        return filename

    grid_config = grid_config or {}
    mesh_config = mesh_config or {}
    cylinder = CylinderGrid(
        diameter_um=grid_config.get("diameter_um", 5),
        length_um=grid_config.get("length_um", 20),
        num_boundary_points_per_z=grid_config.get("num_boundary_points_per_z", 20),
        num_z_levels=grid_config.get("num_z_levels", 20),
        num_inner_points=grid_config.get("num_inner_points", 20),
        min_distance_um=grid_config.get("min_distance_um", 1.0),
    )
    cylinder.save_structured_solve_mesh(
        filename,
        num_radial_layers=mesh_config.get("num_radial_layers", 5),
        num_theta_points=mesh_config.get("num_theta_points", 24),
        num_axial_layers=mesh_config.get("num_axial_layers", 12),
        radial_cluster_power=mesh_config.get("radial_cluster_power", 2.0),
        axial_cluster_power=mesh_config.get("axial_cluster_power", 2.0),
    )
    logger.info("Generated default cylinder solve mesh at %s", filename)
    return filename


def select_solve_mesh_file(
    output_prefix: str,
    mesh_file: str | None,
    grid_config: dict | None = None,
    mesh_config: dict | None = None,
) -> str:
    """Prefer a prior refined mesh for the same output prefix, then fall back."""
    grid_config = grid_config or {}
    mesh_config = mesh_config or {}
    if mesh_file:
        # An explicit mesh path always wins so users can force a specific
        # solve mesh without the auto-refinement logic stepping in.
        return ensure_default_cylinder_solve_mesh(mesh_file, grid_config=grid_config, mesh_config=mesh_config)

    # Reuse the most recent mesh produced by the same output prefix if the
    # previous run already generated a follow-up refined solve mesh.
    refined_mesh = f"{output_prefix}_refined_solve_mesh.npz"
    if os.path.exists(refined_mesh):
        logger.info("Using refined solve mesh from prior run: %s", refined_mesh)
        return refined_mesh

    # Otherwise fall back to the configured default solve mesh, generating it
    # on demand when the file is missing.
    default_mesh = DEFAULT_CONFIG.get("mesh_file", "straight_cylinder_solve_mesh.npz")
    return ensure_default_cylinder_solve_mesh(default_mesh, grid_config=grid_config, mesh_config=mesh_config)


def log_refined_mesh_suggestion(prefix: str, solver, refinement) -> None:
    """Log a compact follow-up mesh delta in the same format for every mode."""
    if refinement is None:
        return
    current = solver.current_mesh_recipe()
    if current is None:
        logger.info(
            "%s refined mesh suggestion: radial=%d, theta=%d, axial=%d, cluster(r,a)=%.2f/%.2f",
            prefix,
            refinement.num_radial_layers,
            refinement.num_theta_points,
            refinement.num_axial_layers,
            refinement.radial_cluster_power,
            refinement.axial_cluster_power,
        )
        return

    delta = (
        f"dr{refinement.num_radial_layers - current.num_radial_layers:+d} "
        f"dθ{refinement.num_theta_points - current.num_theta_points:+d} "
        f"dz{refinement.num_axial_layers - current.num_axial_layers:+d} "
        f"kr{refinement.radial_cluster_power - current.radial_cluster_power:+.2f} "
        f"ka{refinement.axial_cluster_power - current.axial_cluster_power:+.2f}"
    )
    logger.info(
        "%s refined mesh delta: %s",
        prefix,
        delta,
    )


def main(argv=None):
    """Command-line entry point for the FEM solver."""
    parser = argparse.ArgumentParser(description="Run the liquid-crystal FEM solver on a cylindrical grid.")
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH, help="YAML configuration file with solver defaults.")
    parser.add_argument("--coordinates-file", default=None, help="Input point-cloud file produced by grid1.py.")
    parser.add_argument("--mesh-file", default=None, help="Body-fitted solve mesh file produced by grid1.py.")
    parser.add_argument("--checkpoint-file", default=None, help="Optional director checkpoint to resume from.")
    parser.add_argument(
        "--simulation-mode",
        default=None,
        choices=["static", "dynamics", "temperature_sweep", "temperature_hysteresis", "qtensor_static", "qtensor_temperature_sweep"],
        help="Choose static minimization, overdamped LC dynamics, or temperature diagnostics.",
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
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Run the built-in validation suite and write a validation report bundle.",
    )
    parser.add_argument(
        "--temperature-enabled",
        dest="temperature_enabled",
        action="store_true",
        help="Enable the thermal profile and temperature-dependent couplings.",
    )
    parser.add_argument(
        "--temperature-disabled",
        dest="temperature_enabled",
        action="store_false",
        help="Disable the thermal profile even if the config enables it.",
    )
    parser.add_argument(
        "--temperature-mode",
        default=None,
        choices=["uniform", "linear_z", "linear_r", "linear_z_plus_radial", "oscillatory_z"],
        help="Spatial temperature profile applied to the FEM mesh.",
    )
    parser.add_argument(
        "--temperature-law",
        default=None,
        choices=["linear", "ldg_proxy"],
        help="How temperature softens the material response.",
    )
    parser.add_argument("--base-temperature", type=float, default=None, help="Base thermal field in kelvin.")
    parser.add_argument(
        "--temperature-gradient-z",
        type=float,
        default=None,
        help="Axial temperature gradient in K/m.",
    )
    parser.add_argument(
        "--temperature-gradient-r",
        type=float,
        default=None,
        help="Radial temperature gradient in K/m.",
    )
    parser.add_argument(
        "--temperature-oscillation-amplitude",
        type=float,
        default=None,
        help="Oscillatory temperature amplitude in kelvin.",
    )
    parser.add_argument(
        "--temperature-oscillation-frequency",
        type=float,
        default=None,
        help="Oscillation frequency for the thermal profile in Hz.",
    )
    parser.add_argument(
        "--temperature-phase",
        type=float,
        default=None,
        help="Phase offset for oscillatory thermal profiles in radians.",
    )
    parser.add_argument(
        "--temperature-transition",
        type=float,
        default=None,
        help="Approximate nematic-isotropic transition temperature for the thermal softening law.",
    )
    parser.add_argument(
        "--temperature-order-floor",
        type=float,
        default=None,
        help="Lower bound for the temperature softening factor.",
    )
    parser.add_argument(
        "--temperature-elastic-alpha",
        type=float,
        default=None,
        help="Linear elastic softening coefficient in 1/K.",
    )
    parser.add_argument(
        "--temperature-anchoring-alpha",
        type=float,
        default=None,
        help="Anchoring softening coefficient in 1/K.",
    )
    parser.add_argument(
        "--temperature-sidewall-anchoring-alpha",
        type=float,
        default=None,
        help="Sidewall anchoring softening coefficient in 1/K.",
    )
    parser.add_argument(
        "--temperature-cap-anchoring-alpha",
        type=float,
        default=None,
        help="Cap anchoring softening coefficient in 1/K.",
    )
    parser.add_argument(
        "--temperature-mobility-gamma",
        type=float,
        default=None,
        help="Mobility temperature sensitivity in 1/K.",
    )
    parser.add_argument(
        "--temperature-noise-multiplier",
        type=float,
        default=None,
        help="Extra multiplier on the thermal noise amplitude.",
    )
    parser.add_argument(
        "--temperature-profile",
        default=None,
        help="Named temperature preset to load from the thermal profile YAML file.",
    )
    parser.add_argument(
        "--temperature-profile-file",
        default=None,
        help="YAML file containing named temperature presets.",
    )
    parser.add_argument(
        "--qtensor-preset",
        default=None,
        choices=["5CB_ldg_room_temperature"],
        help="Named Landau-de Gennes material preset for the Q-tensor solver.",
    )
    parser.add_argument(
        "--qtensor-temperature",
        type=float,
        default=None,
        help="Static Q-tensor solve temperature in kelvin.",
    )
    parser.add_argument(
        "--qtensor-max-iterations",
        type=int,
        default=None,
        help="Maximum Q-tensor minimizer iterations.",
    )
    parser.add_argument(
        "--qtensor-tolerance",
        type=float,
        default=None,
        help="Convergence tolerance for the Q-tensor minimizer.",
    )
    parser.add_argument(
        "--temperature-sweep-start",
        type=float,
        default=None,
        help="Start temperature for the temperature sweep in kelvin.",
    )
    parser.add_argument(
        "--temperature-sweep-stop",
        type=float,
        default=None,
        help="Stop temperature for the temperature sweep in kelvin.",
    )
    parser.add_argument(
        "--temperature-sweep-count",
        type=int,
        default=None,
        help="Number of temperature points in the sweep.",
    )
    parser.set_defaults(temperature_enabled=None)
    args = parser.parse_args(argv)

    config = load_solver_config(args.config)
    coordinates_file = args.coordinates_file or config["coordinates_file"]
    configured_mesh_file = config.get("mesh_file")
    mesh_file = args.mesh_file if args.mesh_file is not None else (
        configured_mesh_file if configured_mesh_file != DEFAULT_CONFIG.get("mesh_file") else None
    )
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
    temperature_sweep_config = config.get("temperature_sweep", {})
    temperature_sweep_start = (
        args.temperature_sweep_start
        if args.temperature_sweep_start is not None
        else temperature_sweep_config.get("start_temperature_K", 295.0)
    )
    temperature_sweep_stop = (
        args.temperature_sweep_stop
        if args.temperature_sweep_stop is not None
        else temperature_sweep_config.get("stop_temperature_K", 310.0)
    )
    temperature_sweep_count = (
        args.temperature_sweep_count
        if args.temperature_sweep_count is not None
        else temperature_sweep_config.get("num_points", 9)
    )
    temperature_sweep_run_time = int(
        args.run_time if args.run_time is not None else temperature_sweep_config.get("max_iterations", run_time)
    )
    temperature_sweep_solver_method = args.solver_method or temperature_sweep_config.get("solver_method", solver_method)
    default_temperature_config = DEFAULT_CONFIG.get("temperature", {})
    temperature_config = dict(config.get("temperature", {}))
    temperature_profile_name = args.temperature_profile or temperature_config.get("profile_name")
    temperature_profile_file = args.temperature_profile_file or temperature_config.get("profile_file")
    qtensor_config = config.get("qtensor", {})
    qtensor_preset = args.qtensor_preset or qtensor_config.get("material_preset", "5CB_ldg_room_temperature")
    qtensor_temperature = args.qtensor_temperature if args.qtensor_temperature is not None else qtensor_config.get("temperature_K", 298.15)
    qtensor_max_iterations = (
        args.qtensor_max_iterations if args.qtensor_max_iterations is not None else qtensor_config.get("max_iterations", 200)
    )
    qtensor_tolerance = args.qtensor_tolerance if args.qtensor_tolerance is not None else qtensor_config.get("tolerance", 1.0e-8)
    if temperature_profile_name:
        temperature_config = load_temperature_profile(temperature_profile_name, temperature_profile_file)
        for key, value in dict(config.get("temperature", {})).items():
            if key in {"profile_name", "profile_file"}:
                continue
            if default_temperature_config.get(key) != value:
                temperature_config[key] = value
    if args.temperature_enabled is not None:
        temperature_config["enabled"] = args.temperature_enabled
    if args.temperature_mode is not None:
        temperature_config["mode"] = args.temperature_mode
    if args.temperature_law is not None:
        temperature_config["law"] = args.temperature_law
    if args.base_temperature is not None:
        temperature_config["base_temperature_K"] = args.base_temperature
    if args.temperature_gradient_z is not None:
        temperature_config["gradient_z_K_per_m"] = args.temperature_gradient_z
    if args.temperature_gradient_r is not None:
        temperature_config["gradient_r_K_per_m"] = args.temperature_gradient_r
    if args.temperature_oscillation_amplitude is not None:
        temperature_config["oscillation_amplitude_K"] = args.temperature_oscillation_amplitude
    if args.temperature_oscillation_frequency is not None:
        temperature_config["oscillation_frequency_hz"] = args.temperature_oscillation_frequency
    if args.temperature_phase is not None:
        temperature_config["phase_rad"] = args.temperature_phase
    if args.temperature_transition is not None:
        temperature_config["transition_temperature_K"] = args.temperature_transition
    if args.temperature_order_floor is not None:
        temperature_config["order_floor"] = args.temperature_order_floor
    if args.temperature_elastic_alpha is not None:
        temperature_config["elastic_alpha_per_K"] = args.temperature_elastic_alpha
    if args.temperature_anchoring_alpha is not None:
        temperature_config["anchoring_alpha_per_K"] = args.temperature_anchoring_alpha
    if args.temperature_sidewall_anchoring_alpha is not None:
        temperature_config["sidewall_anchoring_alpha_per_K"] = args.temperature_sidewall_anchoring_alpha
    if args.temperature_cap_anchoring_alpha is not None:
        temperature_config["cap_anchoring_alpha_per_K"] = args.temperature_cap_anchoring_alpha
    if args.temperature_mobility_gamma is not None:
        temperature_config["mobility_gamma_per_K"] = args.temperature_mobility_gamma
    if args.temperature_noise_multiplier is not None:
        temperature_config["noise_multiplier"] = args.temperature_noise_multiplier

    coordinates_file = ensure_default_cylinder_grid(coordinates_file, grid_config=config.get("grid"))
    # The plotting point cloud and the solve mesh are intentionally separate:
    # the plot grid keeps the old compatibility path, while the solve mesh
    # carries the actual FEM connectivity.
    mesh_file = select_solve_mesh_file(
        output_prefix=output_prefix,
        mesh_file=mesh_file,
        grid_config=config.get("grid"),
        mesh_config=config.get("mesh"),
    )

    if checkpoint_file is None:
        checkpoint_files = [name for name in os.listdir() if name.startswith("checkpoint_iter_")]
        if checkpoint_files:
            checkpoint_files.sort(key=lambda name: int(re.findall(r"\d+", name)[0]))
            checkpoint_file = checkpoint_files[-1]
            logger.info("Found checkpoint %s; resuming the FEM solver from that state.", checkpoint_file)

    if args.validate:
        solver = LiquidCrystalFEMSolver(
            coordinates_file=coordinates_file,
            mesh_file=mesh_file,
            material_preset=material_preset,
            anchoring_preset=anchoring_preset,
            W_side=sidewall_anchoring_strength,
            W_caps=cap_anchoring_strength,
            temperature_profile=temperature_config,
            solver_method=solver_method,
            random_seed=random_seed,
        )
        validation_results = solver.collect_validation_results()
        solver.save_validation_report(validation_results, output_prefix=output_prefix)
        logger.info("Wrote validation report bundle to %s_validation.[csv|txt]", output_prefix)
        for result in validation_results:
            logger.info(
                "%s: %s",
                result.name,
                solver.energy_summary(result.excess_energy),
            )
    elif simulation_mode == "qtensor_static":
        solver, result = run_qtensor_static(
            coordinates_file=coordinates_file,
            mesh_file=mesh_file,
            output_prefix=output_prefix,
            temperature_K=qtensor_temperature,
            temperature_profile=temperature_config,
            qtensor_preset=qtensor_preset,
            anchoring_preset=anchoring_preset,
            W_side=sidewall_anchoring_strength,
            W_caps=cap_anchoring_strength,
            max_iterations=qtensor_max_iterations,
            tolerance=qtensor_tolerance,
            random_seed=random_seed,
        )
        logger.info(
            "Final Q-tensor excess free energy: %.3e J/m^3 (%.6e J, %.3e kBT)",
            result.energy_density_j_m3,
            result.energy_j,
            result.energy_kbt,
        )
        logger.info(
            "Q-tensor order summary: mean S=%.6f, min S=%.6f, max S=%.6f, isotropic fraction=%.3f",
            result.mean_scalar_order,
            result.min_scalar_order,
            result.max_scalar_order,
            result.isotropic_fraction,
        )
        return
    elif simulation_mode == "qtensor_temperature_sweep":
        n_points = max(int(temperature_sweep_count), 1)
        temperatures = np.linspace(float(temperature_sweep_start), float(temperature_sweep_stop), n_points)
        solver = LandauDeGennesQTensorSolver(
            coordinates_file=coordinates_file,
            mesh_file=mesh_file,
            temperature_profile=temperature_config,
            temperature_K=qtensor_temperature,
            qtensor_preset=qtensor_preset,
            anchoring_preset=anchoring_preset,
            W_side=sidewall_anchoring_strength,
            W_caps=cap_anchoring_strength,
            max_iterations=temperature_sweep_run_time,
            tolerance=qtensor_tolerance,
            random_seed=random_seed,
        )
        sweep = solver.run_temperature_sweep(temperatures, output_prefix=output_prefix)
        logger.info(
            "Q-tensor temperature sweep completed from %.3f K to %.3f K; final mean S=%.6f, isotropic fraction=%.3f",
            sweep.temperatures_K[0],
            sweep.temperatures_K[-1],
            sweep.mean_scalar_order[-1],
            sweep.isotropic_fraction[-1],
        )
        return
    elif simulation_mode == "dynamics":
        solver, result = run_fem_dynamics(
            coordinates_file=coordinates_file,
            mesh_file=mesh_file,
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
            temperature_profile=temperature_config,
            solver_method=solver_method,
            random_seed=dynamics_seed,
        )
        logger.info("Final dynamic excess free energy: %s", solver.energy_summary(result.energies[-1]))
        sidewall_mean, sidewall_std, sidewall_min, sidewall_max, _ = solver._sidewall_anchoring_summary(
            directors=result.snapshots[-1] if result.snapshots else None,
            temperature_field=result.temperature_snapshots[-1] if result.temperature_snapshots else None,
        )
        logger.info(
            "Dynamics sidewall anchoring headline (%s): mean %.6e J/m^2, std %.6e, range %.6e -> %.6e",
            solver.side_mode,
            sidewall_mean,
            sidewall_std,
            sidewall_min,
            sidewall_max,
        )
        # Keep the next mesh recipe visible in the console so users can move
        # from a coarse run to the suggested follow-up mesh without opening
        # the JSON artifact by hand.
        refinement = solver.suggest_refined_mesh_parameters(
            directors=result.snapshots[-1] if result.snapshots else None,
            temperature_field=result.temperature_snapshots[-1] if result.temperature_snapshots else None,
        )
        log_refined_mesh_suggestion("Dynamics", solver, refinement)
    elif simulation_mode == "temperature_sweep":
        n_points = max(int(temperature_sweep_count), 1)
        temperatures = np.linspace(float(temperature_sweep_start), float(temperature_sweep_stop), n_points)
        solver, result = run_fem_temperature_sweep(
            coordinates_file=coordinates_file,
            mesh_file=mesh_file,
            checkpoint_file=checkpoint_file,
            temperatures_K=temperatures,
            output_prefix=output_prefix,
            material_preset=material_preset,
            anchoring_preset=anchoring_preset,
            sidewall_anchoring_strength=sidewall_anchoring_strength,
            cap_anchoring_strength=cap_anchoring_strength,
            temperature_profile=temperature_config,
            solver_method=temperature_sweep_solver_method,
            random_seed=random_seed,
            run_time=temperature_sweep_run_time,
        )
        if result.temperatures_K:
            logger.info(
                "Temperature sweep completed from %.3f K to %.3f K; final equilibrium: %s",
                result.temperatures_K[0],
                result.temperatures_K[-1],
                solver.energy_summary(result.excess_energies[-1]),
            )
            logger.info(
                "Order parameter shifted from %.6f to %.6f; principal order from %.6f to %.6f",
                result.order_parameters[0],
                result.order_parameters[-1],
                result.principal_order_parameters[0],
                result.principal_order_parameters[-1],
            )
            logger.info(
                "Sweep material response at endpoints: K_eff %.3f -> %.3f pN",
                result.effective_Keff_pN[0],
                result.effective_Keff_pN[-1],
            )
            midpoint = len(result.temperatures_K) // 2
            logger.info(
                "Sweep headline: order shift %.6f, midpoint %.3f K, Szz %.6f, K_eff %.3f -> %.3f pN",
                result.endpoint_order_shift,
                result.temperatures_K[midpoint],
                result.order_parameters[midpoint],
                result.effective_Keff_pN[0],
                result.effective_Keff_pN[-1],
            )
            sidewall_mean, sidewall_std, sidewall_min, sidewall_max, _ = solver._sidewall_anchoring_summary(
                directors=result.snapshots[-1] if result.snapshots else None,
                temperature_field=result.temperature_snapshots[-1] if result.temperature_snapshots else None,
            )
            logger.info(
                "Sweep sidewall anchoring headline (%s): mean %.6e J/m^2, std %.6e, range %.6e -> %.6e",
                solver.side_mode,
                sidewall_mean,
                sidewall_std,
                sidewall_min,
                sidewall_max,
            )
            # Mirror the summary-file recipe in the console so the next mesh
            # pass is immediately actionable after the run finishes.
            refinement = solver.suggest_refined_mesh_parameters(
                directors=result.snapshots[-1] if result.snapshots else None,
                temperature_field=result.temperature_snapshots[-1] if result.temperature_snapshots else None,
            )
            log_refined_mesh_suggestion("Sweep", solver, refinement)
    elif simulation_mode == "temperature_hysteresis":
        n_points = max(int(temperature_sweep_count), 1)
        temperatures = np.linspace(float(temperature_sweep_start), float(temperature_sweep_stop), n_points)
        solver, result = run_fem_temperature_hysteresis(
            coordinates_file=coordinates_file,
            mesh_file=mesh_file,
            checkpoint_file=checkpoint_file,
            temperatures_K=temperatures,
            output_prefix=output_prefix,
            material_preset=material_preset,
            anchoring_preset=anchoring_preset,
            sidewall_anchoring_strength=sidewall_anchoring_strength,
            cap_anchoring_strength=cap_anchoring_strength,
            temperature_profile=temperature_config,
            solver_method=temperature_sweep_solver_method,
            random_seed=random_seed,
            run_time=temperature_sweep_run_time,
        )
        if result.temperatures_K:
            logger.info(
                "Temperature hysteresis completed from %.3f K to %.3f K; final cooling equilibrium: %s",
                result.temperatures_K[0],
                result.temperatures_K[-1],
                solver.energy_summary(result.cooling.excess_energies[-1]),
            )
            logger.info(
                "Cooling-minus-heating order gap at the final point: principal=%.6f, Szz=%.6f, energy_density=%.6e",
                result.principal_order_gap[-1],
                result.order_parameter_gap[-1],
                result.energy_density_gap[-1],
            )
            logger.info(
                "Hysteresis midpoint at %.3f K: principal gap %.6f, Szz gap %.6f, energy gap %.6e",
                result.midpoint_temperature_K,
                result.midpoint_principal_order_gap,
                result.midpoint_order_parameter_gap,
                result.midpoint_energy_density_gap,
            )
            logger.info(
                "Hysteresis midpoint loop width: %.6f",
                result.midpoint_loop_width,
            )
            logger.info(
                "Hysteresis headline: loop width %.6f at %.3f K",
                result.midpoint_loop_width,
                result.midpoint_temperature_K,
            )
    else:
        solver, energies = run_fem_solver(
            coordinates_file=coordinates_file,
            mesh_file=mesh_file,
            checkpoint_file=checkpoint_file,
            run_time=run_time,
            output_prefix=output_prefix,
            material_preset=material_preset,
            anchoring_preset=anchoring_preset,
            sidewall_anchoring_strength=sidewall_anchoring_strength,
            cap_anchoring_strength=cap_anchoring_strength,
            temperature_profile=temperature_config,
            solver_method=solver_method,
            random_seed=random_seed,
        )
        logger.info("Final continuum excess free energy: %s", solver.energy_summary(energies[-1]))

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")
    if simulation_mode == "dynamics":
        plot_title = "Dynamic FEM Director Field"
    elif simulation_mode == "temperature_sweep":
        plot_title = "Temperature Sweep Final FEM Director Field"
    elif simulation_mode == "temperature_hysteresis":
        plot_title = "Temperature Hysteresis Final FEM Director Field"
    else:
        plot_title = "Relaxed FEM Director Field"
    if args.validate:
        plot_title = "Validation FEM Director Field"
    solver.plot_director_field(ax, title=plot_title)
    fig.savefig(f"{output_prefix}_director_field.jpg", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
