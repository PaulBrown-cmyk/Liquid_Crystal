# Liquid_Crystal

## FEM Solver

The main entry point is `LC.py`, which now runs the finite-element
continuum solver and can either minimize the free energy or evolve overdamped
director dynamics.

The solver now uses a separate body-fitted cylindrical solve mesh stored as
`straight_cylinder_solve_mesh.npz`, while the older point-cloud grid remains
available for plotting and compatibility.

The old Monte Carlo prototype has been archived in `legacy/lc_wkcp2_legacy.py`.

Example:

```bash
python LC.py --config lc_fem_config.yaml --simulation-mode dynamics
```

To sweep temperature-dependent equilibria and save the order-parameter and
anchoring diagnostics:

```bash
python LC.py --config lc_fem_config.yaml --simulation-mode temperature_sweep \
  --temperature-profile coupled_boundary_bulk_transition
```

To run a heating/cooling hysteresis scan with the same temperature window:

```bash
python LC.py --config lc_fem_config.yaml --simulation-mode temperature_hysteresis \
  --temperature-profile coupled_boundary_bulk_transition
```

To run the built-in reference checks and write a compact validation bundle:

```bash
python LC.py --config lc_fem_config.yaml --validate
```

To try one of the named thermal presets:

```bash
python LC.py --config lc_fem_config.yaml --temperature-profile gradient_z
```

Other included thermal presets are `uniform`, `oscillatory_z`,
`boundary_heated`, `hot_wall_weak_sidewall`, `near_transition_drive`, and
`transition_cycle`.

Useful config keys:

- `material_preset`: `5CB_room_temperature`, `5CB_anisotropic_room_temperature`, `generic_nematic`, or `generic_cholesteric`
- `anchoring_preset`: `planar_side_homeotropic_caps`, `planar_all`, `homeotropic_all`, or `free`
- `sidewall_anchoring_strength` and `cap_anchoring_strength`: independent Rapini-Papoular strengths for the sidewall and the end caps
- `mesh_file`: body-fitted tetrahedral solve mesh used by the FEM backend
- `solver_method`: `trust-krylov`, `newton-cg`, or `lbfgs`
- `simulation_mode`: `static`, `dynamics`, `temperature_sweep`, `temperature_hysteresis`, `qtensor_static`, or `qtensor_temperature_sweep`
- `temperature_sweep`: `start_temperature_K`, `stop_temperature_K`, and `num_points`
- `temperature`: optional thermal profile with `enabled`, `mode`, `law`, gradients, and coupling strengths, including separate sidewall and cap anchoring softening
- `temperature-profile`: named preset loaded from `temperature_profiles.yaml`
- `dynamics`: `total_time`, `time_step`, `mobility`, `thermal_noise_strength`, `snapshot_interval`, `histogram_bins`, and `random_seed`
- `qtensor`: first-pass Landau-de Gennes settings for the Q-tensor backend, including temperature and minimizer controls
- `grid`: default cylinder dimensions and sampling used when the plotting grid is missing
- `mesh`: default body-fitted solve-mesh resolution and clustering used when the solve mesh is missing
- If a prior run with the same `output_prefix` wrote `*_refined_solve_mesh.npz`,
  the CLI prefers that refined mesh automatically on the next run unless you
  pass an explicit `--mesh-file`

For the default 5CB preset, the solver uses room-temperature literature values
K11 ≈ 5.9 pN, K22 ≈ 4.5 pN, K33 ≈ 9.9 pN, and a one-constant equivalent of
about 7.9 pN. Because 5CB is a nematic rather than a cholesteric, the default
`q0` is zero. The `generic_cholesteric` preset keeps the same simple
one-constant structure but sets a nonzero `q0` so we have a clean extension
point for more complicated chiral dynamics later.
The `5CB_anisotropic_room_temperature` preset switches to the full anisotropic
Oseen-Frank bulk form while keeping the same room-temperature 5CB constants.
The opt-in Q-tensor backend uses a Landau-de Gennes model with 5CB
coefficients `A0 = 0.044e6 J/m^3/K`, `B = 0.816e6 J/m^3`, `C = 0.45e6 J/m^3`,
`L1 = 6 pN`, `L2 = 18 pN`, `T* = 307 K`, and `T_NI = 308.5 K`. The surface
term is written as a preferred-surface Q-tensor boundary free energy, so the
anchoring target stays in the same Q basis as the solver variables. Use
`qtensor_static` to relax a single state or `qtensor_temperature_sweep` to
watch the scalar order soften across temperature.

For validation, the most useful quantity is usually the excess free-energy
density in `J/m^3`. The code also reports a rough `K/L^2` scale so you can
compare the result against a typical continuum elastic magnitude for the
geometry.

The validation command writes `*_validation.csv` and `*_validation.txt`
alongside the usual director-field plot so you can compare the benchmark
cases in a terminal or spreadsheet.

For dynamics runs, the solver writes a diagnostics bundle with the trace CSV,
trace plot, alignment CSV, alignment plot, snapshot-angle histogram CSV,
snapshot-angle histogram heatmap, a color-mapped final director field, 2D
director-orientation maps in rho-z and x-z projections, 2D local alignment
maps in rho-z and x-z projections, 2D temperature maps in rho-z and x-z
projections, and matching x-y basal-plane maps for each of those fields, plus
the GIF movie. The bundle also includes theta-z surface-unwrapped maps for the
same director, alignment, and temperature fields on the cylinder wall, plus a
mode-labeled sidewall anchoring-energy-density map and CSV on the unwrapped
wall.
The dynamics bundle also writes a mesh refinement-hint CSV and projected map
so you can see where the structured cylinder would benefit from additional
local resolution.
It also writes a refined follow-up mesh file and a JSON suggestion for the
next mesh pass.
The console log also prints the follow-up mesh delta so the next run is easy
to compare without opening the JSON file.

For temperature-sweep runs, the solver writes `*_temperature_sweep.csv`,
`*_temperature_sweep.jpg`, `*_temperature_sweep_summary.txt`, and the final
director field at the last temperature in the sweep. The CSV includes the
equilibrium order parameter, principal order parameter, excess energy
density, the bulk/anchoring softening scales, and temperature-dependent `K_i`
at each temperature. The summary file highlights the midpoint temperature,
midpoint order parameters, the endpoint order-shift magnitude, and the
endpoint `K_eff` range. The bundle also includes a color-mapped final director
field plus 2D director-orientation, local alignment, and temperature maps in
both rho-z and x-z projections, along with x-y basal-plane views and theta-z
surface-unwrapped views. The temperature-sweep bundle also writes a
mode-labeled sidewall anchoring diagnostic on the unwrapped wall. It also
writes a mesh refinement-hint CSV and map for the final swept state.
The sweep bundle also writes a refined follow-up mesh file and a JSON
suggestion for the next mesh pass.
The console log also prints the same follow-up mesh delta as a compact
headline.

For Q-tensor runs, `qtensor_static` writes `*_qtensor_summary.txt`,
`*_qtensor_field.csv`, `*_qtensor_director_field.jpg`, and
`*_qtensor_order_map.jpg`. The `qtensor_temperature_sweep` mode writes the
same bundle for the final state plus `*_qtensor_temperature_sweep.csv`,
`*_qtensor_temperature_sweep.jpg`, and
`*_qtensor_temperature_sweep_summary.txt` so you can watch the scalar order
soften across the 5CB transition window.

For hysteresis runs, the solver writes `*_temperature_hysteresis.csv`,
`*_temperature_hysteresis.jpg`, `*_temperature_hysteresis_summary.txt`, and
the final director field at the end of the cooling leg. The CSV contains paired
heating/cooling rows and the cooling-minus-heating gaps so you can spot
irreversible shifts or loop width, while the summary file highlights the
midpoint and maximum loop gaps. The summary also includes a compact midpoint
loop-width metric based on the order-parameter gaps.

If you enable the thermal profile, the solver also applies temperature-dependent
softening to elastic constants and surface anchoring, and it scales the
stochastic forcing and mobility with the local temperature field.

The temperature sweep mode relaxes the director field at a sequence of
temperatures and writes a sweep CSV/plot bundle. It is the easiest way to see
order-parameter equilibria shift as temperature crosses the transition window.
The console log also prints a one-line sweep headline for the endpoint order
shift, midpoint temperature, and endpoint `K_eff` range.

The included `temperature_profiles.yaml` file provides ready-to-run `uniform`,
`gradient_z`, `oscillatory_z`, `boundary_heated`, `hot_wall_weak_sidewall`,
`near_transition_drive`, `transition_cycle`, and
`coupled_boundary_bulk_transition` examples.

For a static minimization run, pass `--simulation-mode static`.

For regression checks, run:

```bash
python -m unittest test_lc_validation
```
