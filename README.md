# Liquid_Crystal

## FEM Solver

The main entry point is `LC.py`, which now runs the finite-element
continuum solver and can either minimize the free energy or evolve overdamped
director dynamics.

The old Monte Carlo prototype has been archived in `legacy/lc_wkcp2_legacy.py`.

Example:

```bash
python LC.py --config lc_fem_config.yaml --simulation-mode dynamics
```

Useful config keys:

- `material_preset`: `5CB_room_temperature`, `5CB_anisotropic_room_temperature`, `generic_nematic`, or `generic_cholesteric`
- `anchoring_preset`: `planar_side_homeotropic_caps`, `planar_all`, `homeotropic_all`, or `free`
- `sidewall_anchoring_strength` and `cap_anchoring_strength`: independent Rapini-Papoular strengths for the sidewall and the end caps
- `solver_method`: `trust-krylov`, `newton-cg`, or `lbfgs`
- `simulation_mode`: `static` or `dynamics`
- `dynamics`: `total_time`, `time_step`, `mobility`, `thermal_noise_strength`, `snapshot_interval`, `histogram_bins`, and `random_seed`
- `grid`: default cylinder dimensions and sampling used when the grid file is missing

For the default 5CB preset, the solver uses room-temperature literature values
K11 ≈ 5.9 pN, K22 ≈ 4.5 pN, K33 ≈ 9.9 pN, and a one-constant equivalent of
about 7.9 pN. Because 5CB is a nematic rather than a cholesteric, the default
`q0` is zero. The `generic_cholesteric` preset keeps the same simple
one-constant structure but sets a nonzero `q0` so we have a clean extension
point for more complicated chiral dynamics later.
The `5CB_anisotropic_room_temperature` preset switches to the full anisotropic
Oseen-Frank bulk form while keeping the same room-temperature 5CB constants.

For validation, the most useful quantity is usually the excess free-energy
density in `J/m^3`. The code also reports a rough `K/L^2` scale so you can
compare the result against a typical continuum elastic magnitude for the
geometry.

For dynamics runs, the solver writes a diagnostics bundle with the trace CSV,
trace plot, snapshot-angle histogram CSV, snapshot-angle histogram heatmap, the
final director field, and the GIF movie.

For a static minimization run, pass `--simulation-mode static`.
