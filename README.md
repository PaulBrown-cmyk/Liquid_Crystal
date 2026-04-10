# Liquid_Crystal

## FEM Solver

The main entry point is `lc_wkcp2.py`, which now runs the finite-element
continuum solver by default.

Example:

```bash
python lc_wkcp2.py --config lc_fem_config.yaml
```

Useful config keys:

- `anchoring_preset`: `planar_side_homeotropic_caps`, `planar_all`, `homeotropic_all`, or `free`
- `solver_method`: `trust-krylov`, `newton-cg`, or `lbfgs`
- `grid`: default cylinder dimensions and sampling used when the grid file is missing
