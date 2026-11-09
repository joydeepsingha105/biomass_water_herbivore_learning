# Numerical time integration of the BWHL model

This folder contains **`BWHL_BWHLP_DNS.ipynb`**, a Julia/Jupyter notebook for direct numerical time integration of the vegetation–water–herbivore–learning (BWHL) model. The notebook includes two independent examples:

- **Four-dimensional model (`B`, `W`, `H`, `L`):** precipitation `P` is held constant. The example uses `P = 95` and integrates the model from `t = 0` to `t = 60`, displaying the time series of all four variables and the final numerical state.
- **Five-dimensional model (`B`, `W`, `H`, `L`, `P`):** precipitation is an additional dynamical variable satisfying `dP/dt = -r(P - 95)`. The example uses `r = 3.1`, `P(0) = 449.949712`, and integrates from `t = 0` to `t = 100`. It plots biomass, precipitation, herbivore density, and learning; soil water is integrated but not included in the displayed figure.

## Numerical approach

Both examples solve an initial-value problem with the adaptive fifth-order Runge–Kutta method **Tsit5**, using relative tolerance `1e-12` and absolute tolerance `1e-13`. They illustrate transient dynamics and the approach toward long-time states for selected initial conditions; they do **not** calculate basin boundaries or bifurcation branches.

## Running the notebook

1. Open `BWHL_BWHLP_DNS.ipynb` in a Jupyter environment with a Julia kernel.
2. Install the Julia packages `DifferentialEquations` and `Plots` if needed. The 4D cell also imports `LinearAlgebra` and `Printf` (standard Julia libraries).
3. Run either model section independently; edit the parameter values, initial conditions, or integration time to explore other scenarios.

The two examples redefine `RHS!` for different state-vector dimensions, so run each section from its own model definition onward.

## Reproducibility note

The numerical parameters and initial conditions are given explicitly in the notebook. In both examples, the evaporation term is implemented as `N*W/(1 + R*B/K_B)`; ensure that the equation in the associated manuscript matches this implemented form before interpreting or reproducing results.
