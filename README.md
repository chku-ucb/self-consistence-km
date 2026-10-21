# Self-consistence-km: Numerical simulations and analytical framework for higher-order kuramoto models.

## Prerequisites & Setup

Before running the code, ensure that you have Julia installed and updated to a recent version. Since these scripts utilize specific multi-threading and package capabilities, you'll need to set up the Julia environment first.

To initialize and update the project environment with the required dependencies:

1. Open your terminal and navigate to the project root directory (`self-consistence-km/`).
2. Start the Julia REPL by running:
   ```bash
   julia
   ```
3. Enter the Julia package manager mode by pressing `]`.
4. Activate the enclosed environment and install/update the packages by running:
   ```julia
   activate env/
   instantiate
   update
   ```
   *Note: The scripts themselves already activate the `env` folder automatically via `Pkg.activate(env_dir)`, but instantiating it ahead of time ensures all dependencies are downloaded and precompiled.*
5. Press `Backspace` to return to the standard Julia prompt, and type `exit()` (or press `Ctrl+D`) to exit.

## Experiment Scripts Overview

Below is a brief overview of the main experimental scripts in this repository. The purpose is declared at the top of each file.

*   **`uniform_exp.jl`**
    This file is for the experiment for validating the critical coupling constant and hysteresis with a **uniform** distribution.

*   **`power_law_exp.jl`**
    This file is for the experiment for validating the critical coupling constant and hysteresis with a **power-law** distribution.

*   **`subcrit_exp.jl`**
    This file is for the experiment for validating the subcritical point with a **power-law** degree distribution.
    *Usage Note:* This script requires `rK3` and `Gamma` as command-line arguments. Example: `julia subcrit_exp.jl 1.5 2.5`

*   **`rewire_exp.jl`**
    This file is for the experiment for validating the change of the subcritical point depending on the triadic interaction structure. 

*   **`utils.jl`**
    This heavily-documented utility file sets up the shared configurations, basic math functions, Chung-Lu and configuration models for network generations, and Runge-Kutta numerical integration steps used by the above experimental scripts. Main components:
    *   **Network generation:** Chung-Lu graphs and 3-uniform hyperedges (triangles) use the Miller-Hagberg geometric-skip algorithm (Miller & Hagberg, WAW 2011), giving O(N + M) expected time for pairwise edges and O(N² + M₃) for triangles instead of O(N²) and O(N³). Configuration-model generators (`create_L_CF`, `create_3D_CF`) are also provided.
    *   **Critical couplings:** `Un_gen` computes the leading eigenpair of the adjacency matrix by sparse power iteration on the adjacency list, so no dense N×N matrix is built. `den_cal` and `num_cal` give the eigenvector-based K3 threshold, and `K3crit_HMF` gives the heterogeneous mean-field (HMF) estimate.
    *   **Simulation:** `RK4_step_fast!` (optionally with pre-allocated `caches`), `evolve_RK4!` (integrates at fixed K2, K3 and returns the time-averaged R), `RK4_step` and `Heun_step`.
    *   **Self-consistent theory:** `R1_th_Int` (FDA), `R1_th_sum` (TAT on the actual frequency sample) and `hmf_fixed_point` / `compute_HMF_sweep` (HMF forward/backward sweeps). All are solved by damped fixed-point iteration with a convergence tolerance.
    *   **Rewiring:** `rewire_3D_Ω` swaps hyperedge vertices to raise or lower K3crit while preserving the triadic degree sequence.

## Running the Code

To run any script (utilizing multi-threading if desired by setting the `JULIA_NUM_THREADS` environment variable), you can execute it from the command line:

```bash
# Example for running the uniform distribution experiment
julia uniform_exp.jl

# Example for running a script requiring arguments
julia subcrit_exp.jl <rK3> <Gamma>
```
