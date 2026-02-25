# Self-Consistence KM

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
    This heavily-documented utility file sets up the shared configurations, basic math functions, Chung-Lu and configuration models for network generations, and Runge-Kutta numerical integration steps used by the above experimental scripts.

## Running the Code

To run any script (utilizing multi-threading if desired by setting the `JULIA_NUM_THREADS` environment variable), you can execute it from the command line:

```bash
# Example for running the uniform distribution experiment
julia uniform_exp.jl

# Example for running a script requiring arguments
julia subcrit_exp.jl <rK3> <Gamma>
```
