This is a complete documentation set for your Factor Model Optimization framework. 

The documentation is divided into a "Quick Start" guide and detailed file-by-file documentation, adhering to the requested Markdown format.

---

## Project Documentation: Factor Model Optimization Framework

### 1. Quick Start Guide

This guide is designed to get a user running a basic optimization simulation within minutes.

#### Prerequisites

1. Python 3.9+ installed.

2. All six project files (`factor_optimizer.py`, `data_sampler.py`, `simulation_runner.py`, and their test counterparts) placed in the same directory.

3. Required external libraries installed:
   
   Bash
   
   ```
   pip install numpy cvxpy clarabel pytest
   ```
   
   *(Note: `clarabel` is the default high-performance solver used.)*

#### Step 1: Verify Installation and Dependencies

Run the primary test file to confirm all modules are correctly imported and the core optimization logic is sound.

Bash

```
pytest test_factor_optimizer.py
```

*(Expected Output: All tests passed.)*

#### Step 2: Run a Basic Simulation

The `simulation_runner.py` file contains the main execution logic and a demonstration block. Run it directly:

Bash

```
python simulation_runner.py
```

*(Expected Output: A summary of two optimization runs—one "full investment only" and one "long-only"—showing risk, positions, and solve time for a 50-asset, 5-factor problem.)*

#### Step 3: Configure a Custom Scenario

To set up a custom analysis, modify the `if __name__ == "__main__":` block in `simulation_runner.py`.

**Example: Running a Dollar-Neutral Portfolio with Position Limits**

Python

```
# Inside simulation_runner.py (Example 1 block)

runner = SimulationRunner(p=75, k=10, n_samples=2000, seed=42)

# Custom Scenario: Must be Dollar-Neutral AND limit any short position to 2%
custom_builder = runner.builder.create('market_neutral_limited')
custom_builder = runner.builder.add_dollar_neutral(custom_builder)
custom_builder = runner.builder.add_box_constraints(custom_builder, lower=-0.02, upper=None) 

runner.add_scenario('custom_neutral', custom_scenario=custom_builder)

results = runner.run(verbose=False)
runner.print_summary()
```

---

### 2. Detailed File Documentation

This section provides comprehensive documentation for the individual modules, detailing classes, methods, and their relationships.

### A. Core Optimization Engine

#### `factor_optimizer.py`

This module contains the high-performance solver for minimum variance portfolios under the factor model structure.

| **Class / Function**        | **Description**                                                                                                                                                                                                                                                                                                                                           |
| --------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **`FactorModelOptimizer`**  | The main class responsible for solving the QP. It implements the key optimization innovation: transforming the $\mathbf{w}^T \mathbf{C} \mathbf{w}$ objective into a sparse, numerically stable form $\mathbf{y}^T \mathbf{F} \mathbf{y} + \mathbf{w}^T \mathbf{D} \mathbf{w}$ by introducing the auxiliary variable $\mathbf{y} = \mathbf{B}\mathbf{w}$. |
| `add_constraints()`         | Generic method to add equality (`eq`) or inequality (`ineq`) constraints.                                                                                                                                                                                                                                                                                 |
| `solve()`                   | Executes the optimization using the specified solver (defaults to `CLARABEL`).                                                                                                                                                                                                                                                                            |
| `get_covariance_matrix()`   | **Safely** returns the dense covariance matrix $\mathbf{C}$ or, preferably, an efficient subset, avoiding costly $O(p^3)$ or $O(kp^2)$ computation for large $\mathbf{p}$.                                                                                                                                                                                |
| `evaluate_objective()`      | Efficiently calculates the objective value for any vector $\mathbf{w}$ in $O(kp)$ time without forming $\mathbf{C}$.                                                                                                                                                                                                                                      |
| `verify_kkt_stationarity()` | Rigorous check confirming the solution satisfies the Karush-Kuhn-Tucker conditions.                                                                                                                                                                                                                                                                       |
| **`ProblemData`**           | Dataclass container for the input matrices ($\mathbf{B}, \mathbf{F}, \mathbf{D}$) and constraint matrices.                                                                                                                                                                                                                                                |
| `generate_problem_data()`   | Utility function to quickly generate synthetic $\mathbf{B}, \mathbf{F}, \mathbf{D}$ matrices for testing.                                                                                                                                                                                                                                                 |

---

### B. Data Simulation and Sampling

#### `data_sampler.py`

This module is responsible for statistically modeling and generating the inputs ($\mathbf{B}, \mathbf{F}, \mathbf{D}$) for the optimization engine.

| **Class / Function**             | **Description**                                                                                                                                                                                      |
| -------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **`DistributionFactory`**        | A static class acting as a registry and factory for statistical distributions. It allows users to define parameters (e.g., `mean`, `std`) and returns a parameter-locked sampler function (closure). |
| `DistributionFactory.create()`   | Standard creation method. Returns a callable that generates $N$ samples from a fixed distribution (e.g., `normal`, `uniform`, `constant`).                                                           |
| `DistributionFactory.register()` | Allows users to add custom statistical functions (e.g., Student's t, Truncated Normal) to the registry at runtime, making the module highly extensible.                                              |
| **`DataSampler`**                | The main class for orchestrating the generation of the final optimization matrices from the configured samplers.                                                                                     |
| `add_factor_samplers()`          | Configures the $\mathbf{B}$ (loadings) and $\mathbf{F}$ (factor covariance) inputs using the created sampler functions.                                                                              |
| `add_idiosyncratic_sampler()`    | Configures the $\mathbf{D}$ (idiosyncratic covariance) input.                                                                                                                                        |
| `generate()`                     | Executes all samplers and assembles the final $\mathbf{B}, \mathbf{F}, \mathbf{D}$ matrices, ready for the optimizer.                                                                                |
| **`FactorModelData`**            | Dataclass container for the output matrices, including validation methods to ensure $\mathbf{D}$ is positive definite and $\mathbf{F}$ is positive semidefinite.                                     |

---

### C. Simulation Runner and Orchestration

#### `simulation_runner.py`

This module provides the high-level framework for defining optimization experiments, comparing results, and handling all project infrastructure.

| **Class / Function**     | **Description**                                                                                                                                                                                                         |
| ------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **`SimulationRunner`**   | The main entry point for running multi-scenario optimization tests. It encapsulates configuration, data generation, and result storage.                                                                                 |
| `configure_data()`       | Sets the statistical parameters for the underlying `DataSampler` before generation.                                                                                                                                     |
| `add_scenario()`         | Adds a constraint set to the run queue, either using predefined types (e.g., `LONG_ONLY_FULLY_INVESTED`) or custom `Scenario` objects.                                                                                  |
| `run()`                  | Executes all added scenarios sequentially using the `FactorModelOptimizer`.                                                                                                                                             |
| `print_summary()`        | Prints a formatted table comparing key portfolio metrics (Risk, Net Exposure, Solve Time) across all scenarios.                                                                                                         |
| **`ScenarioBuilder`**    | A helper utility for easily constructing standard constraint matrices ($\mathbf{A}, \mathbf{b}$) for common financial requirements (e.g., $\mathbf{w} \ge 0$, $\sum \mathbf{w} = 1$, sector limits, factor neutrality). |
| **`OptimizationResult`** | Dataclass to store and compute high-level portfolio statistics (e.g., Long/Short Exposure, Net/Gross Exposure, number of positions) from the raw weights ($\mathbf{w}$).                                                |

---

### D. Testing Modules (Unit and Integration)

The testing modules ensure the functional correctness, efficiency, and robustness of the entire framework.

#### `test_factor_optimizer.py`

- **Focus:** Unit testing the core mathematical and solver functionality of `FactorModelOptimizer`.

- **Key Tests:** Verification that the fast `evaluate_objective` matches the slow ground-truth calculation; tests for KKT stationarity passing; checks on covariance matrix slicing; solver mechanics.

#### `test_data_sampler.py`

- **Focus:** Comprehensive testing of statistical parameterization and data assembly.

- **Key Tests:** Validation of `DistributionFactory` registration, parameter validation (missing arguments), and creation of known distributions; tests ensuring generated matrices ($\mathbf{B}, \mathbf{F}, \mathbf{D}$) have the correct dimensions and properties (e.g., positive variance).

#### `test_simulation_runner.py`

- **Focus:** Integration testing and framework correctness.

- **Key Tests:** Verification that the `ScenarioBuilder` correctly outputs matrix forms; tests confirming that constrained portfolios have higher risk than unconstrained ones; checks that constraints (e.g., $\sum \mathbf{w} = 1$, $\mathbf{w} \ge 0$) are satisfied in the final optimization result.
