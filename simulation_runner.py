"""
simulation_runner.py

Portfolio optimization simulation framework using factor models.

This module orchestrates:
1. Data generation via DataSampler
2. Portfolio optimization via FactorModelOptimizer
3. Scenario analysis with different constraint sets
4. Results comparison and visualization

Example:
    >>> runner = SimulationRunner(p=100, k=10)
    >>> runner.add_scenario('long_only', long_only=True)
    >>> runner.add_scenario('unconstrained', long_only=False)
    >>> results = runner.run()
    >>> runner.print_summary()
"""

import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
import time
from enum import Enum

# Import dependencies
try:
    from factor_optimizer import FactorModelOptimizer, ProblemData
    from data_sampler import DataSampler, DistributionFactory, FactorModelData
except ImportError as e:
    raise ImportError(
        f"Required module not found: {e}. "
        "Ensure factor_optimizer.py and data_sampler.py are in the Python path."
    )


class ScenarioType(Enum):
    """Predefined scenario types for common portfolio constraints."""
    TRULY_UNCONSTRAINED = "truly_unconstrained"  # No constraints (returns w=0)
    FULL_INVESTMENT_ONLY = "full_investment_only"  # Only sum(w)=1 constraint
    UNCONSTRAINED = "unconstrained"  # Alias for FULL_INVESTMENT_ONLY
    LONG_ONLY = "long_only"
    FULLY_INVESTED = "fully_invested"
    LONG_ONLY_FULLY_INVESTED = "long_only_fully_invested"
    DOLLAR_NEUTRAL = "dollar_neutral"
    SECTOR_CONSTRAINED = "sector_constrained"



@dataclass
class OptimizationResult:
    """
    Container for optimization results from a single scenario.
    
    Attributes:
        scenario_name: Identifier for the scenario
        weights: Optimal portfolio weights
        objective: Optimal objective value (portfolio variance)
        risk: Portfolio volatility (standard deviation)
        solve_time: Time taken to solve (seconds)
        status: Solver status string
        constraints_satisfied: Whether all constraints are satisfied
        metadata: Additional information
    """
    scenario_name: str
    weights: np.ndarray
    objective: float
    risk: float
    solve_time: float
    status: str
    constraints_satisfied: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def n_positions(self) -> int:
        """Number of non-zero positions."""
        return np.sum(np.abs(self.weights) > 1e-6)
    
    @property
    def long_exposure(self) -> float:
        """Sum of positive weights."""
        return np.sum(self.weights[self.weights > 0])
    
    @property
    def short_exposure(self) -> float:
        """Sum of negative weights (absolute value)."""
        return np.abs(np.sum(self.weights[self.weights < 0]))
    
    @property
    def net_exposure(self) -> float:
        """Net exposure (long - short)."""
        return self.long_exposure - self.short_exposure
    
    @property
    def gross_exposure(self) -> float:
        """Gross exposure (long + short)."""
        return self.long_exposure + self.short_exposure


@dataclass
class Scenario:
    """
    Defines a portfolio optimization scenario with specific constraints.
    
    Attributes:
        name: Scenario identifier
        equality_constraints: List of (A, b) tuples for A @ w == b
        inequality_constraints: List of (A, b) tuples for A @ w <= b
        description: Human-readable description
    """
    name: str
    equality_constraints: List[Tuple[np.ndarray, np.ndarray]] = field(default_factory=list)
    inequality_constraints: List[Tuple[np.ndarray, np.ndarray]] = field(default_factory=list)
    description: str = ""


class ScenarioBuilder:
    """
    Helper class for building common constraint scenarios.
    
    Example:
        >>> builder = ScenarioBuilder(p=100)
        >>> scenario = builder.long_only_fully_invested()
        >>> # Or build custom
        >>> scenario = builder.create('custom')
        >>> scenario = builder.add_fully_invested(scenario)
        >>> scenario = builder.add_box_constraints(scenario, lower=0, upper=0.1)
    """
    
    def __init__(self, p: int):
        """
        Initialize builder.
        
        Args:
            p: Number of securities
        """
        self.p = p
    
    def create(self, name: str, description: str = "") -> Scenario:
        """Create empty scenario."""
        return Scenario(name=name, description=description)
    
    def add_fully_invested(self, scenario: Scenario) -> Scenario:
        """Add constraint: sum(w) = 1."""
        A = np.ones((1, self.p))
        b = np.array([1.0])
        scenario.equality_constraints.append((A, b))
        return scenario
    
    def add_dollar_neutral(self, scenario: Scenario) -> Scenario:
        """Add constraint: sum(w) = 0."""
        A = np.ones((1, self.p))
        b = np.array([0.0])
        scenario.equality_constraints.append((A, b))
        return scenario
    
    def add_long_only(self, scenario: Scenario) -> Scenario:
        """Add constraint: w >= 0."""
        A = -np.eye(self.p)
        b = np.zeros(self.p)
        scenario.inequality_constraints.append((A, b))
        return scenario
    
    def add_short_only(self, scenario: Scenario) -> Scenario:
        """Add constraint: w <= 0."""
        A = np.eye(self.p)
        b = np.zeros(self.p)
        scenario.inequality_constraints.append((A, b))
        return scenario
    
    def add_box_constraints(self, scenario: Scenario, 
                           lower: Optional[float] = None,
                           upper: Optional[float] = None) -> Scenario:
        """
        Add box constraints: lower <= w <= upper.
        
        Args:
            scenario: Scenario to modify
            lower: Lower bound (None for no bound)
            upper: Upper bound (None for no bound)
        """
        if lower is not None:
            A = -np.eye(self.p)
            b = -np.ones(self.p) * lower
            scenario.inequality_constraints.append((A, b))
        
        if upper is not None:
            A = np.eye(self.p)
            b = np.ones(self.p) * upper
            scenario.inequality_constraints.append((A, b))
        
        return scenario
    
    def add_sector_constraints(self, scenario: Scenario, 
                              sector_matrix: np.ndarray,
                              sector_limits: np.ndarray) -> Scenario:
        """
        Add sector exposure limits.
        
        Args:
            scenario: Scenario to modify
            sector_matrix: (n_sectors, p) one-hot encoding of sector membership
            sector_limits: (n_sectors,) maximum exposure per sector
        """
        scenario.inequality_constraints.append((sector_matrix, sector_limits))
        return scenario
    
    def add_factor_neutrality(self, scenario: Scenario, 
                             factor_exposures: np.ndarray,
                             factors_to_neutralize: Optional[List[int]] = None) -> Scenario:
        """
        Add factor neutrality constraints.
        
        Args:
            scenario: Scenario to modify
            factor_exposures: (k, p) factor loading matrix
            factors_to_neutralize: List of factor indices (None = all)
        """
        if factors_to_neutralize is None:
            factors_to_neutralize = list(range(factor_exposures.shape[0]))
        
        A = factor_exposures[factors_to_neutralize, :]
        b = np.zeros(len(factors_to_neutralize))
        scenario.equality_constraints.append((A, b))
        return scenario
    
    # Convenience methods for common scenarios
    
    def truly_unconstrained(self) -> Scenario:
        """
        Truly unconstrained portfolio (no constraints at all).
        
        NOTE: This will return w=0 (all zeros) as the optimal solution since
        w=0 minimizes variance (gives zero variance). This is mathematically
        correct but practically useless. Included for educational/testing purposes.
        
        Expected result: All weights = 0, Objective = 0, Risk = 0
        """
        return self.create('truly_unconstrained', 
                         'Minimum variance, NO constraints (expect zero solution)')
    
    def full_investment_only(self) -> Scenario:
        """
        Fully invested portfolio with no other restrictions.
        
        This is what is typically meant by "unconstrained" in portfolio optimization:
        - Must be fully invested (sum of weights = 1)
        - No restrictions on long/short positions
        - No position size limits
        """
        scenario = self.create('full_investment_only', 
                             'Minimum variance, fully invested, no sign/position restrictions')
        return self.add_fully_invested(scenario)
    
    def unconstrained(self) -> Scenario:
        """
        Alias for full_investment_only() for backward compatibility.
        
        Returns fully invested portfolio with no other constraints.
        This is the standard interpretation of "unconstrained" in portfolio optimization.
        """
        return self.full_investment_only()
    
    def long_only(self) -> Scenario:
        """Long-only portfolio."""
        scenario = self.create('long_only', 'Long-only minimum variance')
        return self.add_long_only(scenario)
    
    def fully_invested(self) -> Scenario:
        """Fully invested portfolio."""
        scenario = self.create('fully_invested', 'Fully invested (sum=1)')
        return self.add_fully_invested(scenario)
    
    def long_only_fully_invested(self) -> Scenario:
        """Long-only, fully invested portfolio."""
        scenario = self.create('long_only_fully_invested', 'Long-only, fully invested')
        scenario = self.add_fully_invested(scenario)
        scenario = self.add_long_only(scenario)
        return scenario
    
    def dollar_neutral(self) -> Scenario:
        """Dollar-neutral (long-short) portfolio."""
        scenario = self.create('dollar_neutral', 'Dollar-neutral (sum=0)')
        return self.add_dollar_neutral(scenario)
    
    def position_limited(self, max_position: float = 0.05) -> Scenario:
        """Position-limited, fully invested portfolio."""
        scenario = self.create(
            f'position_limited_{int(max_position*100)}pct',
            f'Fully invested, max {max_position*100}% per position'
        )
        scenario = self.add_fully_invested(scenario)
        scenario = self.add_box_constraints(scenario, lower=0, upper=max_position)
        return scenario


class SimulationRunner:
    """
    Main class for running portfolio optimization simulations.
    
    Example:
        >>> # Setup
        >>> runner = SimulationRunner(p=100, k=10, seed=42)
        >>> 
        >>> # Configure data generation
        >>> runner.configure_data(
        ...     beta_params={'dist': 'normal', 'mean': 0, 'std': 0.5},
        ...     factor_vol_params={'dist': 'uniform', 'low': 0.15, 'high': 0.25}
        ... )
        >>> 
        >>> # Add scenarios
        >>> runner.add_scenario('long_only', ScenarioType.LONG_ONLY_FULLY_INVESTED)
        >>> runner.add_scenario('unconstrained', ScenarioType.UNCONSTRAINED)
        >>> 
        >>> # Run
        >>> results = runner.run()
        >>> runner.print_summary()
    """
    
    def __init__(self, p: int, k: int, n_samples: int = 1000, seed: Optional[int] = None):
        """
        Initialize simulation runner.
        
        Args:
            p: Number of securities
            k: Number of factors
            n_samples: Number of samples for data generation
            seed: Random seed for reproducibility
        """
        self.p = p
        self.k = k
        self.n_samples = n_samples
        self.seed = seed
        
        self.builder = ScenarioBuilder(p)
        self.scenarios: Dict[str, Scenario] = {}
        self.results: Dict[str, OptimizationResult] = {}
        
        # Data generation configuration
        self._data_config = {
            'beta_params': {'dist': 'normal', 'mean': 0, 'std': 1},
            'factor_vol_params': {'dist': 'uniform', 'low': 0.1, 'high': 0.3},
            'idio_vol_params': {'dist': 'uniform', 'low': 0.05, 'high': 0.2}
        }
        
        self._data: Optional[FactorModelData] = None
        self._verbose = True
    
    def configure_data(self, beta_params: Optional[Dict] = None,
                      factor_vol_params: Optional[Dict] = None,
                      idio_vol_params: Optional[Dict] = None):
        """
        Configure data generation parameters.
        
        Args:
            beta_params: Parameters for beta distribution (must include 'dist' key)
            factor_vol_params: Parameters for factor volatility
            idio_vol_params: Parameters for idiosyncratic volatility
        """
        if beta_params:
            self._data_config['beta_params'] = beta_params
        if factor_vol_params:
            self._data_config['factor_vol_params'] = factor_vol_params
        if idio_vol_params:
            self._data_config['idio_vol_params'] = idio_vol_params
    
    def add_scenario(self, name: str, scenario_type: Optional[ScenarioType] = None,
                    custom_scenario: Optional[Scenario] = None):
        """
        Add a scenario to run.
        
        Args:
            name: Scenario identifier
            scenario_type: Use predefined scenario type
            custom_scenario: Use custom scenario object
            
        Example:
            >>> # Predefined
            >>> runner.add_scenario('long_only', ScenarioType.LONG_ONLY)
            >>> 
            >>> # Custom
            >>> scenario = runner.builder.create('custom')
            >>> scenario = runner.builder.add_fully_invested(scenario)
            >>> runner.add_scenario('custom', custom_scenario=scenario)
        """
        if custom_scenario:
            self.scenarios[name] = custom_scenario
        elif scenario_type:
            # Map scenario type to builder method
            scenario_map = {
                ScenarioType.TRULY_UNCONSTRAINED: self.builder.truly_unconstrained,
                ScenarioType.FULL_INVESTMENT_ONLY: self.builder.full_investment_only,
                ScenarioType.UNCONSTRAINED: self.builder.unconstrained,
                ScenarioType.LONG_ONLY: self.builder.long_only,
                ScenarioType.FULLY_INVESTED: self.builder.fully_invested,
                ScenarioType.LONG_ONLY_FULLY_INVESTED: self.builder.long_only_fully_invested,
                ScenarioType.DOLLAR_NEUTRAL: self.builder.dollar_neutral,
            }
            
            if scenario_type not in scenario_map:
                raise ValueError(f"Unsupported scenario type: {scenario_type}")
            
            self.scenarios[name] = scenario_map[scenario_type]()
        else:
            raise ValueError("Must provide either scenario_type or custom_scenario")
    
    def generate_data(self, force_regenerate: bool = False) -> FactorModelData:
        """
        Generate or retrieve factor model data.
        
        Args:
            force_regenerate: Force new data generation even if cached
            
        Returns:
            FactorModelData object
        """
        if self._data is not None and not force_regenerate:
            return self._data
        
        if self._verbose:
            print("Generating factor model data...")
        
        self._data = DataSampler.quick_generate(
            p=self.p,
            k=self.k,
            n_samples=self.n_samples,
            seed=self.seed,
            **self._data_config
        )
        
        if self._verbose:
            print(f"  Generated: B{self._data.B.shape}, F{self._data.F.shape}, D{self._data.D.shape}")
        
        return self._data
    
    def run(self, verbose: bool = True) -> Dict[str, OptimizationResult]:
        """
        Run all scenarios.
        
        Args:
            verbose: Print progress information
            
        Returns:
            Dictionary mapping scenario names to results
        """
        self._verbose = verbose
        
        if not self.scenarios:
            raise RuntimeError("No scenarios added. Call add_scenario() first.")
        
        # Generate data
        data = self.generate_data()
        
        if verbose:
            print(f"\n{'='*70}")
            print(f"Running {len(self.scenarios)} scenarios")
            print(f"{'='*70}\n")
        
        # Run each scenario
        for name, scenario in self.scenarios.items():
            if verbose:
                print(f"Scenario: {name}")
                if scenario.description:
                    print(f"  {scenario.description}")
            
            result = self._run_scenario(name, scenario, data)
            self.results[name] = result
            
            if verbose:
                print(f"  Status: {result.status}")
                print(f"  Risk: {result.risk:.4f}")
                print(f"  Positions: {result.n_positions}/{self.p}")
                
                # Special note for truly unconstrained scenarios
                if 'truly_unconstrained' in name.lower() and result.n_positions == 0:
                    print(f"  Note: Zero solution is EXPECTED (no constraints → w=0 is optimal)")
                    print(f"        Objective: {result.objective:.8f} (should be ~0)")
                
                print(f"  Time: {result.solve_time:.3f}s\n")
        
        if verbose:
            print(f"{'='*70}\n")
        
        return self.results
    
    def _run_scenario(self, name: str, scenario: Scenario, 
                     data: FactorModelData) -> OptimizationResult:
        """Run a single optimization scenario."""
        # Create optimizer
        opt = FactorModelOptimizer(data.B, data.F, data.D)
        
        # Add constraints
        for A, b in scenario.equality_constraints:
            opt.add_equality_constraint(A, b)
        
        for A, b in scenario.inequality_constraints:
            opt.add_inequality_constraint(A, b)
        
        # Solve
        t0 = time.time()
        status = opt.solve(verbose=False)
        solve_time = time.time() - t0
        
        # Extract results
        if opt.is_solved:
            weights = opt.solution
            objective = opt.objective_value
            risk = np.sqrt(2 * objective)
            
            # Verify constraints
            constraints_satisfied = self._verify_constraints(weights, scenario)
        else:
            weights = np.zeros(self.p)
            objective = np.inf
            risk = np.inf
            constraints_satisfied = False
        
        return OptimizationResult(
            scenario_name=name,
            weights=weights,
            objective=objective,
            risk=risk,
            solve_time=solve_time,
            status=status,
            constraints_satisfied=constraints_satisfied,
            metadata={'n_eq_constraints': len(scenario.equality_constraints),
                     'n_ineq_constraints': len(scenario.inequality_constraints)}
        )
    
    def _verify_constraints(self, weights: np.ndarray, scenario: Scenario, 
                           tol: float = 1e-4) -> bool:
        """Verify that solution satisfies constraints."""
        # Check equality constraints
        for A, b in scenario.equality_constraints:
            residual = np.linalg.norm(A @ weights - b)
            if residual > tol:
                return False
        
        # Check inequality constraints
        for A, b in scenario.inequality_constraints:
            violations = A @ weights - b
            if np.any(violations > tol):
                return False
        
        return True
    
    def print_summary(self):
        """Print formatted summary of all results."""
        if not self.results:
            print("No results to display. Run simulation first.")
            return
        
        print(f"\n{'='*70}")
        print("SIMULATION SUMMARY")
        print(f"{'='*70}\n")
        
        # Summary table
        print(f"{'Scenario':<25} {'Risk':>10} {'Positions':>10} {'Net Exp':>10} {'Time':>10}")
        print("-" * 70)
        
        for name, result in self.results.items():
            print(f"{name:<25} {result.risk:>10.4f} {result.n_positions:>10} "
                  f"{result.net_exposure:>10.2f} {result.solve_time:>10.3f}s")
            
            # Add note for truly unconstrained scenarios (expect zero solution)
            if 'truly_unconstrained' in name.lower() and result.n_positions == 0:
                print(f"{'':>25} {'(Expected: all zeros, obj=0)':<45}")
        
        print("\n" + "="*70 + "\n")
    
    def compare_scenarios(self, scenario1: str, scenario2: str):
        """
        Compare two scenarios in detail.
        
        Args:
            scenario1: First scenario name
            scenario2: Second scenario name
        """
        if scenario1 not in self.results or scenario2 not in self.results:
            raise ValueError("Both scenarios must be run first")
        
        r1 = self.results[scenario1]
        r2 = self.results[scenario2]
        
        print(f"\n{'='*70}")
        print(f"Comparing: {scenario1} vs {scenario2}")
        print(f"{'='*70}\n")
        
        metrics = [
            ('Risk', r1.risk, r2.risk, '.4f'),
            ('Objective', r1.objective, r2.objective, '.6f'),
            ('Positions', r1.n_positions, r2.n_positions, 'd'),
            ('Long Exposure', r1.long_exposure, r2.long_exposure, '.4f'),
            ('Short Exposure', r1.short_exposure, r2.short_exposure, '.4f'),
            ('Net Exposure', r1.net_exposure, r2.net_exposure, '.4f'),
            ('Gross Exposure', r1.gross_exposure, r2.gross_exposure, '.4f'),
            ('Solve Time', r1.solve_time, r2.solve_time, '.3f'),
        ]
        
        print(f"{'Metric':<20} {scenario1:>15} {scenario2:>15} {'Difference':>15}")
        print("-" * 70)
        
        for name, val1, val2, fmt in metrics:
            diff = val2 - val1
            print(f"{name:<20} {val1:>15{fmt}} {val2:>15{fmt}} {diff:>15{fmt}}")
        
        print("\n" + "="*70 + "\n")
    
    def get_result(self, scenario_name: str) -> OptimizationResult:
        """Get result for a specific scenario."""
        if scenario_name not in self.results:
            raise ValueError(f"Scenario '{scenario_name}' not found or not run yet")
        return self.results[scenario_name]


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    print("="*70)
    print("Portfolio Optimization Simulation")
    print("="*70)
    
    # Example 1: Basic usage with predefined scenarios
    print("\n--- Example 1: Basic Usage ---\n")
    
    runner = SimulationRunner(p=50, k=5, n_samples=1000, seed=42)
    
    # Add scenarios demonstrating different constraint levels
    runner.add_scenario('truly_unconstrained', ScenarioType.TRULY_UNCONSTRAINED)
    runner.add_scenario('full_investment_only', ScenarioType.FULL_INVESTMENT_ONLY)
    runner.add_scenario('long_only', ScenarioType.LONG_ONLY_FULLY_INVESTED)
    runner.add_scenario('position_limited', 
                       custom_scenario=runner.builder.position_limited(0.05))
    
    # Run
    results = runner.run(verbose=True)
    runner.print_summary()
    
    # Compare scenarios
    print("NOTE: truly_unconstrained returns w=0 (educational example)")
    print("      full_investment_only is the practical 'unconstrained' case\n")
    runner.compare_scenarios('full_investment_only', 'long_only')
    
    # Example 2: Custom data configuration
    print("\n--- Example 2: Custom Configuration ---\n")
    
    runner2 = SimulationRunner(p=100, k=10, seed=123)
    
    # Custom data generation
    runner2.configure_data(
        beta_params={'dist': 'normal', 'mean': 0.5, 'std': 0.3},
        factor_vol_params={'dist': 'lognormal', 'mean': -2, 'std': 0.5}
    )
    
    # Add custom scenario
    custom = runner2.builder.create('custom', 'Custom constraints')
    custom = runner2.builder.add_fully_invested(custom)
    custom = runner2.builder.add_box_constraints(custom, lower=-0.02, upper=0.05)
    runner2.add_scenario('custom', custom_scenario=custom)
    
    # Run
    results2 = runner2.run(verbose=False)
    print(f"Custom scenario risk: {results2['custom'].risk:.4f}")
    
    print("\n" + "="*70)

    ####################################3################################
    #### CUSTOM STRATEGIES BELOW THIS LINE - DO NOT DELETE ##############
    ####################################3################################

    runner = SimulationRunner(p=75, k=10, n_samples=2000, seed=42)

    # CORRECTED: Fully-invested with limited shorts
    custom_builder = runner.builder.create('120_20_strategy')
    custom_builder = runner.builder.add_fully_invested(custom_builder)  # sum(w) = 1
    custom_builder = runner.builder.add_box_constraints(
        custom_builder, 
        lower=-0.02,  # Max 2% short per positionS
        upper=None    # No upper limit (allows concentration in longs)
    )

    runner.add_scenario('custom_limited_shorts', custom_scenario=custom_builder)
    runner.add_scenario('long_only', ScenarioType.LONG_ONLY_FULLY_INVESTED)
    results = runner.run(verbose=False)
    runner.print_summary()
    runner.compare_scenarios('custom_limited_shorts', 'long_only')
