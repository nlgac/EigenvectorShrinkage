"""
test_simulation_runner.py

Comprehensive test suite for simulation_runner module.

Tests cover:
- ScenarioBuilder constraint construction
- SimulationRunner orchestration
- OptimizationResult metrics
- Scenario comparisons
- Integration with FactorModelOptimizer
- Edge cases and error handling

Usage:
    pytest test_simulation_runner.py -v
    pytest test_simulation_runner.py -v --cov=simulation_runner
"""

import pytest
import numpy as np
import time
from simulation_runner import (
    SimulationRunner,
    ScenarioBuilder,
    ScenarioType,
    Scenario,
    OptimizationResult
)
from data_sampler import DataSampler, FactorModelData


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture(scope="module")
def small_data():
    """Small factor model data for quick tests."""
    return DataSampler.quick_generate(p=20, k=3, n_samples=100, seed=42)


@pytest.fixture(scope="module")
def medium_data():
    """Medium factor model data for realistic tests."""
    return DataSampler.quick_generate(p=50, k=5, n_samples=500, seed=42)


@pytest.fixture
def small_runner():
    """Runner with small problem for fast tests."""
    return SimulationRunner(p=20, k=3, n_samples=100, seed=42)


@pytest.fixture
def medium_runner():
    """Runner with medium problem for realistic tests."""
    return SimulationRunner(p=50, k=5, n_samples=500, seed=42)


@pytest.fixture
def builder_small():
    """ScenarioBuilder for small problem."""
    return ScenarioBuilder(p=20)


# ============================================================================
# ScenarioBuilder Tests
# ============================================================================

class TestScenarioBuilder:
    """Tests for ScenarioBuilder functionality."""
    
    def test_initialization(self):
        """Test ScenarioBuilder initialization."""
        builder = ScenarioBuilder(p=100)
        assert builder.p == 100
    
    def test_create_empty_scenario(self, builder_small):
        """Test creating empty scenario."""
        scenario = builder_small.create('test', 'Test scenario')
        
        assert scenario.name == 'test'
        assert scenario.description == 'Test scenario'
        assert len(scenario.equality_constraints) == 0
        assert len(scenario.inequality_constraints) == 0
    
    def test_add_fully_invested(self, builder_small):
        """Test adding fully invested constraint."""
        scenario = builder_small.create('test')
        scenario = builder_small.add_fully_invested(scenario)
        
        assert len(scenario.equality_constraints) == 1
        A, b = scenario.equality_constraints[0]
        assert A.shape == (1, 20)
        assert np.allclose(A, np.ones((1, 20)))
        assert np.isclose(b[0], 1.0)
    
    def test_add_dollar_neutral(self, builder_small):
        """Test adding dollar neutral constraint."""
        scenario = builder_small.create('test')
        scenario = builder_small.add_dollar_neutral(scenario)
        
        assert len(scenario.equality_constraints) == 1
        A, b = scenario.equality_constraints[0]
        assert np.allclose(A, np.ones((1, 20)))
        assert np.isclose(b[0], 0.0)
    
    def test_add_long_only(self, builder_small):
        """Test adding long-only constraint."""
        scenario = builder_small.create('test')
        scenario = builder_small.add_long_only(scenario)
        
        assert len(scenario.inequality_constraints) == 1
        A, b = scenario.inequality_constraints[0]
        assert A.shape == (20, 20)
        assert np.allclose(A, -np.eye(20))
        assert np.allclose(b, np.zeros(20))
    
    def test_add_short_only(self, builder_small):
        """Test adding short-only constraint."""
        scenario = builder_small.create('test')
        scenario = builder_small.add_short_only(scenario)
        
        assert len(scenario.inequality_constraints) == 1
        A, b = scenario.inequality_constraints[0]
        assert A.shape == (20, 20)
        assert np.allclose(A, np.eye(20))
        assert np.allclose(b, np.zeros(20))
    
    def test_add_box_constraints_both_bounds(self, builder_small):
        """Test adding box constraints with both bounds."""
        scenario = builder_small.create('test')
        scenario = builder_small.add_box_constraints(scenario, lower=-0.05, upper=0.10)
        
        assert len(scenario.inequality_constraints) == 2
        # Check dimensions
        for A, b in scenario.inequality_constraints:
            assert A.shape == (20, 20)
            assert b.shape == (20,)
    
    def test_add_box_constraints_lower_only(self, builder_small):
        """Test adding only lower bound."""
        scenario = builder_small.create('test')
        scenario = builder_small.add_box_constraints(scenario, lower=0, upper=None)
        
        assert len(scenario.inequality_constraints) == 1
    
    def test_add_box_constraints_upper_only(self, builder_small):
        """Test adding only upper bound."""
        scenario = builder_small.create('test')
        scenario = builder_small.add_box_constraints(scenario, lower=None, upper=0.05)
        
        assert len(scenario.inequality_constraints) == 1
    
    def test_add_sector_constraints(self, builder_small):
        """Test adding sector constraints."""
        # Create sector matrix (2 sectors)
        sector_matrix = np.zeros((2, 20))
        sector_matrix[0, :10] = 1  # First 10 in sector 1
        sector_matrix[1, 10:] = 1  # Last 10 in sector 2
        sector_limits = np.array([0.6, 0.4])
        
        scenario = builder_small.create('test')
        scenario = builder_small.add_sector_constraints(scenario, sector_matrix, sector_limits)
        
        assert len(scenario.inequality_constraints) == 1
        A, b = scenario.inequality_constraints[0]
        assert A.shape == (2, 20)
        assert b.shape == (2,)
    
    def test_add_factor_neutrality_all_factors(self, small_data):
        """Test adding factor neutrality for all factors."""
        builder = ScenarioBuilder(p=20)
        scenario = builder.create('test')
        scenario = builder.add_factor_neutrality(scenario, small_data.B)
        
        assert len(scenario.equality_constraints) == 1
        A, b = scenario.equality_constraints[0]
        assert A.shape == (3, 20)  # All 3 factors
        assert np.allclose(b, np.zeros(3))
    
    def test_add_factor_neutrality_subset(self, small_data):
        """Test adding factor neutrality for subset of factors."""
        builder = ScenarioBuilder(p=20)
        scenario = builder.create('test')
        scenario = builder.add_factor_neutrality(
            scenario, small_data.B, 
            factors_to_neutralize=[0, 2]  # Only factors 0 and 2
        )
        
        assert len(scenario.equality_constraints) == 1
        A, b = scenario.equality_constraints[0]
        assert A.shape == (2, 20)  # Only 2 factors
    
    def test_convenience_truly_unconstrained(self, builder_small):
        """Test truly unconstrained scenario builder (no constraints, returns w=0)."""
        scenario = builder_small.truly_unconstrained()
        
        assert scenario.name == 'truly_unconstrained'
        assert len(scenario.equality_constraints) == 0
        assert len(scenario.inequality_constraints) == 0
        assert 'expect zero solution' in scenario.description.lower()
    
    def test_convenience_full_investment_only(self, builder_small):
        """Test full_investment_only scenario builder."""
        scenario = builder_small.full_investment_only()
        
        assert scenario.name == 'full_investment_only'
        assert len(scenario.equality_constraints) == 1  # sum(w) = 1
        assert len(scenario.inequality_constraints) == 0
    
    def test_convenience_unconstrained(self, builder_small):
        """Test unconstrained scenario builder (now an alias for full_investment_only)."""
        scenario = builder_small.unconstrained()
        
        # unconstrained() is now an alias for full_investment_only()
        assert scenario.name == 'full_investment_only'
        assert len(scenario.equality_constraints) == 1  # sum(w) = 1
        assert len(scenario.inequality_constraints) == 0
    
    def test_convenience_long_only(self, builder_small):
        """Test long-only scenario builder."""
        scenario = builder_small.long_only()
        
        assert scenario.name == 'long_only'
        assert len(scenario.inequality_constraints) == 1
    
    def test_convenience_fully_invested(self, builder_small):
        """Test fully invested scenario builder."""
        scenario = builder_small.fully_invested()
        
        assert scenario.name == 'fully_invested'
        assert len(scenario.equality_constraints) == 1
    
    def test_convenience_long_only_fully_invested(self, builder_small):
        """Test long-only fully invested scenario builder."""
        scenario = builder_small.long_only_fully_invested()
        
        assert scenario.name == 'long_only_fully_invested'
        assert len(scenario.equality_constraints) == 1  # Fully invested
        assert len(scenario.inequality_constraints) == 1  # Long only
    
    def test_convenience_dollar_neutral(self, builder_small):
        """Test dollar neutral scenario builder."""
        scenario = builder_small.dollar_neutral()
        
        assert scenario.name == 'dollar_neutral'
        assert len(scenario.equality_constraints) == 1
        A, b = scenario.equality_constraints[0]
        assert np.isclose(b[0], 0.0)
    
    def test_convenience_position_limited(self, builder_small):
        """Test position limited scenario builder."""
        scenario = builder_small.position_limited(max_position=0.05)
        
        assert 'position_limited' in scenario.name
        assert len(scenario.equality_constraints) == 1  # Fully invested
        assert len(scenario.inequality_constraints) == 2  # Lower and upper bounds
    
    def test_fluent_interface(self, builder_small):
        """Test that builder methods return scenario for chaining."""
        # Note: Python doesn't have method chaining by default,
        # but methods return scenario which enables this pattern
        scenario = builder_small.create('complex')
        scenario = builder_small.add_fully_invested(scenario)
        scenario = builder_small.add_long_only(scenario)
        
        assert len(scenario.equality_constraints) == 1
        assert len(scenario.inequality_constraints) == 1


# ============================================================================
# OptimizationResult Tests
# ============================================================================

class TestOptimizationResult:
    """Tests for OptimizationResult dataclass."""
    
    def test_basic_properties(self):
        """Test basic result properties."""
        weights = np.array([0.5, 0.3, 0.2, 0.0, 0.0])
        result = OptimizationResult(
            scenario_name='test',
            weights=weights,
            objective=0.05,
            risk=np.sqrt(0.1),
            solve_time=0.1,
            status='optimal'
        )
        
        assert result.scenario_name == 'test'
        assert result.status == 'optimal'
        assert np.allclose(result.weights, weights)
    
    def test_n_positions(self):
        """Test n_positions property."""
        weights = np.array([0.5, 0.3, 0.0, 1e-7, -0.2])
        result = OptimizationResult('test', weights, 0.05, 0.2, 0.1, 'optimal')
        
        assert result.n_positions == 3  # Only significant positions
    
    def test_long_exposure(self):
        """Test long exposure calculation."""
        weights = np.array([0.5, 0.3, 0.0, -0.2])
        result = OptimizationResult('test', weights, 0.05, 0.2, 0.1, 'optimal')
        
        assert np.isclose(result.long_exposure, 0.8)
    
    def test_short_exposure(self):
        """Test short exposure calculation."""
        weights = np.array([0.5, 0.3, -0.2, -0.1])
        result = OptimizationResult('test', weights, 0.05, 0.2, 0.1, 'optimal')
        
        assert np.isclose(result.short_exposure, 0.3)
    
    def test_net_exposure(self):
        """Test net exposure calculation."""
        weights = np.array([0.5, 0.3, -0.2, -0.1])
        result = OptimizationResult('test', weights, 0.05, 0.2, 0.1, 'optimal')
        
        assert np.isclose(result.net_exposure, 0.5)  # 0.8 - 0.3
    
    def test_gross_exposure(self):
        """Test gross exposure calculation."""
        weights = np.array([0.5, 0.3, -0.2, -0.1])
        result = OptimizationResult('test', weights, 0.05, 0.2, 0.1, 'optimal')
        
        assert np.isclose(result.gross_exposure, 1.1)  # 0.8 + 0.3
    
    def test_all_long_portfolio(self):
        """Test metrics for all-long portfolio."""
        weights = np.array([0.4, 0.3, 0.2, 0.1])
        result = OptimizationResult('test', weights, 0.05, 0.2, 0.1, 'optimal')
        
        assert np.isclose(result.long_exposure, 1.0)
        assert np.isclose(result.short_exposure, 0.0)
        assert np.isclose(result.net_exposure, 1.0)
        assert np.isclose(result.gross_exposure, 1.0)
    
    def test_market_neutral_portfolio(self):
        """Test metrics for market neutral portfolio."""
        weights = np.array([0.5, 0.5, -0.5, -0.5])
        result = OptimizationResult('test', weights, 0.05, 0.2, 0.1, 'optimal')
        
        assert np.isclose(result.net_exposure, 0.0)
        assert np.isclose(result.gross_exposure, 2.0)


# ============================================================================
# SimulationRunner Tests
# ============================================================================

class TestSimulationRunner:
    """Tests for SimulationRunner class."""
    
    def test_initialization(self):
        """Test SimulationRunner initialization."""
        runner = SimulationRunner(p=50, k=5, n_samples=1000, seed=42)
        
        assert runner.p == 50
        assert runner.k == 5
        assert runner.n_samples == 1000
        assert runner.seed == 42
        assert isinstance(runner.builder, ScenarioBuilder)
    
    def test_configure_data(self, small_runner):
        """Test data configuration."""
        small_runner.configure_data(
            beta_params={'dist': 'uniform', 'low': -1, 'high': 1},
            factor_vol_params={'dist': 'constant', 'c': 0.2}
        )
        
        assert small_runner._data_config['beta_params']['dist'] == 'uniform'
        assert small_runner._data_config['factor_vol_params']['dist'] == 'constant'
    
    def test_add_scenario_predefined(self, small_runner):
        """Test adding predefined scenario."""
        small_runner.add_scenario('long_only', ScenarioType.LONG_ONLY)
        
        assert 'long_only' in small_runner.scenarios
        assert isinstance(small_runner.scenarios['long_only'], Scenario)
    
    def test_add_scenario_custom(self, small_runner):
        """Test adding custom scenario."""
        custom = small_runner.builder.create('custom')
        custom = small_runner.builder.add_fully_invested(custom)
        
        small_runner.add_scenario('custom', custom_scenario=custom)
        
        assert 'custom' in small_runner.scenarios
    
    def test_add_scenario_invalid(self, small_runner):
        """Test error when adding scenario without type or custom."""
        with pytest.raises(ValueError, match="Must provide either"):
            small_runner.add_scenario('invalid')
    
    def test_generate_data(self, small_runner):
        """Test data generation."""
        data = small_runner.generate_data()
        
        assert isinstance(data, FactorModelData)
        assert data.B.shape == (3, 20)
        assert data.F.shape == (3, 3)
        assert data.D.shape == (20, 20)
    
    def test_generate_data_caching(self, small_runner):
        """Test that data is cached."""
        data1 = small_runner.generate_data()
        data2 = small_runner.generate_data()
        
        # Should be the same object
        assert data1 is data2
    
    def test_generate_data_force_regenerate(self, small_runner):
        """Test force regeneration of data."""
        data1 = small_runner.generate_data()
        data2 = small_runner.generate_data(force_regenerate=True)
        
        # Should be different objects
        assert data1 is not data2
    
    def test_run_without_scenarios(self, small_runner):
        """Test that run fails without scenarios."""
        with pytest.raises(RuntimeError, match="No scenarios added"):
            small_runner.run()
    
    def test_run_single_scenario(self, small_runner):
        """Test running single scenario."""
        small_runner.add_scenario('long_only', ScenarioType.LONG_ONLY_FULLY_INVESTED)
        results = small_runner.run(verbose=False)
        
        assert 'long_only' in results
        assert isinstance(results['long_only'], OptimizationResult)
        assert results['long_only'].status in ['optimal', 'optimal_inaccurate']
    
    def test_run_multiple_scenarios(self, small_runner):
        """Test running multiple scenarios."""
        small_runner.add_scenario('unconstrained', ScenarioType.UNCONSTRAINED)
        small_runner.add_scenario('long_only', ScenarioType.LONG_ONLY_FULLY_INVESTED)
        
        results = small_runner.run(verbose=False)
        
        assert len(results) == 2
        assert all(isinstance(r, OptimizationResult) for r in results.values())
    
    def test_results_stored(self, small_runner):
        """Test that results are stored in runner."""
        small_runner.add_scenario('test', ScenarioType.FULLY_INVESTED)
        small_runner.run(verbose=False)
        
        assert len(small_runner.results) == 1
        assert 'test' in small_runner.results
    
    def test_get_result(self, small_runner):
        """Test getting specific result."""
        small_runner.add_scenario('test', ScenarioType.FULLY_INVESTED)
        small_runner.run(verbose=False)
        
        result = small_runner.get_result('test')
        assert isinstance(result, OptimizationResult)
    
    def test_get_result_not_found(self, small_runner):
        """Test error when getting non-existent result."""
        with pytest.raises(ValueError, match="not found"):
            small_runner.get_result('nonexistent')
    
    def test_compare_scenarios(self, small_runner):
        """Test scenario comparison (should not raise error)."""
        small_runner.add_scenario('s1', ScenarioType.UNCONSTRAINED)
        small_runner.add_scenario('s2', ScenarioType.LONG_ONLY)
        small_runner.run(verbose=False)
        
        # Should not raise error
        small_runner.compare_scenarios('s1', 's2')
    
    def test_compare_scenarios_not_run(self, small_runner):
        """Test comparison fails for unrun scenarios."""
        small_runner.add_scenario('s1', ScenarioType.UNCONSTRAINED)
        
        with pytest.raises(ValueError, match="must be run first"):
            small_runner.compare_scenarios('s1', 's2')
    
    def test_print_summary(self, small_runner, capsys):
        """Test print summary (should not raise error)."""
        small_runner.add_scenario('test', ScenarioType.FULLY_INVESTED)
        small_runner.run(verbose=False)
        
        small_runner.print_summary()
        captured = capsys.readouterr()
        
        assert 'SIMULATION SUMMARY' in captured.out
        assert 'test' in captured.out
    
    def test_print_summary_no_results(self, small_runner, capsys):
        """Test print summary with no results."""
        small_runner.print_summary()
        captured = capsys.readouterr()
        
        assert 'No results' in captured.out


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    """Integration tests with actual optimization."""
    
    def test_unconstrained_vs_constrained(self, medium_runner):
        """Test that unconstrained has lower risk than constrained."""
        medium_runner.add_scenario('unconstrained', ScenarioType.UNCONSTRAINED)
        medium_runner.add_scenario('long_only', ScenarioType.LONG_ONLY_FULLY_INVESTED)
        
        results = medium_runner.run(verbose=False)
        
        # Unconstrained should have lower or equal risk
        assert results['unconstrained'].risk <= results['long_only'].risk
    
    def test_fully_invested_constraint_satisfied(self, small_runner):
        """Test that fully invested constraint is satisfied."""
        small_runner.add_scenario('test', ScenarioType.LONG_ONLY_FULLY_INVESTED)
        results = small_runner.run(verbose=False)
        
        weights = results['test'].weights
        assert np.isclose(weights.sum(), 1.0, atol=1e-4)
    
    def test_long_only_constraint_satisfied(self, small_runner):
        """Test that long-only constraint is satisfied."""
        small_runner.add_scenario('test', ScenarioType.LONG_ONLY_FULLY_INVESTED)
        results = small_runner.run(verbose=False)
        
        weights = results['test'].weights
        assert np.all(weights >= -1e-6)  # All non-negative (with tolerance)
    
    def test_position_limits_satisfied(self, small_runner):
        """Test that position limits are satisfied."""
        scenario = small_runner.builder.position_limited(max_position=0.1)
        small_runner.add_scenario('limited', custom_scenario=scenario)
        
        results = small_runner.run(verbose=False)
        weights = results['limited'].weights
        
        assert np.all(weights >= -1e-6)
        assert np.all(weights <= 0.1 + 1e-6)
    
    def test_dollar_neutral_satisfied(self, small_runner):
        """Test that dollar neutral constraint is satisfied."""
        small_runner.add_scenario('neutral', ScenarioType.DOLLAR_NEUTRAL)
        results = small_runner.run(verbose=False)
        
        weights = results['neutral'].weights
        assert np.isclose(weights.sum(), 0.0, atol=1e-4)
    
    def test_solve_times_recorded(self, small_runner):
        """Test that solve times are recorded."""
        small_runner.add_scenario('test', ScenarioType.FULLY_INVESTED)
        results = small_runner.run(verbose=False)
        
        assert results['test'].solve_time > 0
        assert results['test'].solve_time < 10  # Shouldn't take too long
    
    def test_reproducibility(self):
        """Test that same seed produces same results."""
        runner1 = SimulationRunner(p=30, k=5, seed=42)
        runner1.add_scenario('test', ScenarioType.LONG_ONLY_FULLY_INVESTED)
        results1 = runner1.run(verbose=False)
        
        runner2 = SimulationRunner(p=30, k=5, seed=42)
        runner2.add_scenario('test', ScenarioType.LONG_ONLY_FULLY_INVESTED)
        results2 = runner2.run(verbose=False)
        
        assert np.allclose(results1['test'].weights, results2['test'].weights, atol=1e-4)
    
    def test_multiple_constraints_interaction(self, small_runner):
        """Test portfolio with multiple interacting constraints."""
        scenario = small_runner.builder.create('complex')
        scenario = small_runner.builder.add_fully_invested(scenario)
        scenario = small_runner.builder.add_box_constraints(scenario, lower=0, upper=0.2)
        
        small_runner.add_scenario('complex', custom_scenario=scenario)
        results = small_runner.run(verbose=False)
        
        weights = results['complex'].weights
        
        # Check all constraints satisfied
        assert np.isclose(weights.sum(), 1.0, atol=1e-4)
        assert np.all(weights >= -1e-6)
        assert np.all(weights <= 0.2 + 1e-6)


# ============================================================================
# Edge Cases
# ============================================================================

class TestEdgeCases:
    """Edge cases and boundary conditions."""
    
    def test_very_small_problem(self):
        """Test with minimal problem size."""
        runner = SimulationRunner(p=3, k=1, n_samples=50, seed=42)
        runner.add_scenario('test', ScenarioType.FULLY_INVESTED)
        
        results = runner.run(verbose=False)
        assert results['test'].status in ['optimal', 'optimal_inaccurate']
    
    def test_many_scenarios(self, small_runner):
        """Test running many scenarios."""
        for i in range(10):
            small_runner.add_scenario(f'scenario_{i}', ScenarioType.FULLY_INVESTED)
        
        results = small_runner.run(verbose=False)
        assert len(results) == 10
    
    def test_infeasible_scenario(self, small_runner):
        """Test handling of infeasible scenario."""
        # Create contradictory constraints
        scenario = small_runner.builder.create('infeasible')
        scenario = small_runner.builder.add_fully_invested(scenario)  # sum = 1
        scenario = small_runner.builder.add_dollar_neutral(scenario)   # sum = 0
        
        small_runner.add_scenario('infeasible', custom_scenario=scenario)
        results = small_runner.run(verbose=False)
        
        # Should still complete, but may not be optimal
        assert 'infeasible' in results
    
    def test_tight_constraints(self, small_runner):
        """Test with very tight position constraints."""
        scenario = small_runner.builder.create('tight')
        scenario = small_runner.builder.add_fully_invested(scenario)
        scenario = small_runner.builder.add_box_constraints(scenario, lower=0.04, upper=0.06)
        
        small_runner.add_scenario('tight', custom_scenario=scenario)
        results = small_runner.run(verbose=False)
        
        # Should find solution even with tight constraints
        weights = results['tight'].weights
        assert np.isclose(weights.sum(), 1.0, atol=1e-3)


# ============================================================================
# Performance Tests
# ============================================================================

class TestPerformance:
    """Performance tests (marked as slow)."""
    
    @pytest.mark.slow
    def test_large_problem_performance(self):
        """Test performance with large problem."""
        runner = SimulationRunner(p=500, k=50, n_samples=100, seed=42)
        runner.add_scenario('test', ScenarioType.LONG_ONLY_FULLY_INVESTED)
        
        t0 = time.time()
        results = runner.run(verbose=False)
        elapsed = time.time() - t0
        
        assert elapsed < 60  # Should complete within 1 minute
        assert results['test'].status in ['optimal', 'optimal_inaccurate']
    
    @pytest.mark.slow
    def test_many_scenarios_performance(self):
        """Test performance with many scenarios."""
        runner = SimulationRunner(p=100, k=10, seed=42)
        
        for i in range(20):
            runner.add_scenario(f's_{i}', ScenarioType.FULLY_INVESTED)
        
        t0 = time.time()
        results = runner.run(verbose=False)
        elapsed = time.time() - t0
        
        assert len(results) == 20
        assert elapsed < 30  # Should complete within 30 seconds


# ============================================================================
# Pytest Configuration
# ============================================================================

def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line(
        "markers",
        "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])