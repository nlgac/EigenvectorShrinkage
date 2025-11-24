"""
test_factor_optimizer.py

Robust unit tests for FactorModelOptimizer using Pytest.
Includes tests for math logic, solver mechanics, and verification utilities.

Usage: 
    pytest test_factor_optimizer.py
"""

import pytest
import numpy as np
import cvxpy as cp
from factor_optimizer import FactorModelOptimizer, generate_problem_data

# --- Fixtures ---

@pytest.fixture(scope="module")
def problem_data():
    """Generates consistent test data used across all tests."""
    return generate_problem_data(p=100, k=10, m_eq=5, m_in=10)

@pytest.fixture(scope="module")
def ground_truth_C(problem_data):
    """Computes the true C matrix for validation comparisons."""
    d = problem_data
    return (d.B.T @ d.F @ d.B) + d.D

@pytest.fixture
def optimizer(problem_data):
    """Returns a fresh optimizer instance with constraints added."""
    d = problem_data
    opt = FactorModelOptimizer(d.B, d.F, d.D)
    opt.add_equality_constraint(d.A_eq, d.b_eq)
    opt.add_inequality_constraint(d.A_in, d.b_in)
    return opt

# --- Core Logic Tests ---

def test_init_validation():
    """Test that invalid matrix shapes raise errors."""
    with pytest.raises(ValueError):
        # D must be square p x p
        FactorModelOptimizer(np.zeros((2,2)), np.eye(2), np.eye(3))

def test_evaluate_objective_accuracy(optimizer, problem_data, ground_truth_C):
    """
    Verifies that the O(k) evaluate_objective method matches 
    the O(p^2) manual matrix multiplication result.
    """
    w = np.random.rand(problem_data.p)
    
    # 1. Fast calculation
    val_fast = optimizer.evaluate_objective(w)
    
    # 2. Slow "ground truth" calculation: 0.5 * w.T @ C @ w
    val_truth = 0.5 * (w.T @ ground_truth_C @ w)
    
    assert np.isclose(val_fast, val_truth)

def test_solve_mechanics(optimizer):
    """Tests the solving process and status reporting."""
    status = optimizer.solve(solver='CLARABEL', verbose=False)
    
    assert status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]
    assert optimizer.w_solution is not None
    assert optimizer.y_solution is not None
    assert len(optimizer.w_solution) == optimizer.p

def test_constraint_satisfaction(optimizer, problem_data):
    """Checks if the solution actually satisfies the linear constraints."""
    optimizer.solve()
    w = optimizer.w_solution
    d = problem_data
    
    # Equality: ||Ax - b|| ~ 0
    eq_resid = np.linalg.norm(d.A_eq @ w - d.b_eq)
    assert eq_resid < 1e-4
    
    # Inequality: Ax - b <= 0
    in_resid = np.max(d.A_in @ w - d.b_in)
    assert in_resid < 1e-4

# --- Feature & Verification Tests ---

def test_kkt_verification(optimizer):
    """Tests that the KKT check returns True for an optimal solution."""
    optimizer.solve()
    assert optimizer.verify_kkt_stationarity(tolerance=1e-3) == True

def test_get_C_matrix(optimizer, ground_truth_C):
    """Tests lazy C matrix generation and slicing."""
    # Slice
    slice_5 = optimizer.get_covariance_matrix(subset=5)
    assert np.allclose(slice_5, ground_truth_C[:5, :5])
    
    # Safety Catch
    assert optimizer.get_covariance_matrix(subset=None, force_full=False) is None
    
    # Full
    full = optimizer.get_covariance_matrix(subset=None, force_full=True)
    assert np.allclose(full, ground_truth_C)

def test_cross_check(optimizer):
    """
    Tests cross-solver verification. 
    Note: Wraps in try/except in case OSQP is not installed in the test env.
    """
    optimizer.solve()
    try:
        # We expect OSQP to agree with CLARABEL for this Convex QP
        result = optimizer.cross_validate_solver('OSQP', tolerance=1e-2)
        if result is not None: # If OSQP ran
            assert result == True
    except ImportError:
        pytest.skip("OSQP solver not available")

def test_closed_form_mvp_equality(problem_data, ground_truth_C):
        """
        Confirms numerical solution for the fully-invested MVP problem
        matches the analytical closed-form solution (C^-1 * 1).
        """
        # 1. Setup Data
        C = ground_truth_C
        p = problem_data.p
        
        # 2. Calculate Analytical Solution (w = C^-1 @ 1 / (1.T @ C^-1 @ 1))
        # Note: We compute the inverse directly since p is small (100) in the fixture.
        ones = np.ones(p)
        
        # Use np.linalg.solve(C, ones) which is numerically superior to inverting C
        # Solution to C^-1 @ 1 is the vector w' such that C @ w' = 1
        w_prime = np.linalg.solve(C, ones)
        
        # w_MVP = w' / (1.T @ w')
        denominator = ones.T @ w_prime
        w_analytical = w_prime / denominator
        
        # 3. Configure and Solve Numerically
        d = problem_data
        
        # Use a fresh optimizer instance to ensure only the MVP constraint is active
        opt_fresh = FactorModelOptimizer(d.B, d.F, d.D)
        
        A_eq = np.ones((1, p))
        b_eq = np.array([1.0])
        opt_fresh.add_equality_constraint(A_eq, b_eq)
        
        status = opt_fresh.solve(solver='CLARABEL', verbose=False)
        w_numerical = opt_fresh.solution
        
        # 4. Assert Close Equality
        assert status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]
        assert np.allclose(w_analytical, w_numerical, atol=1e-5), \
            f"MVP solutions disagree. Analytical Norm: {np.linalg.norm(w_analytical):.6f}, Numerical Norm: {np.linalg.norm(w_numerical):.6f}"