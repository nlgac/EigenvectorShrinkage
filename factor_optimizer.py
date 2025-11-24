"""
factor_optimizer.py

Efficient Convex Optimizer for Factor Models with Low-Rank Structure.

Solves: min 0.5 * w.T @ (B.T @ F @ B + D) @ w subject to linear constraints
Exploits low-rank structure by reformulating as a conic problem, avoiding 
dense covariance matrix formation. Handles thousands of variables in seconds.

Key Innovation: Introduces auxiliary variable y = Bw to transform the dense 
quadratic program into a sparse problem with norm-based objectives.
"""

import cvxpy as cp
import numpy as np
import time
from typing import Optional, Dict, Tuple, List
from dataclasses import dataclass
from enum import Enum


class ConstraintType(Enum):
    """Constraint types for cleaner type handling."""
    EQUALITY = 'eq'
    INEQUALITY = 'ineq'


@dataclass
class ProblemData:
    """Container for factor model problem data."""
    B: np.ndarray  # (k, p) Factor loadings
    F: np.ndarray  # (k, k) Factor covariance (diagonal)
    D: np.ndarray  # (p, p) Idiosyncratic variance (diagonal)
    A_eq: Optional[np.ndarray] = None  # (m_eq, p) Equality constraint matrix
    b_eq: Optional[np.ndarray] = None  # (m_eq,) Equality RHS
    A_in: Optional[np.ndarray] = None  # (m_in, p) Inequality constraint matrix
    b_in: Optional[np.ndarray] = None  # (m_in,) Inequality RHS
    
    @property
    def dimensions(self) -> Tuple[int, int]:
        """Returns (k, p) dimensions."""
        return self.B.shape
    
    @property
    def k(self) -> int:
        """Number of factors."""
        return self.B.shape[0]
    
    @property
    def p(self) -> int:
        """Number of assets."""
        return self.B.shape[1]


def generate_problem_data(p: int, k: int, m_eq: int = 0, m_in: int = 0, 
                         seed: Optional[int] = None) -> ProblemData:
    """
    Generates synthetic factor model optimization data.
    
    Args:
        p: Number of assets (variables)
        k: Number of factors
        m_eq: Number of equality constraints
        m_in: Number of inequality constraints
        seed: Random seed for reproducibility
        
    Returns:
        ProblemData object containing all matrices
    """
    if seed is not None:
        np.random.seed(seed)
    
    print(f"Generating data: p={p}, k={k}, m_eq={m_eq}, m_in={m_in}")
    
    # Factor structure
    B = np.random.randn(k, p)
    F = np.diag(np.random.rand(k) * 10)  # Diagonal factor covariance
    D = np.diag(np.random.rand(p) * 5 + 0.1)  # Positive definite idiosyncratic variance
    
    # Constraints (None if counts are 0)
    A_eq = np.random.randn(m_eq, p) if m_eq > 0 else None
    b_eq = np.random.randn(m_eq) if m_eq > 0 else None
    A_in = np.random.randn(m_in, p) if m_in > 0 else None
    b_in = np.abs(np.random.randn(m_in)) if m_in > 0 else None  # Ensure feasibility
    
    # Add fully-invested constraint if equality constraints exist
    if m_eq > 0:
        A_eq[0, :] = 1.0
        b_eq[0] = 1.0
    
    return ProblemData(B, F, D, A_eq, b_eq, A_in, b_in)


class FactorModelOptimizer:
    """
    Efficient solver for factor model portfolio optimization.
    
    Reformulates the problem using auxiliary variable y = Bw:
        Minimize: 0.5 * ||F^0.5 y||^2 + 0.5 * ||D^0.5 w||^2
        Subject to: y = Bw, and user-specified linear constraints
    
    This avoids forming the dense covariance matrix C = B.T @ F @ B + D.
    
    Example:
        >>> data = generate_problem_data(p=1000, k=50, m_eq=5, m_in=10)
        >>> opt = FactorModelOptimizer(data.B, data.F, data.D)
        >>> opt.add_constraints(data.A_eq, data.b_eq, 'eq')
        >>> opt.add_constraints(data.A_in, data.b_in, 'ineq')
        >>> opt.solve()
        >>> w_optimal = opt.solution
    """
    
    def __init__(self, B: np.ndarray, F: np.ndarray, D: np.ndarray):
        """
        Initialize optimizer with factor model structure.
        
        Args:
            B: (k, p) Factor loading matrix
            F: (k, k) Diagonal factor covariance matrix (must be PSD)
            D: (p, p) Diagonal idiosyncratic variance matrix (must be PD)
            
        Raises:
            ValueError: If dimensions mismatch or matrices aren't valid
        """
        self.B = B
        self.F = F
        self.D = D
        self.k, self.p = B.shape
        
        # Validate dimensions and properties
        self._validate_inputs()
        
        # Extract and cache diagonal square roots for efficiency
        self._D_diag = np.diag(D)
        self._F_diag = np.diag(F)
        self._D_sqrt = np.sqrt(self._D_diag)
        self._F_sqrt = np.sqrt(self._F_diag)
        
        # CVXPY variables (handle k=0 case)
        self.w = cp.Variable(self.p, name="w")
        self.y = cp.Variable(self.k, name="y") if self.k > 0 else None
        
        # Problem state
        self._constraints: List[Tuple[np.ndarray, np.ndarray, ConstraintType]] = []
        self._cvx_constraints: List[cp.Constraint] = []
        self._problem: Optional[cp.Problem] = None
        self._C_cache: Optional[np.ndarray] = None  # Cached full covariance matrix
    
    def _validate_inputs(self):
        """Validates matrix dimensions and properties."""
        if self.F.shape != (self.k, self.k):
            raise ValueError(f"F shape {self.F.shape} inconsistent with B shape {self.B.shape}")
        if self.D.shape != (self.p, self.p):
            raise ValueError(f"D shape {self.D.shape} inconsistent with B shape {self.B.shape}")
        
        D_diag = np.diag(self.D)
        F_diag = np.diag(self.F)
        
        if np.any(D_diag <= 0):
            raise ValueError("D must be positive definite (all diagonal elements > 0)")
        if np.any(F_diag < 0):
            raise ValueError("F must be positive semidefinite (all diagonal elements >= 0)")
    
    def add_constraints(self, A: np.ndarray, b: np.ndarray, 
                       constraint_type: str = 'eq'):
        """
        Add linear constraints to the problem.
        
        Args:
            A: (m, p) Constraint matrix
            b: (m,) Right-hand side vector
            constraint_type: 'eq' for equality (A @ w == b) or 'ineq' for inequality (A @ w <= b)
            
        Raises:
            ValueError: If dimensions mismatch or invalid constraint type
        """
        if A.shape[1] != self.p:
            raise ValueError(f"A has {A.shape[1]} columns, expected {self.p}")
        if A.shape[0] != len(b):
            raise ValueError(f"A has {A.shape[0]} rows but b has length {len(b)}")
        
        # Validate constraint type
        if constraint_type not in ['eq', 'ineq']:
            raise ValueError(f"Invalid constraint_type '{constraint_type}'. Use 'eq' or 'ineq'")
        
        ctype = ConstraintType.EQUALITY if constraint_type == 'eq' else ConstraintType.INEQUALITY
        
        self._constraints.append((A, b, ctype))
        self._problem = None  # Invalidate cached problem
    
    def add_equality_constraint(self, A: np.ndarray, b: np.ndarray):
        """Add equality constraint A @ w == b."""
        self.add_constraints(A, b, 'eq')
    
    def add_inequality_constraint(self, A: np.ndarray, b: np.ndarray):
        """Add inequality constraint A @ w <= b."""
        self.add_constraints(A, b, 'ineq')
    
    def _build_problem(self):
        """Constructs the CVXPY problem with sparse reformulation."""
        # Objective: 0.5 * ||D^0.5 w||^2 + 0.5 * ||F^0.5 y||^2
        # Using element-wise multiplication for efficiency
        obj_w = cp.sum_squares(cp.multiply(self._D_sqrt, self.w))
        
        if self.k > 0:
            # Normal case with factors
            obj_y = cp.sum_squares(cp.multiply(self._F_sqrt, self.y))
            objective = cp.Minimize(0.5 * (obj_w + obj_y))
            
            # Core constraint linking auxiliary variable: y = Bw
            constraints = [self.y == self.B @ self.w]
        else:
            # Degenerate case: no factors, just diagonal covariance
            objective = cp.Minimize(0.5 * obj_w)
            constraints = []
        
        self._cvx_constraints = []
        
        # Add user constraints
        for A, b, ctype in self._constraints:
            if ctype == ConstraintType.EQUALITY:
                cons = A @ self.w == b
            else:  # INEQUALITY
                cons = A @ self.w <= b
            
            constraints.append(cons)
            self._cvx_constraints.append(cons)
        
        self._problem = cp.Problem(objective, constraints)
    
    def solve(self, solver: str = 'CLARABEL', verbose: bool = False, 
             **solver_kwargs) -> str:
        """
        Solve the optimization problem.
        
        Args:
            solver: Solver to use (default: CLARABEL, also supports OSQP, SCS, etc.)
            verbose: Print detailed solver output
            **solver_kwargs: Additional arguments passed to solver (e.g., eps_abs, max_iter)
            
        Returns:
            Solver status string ('optimal', 'infeasible', etc.)
        """
        if self._problem is None:
            self._build_problem()
        
        print(f"\n{'='*60}")
        print(f"Solving with {solver}")
        print(f"{'='*60}")
        
        t_start = time.time()
        
        try:
            self._problem.solve(solver=solver, verbose=verbose, **solver_kwargs)
            elapsed = time.time() - t_start
            
            status = self._problem.status
            print(f"Status: {status}")
            print(f"Time: {elapsed:.4f}s")
            
            if self.is_solved:
                print(f"Objective: {self._problem.value:.8f}")
            else:
                print("⚠ Warning: Optimal solution not found")
                
        except cp.SolverError as e:
            print(f"❌ Solver Error: {e}")
            return "error"
        
        return self._problem.status
    
    @property
    def is_solved(self) -> bool:
        """Returns True if problem has been solved to optimality."""
        return (self._problem is not None and 
                self._problem.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE])
    
    @property
    def solution(self) -> Optional[np.ndarray]:
        """Returns optimal w vector if solved, else None."""
        return self.w.value if self.is_solved else None
    
    @property
    def w_solution(self) -> Optional[np.ndarray]:
        """Alias for solution property."""
        return self.solution
    
    @property
    def y_solution(self) -> Optional[np.ndarray]:
        """Returns optimal auxiliary variable y if solved, else None."""
        if self.k == 0:
            return None  # No auxiliary variable when k=0
        return self.y.value if self.is_solved else None
    
    @property
    def objective_value(self) -> Optional[float]:
        """Returns optimal objective value if solved, else None."""
        return self._problem.value if self.is_solved else None
    
    def evaluate_objective(self, w: np.ndarray) -> float:
        """
        Efficiently evaluates objective 0.5 * w.T @ C @ w without forming C.
        
        Complexity: O(kp + p) vs O(p^2) for naive approach.
        
        Args:
            w: (p,) weight vector to evaluate
            
        Returns:
            Objective value at w
        """
        if w.shape != (self.p,):
            raise ValueError(f"w must have shape ({self.p},), got {w.shape}")
        
        y = self.B @ w  # O(kp)
        term_factor = np.dot(self._F_diag, y**2)  # O(k)
        term_idio = np.dot(self._D_diag, w**2)    # O(p)
        
        return 0.5 * (term_factor + term_idio)
    
    def get_covariance_matrix(self, subset: Optional[int] = 10, 
                            force_full: bool = False) -> Optional[np.ndarray]:
        """
        Returns the covariance matrix C = B.T @ F @ B + D (or a subset).
        
        Warning: Full matrix computation is O(kp^2) and should be avoided for large p.
        
        Args:
            subset: Return only top-left (subset x subset) block (default: 10)
            force_full: Must be True to compute full matrix when subset is None
            
        Returns:
            Covariance matrix or None if refused
        """
        if subset is not None:
            # Compute only the requested slice - much more efficient
            n = min(subset, self.p)
            B_sub = self.B[:, :n]
            return (B_sub.T @ self.F @ B_sub) + np.diag(self._D_diag[:n])
        
        # Full matrix requested
        if not force_full:
            print(f"⚠ Full {self.p}x{self.p} matrix computation is expensive.")
            print("  Set force_full=True to proceed or use subset parameter.")
            return None
        
        if self._C_cache is None:
            print(f"Computing full {self.p}x{self.p} covariance matrix...")
            self._C_cache = (self.B.T @ self.F @ self.B) + self.D
        
        return self._C_cache
    
    def verify_kkt_stationarity(self, tolerance: float = 1e-4) -> bool:
        """
        Verifies KKT stationarity condition for optimality.
        
        Checks: ∇L = ∇f(w*) + Σ λᵢ ∇gᵢ(w*) ≈ 0
        where f is objective and gᵢ are constraints.
        
        Args:
            tolerance: Maximum acceptable norm of gradient
            
        Returns:
            True if KKT conditions satisfied within tolerance
        """
        if not self.is_solved:
            print("❌ Model not solved - cannot verify KKT conditions")
            return False
        
        w_star = self.w.value
        
        # Gradient of objective: ∇f = C @ w = (B.T @ F @ B + D) @ w
        grad_obj = self.B.T @ (self.F @ (self.B @ w_star)) + self._D_diag * w_star
        
        # Add constraint gradients weighted by dual variables
        grad_lagrangian = grad_obj.copy()
        
        for (A, _, _), cvx_cons in zip(self._constraints, self._cvx_constraints):
            dual = cvx_cons.dual_value
            if dual is not None:
                grad_lagrangian += A.T @ dual
        
        residual = np.linalg.norm(grad_lagrangian)
        passed = residual < tolerance
        
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"\nKKT Stationarity Check:")
        print(f"  Gradient norm: {residual:.2e}")
        print(f"  Tolerance: {tolerance:.2e}")
        print(f"  Status: {status}")
        
        return passed
    
    def cross_validate_solver(self, alt_solver: str = 'OSQP', 
                             tolerance: float = 1e-3) -> bool:
        """
        Validates solution by re-solving with alternative solver.
        
        Args:
            alt_solver: Alternative solver to use for validation
            tolerance: Maximum acceptable solution difference
            
        Returns:
            True if solutions agree within tolerance
        """
        if not self.is_solved:
            print("❌ No solution to validate")
            return False
        
        w_original = self.w.value.copy()
        obj_original = self._problem.value
        
        print(f"\nCross-validation with {alt_solver}...")
        
        try:
            self._problem.solve(solver=alt_solver, verbose=False)
            
            if not self.is_solved:
                print(f"  ⚠ {alt_solver} failed to converge")
                self.w.value = w_original
                return False
            
            diff_norm = np.linalg.norm(self.w.value - w_original)
            diff_obj = abs(self._problem.value - obj_original)
            passed = diff_norm < tolerance
            
            status = "✓ PASS" if passed else "✗ FAIL"
            print(f"  Solution difference: {diff_norm:.2e}")
            print(f"  Objective difference: {diff_obj:.2e}")
            print(f"  Status: {status}")
            
            return passed
            
        except Exception as e:
            print(f"  ❌ Cross-validation failed: {e}")
            return False
        finally:
            # Always restore original solution
            self.w.value = w_original
    
    def verify_solution(self, kkt_tol: float = 1e-4, 
                       cross_validate: bool = True,
                       alt_solver: str = 'OSQP') -> Dict[str, bool]:
        """
        Comprehensive solution verification.
        
        Args:
            kkt_tol: Tolerance for KKT stationarity check
            cross_validate: Whether to cross-validate with alternative solver
            alt_solver: Alternative solver for cross-validation
            
        Returns:
            Dictionary with verification results
        """
        results = {}
        
        if not self.is_solved:
            print("❌ No solution to verify")
            return {'solved': False}
        
        print(f"\n{'='*60}")
        print("Solution Verification")
        print(f"{'='*60}")
        
        # KKT stationarity
        results['kkt_satisfied'] = self.verify_kkt_stationarity(kkt_tol)
        
        # Objective consistency
        w = self.solution
        obj_cvxpy = self.objective_value
        obj_manual = self.evaluate_objective(w)
        obj_diff = abs(obj_cvxpy - obj_manual)
        
        print(f"\nObjective Consistency:")
        print(f"  CVXPY value: {obj_cvxpy:.8f}")
        print(f"  Manual eval: {obj_manual:.8f}")
        print(f"  Difference: {obj_diff:.2e}")
        
        results['objective_consistent'] = obj_diff < 1e-6
        
        # Cross-validation
        if cross_validate:
            results['cross_validation'] = self.cross_validate_solver(alt_solver)
        
        print(f"\n{'='*60}")
        all_passed = all(results.values())
        print(f"Overall: {'✓ ALL CHECKS PASSED' if all_passed else '⚠ SOME CHECKS FAILED'}")
        print(f"{'='*60}\n")
        
        return results


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    # Generate synthetic problem
    data = generate_problem_data(p=1000, k=50, m_eq=5, m_in=10, seed=42)
    
    # Initialize optimizer
    opt = FactorModelOptimizer(data.B, data.F, data.D)
    
    # Add constraints
    if data.A_eq is not None:
        opt.add_equality_constraint(data.A_eq, data.b_eq)
    if data.A_in is not None:
        opt.add_inequality_constraint(data.A_in, data.b_in)
    
    # Solve
    opt.solve(solver='CLARABEL')
    
    # Comprehensive verification
    if opt.is_solved:
        opt.verify_solution(cross_validate=True)
        
        # Display solution statistics
        w = opt.solution
        print("\nSolution Statistics:")
        print(f"  ||w||: {np.linalg.norm(w):.6f}")
        print(f"  min(w): {w.min():.6f}")
        print(f"  max(w): {w.max():.6f}")
        print(f"  mean(w): {w.mean():.6f}")
        
        # Show small slice of covariance matrix
        C_slice = opt.get_covariance_matrix(subset=5)
        print(f"\nCovariance matrix (5×5 slice):")
        print(C_slice)