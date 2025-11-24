"""
data_sampler.py

Flexible statistical sampling framework for factor model simulation.

This module provides:
1. DistributionFactory: Registry-based sampler creation with validation
2. DataSampler: Generates synthetic factor model data (B, F, D matrices)
3. Built-in distributions: normal, uniform, beta, constant

Example:
    >>> # Create samplers
    >>> beta_sampler = DistributionFactory.create(1000, 'normal', mean=0.5, std=0.2)
    >>> vol_sampler = DistributionFactory.create(1000, 'uniform', low=0.1, high=0.3)
    >>> 
    >>> # Generate data
    >>> sampler = DataSampler(p=100, k=10, n=1000)
    >>> sampler.add_factor_samplers([beta_sampler] * 10, vol_sampler)
    >>> sampler.add_idiosyncratic_sampler(vol_sampler)
    >>> data = sampler.generate()
"""

import numpy as np
import warnings
from typing import Callable, Dict, List, Optional, Any, Union
from dataclasses import dataclass
import inspect


# Type alias for clarity
SamplerFunc = Callable[[], np.ndarray]


@dataclass
class FactorModelData:
    """
    Container for factor model matrices compatible with FactorModelOptimizer.
    
    Attributes:
        B: (k, p) Factor loading matrix
        F: (k, k) Factor covariance matrix (diagonal)
        D: (p, p) Idiosyncratic covariance matrix (diagonal)
        metadata: Optional dictionary with generation parameters
    """
    B: np.ndarray
    F: np.ndarray
    D: np.ndarray
    metadata: Optional[Dict[str, Any]] = None
    
    @property
    def k(self) -> int:
        """Number of factors."""
        return self.B.shape[0]
    
    @property
    def p(self) -> int:
        """Number of securities."""
        return self.B.shape[1]
    
    def validate(self) -> bool:
        """Validates matrix dimensions and properties."""
        if self.B.shape != (self.k, self.p):
            raise ValueError(f"B has wrong shape: {self.B.shape} vs expected ({self.k}, {self.p})")
        if self.F.shape != (self.k, self.k):
            raise ValueError(f"F has wrong shape: {self.F.shape}")
        if self.D.shape != (self.p, self.p):
            raise ValueError(f"D has wrong shape: {self.D.shape}")
        
        # Check F and D are diagonal
        if not np.allclose(self.F, np.diag(np.diag(self.F))):
            raise ValueError("F must be diagonal")
        if not np.allclose(self.D, np.diag(np.diag(self.D))):
            raise ValueError("D must be diagonal")
        
        # Check positive definiteness
        if np.any(np.diag(self.D) <= 0):
            raise ValueError("D must be positive definite")
        if np.any(np.diag(self.F) < 0):
            raise ValueError("F must be positive semidefinite")
        
        return True


class DistributionFactory:
    """
    Factory for creating parameterized sampling functions.
    
    Supports built-in distributions and custom registration. All samplers
    follow the signature: (n: int, **params) -> np.ndarray
    
    Built-in distributions:
        - normal: mean, std
        - uniform: low, high
        - beta: a, b, low (default 0), high (default 1)
        - constant: c
        - lognormal: mean, std
        - exponential: scale
    
    Example:
        >>> # Create a normal sampler
        >>> sampler = DistributionFactory.create(1000, 'normal', mean=0, std=1)
        >>> samples = sampler()  # Returns 1000 samples
        >>> 
        >>> # Register custom distribution
        >>> DistributionFactory.register('custom', lambda n, a, b: a + b * np.random.randn(n))
        >>> custom_sampler = DistributionFactory.create(1000, 'custom', a=5, b=2)
    """
    
    # Registry: distribution_name -> sampling_function
    _registry: Dict[str, Callable] = {
        'normal': lambda n, mean, std: np.random.normal(mean, std, n),
        'uniform': lambda n, low, high: np.random.uniform(low, high, n),
        'beta': lambda n, a, b, low=0.0, high=1.0: (
            low + (high - low) * np.random.beta(a, b, n)
        ),
        'constant': lambda n, c: np.full(n, c, dtype=float),
        'lognormal': lambda n, mean, std: np.random.lognormal(mean, std, n),
        'exponential': lambda n, scale: np.random.exponential(scale, n),
    }
    
    @classmethod
    def list_distributions(cls) -> List[str]:
        """Returns list of available distribution names."""
        return sorted(cls._registry.keys())
    
    @classmethod
    def get_signature(cls, dist_name: str) -> str:
        """Returns the signature of a distribution's sampling function."""
        dist_name = dist_name.lower()
        if dist_name not in cls._registry:
            raise ValueError(f"Unknown distribution: '{dist_name}'")
        
        func = cls._registry[dist_name]
        sig = inspect.signature(func)
        return f"{dist_name}{sig}"
    
    @classmethod
    def register(cls, name: str, sampling_func: Callable, overwrite: bool = False):
        """
        Register a custom distribution.
        
        Args:
            name: Distribution name (case-insensitive)
            sampling_func: Function with signature (n: int, **params) -> np.ndarray
            overwrite: Allow overwriting existing distributions
            
        Example:
            >>> def truncated_normal(n, mean, std, low, high):
            ...     samples = np.random.normal(mean, std, n)
            ...     return np.clip(samples, low, high)
            >>> DistributionFactory.register('truncnorm', truncated_normal)
        """
        name = name.lower()
        
        # Validate function signature
        sig = inspect.signature(sampling_func)
        params = list(sig.parameters.keys())
        if not params or params[0] != 'n':
            raise ValueError("Sampling function must have 'n' as first parameter")
        
        # Check for overwrite
        if name in cls._registry and not overwrite:
            raise ValueError(f"Distribution '{name}' already exists. Use overwrite=True to replace.")
        
        cls._registry[name] = sampling_func
    
    @classmethod
    def create(cls, n_samples: int, dist_name: str, **params) -> SamplerFunc:
        """
        Create a parameterized sampler function.
        
        Args:
            n_samples: Number of samples to generate
            dist_name: Distribution name (case-insensitive)
            **params: Distribution-specific parameters
            
        Returns:
            Sampler function with no arguments that returns n_samples values
            
        Raises:
            ValueError: If distribution unknown or parameters invalid
        """
        dist_name = dist_name.lower()
        
        if dist_name not in cls._registry:
            available = ', '.join(cls.list_distributions())
            raise ValueError(f"Unknown distribution '{dist_name}'. Available: {available}")
        
        func = cls._registry[dist_name]
        sig = inspect.signature(func)
        
        # Validate parameters
        cls._validate_parameters(dist_name, sig, params)
        
        # Create closure with fixed parameters
        def sampler() -> np.ndarray:
            return func(n_samples, **params)
        
        sampler.__name__ = f"{dist_name}_sampler_{n_samples}"
        sampler.__doc__ = f"Sampler for {dist_name}({params}) returning {n_samples} samples"
        
        return sampler
    
    @classmethod
    def _validate_parameters(cls, dist_name: str, sig: inspect.Signature, params: Dict):
        """Validates provided parameters against function signature."""
        bound_params = list(sig.parameters.keys())[1:]  # Skip 'n'
        
        # Check for missing required parameters
        required = [
            name for name, param in list(sig.parameters.items())[1:]
            if param.default == inspect.Parameter.empty
        ]
        
        missing = set(required) - set(params.keys())
        if missing:
            raise ValueError(
                f"Missing required parameters for '{dist_name}': {missing}. "
                f"Signature: {cls.get_signature(dist_name)}"
            )
        
        # Warn about unknown parameters
        unknown = set(params.keys()) - set(bound_params)
        if unknown:
            warnings.warn(
                f"Unknown parameters for '{dist_name}' will be ignored: {unknown}",
                UserWarning
            )
    
    @classmethod
    def create_transformed(cls, n_samples: int, dist_name: str, 
                          transform: Callable[[np.ndarray], np.ndarray],
                          **params) -> SamplerFunc:
        """
        Create a sampler with post-processing transformation.
        
        Args:
            n_samples: Number of samples
            dist_name: Base distribution name
            transform: Function to apply to sampled values
            **params: Distribution parameters
            
        Example:
            >>> # Sample normal and take absolute value
            >>> sampler = DistributionFactory.create_transformed(
            ...     1000, 'normal', np.abs, mean=0, std=1
            ... )
        """
        base_sampler = cls.create(n_samples, dist_name, **params)
        
        def transformed_sampler() -> np.ndarray:
            return transform(base_sampler())
        
        return transformed_sampler


class DataSampler:
    """
    Generates synthetic factor model data (B, F, D matrices).
    
    Provides flexible interface for specifying how each component is sampled.
    
    Example:
        >>> # Basic usage
        >>> sampler = DataSampler(p=100, k=10, n=1000)
        >>> 
        >>> # Add factor samplers (one per factor)
        >>> beta_samplers = [
        ...     DistributionFactory.create(1000, 'normal', mean=0, std=1)
        ...     for _ in range(10)
        ... ]
        >>> factor_vol = DistributionFactory.create(1000, 'uniform', low=0.1, high=0.3)
        >>> sampler.add_factor_samplers(beta_samplers, factor_vol)
        >>> 
        >>> # Add idiosyncratic sampler
        >>> idio_vol = DistributionFactory.create(1000, 'uniform', low=0.05, high=0.2)
        >>> sampler.add_idiosyncratic_sampler(idio_vol)
        >>> 
        >>> # Generate data
        >>> data = sampler.generate()
    """
    
    def __init__(self, p: int, k: int, n: int, seed: Optional[int] = None):
        """
        Initialize data sampler.
        
        Args:
            p: Number of securities (assets)
            k: Number of factors
            n: Number of samples for each distribution
            seed: Random seed for reproducibility
        """
        self.p = p
        self.k = k
        self.n = n
        
        if seed is not None:
            np.random.seed(seed)
        
        # Sampler storage
        self._beta_samplers: List[SamplerFunc] = []
        self._factor_vol_sampler: Optional[SamplerFunc] = None
        self._idio_vol_sampler: Optional[SamplerFunc] = None
        
        # Configuration
        self._normalize_beta = False
        self._ensure_positive_vols = True
    
    def add_factor_samplers(self, beta_samplers: List[SamplerFunc], 
                           factor_vol_sampler: SamplerFunc):
        """
        Configure factor loading and volatility samplers.
        
        Args:
            beta_samplers: List of k samplers for factor loadings (one per factor)
            factor_vol_sampler: Single sampler for factor volatilities
            
        Raises:
            ValueError: If number of beta samplers doesn't match k
        """
        if len(beta_samplers) != self.k:
            raise ValueError(f"Need {self.k} beta samplers, got {len(beta_samplers)}")
        
        self._beta_samplers = beta_samplers
        self._factor_vol_sampler = factor_vol_sampler
    
    def add_idiosyncratic_sampler(self, vol_sampler: SamplerFunc):
        """
        Configure idiosyncratic volatility sampler.
        
        Args:
            vol_sampler: Sampler for idiosyncratic volatilities
        """
        self._idio_vol_sampler = vol_sampler
    
    def configure(self, normalize_beta: bool = False, 
                 ensure_positive_vols: bool = True):
        """
        Set generation options.
        
        Args:
            normalize_beta: Normalize beta to mean 0, std 1 per factor
            ensure_positive_vols: Take absolute value of volatilities
        """
        self._normalize_beta = normalize_beta
        self._ensure_positive_vols = ensure_positive_vols
    
    def generate(self) -> FactorModelData:
        """
        Generate factor model matrices.
        
        Returns:
            FactorModelData with B, F, D matrices
            
        Raises:
            RuntimeError: If samplers not configured
        """
        if not self._beta_samplers:
            raise RuntimeError("Factor samplers not configured. Call add_factor_samplers() first.")
        if self._idio_vol_sampler is None:
            raise RuntimeError("Idiosyncratic sampler not configured. Call add_idiosyncratic_sampler() first.")
        
        # Generate B matrix (k, p)
        B = self._generate_factor_loadings()
        
        # Generate F matrix (k, k) - diagonal
        F = self._generate_factor_covariance()
        
        # Generate D matrix (p, p) - diagonal
        D = self._generate_idiosyncratic_covariance()
        
        # Create metadata
        metadata = {
            'n_samples': self.n,
            'normalized_beta': self._normalize_beta,
            'seed': np.random.get_state()[1][0] if hasattr(np.random.get_state()[1], '__getitem__') else None
        }
        
        data = FactorModelData(B, F, D, metadata)
        data.validate()
        
        return data
    
    def _generate_factor_loadings(self) -> np.ndarray:
        """Generate factor loading matrix B."""
        B = np.zeros((self.k, self.p))
        
        for factor_idx in range(self.k):
            sampler = self._beta_samplers[factor_idx]
            samples = sampler()
            
            # Randomly assign samples to securities
            B[factor_idx, :] = np.random.choice(samples, size=self.p, replace=True)
        
        # Optional normalization
        if self._normalize_beta:
            for factor_idx in range(self.k):
                factor_loadings = B[factor_idx, :]
                if factor_loadings.std() > 1e-8:
                    B[factor_idx, :] = (factor_loadings - factor_loadings.mean()) / factor_loadings.std()
        
        return B
    
    def _generate_factor_covariance(self) -> np.ndarray:
        """Generate diagonal factor covariance matrix F."""
        samples = self._factor_vol_sampler()
        factor_vols = np.random.choice(samples, size=self.k, replace=True)
        
        if self._ensure_positive_vols:
            factor_vols = np.abs(factor_vols)
        
        # Convert volatilities to variances
        factor_vars = factor_vols ** 2
        return np.diag(factor_vars)
    
    def _generate_idiosyncratic_covariance(self) -> np.ndarray:
        """Generate diagonal idiosyncratic covariance matrix D."""
        samples = self._idio_vol_sampler()
        idio_vols = np.random.choice(samples, size=self.p, replace=True)
        
        if self._ensure_positive_vols:
            idio_vols = np.abs(idio_vols)
        
        # Convert volatilities to variances
        idio_vars = idio_vols ** 2
        return np.diag(idio_vars)
    
    @classmethod
    def quick_generate(cls, p: int, k: int, 
                      beta_params: Dict[str, Any] = None,
                      factor_vol_params: Dict[str, Any] = None,
                      idio_vol_params: Dict[str, Any] = None,
                      n_samples: int = 1000,
                      seed: Optional[int] = None) -> FactorModelData:
        """
        Convenience method for quick data generation with defaults.
        
        Args:
            p: Number of securities
            k: Number of factors
            beta_params: Parameters for beta distribution (default: normal(0, 1))
            factor_vol_params: Parameters for factor vol (default: uniform(0.1, 0.3))
            idio_vol_params: Parameters for idio vol (default: uniform(0.05, 0.2))
            n_samples: Number of samples per distribution
            seed: Random seed
            
        Returns:
            FactorModelData
            
        Example:
            >>> data = DataSampler.quick_generate(
            ...     p=100, k=10,
            ...     beta_params={'dist': 'normal', 'mean': 0.5, 'std': 0.2},
            ...     seed=42
            ... )
        """
        # Set defaults
        beta_params = beta_params or {'dist': 'normal', 'mean': 0, 'std': 1}
        factor_vol_params = factor_vol_params or {'dist': 'uniform', 'low': 0.1, 'high': 0.3}
        idio_vol_params = idio_vol_params or {'dist': 'uniform', 'low': 0.05, 'high': 0.2}
        
        # Extract distribution names
        beta_dist = beta_params.pop('dist', 'normal')
        factor_vol_dist = factor_vol_params.pop('dist', 'uniform')
        idio_vol_dist = idio_vol_params.pop('dist', 'uniform')
        
        # Create sampler
        sampler = cls(p, k, n_samples, seed)
        
        # Create factor samplers (all factors use same distribution)
        beta_samplers = [
            DistributionFactory.create(n_samples, beta_dist, **beta_params)
            for _ in range(k)
        ]
        factor_vol_sampler = DistributionFactory.create(n_samples, factor_vol_dist, **factor_vol_params)
        idio_vol_sampler = DistributionFactory.create(n_samples, idio_vol_dist, **idio_vol_params)
        
        # Configure and generate
        sampler.add_factor_samplers(beta_samplers, factor_vol_sampler)
        sampler.add_idiosyncratic_sampler(idio_vol_sampler)
        
        return sampler.generate()


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    print("="*70)
    print("Data Sampler Examples")
    print("="*70)
    
    # Example 1: Quick generation
    print("\n1. Quick Generation:")
    data = DataSampler.quick_generate(p=50, k=5, seed=42)
    print(f"   Generated: B{data.B.shape}, F{data.F.shape}, D{data.D.shape}")
    print(f"   Beta range: [{data.B.min():.3f}, {data.B.max():.3f}]")
    print(f"   Factor vol range: [{np.sqrt(np.diag(data.F)).min():.3f}, {np.sqrt(np.diag(data.F)).max():.3f}]")
    
    # Example 2: Custom configuration
    print("\n2. Custom Configuration:")
    sampler = DataSampler(p=100, k=10, n=1000, seed=42)
    
    # Create custom samplers
    beta_samplers = [
        DistributionFactory.create(1000, 'normal', mean=i*0.1, std=0.2)
        for i in range(10)
    ]
    factor_vol = DistributionFactory.create(1000, 'lognormal', mean=-2, std=0.5)
    idio_vol = DistributionFactory.create(1000, 'uniform', low=0.05, high=0.15)
    
    sampler.add_factor_samplers(beta_samplers, factor_vol)
    sampler.add_idiosyncratic_sampler(idio_vol)
    sampler.configure(normalize_beta=True)
    
    data = sampler.generate()
    print(f"   Generated: B{data.B.shape}, F{data.F.shape}, D{data.D.shape}")
    print(f"   Validation: {data.validate()}")
    
    # Example 3: List available distributions
    print("\n3. Available Distributions:")
    for dist in DistributionFactory.list_distributions():
        print(f"   - {DistributionFactory.get_signature(dist)}")
    
    print("\n" + "="*70)