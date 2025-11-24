"""
test_data_sampler.py

Comprehensive test suite for data_sampler module.

Tests cover:
- DistributionFactory registration and creation
- Parameter validation and error handling
- DataSampler matrix generation
- FactorModelData validation
- Custom distributions
- Edge cases and boundary conditions

Usage:
    pytest test_data_sampler.py -v
    pytest test_data_sampler.py -v --cov=data_sampler
"""

import pytest
import numpy as np
from typing import Callable
from data_sampler import (
    DistributionFactory,
    DataSampler,
    FactorModelData,
    SamplerFunc
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture(scope="module")
def sample_sizes():
    """Common sample sizes for testing."""
    return {'small': 100, 'medium': 1000, 'large': 10000}


@pytest.fixture
def reset_factory():
    """Reset factory registry after tests that modify it."""
    # Store original registry
    original = DistributionFactory._registry.copy()
    yield
    # Restore
    DistributionFactory._registry = original


@pytest.fixture
def basic_samplers():
    """Create basic samplers for common tests."""
    return {
        'normal': DistributionFactory.create(1000, 'normal', mean=0, std=1),
        'uniform': DistributionFactory.create(1000, 'uniform', low=0, high=1),
        'constant': DistributionFactory.create(1000, 'constant', c=0.5)
    }


@pytest.fixture
def simple_factor_data():
    """Generate simple factor model data for testing."""
    return DataSampler.quick_generate(p=20, k=3, n_samples=100, seed=42)


# ============================================================================
# DistributionFactory Tests
# ============================================================================

class TestDistributionFactory:
    """Tests for DistributionFactory functionality."""
    
    def test_list_distributions(self):
        """Test listing available distributions."""
        dists = DistributionFactory.list_distributions()
        
        assert isinstance(dists, list)
        assert len(dists) > 0
        assert 'normal' in dists
        assert 'uniform' in dists
        assert all(isinstance(d, str) for d in dists)
    
    def test_get_signature(self):
        """Test getting distribution signatures."""
        sig = DistributionFactory.get_signature('normal')
        
        assert isinstance(sig, str)
        assert 'normal' in sig
        assert 'mean' in sig
        assert 'std' in sig
    
    def test_get_signature_invalid_distribution(self):
        """Test signature request for unknown distribution."""
        with pytest.raises(ValueError, match="Unknown distribution"):
            DistributionFactory.get_signature('nonexistent')
    
    def test_create_normal_sampler(self, sample_sizes):
        """Test creating normal distribution sampler."""
        n = sample_sizes['medium']
        sampler = DistributionFactory.create(n, 'normal', mean=5.0, std=2.0)
        
        assert callable(sampler)
        samples = sampler()
        assert samples.shape == (n,)
        assert np.abs(samples.mean() - 5.0) < 0.1  # Should be close to mean
        assert np.abs(samples.std() - 2.0) < 0.1   # Should be close to std
    
    def test_create_uniform_sampler(self, sample_sizes):
        """Test creating uniform distribution sampler."""
        n = sample_sizes['medium']
        sampler = DistributionFactory.create(n, 'uniform', low=2.0, high=8.0)
        
        samples = sampler()
        assert samples.shape == (n,)
        assert np.all(samples >= 2.0)
        assert np.all(samples <= 8.0)
        assert np.abs(samples.mean() - 5.0) < 0.1  # Mean should be ~5
    
    def test_create_constant_sampler(self, sample_sizes):
        """Test creating constant sampler."""
        n = sample_sizes['small']
        sampler = DistributionFactory.create(n, 'constant', c=3.14)
        
        samples = sampler()
        assert samples.shape == (n,)
        assert np.allclose(samples, 3.14)
    
    def test_create_beta_sampler(self, sample_sizes):
        """Test creating beta distribution sampler."""
        n = sample_sizes['medium']
        sampler = DistributionFactory.create(n, 'beta', a=2, b=5, low=0, high=10)
        
        samples = sampler()
        assert samples.shape == (n,)
        assert np.all(samples >= 0)
        assert np.all(samples <= 10)
    
    def test_create_lognormal_sampler(self, sample_sizes):
        """Test creating lognormal sampler."""
        n = sample_sizes['medium']
        sampler = DistributionFactory.create(n, 'lognormal', mean=0, std=1)
        
        samples = sampler()
        assert samples.shape == (n,)
        assert np.all(samples > 0)  # Lognormal is always positive
    
    def test_create_exponential_sampler(self, sample_sizes):
        """Test creating exponential sampler."""
        n = sample_sizes['medium']
        sampler = DistributionFactory.create(n, 'exponential', scale=2.0)
        
        samples = sampler()
        assert samples.shape == (n,)
        assert np.all(samples >= 0)
        assert np.abs(samples.mean() - 2.0) < 0.2
    
    def test_case_insensitive_distribution_names(self):
        """Test that distribution names are case-insensitive."""
        sampler1 = DistributionFactory.create(100, 'NORMAL', mean=0, std=1)
        sampler2 = DistributionFactory.create(100, 'Normal', mean=0, std=1)
        sampler3 = DistributionFactory.create(100, 'normal', mean=0, std=1)
        
        assert callable(sampler1)
        assert callable(sampler2)
        assert callable(sampler3)
    
    def test_missing_required_parameter(self):
        """Test error when required parameter is missing."""
        with pytest.raises(ValueError, match="Missing required parameter"):
            DistributionFactory.create(100, 'normal', mean=0)  # Missing 'std'
    
    def test_unknown_distribution(self):
        """Test error for unknown distribution."""
        with pytest.raises(ValueError, match="Unknown distribution"):
            DistributionFactory.create(100, 'gaussian', mean=0, std=1)
    
    def test_unknown_parameter_warning(self):
        """Test warning for unknown parameters."""
        with pytest.warns(UserWarning, match="Unknown parameters"):
            sampler = DistributionFactory.create(
                100, 'normal', 
                mean=0, std=1, 
                unknown_param=42  # This should trigger warning
            )
            assert callable(sampler)
    
    def test_register_custom_distribution(self, reset_factory):
        """Test registering a custom distribution."""
        def custom_dist(n, a, b):
            return np.random.randn(n) * b + a
        
        DistributionFactory.register('custom', custom_dist)
        
        assert 'custom' in DistributionFactory.list_distributions()
        
        sampler = DistributionFactory.create(100, 'custom', a=5, b=2)
        samples = sampler()
        assert samples.shape == (100,)
    
    def test_register_without_n_parameter(self, reset_factory):
        """Test that registration fails without 'n' parameter."""
        def bad_dist(mean, std):  # Missing 'n'
            return np.random.randn(100)
        
        with pytest.raises(ValueError, match="must have 'n' as first parameter"):
            DistributionFactory.register('bad', bad_dist)
    
    def test_register_overwrite_protection(self, reset_factory):
        """Test that overwriting is protected by default."""
        def new_normal(n, mean, std):
            return np.random.randn(n)
        
        with pytest.raises(ValueError, match="already exists"):
            DistributionFactory.register('normal', new_normal, overwrite=False)
    
    def test_register_overwrite_allowed(self, reset_factory):
        """Test overwriting with explicit permission."""
        def new_normal(n, mean, std):
            return np.ones(n) * mean
        
        DistributionFactory.register('normal', new_normal, overwrite=True)
        
        sampler = DistributionFactory.create(100, 'normal', mean=5, std=1)
        samples = sampler()
        assert np.allclose(samples, 5.0)  # Uses new implementation
    
    def test_create_transformed(self, sample_sizes):
        """Test creating transformed sampler."""
        n = sample_sizes['medium']
        
        # Create sampler that takes absolute value
        sampler = DistributionFactory.create_transformed(
            n, 'normal',
            transform=np.abs,
            mean=0, std=1
        )
        
        samples = sampler()
        assert samples.shape == (n,)
        assert np.all(samples >= 0)  # All positive due to abs()
    
    def test_sampler_reproducibility(self):
        """Test that samplers are deterministic with same seed."""
        np.random.seed(42)
        sampler = DistributionFactory.create(100, 'normal', mean=0, std=1)
        samples1 = sampler()
        
        np.random.seed(42)
        sampler = DistributionFactory.create(100, 'normal', mean=0, std=1)
        samples2 = sampler()
        
        assert np.allclose(samples1, samples2)


# ============================================================================
# DataSampler Tests
# ============================================================================

class TestDataSampler:
    """Tests for DataSampler class."""
    
    def test_initialization(self):
        """Test DataSampler initialization."""
        sampler = DataSampler(p=50, k=5, n=1000, seed=42)
        
        assert sampler.p == 50
        assert sampler.k == 5
        assert sampler.n == 1000
    
    def test_add_factor_samplers_correct_count(self, basic_samplers):
        """Test adding correct number of factor samplers."""
        sampler = DataSampler(p=20, k=3, n=100)
        
        beta_samplers = [basic_samplers['normal'] for _ in range(3)]
        factor_vol = basic_samplers['uniform']
        
        sampler.add_factor_samplers(beta_samplers, factor_vol)
        
        assert len(sampler._beta_samplers) == 3
        assert sampler._factor_vol_sampler is not None
    
    def test_add_factor_samplers_wrong_count(self, basic_samplers):
        """Test error with wrong number of beta samplers."""
        sampler = DataSampler(p=20, k=3, n=100)
        
        beta_samplers = [basic_samplers['normal'] for _ in range(5)]  # Wrong count
        factor_vol = basic_samplers['uniform']
        
        with pytest.raises(ValueError, match="Need 3 beta samplers"):
            sampler.add_factor_samplers(beta_samplers, factor_vol)
    
    def test_add_idiosyncratic_sampler(self, basic_samplers):
        """Test adding idiosyncratic sampler."""
        sampler = DataSampler(p=20, k=3, n=100)
        sampler.add_idiosyncratic_sampler(basic_samplers['uniform'])
        
        assert sampler._idio_vol_sampler is not None
    
    def test_configure(self):
        """Test configuration options."""
        sampler = DataSampler(p=20, k=3, n=100)
        sampler.configure(normalize_beta=True, ensure_positive_vols=False)
        
        assert sampler._normalize_beta is True
        assert sampler._ensure_positive_vols is False
    
    def test_generate_without_configuration(self):
        """Test that generation fails without configuration."""
        sampler = DataSampler(p=20, k=3, n=100)
        
        with pytest.raises(RuntimeError, match="Factor samplers not configured"):
            sampler.generate()
    
    def test_generate_without_idiosyncratic(self, basic_samplers):
        """Test that generation fails without idiosyncratic sampler."""
        sampler = DataSampler(p=20, k=3, n=100)
        
        beta_samplers = [basic_samplers['normal'] for _ in range(3)]
        sampler.add_factor_samplers(beta_samplers, basic_samplers['uniform'])
        
        with pytest.raises(RuntimeError, match="Idiosyncratic sampler not configured"):
            sampler.generate()
    
    def test_generate_basic(self, basic_samplers):
        """Test basic data generation."""
        sampler = DataSampler(p=20, k=3, n=100, seed=42)
        
        beta_samplers = [basic_samplers['normal'] for _ in range(3)]
        sampler.add_factor_samplers(beta_samplers, basic_samplers['uniform'])
        sampler.add_idiosyncratic_sampler(basic_samplers['uniform'])
        
        data = sampler.generate()
        
        assert isinstance(data, FactorModelData)
        assert data.B.shape == (3, 20)
        assert data.F.shape == (3, 3)
        assert data.D.shape == (20, 20)
    
    def test_generate_dimensions(self):
        """Test generated data has correct dimensions."""
        for p in [10, 50, 100]:
            for k in [2, 5, 10]:
                data = DataSampler.quick_generate(p=p, k=k, seed=42)
                
                assert data.B.shape == (k, p)
                assert data.F.shape == (k, k)
                assert data.D.shape == (p, p)
    
    def test_generate_with_normalization(self, basic_samplers):
        """Test generation with beta normalization."""
        sampler = DataSampler(p=50, k=5, n=1000, seed=42)
        
        beta_samplers = [basic_samplers['normal'] for _ in range(5)]
        sampler.add_factor_samplers(beta_samplers, basic_samplers['uniform'])
        sampler.add_idiosyncratic_sampler(basic_samplers['uniform'])
        sampler.configure(normalize_beta=True)
        
        data = sampler.generate()
        
        # Check normalization (approximately mean 0, std 1)
        for factor_idx in range(5):
            beta = data.B[factor_idx, :]
            assert np.abs(beta.mean()) < 0.2
            assert np.abs(beta.std() - 1.0) < 0.2
    
    def test_generate_positive_volatilities(self, basic_samplers):
        """Test that volatilities are positive."""
        sampler = DataSampler(p=20, k=3, n=100, seed=42)
        
        beta_samplers = [basic_samplers['normal'] for _ in range(3)]
        sampler.add_factor_samplers(beta_samplers, basic_samplers['normal'])  # Can be negative
        sampler.add_idiosyncratic_sampler(basic_samplers['normal'])
        sampler.configure(ensure_positive_vols=True)
        
        data = sampler.generate()
        
        # All variances should be positive
        assert np.all(np.diag(data.F) >= 0)
        assert np.all(np.diag(data.D) > 0)
    
    def test_quick_generate(self):
        """Test quick_generate convenience method."""
        data = DataSampler.quick_generate(
            p=30, k=5,
            beta_params={'dist': 'normal', 'mean': 0.5, 'std': 0.2},
            factor_vol_params={'dist': 'uniform', 'low': 0.1, 'high': 0.3},
            idio_vol_params={'dist': 'uniform', 'low': 0.05, 'high': 0.15},
            n_samples=1000,
            seed=42
        )
        
        assert isinstance(data, FactorModelData)
        assert data.B.shape == (5, 30)
        assert data.F.shape == (5, 5)
        assert data.D.shape == (30, 30)
    
    def test_quick_generate_defaults(self):
        """Test quick_generate with default parameters."""
        data = DataSampler.quick_generate(p=20, k=3, seed=42)
        
        assert data.B.shape == (3, 20)
        assert data.metadata is not None
    
    def test_reproducibility_with_seed(self):
        """Test that same seed produces same data."""
        data1 = DataSampler.quick_generate(p=20, k=3, seed=42)
        data2 = DataSampler.quick_generate(p=20, k=3, seed=42)
        
        assert np.allclose(data1.B, data2.B)
        assert np.allclose(data1.F, data2.F)
        assert np.allclose(data1.D, data2.D)
    
    def test_different_seeds_produce_different_data(self):
        """Test that different seeds produce different data."""
        data1 = DataSampler.quick_generate(p=20, k=3, seed=42)
        data2 = DataSampler.quick_generate(p=20, k=3, seed=123)
        
        assert not np.allclose(data1.B, data2.B)


# ============================================================================
# FactorModelData Tests
# ============================================================================

class TestFactorModelData:
    """Tests for FactorModelData dataclass."""
    
    def test_properties(self, simple_factor_data):
        """Test FactorModelData properties."""
        assert simple_factor_data.k == 3
        assert simple_factor_data.p == 20
        assert simple_factor_data.B.shape == (3, 20)
    
    def test_validate_correct_data(self, simple_factor_data):
        """Test validation passes for correct data."""
        assert simple_factor_data.validate() is True
    
    def test_validate_F_incompatible_with_B(self, simple_factor_data):
        """Test validation fails when F shape doesn't match B."""
        data = simple_factor_data
        # Change B so k=5, but F is still (3,3)
        data.B = np.zeros((5, 20))
        
        with pytest.raises(ValueError, match="F has wrong shape"):
            data.validate()
    
    def test_validate_D_incompatible_with_B(self, simple_factor_data):
        """Test validation fails when D shape doesn't match B."""
        data = simple_factor_data
        # Change B so p=30, but D is still (20,20)
        data.B = np.zeros((3, 30))
        
        with pytest.raises(ValueError, match="D has wrong shape"):
            data.validate()
    
    def test_validate_wrong_F_shape(self, simple_factor_data):
        """Test validation fails when F has wrong dimensions."""
        data = simple_factor_data
        # B expects k=3, change F to wrong size
        data.F = np.eye(5)
        
        with pytest.raises(ValueError, match="F has wrong shape"):
            data.validate()
    
    def test_validate_wrong_D_shape(self, simple_factor_data):
        """Test validation fails when D has wrong dimensions."""
        data = simple_factor_data
        # B expects p=20, change D to wrong size
        data.D = np.eye(30)
        
        with pytest.raises(ValueError, match="D has wrong shape"):
            data.validate()
    
    def test_validate_non_diagonal_F(self, simple_factor_data):
        """Test validation fails for non-diagonal F."""
        data = simple_factor_data
        data.F = np.ones((3, 3))  # Not diagonal
        
        with pytest.raises(ValueError, match="F must be diagonal"):
            data.validate()
    
    def test_validate_non_diagonal_D(self, simple_factor_data):
        """Test validation fails for non-diagonal D."""
        data = simple_factor_data
        data.D = np.ones((20, 20))  # Not diagonal
        
        with pytest.raises(ValueError, match="D must be diagonal"):
            data.validate()
    
    def test_validate_non_positive_D(self, simple_factor_data):
        """Test validation fails for non-positive D."""
        data = simple_factor_data
        data.D = np.diag([-0.1] * 20)  # Negative values
        
        with pytest.raises(ValueError, match="D must be positive definite"):
            data.validate()
    
    def test_validate_negative_F(self, simple_factor_data):
        """Test validation fails for negative F."""
        data = simple_factor_data
        data.F = np.diag([-0.1] * 3)  # Negative values
        
        with pytest.raises(ValueError, match="F must be positive semidefinite"):
            data.validate()
    
    def test_metadata_populated(self):
        """Test that metadata is populated during generation."""
        data = DataSampler.quick_generate(p=20, k=3, n_samples=500, seed=42)
        
        assert data.metadata is not None
        assert 'n_samples' in data.metadata
        assert data.metadata['n_samples'] == 500

# ============================================================================
# Edge Cases and Integration Tests
# ============================================================================

class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""
    
    def test_single_security(self):
        """Test with single security (p=1)."""
        data = DataSampler.quick_generate(p=1, k=2, seed=42)
        
        assert data.B.shape == (2, 1)
        assert data.D.shape == (1, 1)
        assert data.validate()
    
    def test_single_factor(self):
        """Test with single factor (k=1)."""
        data = DataSampler.quick_generate(p=20, k=1, seed=42)
        
        assert data.B.shape == (1, 20)
        assert data.F.shape == (1, 1)
        assert data.validate()
    
    def test_large_problem(self):
        """Test with large dimensions."""
        data = DataSampler.quick_generate(p=500, k=50, n_samples=100, seed=42)
        
        assert data.B.shape == (50, 500)
        assert data.validate()
    
    def test_many_factors(self):
        """Test with many factors relative to securities."""
        data = DataSampler.quick_generate(p=20, k=15, seed=42)
        
        assert data.B.shape == (15, 20)
        assert data.validate()
    
    def test_zero_variance_constant_distribution(self):
        """Test with constant (zero variance) distributions."""
        sampler = DataSampler(p=10, k=3, n=100, seed=42)
        
        beta_samplers = [
            DistributionFactory.create(100, 'constant', c=i)
            for i in range(3)
        ]
        factor_vol = DistributionFactory.create(100, 'constant', c=0.2)
        idio_vol = DistributionFactory.create(100, 'constant', c=0.1)
        
        sampler.add_factor_samplers(beta_samplers, factor_vol)
        sampler.add_idiosyncratic_sampler(idio_vol)
        
        data = sampler.generate()
        
        # Should still be valid
        assert data.validate()
        
        # All factor volatilities should be the same
        assert np.allclose(np.diag(data.F), 0.2**2)
        assert np.allclose(np.diag(data.D), 0.1**2)
    
    def test_different_distributions_per_factor(self):
        """Test using different distributions for each factor."""
        sampler = DataSampler(p=30, k=3, n=1000, seed=42)
        
        beta_samplers = [
            DistributionFactory.create(1000, 'normal', mean=0, std=1),
            DistributionFactory.create(1000, 'uniform', low=-1, high=1),
            DistributionFactory.create(1000, 'beta', a=2, b=5, low=0, high=1)
        ]
        factor_vol = DistributionFactory.create(1000, 'lognormal', mean=-2, std=0.5)
        idio_vol = DistributionFactory.create(1000, 'exponential', scale=0.1)
        
        sampler.add_factor_samplers(beta_samplers, factor_vol)
        sampler.add_idiosyncratic_sampler(idio_vol)
        
        data = sampler.generate()
        
        assert data.validate()
        assert data.B.shape == (3, 30)


# ============================================================================
# Performance Tests (Optional, marked as slow)
# ============================================================================

class TestPerformance:
    """Performance and stress tests."""
    
    @pytest.mark.slow
    def test_large_sample_count(self):
        """Test with large number of samples."""
        data = DataSampler.quick_generate(
            p=100, k=10, 
            n_samples=100000,  # Large sample count
            seed=42
        )
        
        assert data.validate()
    
    @pytest.mark.slow
    def test_many_quick_generations(self):
        """Test multiple quick generations."""
        for i in range(100):
            data = DataSampler.quick_generate(p=50, k=5, seed=i)
            assert data.validate()


# ============================================================================
# Pytest Configuration
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])