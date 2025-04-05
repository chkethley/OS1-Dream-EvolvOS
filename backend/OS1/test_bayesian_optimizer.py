"""
Test module for Bayesian optimization implementation.
"""

import numpy as np
import pytest
from .bayesian_optimizer import BayesianOptimizer, MultiObjectiveBayesianOptimizer

def test_function(x: dict) -> float:
    """Simple test function for optimization."""
    # Rosenbrock function
    x1 = x["x1"]
    x2 = x["x2"]
    return -(100 * (x2 - x1**2)**2 + (1 - x1)**2)

def multi_objective_test_function(x: dict) -> list:
    """Multi-objective test function."""
    x1 = x["x1"]
    x2 = x["x2"]
    
    # Two objectives:
    # 1. Minimize Rosenbrock function
    f1 = -(100 * (x2 - x1**2)**2 + (1 - x1)**2)
    
    # 2. Minimize sphere function
    f2 = -(x1**2 + x2**2)
    
    return [f1, f2]

def test_bayesian_optimizer_initialization():
    """Test initialization of BayesianOptimizer."""
    param_ranges = {
        "x1": (-2.0, 2.0),
        "x2": (-2.0, 2.0)
    }
    
    optimizer = BayesianOptimizer(
        parameter_ranges=param_ranges,
        n_init_points=5
    )
    
    assert optimizer.parameter_ranges == param_ranges
    assert optimizer.n_init_points == 5
    assert optimizer.acquisition_func == "ucb"
    assert len(optimizer.X) == 0
    assert len(optimizer.y) == 0

def test_bayesian_optimizer_suggest_params():
    """Test parameter suggestion."""
    param_ranges = {
        "x1": (-2.0, 2.0),
        "x2": (-2.0, 2.0)
    }
    
    optimizer = BayesianOptimizer(param_ranges)
    
    # Test initial random suggestions
    for _ in range(optimizer.n_init_points):
        params = optimizer.suggest_params()
        assert "x1" in params
        assert "x2" in params
        assert -2.0 <= params["x1"] <= 2.0
        assert -2.0 <= params["x2"] <= 2.0
        
        # Update with test function result
        value = test_function(params)
        optimizer.update(params, value)
        
    # Test suggestion after initialization
    params = optimizer.suggest_params()
    assert "x1" in params
    assert "x2" in params
    assert -2.0 <= params["x1"] <= 2.0
    assert -2.0 <= params["x2"] <= 2.0

def test_bayesian_optimizer_optimization():
    """Test full optimization cycle."""
    param_ranges = {
        "x1": (-2.0, 2.0),
        "x2": (-2.0, 2.0)
    }
    
    optimizer = BayesianOptimizer(param_ranges)
    
    n_iterations = 20
    best_value = float('-inf')
    
    for _ in range(n_iterations):
        params = optimizer.suggest_params()
        value = test_function(params)
        optimizer.update(params, value)
        
        if value > best_value:
            best_value = value
            
    # Get best parameters
    best_params, best_val = optimizer.get_best_params()
    
    # Verify improvement
    assert best_val > -10.0  # Reasonable threshold for Rosenbrock function

def test_multi_objective_optimizer():
    """Test multi-objective optimization."""
    param_ranges = {
        "x1": (-2.0, 2.0),
        "x2": (-2.0, 2.0)
    }
    
    optimizer = MultiObjectiveBayesianOptimizer(
        parameter_ranges=param_ranges,
        n_objectives=2,
        weights=[0.5, 0.5]
    )
    
    n_iterations = 20
    
    for _ in range(n_iterations):
        params = optimizer.suggest_params_multi()
        values = multi_objective_test_function(params)
        optimizer.update_multi(params, values)
        
    # Get Pareto front
    pareto_front = optimizer.get_pareto_front()
    
    # Verify Pareto front properties
    assert len(pareto_front) > 0
    for solution in pareto_front:
        assert "params" in solution
        assert "objectives" in solution
        assert len(solution["objectives"]) == 2

def test_optimizer_state_save_load():
    """Test state saving and loading."""
    param_ranges = {
        "x1": (-2.0, 2.0),
        "x2": (-2.0, 2.0)
    }
    
    # Create and train optimizer
    optimizer = BayesianOptimizer(param_ranges)
    for _ in range(5):
        params = optimizer.suggest_params()
        value = test_function(params)
        optimizer.update(params, value)
        
    # Save state
    state = optimizer.get_state()
    
    # Create new optimizer and load state
    new_optimizer = BayesianOptimizer(param_ranges)
    new_optimizer.load_state(state)
    
    # Verify state was loaded correctly
    assert np.array_equal(optimizer.X, new_optimizer.X)
    assert np.array_equal(optimizer.y, new_optimizer.y)
    assert optimizer.parameter_ranges == new_optimizer.parameter_ranges

def test_acquisition_functions():
    """Test different acquisition functions."""
    param_ranges = {
        "x1": (-2.0, 2.0),
        "x2": (-2.0, 2.0)
    }
    
    acquisition_funcs = ["ucb", "ei", "poi"]
    
    for acq_func in acquisition_funcs:
        optimizer = BayesianOptimizer(
            parameter_ranges=param_ranges,
            acquisition_func=acq_func
        )
        
        # Train optimizer
        for _ in range(5):
            params = optimizer.suggest_params()
            value = test_function(params)
            optimizer.update(params, value)
            
        # Verify suggestion works
        params = optimizer.suggest_params()
        assert "x1" in params
        assert "x2" in params

if __name__ == "__main__":
    pytest.main([__file__])