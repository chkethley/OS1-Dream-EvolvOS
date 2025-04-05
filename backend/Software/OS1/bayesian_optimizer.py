"""
Bayesian Optimizer

This module implements Bayesian optimization for hyperparameter tuning
in the EvolvOS component of the Self-Evolving AI system.
"""

import numpy as np
import time
import uuid
import json
from typing import Dict, List, Any, Optional, Tuple, Callable
import random
from scipy.stats import norm

class GaussianProcess:
    """
    Simple Gaussian Process implementation for Bayesian optimization.
    
    This class implements a Gaussian Process regression model used as a
    surrogate model in Bayesian optimization.
    """
    
    def __init__(self, length_scale=1.0, noise=1e-6):
        """
        Initialize the Gaussian Process.
        
        Args:
            length_scale: Length scale parameter for RBF kernel
            noise: Noise level for observations
        """
        self.length_scale = length_scale
        self.noise = noise
        self.X_train = None
        self.y_train = None
        self.K = None  # Kernel matrix
        self.K_inv = None  # Inverse of kernel matrix
        
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit the Gaussian Process to training data.
        
        Args:
            X: Training features (n_samples, n_features)
            y: Training targets (n_samples,)
        """
        self.X_train = X.copy()
        self.y_train = y.copy()
        
        # Compute kernel matrix
        self.K = self._kernel(X, X) + self.noise * np.eye(len(X))
        
        # Compute inverse of kernel matrix
        self.K_inv = np.linalg.inv(self.K)
        
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions with the Gaussian Process.
        
        Args:
            X: Test features (n_samples, n_features)
            
        Returns:
            Tuple of (mean, std) for predictions
        """
        if self.X_train is None or self.y_train is None:
            raise ValueError("Model not fitted. Call fit() first.")
            
        # Compute kernel between test and training points
        K_s = self._kernel(X, self.X_train)
        
        # Compute mean prediction
        mean = K_s @ self.K_inv @ self.y_train
        
        # Compute covariance
        K_ss = self._kernel(X, X)
        var = K_ss - K_s @ self.K_inv @ K_s.T
        
        # Extract diagonal for standard deviation
        std = np.sqrt(np.diag(var))
        
        return mean, std
        
    def _kernel(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """
        Compute RBF kernel matrix between two sets of points.
        
        Args:
            X1: First set of points (n1_samples, n_features)
            X2: Second set of points (n2_samples, n_features)
            
        Returns:
            Kernel matrix (n1_samples, n2_samples)
        """
        # Compute squared Euclidean distance
        X1_norm = np.sum(X1**2, axis=1).reshape(-1, 1)
        X2_norm = np.sum(X2**2, axis=1).reshape(1, -1)
        dist_sq = X1_norm + X2_norm - 2 * X1 @ X2.T
        
        # Compute RBF kernel
        return np.exp(-0.5 * dist_sq / self.length_scale**2)

class AcquisitionFunction:
    """
    Acquisition functions for Bayesian optimization.
    
    This class implements various acquisition functions used to guide
    the exploration-exploitation trade-off in Bayesian optimization.
    """
    
    @staticmethod
    def expected_improvement(mean: np.ndarray, std: np.ndarray, y_best: float, xi: float = 0.01) -> np.ndarray:
        """
        Expected Improvement acquisition function.
        
        Args:
            mean: Predicted mean (n_samples,)
            std: Predicted standard deviation (n_samples,)
            y_best: Best observed value
            xi: Exploration-exploitation trade-off parameter
            
        Returns:
            Expected improvement values (n_samples,)
        """
        # Ensure std is positive
        std = np.maximum(std, 1e-9)
        
        # Compute improvement
        imp = mean - y_best - xi
        
        # Compute Z-score
        z = imp / std
        
        # Compute expected improvement
        ei = imp * norm.cdf(z) + std * norm.pdf(z)
        
        # Return 0 for negative improvement
        return np.maximum(ei, 0)
        
    @staticmethod
    def upper_confidence_bound(mean: np.ndarray, std: np.ndarray, kappa: float = 2.0) -> np.ndarray:
        """
        Upper Confidence Bound acquisition function.
        
        Args:
            mean: Predicted mean (n_samples,)
            std: Predicted standard deviation (n_samples,)
            kappa: Exploration parameter
            
        Returns:
            UCB values (n_samples,)
        """
        return mean + kappa * std
        
    @staticmethod
    def probability_of_improvement(mean: np.ndarray, std: np.ndarray, y_best: float, xi: float = 0.01) -> np.ndarray:
        """
        Probability of Improvement acquisition function.
        
        Args:
            mean: Predicted mean (n_samples,)
            std: Predicted standard deviation (n_samples,)
            y_best: Best observed value
            xi: Exploration-exploitation trade-off parameter
            
        Returns:
            Probability of improvement values (n_samples,)
        """
        # Ensure std is positive
        std = np.maximum(std, 1e-9)
        
        # Compute Z-score
        z = (mean - y_best - xi) / std
        
        # Compute probability of improvement
        return norm.cdf(z)

class HyperparameterSpace:
    """
    Defines the search space for hyperparameters.
    
    This class represents the hyperparameter space for Bayesian optimization,
    handling different types of parameters (continuous, integer, categorical).
    """
    
    def __init__(self):
        """Initialize an empty hyperparameter space."""
        self.params = {}
        self.param_types = {}
        
    def add_continuous_param(self, name: str, low: float, high: float):
        """
        Add a continuous hyperparameter.
        
        Args:
            name: Parameter name
            low: Lower bound
            high: Upper bound
        """
        self.params[name] = (low, high)
        self.param_types[name] = "continuous"
        
    def add_integer_param(self, name: str, low: int, high: int):
        """
        Add an integer hyperparameter.
        
        Args:
            name: Parameter name
            low: Lower bound (inclusive)
            high: Upper bound (inclusive)
        """
        self.params[name] = (low, high)
        self.param_types[name] = "integer"
        
    def add_categorical_param(self, name: str, categories: List):
        """
        Add a categorical hyperparameter.
        
        Args:
            name: Parameter name
            categories: List of possible values
        """
        self.params[name] = categories
        self.param_types[name] = "categorical"
        
    def sample_random(self, n_samples: int = 1) -> List[Dict]:
        """
        Generate random samples from the hyperparameter space.
        
        Args:
            n_samples: Number of samples to generate
            
        Returns:
            List of parameter dictionaries
        """
        samples = []
        
        for _ in range(n_samples):
            sample = {}
            
            for name, param_range in self.params.items():
                param_type = self.param_types[name]
                
                if param_type == "continuous":
                    low, high = param_range
                    sample[name] = np.random.uniform(low, high)
                elif param_type == "integer":
                    low, high = param_range
                    sample[name] = np.random.randint(low, high + 1)
                elif param_type == "categorical":
                    categories = param_range
                    sample[name] = np.random.choice(categories)
                    
            samples.append(sample)
            
        return samples
        
    def to_array(self, params: Dict) -> np.ndarray:
        """
        Convert a parameter dictionary to a numpy array.
        
        Args:
            params: Parameter dictionary
            
        Returns:
            Numpy array representation
        """
        array = []
        
        for name in sorted(self.params.keys()):
            param_type = self.param_types[name]
            value = params[name]
            
            if param_type == "continuous":
                low, high = self.params[name]
                # Normalize to [0, 1]
                normalized = (value - low) / (high - low)
                array.append(normalized)
            elif param_type == "integer":
                low, high = self.params[name]
                # Normalize to [0, 1]
                normalized = (value - low) / (high - low)
                array.append(normalized)
            elif param_type == "categorical":
                categories = self.params[name]
                # One-hot encoding
                for category in categories:
                    array.append(1.0 if value == category else 0.0)
                    
        return np.array(array)
        
    def from_array(self, array: np.ndarray) -> Dict:
        """
        Convert a numpy array back to a parameter dictionary.
        
        Args:
            array: Numpy array representation
            
        Returns:
            Parameter dictionary
        """
        params = {}
        idx = 0
        
        for name in sorted(self.params.keys()):
            param_type = self.param_types[name]
            
            if param_type == "continuous":
                low, high = self.params[name]
                # Denormalize from [0, 1]
                value = low + array[idx] * (high - low)
                params[name] = value
                idx += 1
            elif param_type == "integer":
                low, high = self.params[name]
                # Denormalize from [0, 1] and round to integer
                value = int(round(low + array[idx] * (high - low)))
                # Ensure value is within bounds
                value = max(low, min(high, value))
                params[name] = value
                idx += 1
            elif param_type == "categorical":
                categories = self.params[name]
                # Find category with highest probability
                category_probs = array[idx:idx+len(categories)]
                category_idx = np.argmax(category_probs)
                params[name] = categories[category_idx]
                idx += len(categories)
                
        return params
        
    def get_dimensions(self) -> int:
        """
        Get the dimensionality of the hyperparameter space.
        
        Returns:
            Number of dimensions in the array representation
        """
        dims = 0
        
        for name, param_type in self.param_types.items():
            if param_type in ["continuous", "integer"]:
                dims += 1
            elif param_type == "categorical":
                dims += len(self.params[name])
                
        return dims

class BayesianOptimizer:
    """
    Bayesian optimization for hyperparameter tuning.
    
    This class implements Bayesian optimization for efficient hyperparameter tuning,
    using Gaussian Processes as surrogate models and acquisition functions to guide
    the search process.
    """
    
    def __init__(self, param_space: HyperparameterSpace, 
                objective_fn: Callable[[Dict], float],
                acquisition_type: str = "ei",
                maximize: bool = True,
                exploration_weight: float = 0.1):
        """
        Initialize the Bayesian optimizer.
        
        Args:
            param_space: Hyperparameter search space
            objective_fn: Function to optimize (takes params dict, returns float)
            acquisition_type: Acquisition function type ('ei', 'ucb', or 'pi')
            maximize: Whether to maximize (True) or minimize (False) the objective
            exploration_weight: Weight for exploration vs exploitation
        """
        self.param_space = param_space
        self.objective_fn = objective_fn
        self.acquisition_type = acquisition_type
        self.maximize = maximize
        self.exploration_weight = exploration_weight
        
        # Initialize Gaussian Process
        self.gp = GaussianProcess()
        
        # Initialize history
        self.X_observed = []  # List of parameter dicts
        self.y_observed = []  # List of objective values
        self.X_array = []     # Array representation of parameters
        
        # Set acquisition function
        if acquisition_type == "ei":
            self.acquisition_fn = AcquisitionFunction.expected_improvement
        elif acquisition_type == "ucb":
            self.acquisition_fn = AcquisitionFunction.upper_confidence_bound
        elif acquisition_type == "pi":
            self.acquisition_fn = AcquisitionFunction.probability_of_improvement
        else:
            raise ValueError(f"Unknown acquisition function: {acquisition_type}")
            
        # For tracking optimization
        self.id = str(uuid.uuid4())
        self.start_time = time.time()
        self.best_value = None
        self.best_params = None
        
    def suggest(self, n_samples: int = 10000) -> Dict:
        """
        Suggest next set of parameters to evaluate.
        
        Args:
            n_samples: Number of random samples to evaluate with acquisition function
            
        Returns:
            Dictionary of suggested parameters
        """
        # If no observations yet, return random parameters
        if not self.X_observed:
            return self.param_space.sample_random(1)[0]
            
        # Fit GP to observed data
        X_array = np.array(self.X_array)
        y_array = np.array(self.y_observed)
        
        # If maximizing, negate the objective for the GP
        if self.maximize:
            y_array = -y_array
            
        self.gp.fit(X_array, y_array)
        
        # Generate random candidate points
        candidates = self.param_space.sample_random(n_samples)
        candidates_array = np.array([self.param_space.to_array(c) for c in candidates])
        
        # Predict with GP
        mean, std = self.gp.predict(candidates_array)
        
        # Find best observed value
        if self.maximize:
            y_best = -min(y_array)  # Convert back to original
        else:
            y_best = min(y_array)
            
        # Compute acquisition function values
        if self.acquisition_type == "ucb":
            acq_values = self.acquisition_fn(
                mean if not self.maximize else -mean,
                std,
                kappa=self.exploration_weight
            )
        else:  # ei or pi
            acq_values = self.acquisition_fn(
                mean if not self.maximize else -mean,
                std,
                y_best,
                xi=self.exploration_weight
            )
            
        # Find candidate with highest acquisition value
        best_idx = np.argmax(acq_values)
        best_candidate = candidates[best_idx]
        
        return best_candidate
        
    def observe(self, params: Dict, value: float):
        """
        Record an observation of the objective function.
        
        Args:
            params: Parameter dictionary
            value: Observed objective value
        """
        # Convert parameters to array
        params_array = self.param_space.to_array(params)
        
        # Add to history
        self.X_observed.append(params.copy())
        self.y_observed.append(value)
        self.X_array.append(params_array)
        
        # Update best value
        if self.best_value is None or (self.maximize and value > self.best_value) or (not self.maximize and value < self.best_value):
            self.best_value = value
            self.best_params = params.copy()
            
    def optimize(self, n_iterations: int = 10, initial_points: int = 5) -> Tuple[Dict, float]:
        """
        Run the optimization process.
        
        Args:
            n_iterations: Number of optimization iterations
            initial_points: Number of random points to evaluate initially
            
        Returns:
            Tuple of (best_params, best_value)
        """
        print(f"Starting Bayesian optimization with {n_iterations} iterations")
        
        # Initial random exploration
        initial_params = self.param_space.sample_random(initial_points)
        
        for i, params in enumerate(initial_params):
            print(f"Evaluating initial point {i+1}/{initial_points}")
            value = self.objective_fn(params)
            self.observe(params, value)
            
            print(f"  Parameters: {params}")
            print(f"  Value: {value}")
            
        # Main optimization loop
        for i in range(n_iterations):
            print(f"Iteration {i+1}/{n_iterations}")
            
            # Get suggestion
            suggested_params = self.suggest()
            
            # Evaluate objective
            value = self.objective_fn(suggested_params)
            
            # Record observation
            self.observe(suggested_params, value)
            
            print(f"  Parameters: {suggested_params}")
            print(f"  Value: {value}")
            print(f"  Best value so far: {self.best_value}")
            
        print(f"Optimization completed. Best value: {self.best_value}")
        print(f"Best parameters: {self.best_params}")
        
        return self.best_params, self.best_value
        
    def get_results(self) -> Dict:
        """
        Get optimization results.
        
        Returns:
            Dictionary with optimization results
        """
        return {
            "optimizer_id": self.id,
            "best_value": self.best_value,
            "best_params": self.best_params,
            "n_observations": len(self.y_observed),
            "observation_history": [
                {"params": params, "value": value}
                for params, value in zip(self.X_observed, self.y_observed)
            ],
            "run_time_seconds": time.time() - self.start_time
        }

class EvolvOSOptimizer:
    """
    Specialized Bayesian optimizer for EvolvOS components.
    
    This class provides a convenient interface for optimizing specific
    components of the EvolvOS system using Bayesian optimization.
    """
    
    def __init__(self, component_name: str, eval_function: Callable):
        """
        Initialize the EvolvOS optimizer.
        
        Args:
            component_name: Name of the component to optimize
            eval_function: Function to evaluate component performance
        """
        self.component_name = component_name
        self.eval_function = eval_function
        self.param_space = HyperparameterSpace()
        self.configured = False
        
        # Set default parameters based on component
        self._configure_default_params()
        
    def _configure_default_params(self):
        """Configure default parameters based on component type."""
        if self.component_name == "memory":
            # Memory system parameters
            self.param_space.add_integer_param("volatile_size", 100, 2000)
            self.param_space.add_integer_param("compression_threshold", 10, 100)
            self.param_space.add_continuous_param("retention_factor", 0.1, 0.9)
            self.param_space.add_integer_param("max_batch_size", 5, 50)
            
        elif self.component_name == "dream_system":
            # Dream system parameters
            self.param_space.add_integer_param("agent_count", 3, 10)
            self.param_space.add_integer_param("max_debate_rounds", 2, 8)
            self.param_space.add_continuous_param("consensus_threshold", 0.5, 0.95)
            self.param_space.add_continuous_param("agent_temperature", 0.1, 1.0)
            
        elif self.component_name == "evolvos":
            # EvolvOS parameters
            self.param_space.add_integer_param("evolution_cycles", 1, 10)
            self.param_space.add_integer_param("population_size", 5, 50)
            self.param_space.add_continuous_param("mutation_rate", 0.1, 0.8)
            self.param_space.add_continuous_param("crossover_rate", 0.1, 0.8)
            
        elif self.component_name == "neural_architecture":
            # Neural architecture parameters
            self.param_space.add_integer_param("hidden_layers", 1, 5)
            self.param_space.add_integer_param("neurons_per_layer", 32, 512)
            self.param_space.add_categorical_param("activation", ["relu", "tanh", "sigmoid"])
            self.param_space.add_continuous_param("dropout_rate", 0.0, 0.5)
            self.param_space.add_continuous_param("learning_rate", 0.0001, 0.1)
            
        else:
            # Generic parameters
            self.param_space.add_continuous_param("param1", 0.0, 1.0)
            self.param_space.add_continuous_param("param2", 0.0, 1.0)
            self.param_space.add_integer_param("param3", 1, 10)
            
        self.configured = True
        
    def add_custom_param(self, name: str, param_type: str, *args):
        """
        Add a custom parameter to the optimization space.
        
        Args:
            name: Parameter name
            param_type: Parameter type ('continuous', 'integer', or 'categorical')
            *args: Arguments depending on param_type
        """
        if param_type == "continuous":
            low, high = args
            self.param_space.add_continuous_param(name, low, high)
        elif param_type == "integer":
            low, high = args
            self.param_space.add_integer_param(name, low, high)
        elif param_type == "categorical":
            categories = args[0]
            self.param_space.add_categorical_param(name, categories)
        else:
            raise ValueError(f"Unknown parameter type: {param_type}")
            
    def optimize(self, n_iterations: int = 20, maximize: bool = True) -> Dict:
        """
        Run the optimization process.
        
        Args:
            n_iterations: Number of optimization iterations
            maximize: Whether to maximize or minimize the objective
            
        Returns:
            Optimization results
        """
        if not self.configured:
            raise ValueError("Optimizer not properly configured")
            
        # Create Bayesian optimizer
        optimizer = BayesianOptimizer(
            param_space=self.param_space,
            objective_fn=self.eval_function,
            acquisition_type="ei",
            maximize=maximize,
            exploration_weight=0.1
        )
        
        # Run optimization
        best_params, best_value = optimizer.optimize(
            n_iterations=n_iterations,
            initial_points=max(3, n_iterations // 5)
        )
        
        # Get full results
        results = optimizer.get_results()
        results["component_name"] = self.component_name
        
        return results

# Example usage
def example_usage():
    """Demonstrate usage of the Bayesian optimizer."""
    # Define a test objective function
    def objective_fn(params):
        """Simple test function with noise."""
        x = params["x"]
        y = params["y"]
        # Sphere function with some noise
        value = -(x**2 + y**2) + 0.1 * np.random.randn()
        return value
        
    # Create parameter space
    param_space = HyperparameterSpace()
    param_space.add_continuous_param("x", -5.0, 5.0)
    param_space.add_continuous_param("y", -5.0, 5.0)
    
    # Create optimizer
    optimizer = BayesianOptimizer(
        param_space=param_space,
        objective_fn=objective_fn,
        acquisition_type="ei",
        maximize=True
    )
    
    # Run optimization
    best_params, best_value = optimizer.optimize(n_iterations=10, initial_points=3)
    
    print("\nOptimization results:")
    print(f"Best parameters: {best_params}")
    print(f"Best value: {best_value}")
    
    # Example with EvolvOS optimizer
    def neural_arch_eval(params):
        """Dummy evaluator for neural architecture."""
        # Higher values for balanced architectures
        layers = params["hidden_layers"]
        neurons = params["neurons_per_layer"]
        dropout = params["dropout_rate"]
        
        # Prefer moderate layer count and neuron count
        layer_score = -0.2 * abs(layers - 3)
        neuron_score = -0.0001 * abs(neurons - 128)
        
        # Prefer moderate dropout
        dropout_score = -2.0 * abs(dropout - 0.3)
        
        # Add noise
        score = layer_score + neuron_score + dropout_score + 0.1 * np.random.randn()
        return score
        
    evolvos_opt = EvolvOSOptimizer("neural_architecture", neural_arch_eval)
    results = evolvos_opt.optimize(n_iterations=5)
    
    print("\nEvolvOS Optimizer results:")
    print(f"Best parameters: {results['best_params']}")
    print(f"Best value: {results['best_value']}")
    
    return optimizer, evolvos_opt

if __name__ == "__main__":
    example_usage() 