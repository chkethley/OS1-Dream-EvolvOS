"""
Enhanced Bayesian Optimizer with multi-objective optimization 
for self-evolving system components.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy.stats import norm
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
import logging
import time

logger = logging.getLogger("EvolvOS.BayesianOpt")

class BayesianOptimizer:
    def __init__(self, parameter_ranges: Dict[str, Tuple[float, float]], 
                 n_init_points: int = 5,
                 acquisition_func: str = "ucb",
                 beta: float = 2.0):
        self.parameter_ranges = parameter_ranges
        self.param_names = list(parameter_ranges.keys())
        self.n_init_points = n_init_points
        self.acquisition_func = acquisition_func
        self.beta = beta
        
        # Initialize storage for observations
        self.X = []  # Parameter configurations
        self.y = []  # Observed values
        
        # Initialize GP model with RBF kernel
        self.gp = None
        self._init_gp()
        
        logger.info(f"Initialized Bayesian Optimizer with {len(parameter_ranges)} parameters")
        
    def _init_gp(self):
        """Initialize Gaussian Process model."""
        kernel = ConstantKernel(1.0) * RBF(length_scale=[1.0] * len(self.parameter_ranges))
        self.gp = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=5,
            random_state=42
        )
        
    def _normalize_params(self, params: Dict[str, float]) -> np.ndarray:
        """Normalize parameters to [0, 1] range."""
        normalized = []
        for name in self.param_names:
            min_val, max_val = self.parameter_ranges[name]
            val = params[name]
            normalized.append((val - min_val) / (max_val - min_val))
        return np.array(normalized)
        
    def _denormalize_params(self, normalized: np.ndarray) -> Dict[str, float]:
        """Convert normalized parameters back to original range."""
        params = {}
        for i, name in enumerate(self.param_names):
            min_val, max_val = self.parameter_ranges[name]
            params[name] = normalized[i] * (max_val - min_val) + min_val
        return params
        
    def _acquisition_function(self, x: np.ndarray) -> float:
        """Compute acquisition function value."""
        x = x.reshape(1, -1)
        
        if len(self.X) == 0:
            return 0.0
            
        mean, std = self.gp.predict(x, return_std=True)
        
        if self.acquisition_func == "ucb":
            # Upper Confidence Bound
            return mean[0] + self.beta * std[0]
        elif self.acquisition_func == "ei":
            # Expected Improvement
            y_best = np.max(self.y)
            z = (mean - y_best) / std
            return float((mean - y_best) * norm.cdf(z) + std * norm.pdf(z))
        elif self.acquisition_func == "poi":
            # Probability of Improvement
            y_best = np.max(self.y)
            z = (mean - y_best) / std
            return float(norm.cdf(z))
        else:
            raise ValueError(f"Unknown acquisition function: {self.acquisition_func}")
            
    def _optimize_acquisition(self) -> np.ndarray:
        """Find the maximum of the acquisition function."""
        best_x = None
        best_acq = -np.inf
        
        # Try multiple random starts
        n_random = 10 * len(self.parameter_ranges)
        for _ in range(n_random):
            x0 = np.random.rand(len(self.parameter_ranges))
            
            # Optimize from this starting point
            res = minimize(
                lambda x: -self._acquisition_function(x),
                x0,
                bounds=[(0, 1)] * len(self.parameter_ranges),
                method="L-BFGS-B"
            )
            
            if -res.fun > best_acq:
                best_acq = -res.fun
                best_x = res.x
                
        return best_x
        
    def suggest_params(self) -> Dict[str, float]:
        """Suggest next set of parameters to try."""
        if len(self.X) < self.n_init_points:
            # Initial random exploration
            normalized_params = np.random.rand(len(self.parameter_ranges))
        else:
            # Use Bayesian optimization
            normalized_params = self._optimize_acquisition()
            
        return self._denormalize_params(normalized_params)
        
    def update(self, params: Dict[str, float], value: float):
        """Update the model with new observation."""
        normalized_params = self._normalize_params(params)
        
        # Add to observation history
        if len(self.X) == 0:
            self.X = normalized_params.reshape(1, -1)
            self.y = np.array([value]).reshape(-1, 1)
        else:
            self.X = np.vstack([self.X, normalized_params])
            self.y = np.vstack([self.y, value])
            
        # Update GP model
        self.gp.fit(self.X, self.y)
        
        logger.debug(f"Updated model with new observation. Total observations: {len(self.X)}")
        
    def get_best_params(self) -> Tuple[Dict[str, float], float]:
        """Get the best parameters found so far."""
        if len(self.X) == 0:
            return None, None
            
        best_idx = np.argmax(self.y)
        best_normalized = self.X[best_idx]
        best_value = self.y[best_idx][0]
        
        return self._denormalize_params(best_normalized), best_value
        
    def get_state(self) -> Dict:
        """Get optimizer state for saving."""
        return {
            "X": self.X,
            "y": self.y,
            "parameter_ranges": self.parameter_ranges,
            "acquisition_func": self.acquisition_func,
            "beta": self.beta,
            "gp_params": {
                "kernel": self.gp.kernel_.get_params(),
                "noise": self.gp.alpha
            }
        }
        
    def load_state(self, state: Dict):
        """Load optimizer state."""
        self.X = state["X"]
        self.y = state["y"]
        self.parameter_ranges = state["parameter_ranges"]
        self.acquisition_func = state["acquisition_func"]
        self.beta = state["beta"]
        
        if len(self.X) > 0:
            self._init_gp()
            self.gp.fit(self.X, self.y)
            self.gp.kernel_.set_params(**state["gp_params"]["kernel"])
            self.gp.alpha = state["gp_params"]["noise"]

class MultiObjectiveBayesianOptimizer(BayesianOptimizer):
    """Enhanced optimizer with multi-objective optimization support."""
    
    def __init__(self, parameter_ranges: Dict[str, Tuple[float, float]],
                 n_objectives: int = 2,
                 weights: Optional[List[float]] = None,
                 n_init_points: int = 5,
                 acquisition_func: str = "ucb",
                 beta: float = 2.0):
        # Validate inputs
        if n_objectives < 1:
            raise ValueError("Number of objectives must be positive")
        if weights and len(weights) != n_objectives:
            raise ValueError("Number of weights must match number of objectives")
        if weights and not all(0 <= w <= 1 for w in weights):
            raise ValueError("Weights must be between 0 and 1")
        if weights and abs(sum(weights) - 1.0) > 1e-6:
            raise ValueError("Weights must sum to 1")

        super().__init__(parameter_ranges, n_init_points, acquisition_func, beta)
        self.n_objectives = n_objectives
        self.weights = weights if weights is not None else [1.0/n_objectives] * n_objectives
        
        # Add error recovery state
        self.failed_evaluations = []
        self.recovery_threshold = 3
        self.last_successful_state = None
        
        # Initialize GP models
        self.gp_models = None
        self._init_multi_gp()

    def _validate_objectives(self, values: List[float]) -> bool:
        """Validate objective values."""
        if len(values) != self.n_objectives:
            return False
        return all(isinstance(v, (int, float)) for v in values)

    def _handle_evaluation_error(self, params: Dict[str, float], error: Exception) -> None:
        """Handle failed parameter evaluation."""
        self.failed_evaluations.append({
            "params": params,
            "error": str(error),
            "timestamp": time.time()
        })
        
        # Check if we need to revert to last good state
        recent_failures = [f for f in self.failed_evaluations 
                         if time.time() - f["timestamp"] < 3600]  # Last hour
        if len(recent_failures) >= self.recovery_threshold:
            if self.last_successful_state:
                logger.warning("Multiple evaluation failures detected, reverting to last good state")
                self.load_state(self.last_successful_state)
            else:
                logger.warning("Multiple evaluation failures but no recovery state available")

    def _init_multi_gp(self):
        """Initialize multiple GP models for each objective."""
        self.gp_models = []
        for i in range(self.n_objectives):
            kernel = ConstantKernel(1.0) * RBF(length_scale=[1.0] * len(self.parameter_ranges))
            self.gp_models.append(
                GaussianProcessRegressor(
                    kernel=kernel,
                    n_restarts_optimizer=5,
                    random_state=42+i
                )
            )
            
    def _acquisition_function_multi(self, x: np.ndarray) -> float:
        """Compute weighted acquisition function for multiple objectives."""
        x = x.reshape(1, -1)
        acquisition_values = []
        
        for i, gp in enumerate(self.gp_models):
            if len(self.y_multi[i]) == 0:
                acquisition_values.append(0.0)
                continue
                
            mean, std = gp.predict(x, return_std=True)
            
            if self.acquisition_func == "ucb":
                acq = mean[0] + self.beta * std[0]
            elif self.acquisition_func == "ei":
                y_best = np.max(self.y_multi[i])
                z = (mean[0] - y_best) / std[0]
                acq = float((mean[0] - y_best) * norm.cdf(z) + std[0] * norm.pdf(z))
            else:  # poi
                y_best = np.max(self.y_multi[i])
                z = (mean[0] - y_best) / std[0]
                acq = float(norm.cdf(z))
                
            acquisition_values.append(acq)
            
        # Weighted sum of acquisition values
        return np.sum([w * v for w, v in zip(self.weights, acquisition_values)])
        
    def _update_pareto_front(self):
        """Update the Pareto front of non-dominated solutions."""
        if len(self.X) == 0:
            return
            
        # Convert objectives to array for easier manipulation
        Y = np.array([self.y_multi[i] for i in range(self.n_objectives)]).T
        
        # Identify non-dominated solutions
        is_pareto = np.ones(len(self.X), dtype=bool)
        for i in range(len(self.X)):
            for j in range(len(self.X)):
                if i != j:
                    if np.all(Y[j] >= Y[i]) and np.any(Y[j] > Y[i]):
                        is_pareto[i] = False
                        break
                        
        # Update Pareto front
        self.pareto_front = []
        for i in range(len(self.X)):
            if is_pareto[i]:
                self.pareto_front.append({
                    'params': self._denormalize_params(self.X[i]),
                    'objectives': [float(self.y_multi[obj][i]) for obj in range(self.n_objectives)]
                })
                
    def update_multi(self, params: Dict[str, float], values: List[float]):
        """Update models with multi-objective observations with error handling."""
        try:
            if not self._validate_objectives(values):
                raise ValueError(f"Invalid objective values: {values}")
                
            normalized_params = self._normalize_params(params)
            
            # Update observation history
            if len(self.X) == 0:
                self.X = normalized_params.reshape(1, -1)
                for i in range(self.n_objectives):
                    self.y_multi[i] = [values[i]]
            else:
                self.X = np.vstack([self.X, normalized_params])
                for i in range(self.n_objectives):
                    self.y_multi[i].append(values[i])
                    
            # Update GP models with error checking
            update_success = True
            for i, gp in enumerate(self.gp_models):
                try:
                    y = np.array(self.y_multi[i]).reshape(-1, 1)
                    gp.fit(self.X, y)
                except Exception as e:
                    logger.error(f"Error updating GP model {i}: {str(e)}")
                    update_success = False
                    
            if update_success:
                # Save successful state
                self.last_successful_state = self.get_state()
                
                # Update Pareto front
                self._update_pareto_front()
                
                # Update primary GP with weighted combination
                weighted_y = np.sum([w * np.array(y) for w, y in zip(self.weights, self.y_multi)], axis=0)
                self.y = weighted_y.reshape(-1, 1)
                self.gp.fit(self.X, self.y)
            
        except Exception as e:
            self._handle_evaluation_error(params, e)
            raise

    def suggest_params_multi(self) -> Dict[str, float]:
        """Suggest next parameters using multi-objective optimization with validation."""
        try:
            suggestion = super().suggest_params()
            
            # Validate suggestion
            for param, value in suggestion.items():
                param_range = self.parameter_ranges[param]
                if not param_range[0] <= value <= param_range[1]:
                    logger.warning(f"Invalid parameter value generated for {param}")
                    # Fix the value to be within bounds
                    suggestion[param] = max(param_range[0], min(value, param_range[1]))
                    
            return suggestion
            
        except Exception as e:
            logger.error(f"Error suggesting parameters: {str(e)}")
            # Fall back to random sampling
            return self._sample_params()

    def get_pareto_front(self) -> List[Dict]:
        """Get current Pareto front solutions."""
        return self.pareto_front
        
    def get_state(self) -> Dict:
        """Get optimizer state including multi-objective data."""
        state = super().get_state()
        state.update({
            "n_objectives": self.n_objectives,
            "weights": self.weights,
            "y_multi": self.y_multi,
            "pareto_front": self.pareto_front,
            "gp_models_params": [{
                "kernel": gp.kernel_.get_params(),
                "noise": gp.alpha
            } for gp in self.gp_models]
        })
        return state
        
    def load_state(self, state: Dict):
        """Load optimizer state including multi-objective data."""
        super().load_state(state)
        self.n_objectives = state["n_objectives"]
        self.weights = state["weights"]
        self.y_multi = state["y_multi"]
        self.pareto_front = state["pareto_front"]
        
        if len(self.X) > 0:
            self._init_multi_gp()
            for i, gp in enumerate(self.gp_models):
                y = np.array(self.y_multi[i]).reshape(-1, 1)
                gp.fit(self.X, y)
                gp.kernel_.set_params(**state["gp_models_params"][i]["kernel"])
                gp.alpha = state["gp_models_params"][i]["noise"]