"""
Neural Compressor

This module implements neural compression agents for optimizing memory usage
in the OS1 component of the EvolvOS system.
"""

import numpy as np
import random
import time
import uuid
import json
from typing import Dict, List, Any, Optional, Tuple
import hashlib
import zlib
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
import logging

logger = logging.getLogger("OS1.NeuralCompressor")

class CompressionError(Exception):
    """Custom exception for compression-related errors."""
    pass

class CompressionMetrics:
    """Thread-safe class for tracking compression metrics and performance."""
    
    def __init__(self):
        from threading import Lock
        self.lock = Lock()
        self.compression_ratios = []
        self.processing_times = []
        self.reconstruction_losses = []
        
    def add_metric(self, original_size: int, compressed_size: int, 
                 processing_time: float, reconstruction_loss: float = None):
        """Add compression metrics from a single operation thread-safely."""
        with self.lock:
            compression_ratio = original_size / max(1, compressed_size)
            self.compression_ratios.append(compression_ratio)
            self.processing_times.append(processing_time)
            if reconstruction_loss is not None:
                self.reconstruction_losses.append(reconstruction_loss)
            
    def get_average_metrics(self) -> Dict:
        """Get average performance metrics thread-safely."""
        with self.lock:
            avg_ratio = np.mean(self.compression_ratios) if self.compression_ratios else 0
            avg_time = np.mean(self.processing_times) if self.processing_times else 0
            avg_loss = np.mean(self.reconstruction_losses) if self.reconstruction_losses else 0
            sample_count = len(self.compression_ratios)
        
        return {
            "avg_compression_ratio": avg_ratio,
            "avg_processing_time_ms": avg_time * 1000,  # Convert to ms
            "avg_reconstruction_loss": avg_loss,
            "samples": sample_count
        }
        
    def clear_metrics(self):
        """Clear all metrics thread-safely.""" 
        with self.lock:
            self.compression_ratios.clear()
            self.processing_times.clear()
            self.reconstruction_losses.clear()

class OptimizationError(Exception):
    """Custom exception for optimization-related errors."""
    pass

class BayesianOptimizer:
    """Bayesian optimization for compression hyperparameters."""
    
    def __init__(self, param_ranges: Dict[str, Tuple[float, float]], n_init: int = 5):
        try:
            if not param_ranges:
                raise OptimizationError("Parameter ranges cannot be empty")
                
            for name, (low, high) in param_ranges.items():
                if not isinstance(low, (int, float)) or not isinstance(high, (int, float)):
                    raise OptimizationError(f"Invalid range type for parameter {name}")
                if low >= high:
                    raise OptimizationError(f"Invalid range for parameter {name}: low >= high")
                    
            self.param_ranges = param_ranges
            self.n_init = max(1, n_init)
            
            # Initialize GP
            kernel = ConstantKernel(1.0) * RBF([1.0] * len(param_ranges))
            self.gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)
            
            self.X = []  # Parameter configurations
            self.y = []  # Observed performance
            
        except Exception as e:
            logger.error(f"Failed to initialize optimizer: {str(e)}")
            raise OptimizationError(f"Initialization failed: {str(e)}")
            
    def suggest_params(self) -> Dict[str, float]:
        """Suggest next set of parameters to try with error handling."""
        try:
            if len(self.X) < self.n_init:
                # Initial random exploration
                return self._sample_params()
                
            # Validate data
            if len(self.X) != len(self.y):
                raise OptimizationError("Inconsistent optimization history")
                
            # Fit GP to observed data
            X = np.vstack(self.X)
            y = np.array(self.y)
            
            if np.isnan(y).any() or np.isinf(y).any():
                raise OptimizationError("Invalid performance values in history")
                
            self.gp.fit(X, y)
            
            # Find best parameters using expected improvement
            best_params = None
            best_ei = -float('inf')
            
            for _ in range(100):  # Number of random samples
                params = self._sample_params()
                x = self._params_to_array(params)
                
                # Calculate expected improvement
                mean, std = self.gp.predict([x], return_std=True)
                if std == 0:
                    continue
                    
                best_y = max(self.y)
                z = (mean - best_y) / std
                ei = std * (z * norm.cdf(z) + norm.pdf(z))
                
                if ei > best_ei:
                    best_ei = ei
                    best_params = params
                    
            if best_params is None:
                raise OptimizationError("Failed to find valid parameters")
                
            return best_params
            
        except Exception as e:
            logger.error(f"Error suggesting parameters: {str(e)}")
            # Fall back to random sampling if optimization fails
            return self._sample_params()
            
    def update(self, params: Dict[str, float], performance: float):
        """Update with observed performance with error handling."""
        try:
            # Validate parameters
            if not all(name in self.param_ranges for name in params):
                raise OptimizationError("Invalid parameter names")
                
            # Validate performance
            if not isinstance(performance, (int, float)):
                raise OptimizationError("Performance must be a number")
            if np.isnan(performance) or np.isinf(performance):
                raise OptimizationError("Invalid performance value")
                
            x = self._params_to_array(params)
            self.X.append(x)
            self.y.append(performance)
            
        except Exception as e:
            logger.error(f"Error updating optimizer: {str(e)}")
            raise OptimizationError(f"Update failed: {str(e)}")
            
    def get_best_params(self) -> Dict[str, float]:
        """Get best parameters found so far."""
        try:
            if not self.y:
                raise OptimizationError("No optimization history available")
                
            best_idx = np.argmax(self.y)
            best_x = self.X[best_idx]
            return self._array_to_params(best_x)
            
        except Exception as e:
            logger.error(f"Error getting best parameters: {str(e)}")
            raise OptimizationError(f"Could not get best parameters: {str(e)}")
            
    def _sample_params(self) -> Dict[str, float]:
        """Sample random parameters for initialization."""
        try:
            params = {}
            for name, (low, high) in self.param_ranges.items():
                params[name] = np.random.uniform(low, high)
            return params
        except Exception as e:
            logger.error(f"Error sampling parameters: {str(e)}")
            raise OptimizationError(f"Parameter sampling failed: {str(e)}")

    def _params_to_array(self, params: Dict[str, float]) -> np.ndarray:
        """Convert parameters dict to array."""
        return np.array([params[name] for name in self.param_ranges])
        
    def _array_to_params(self, x: np.ndarray) -> Dict[str, float]:
        """Convert array to parameters dict."""
        return {name: x[i] for i, name in enumerate(self.param_ranges)}

class NeuralCompressor:
    """Neural network based compressor for memory optimization."""
    
    def __init__(self, input_dim: int, compression_ratio: float = 0.5):
        self.input_dim = input_dim
        self.compression_ratio = compression_ratio
        self.compressed_dim = max(1, int(input_dim * compression_ratio))
        self._initialize_model()
        
    def compress(self, data: torch.Tensor) -> torch.Tensor:
        """Compress input data with error handling."""
        try:
            if not isinstance(data, torch.Tensor):
                raise CompressionError("Input must be a torch.Tensor")
                
            if data.dim() != 2:
                raise CompressionError("Input must be 2-dimensional [batch_size, input_dim]")
                
            if data.size(1) != self.input_dim:
                raise CompressionError(f"Input dimension mismatch. Expected {self.input_dim}, got {data.size(1)}")
                
            # Check for NaN or Inf values
            if torch.isnan(data).any() or torch.isinf(data).any():
                raise CompressionError("Input contains NaN or Inf values")
                
            with torch.no_grad():
                compressed = self.encoder(data)
                
            # Validate compression output
            if torch.isnan(compressed).any() or torch.isinf(compressed).any():
                raise CompressionError("Compression produced invalid values")
                
            return compressed
            
        except CompressionError as e:
            logger.error(f"Compression error: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during compression: {str(e)}")
            raise CompressionError(f"Compression failed: {str(e)}")
            
    def decompress(self, compressed_data: torch.Tensor) -> torch.Tensor:
        """Decompress data with error handling."""
        try:
            if not isinstance(compressed_data, torch.Tensor):
                raise CompressionError("Input must be a torch.Tensor")
                
            if compressed_data.dim() != 2:
                raise CompressionError("Input must be 2-dimensional [batch_size, compressed_dim]")
                
            if compressed_data.size(1) != self.compressed_dim:
                raise CompressionError(f"Input dimension mismatch. Expected {self.compressed_dim}, got {compressed_data.size(1)}")
                
            # Check for NaN or Inf values
            if torch.isnan(compressed_data).any() or torch.isinf(compressed_data).any():
                raise CompressionError("Input contains NaN or Inf values")
                
            with torch.no_grad():
                decompressed = self.decoder(compressed_data)
                
            # Validate decompression output
            if torch.isnan(decompressed).any() or torch.isinf(decompressed).any():
                raise CompressionError("Decompression produced invalid values")
                
            return decompressed
            
        except CompressionError as e:
            logger.error(f"Decompression error: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during decompression: {str(e)}")
            raise CompressionError(f"Decompression failed: {str(e)}")
            
    def _initialize_model(self):
        """Initialize the compression model with error handling."""
        try:
            self.encoder = nn.Sequential(
                nn.Linear(self.input_dim, self.compressed_dim * 2),
                nn.ReLU(),
                nn.Linear(self.compressed_dim * 2, self.compressed_dim)
            )
            
            self.decoder = nn.Sequential(
                nn.Linear(self.compressed_dim, self.compressed_dim * 2),
                nn.ReLU(),
                nn.Linear(self.compressed_dim * 2, self.input_dim)
            )
            
        except Exception as e:
            logger.error(f"Failed to initialize compression model: {str(e)}")
            raise CompressionError(f"Model initialization failed: {str(e)}")
            
    def validate_compression(self, original: torch.Tensor, compressed: torch.Tensor, 
                           decompressed: torch.Tensor) -> Dict[str, float]:
        """Validate compression quality and return metrics."""
        try:
            metrics = {}
            
            # Calculate compression ratio
            metrics["compression_ratio"] = compressed.numel() / original.numel()
            
            # Calculate reconstruction error
            metrics["mse_loss"] = F.mse_loss(original, decompressed).item()
            
            # Calculate relative error
            metrics["relative_error"] = torch.norm(original - decompressed) / torch.norm(original)
            
            # Check if compression is within acceptable bounds
            if metrics["relative_error"] > 0.5:  # 50% error threshold
                raise CompressionError(f"Compression quality too low: {metrics['relative_error']:.2%} relative error")
                
            return metrics
            
        except Exception as e:
            logger.error(f"Compression validation failed: {str(e)}")
            raise CompressionError(f"Validation failed: {str(e)}")

class AdaptiveCompressor:
    """Adaptive compression with Bayesian optimization."""
    
    def __init__(self, 
                input_dim: int,
                min_compression_ratio: float = 0.1,
                max_compression_ratio: float = 0.5):
        self.input_dim = input_dim
        
        # Define parameter ranges for Bayesian optimization
        self.param_ranges = {
            "compression_ratio": (min_compression_ratio, max_compression_ratio),
            "learning_rate": (1e-4, 1e-2),
            "dropout": (0.0, 0.3)
        }
        
        # Initialize optimizer
        self.optimizer = BayesianOptimizer(self.param_ranges)
        
        # Track best model
        self.best_model = None
        self.best_performance = float('-inf')
        
    def train_model(self, 
                   data: torch.Tensor,
                   params: Dict[str, float],
                   n_epochs: int = 100,
                   batch_size: int = 32) -> float:
        """Train model with given parameters and return performance."""
        # Create model
        compression_dim = int(self.input_dim * params["compression_ratio"])
        model = NeuralCompressor(
            input_dim=self.input_dim,
            compression_ratio=params["compression_ratio"]
        )
        
        # Create optimizer
        optimizer = torch.optim.Adam(model.encoder.parameters(), lr=params["learning_rate"])
        
        # Training loop
        model.encoder.train()
        for epoch in range(n_epochs):
            total_loss = 0
            
            # Process in batches
            for i in range(0, len(data), batch_size):
                batch = data[i:i+batch_size]
                
                optimizer.zero_grad()
                compressed = model.compress(batch)
                decompressed = model.decompress(compressed)
                
                # Reconstruction loss
                recon_loss = F.mse_loss(decompressed, batch)
                
                # Total loss
                loss = recon_loss
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                
            avg_loss = total_loss / (len(data) / batch_size)
            
            if epoch % 10 == 0:
                logger.debug(f"Epoch {epoch}, Loss: {avg_loss:.4f}")
                
        # Evaluate performance
        model.encoder.eval()
        with torch.no_grad():
            compressed = model.compress(data)
            decompressed = model.decompress(compressed)
            recon_loss = F.mse_loss(decompressed, data).item()
            compression_ratio = compression_dim / self.input_dim
            
            # Performance metric combines reconstruction quality and compression
            performance = 1.0 / (recon_loss * (1 + compression_ratio))
            
        if performance > self.best_performance:
            self.best_performance = performance
            self.best_model = model
            
        return performance
        
    def optimize(self, data: torch.Tensor, n_trials: int = 10):
        """Run Bayesian optimization to find best compression parameters."""
        logger.info("Starting compression optimization")
        
        for trial in range(n_trials):
            # Get suggested parameters
            params = self.optimizer.suggest_params()
            
            # Train and evaluate
            performance = self.train_model(data, params)
            
            # Update optimizer
            self.optimizer.update(params, performance)
            
            logger.info(f"Trial {trial + 1}/{n_trials}, "
                       f"Compression Ratio: {params['compression_ratio']:.3f}, "
                       f"Performance: {performance:.4f}")
            
        return self.best_model
        
    def compress(self, data: torch.Tensor) -> torch.Tensor:
        """Compress data using best model."""
        if self.best_model is None:
            raise ValueError("Must run optimize() first")
            
        self.best_model.encoder.eval()
        with torch.no_grad():
            return self.best_model.compress(data)
            
    def decompress(self, compressed: torch.Tensor) -> torch.Tensor:
        """Decompress data using best model."""
        if self.best_model is None:
            raise ValueError("Must run optimize() first")
            
        self.best_model.encoder.eval()
        with torch.no_grad():
            return self.best_model.decompress(compressed)