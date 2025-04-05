"""
Self Evolution Controller for EvolvOS

Coordinates system-wide evolution and optimization across all components
using Bayesian optimization and multi-objective evolution.
"""

import logging
import time
from typing import Dict, List, Optional, Tuple
import torch
import numpy as np
from dataclasses import dataclass
from collections import defaultdict
from scipy.stats import norm
import os

from .bayesian_optimizer import BayesianOptimizer, MultiObjectiveBayesianOptimizer
from .enhanced_nas_implementation import EnhancedNeuralArchitectureEvolution
from .neural_architecture_evolution import NeuralArchitectureEvolution

logger = logging.getLogger("EvolvOS.SelfEvolution")

@dataclass
class ComponentMetrics:
    accuracy: float
    efficiency: float
    adaptability: float
    last_updated: float

class EvolutionMetrics:
    def __init__(self):
        self.component_metrics = defaultdict(list)
        self.evolution_history = []
        self.optimization_history = defaultdict(list)
        
    def add_component_metrics(self, 
                            component: str, 
                            metrics: ComponentMetrics):
        """Add performance metrics for a component."""
        self.component_metrics[component].append(metrics)
        
    def get_component_trend(self, component: str) -> Dict:
        """Analyze performance trend for a component."""
        if component not in self.component_metrics:
            return {}
            
        metrics = self.component_metrics[component]
        if not metrics:
            return {}
            
        recent = metrics[-5:]  # Last 5 measurements
        
        return {
            "accuracy_trend": np.mean([m.accuracy for m in recent]),
            "efficiency_trend": np.mean([m.efficiency for m in recent]),
            "adaptability_trend": np.mean([m.adaptability for m in recent]),
            "improvement_rate": self._calculate_improvement_rate(recent)
        }
        
    def _calculate_improvement_rate(self, metrics: List[ComponentMetrics]) -> float:
        """Calculate rate of improvement across all metrics."""
        if len(metrics) < 2:
            return 0.0
            
        changes = []
        for i in range(1, len(metrics)):
            acc_change = metrics[i].accuracy - metrics[i-1].accuracy
            eff_change = metrics[i].efficiency - metrics[i-1].efficiency
            adapt_change = metrics[i].adaptability - metrics[i-1].adaptability
            changes.append((acc_change + eff_change + adapt_change) / 3)
            
        return np.mean(changes)

class CompressionPerformance:
    def __init__(self):
        self.compression_ratios = []
        self.reconstruction_losses = []
        self.processing_times = []
        self.memory_savings = []
        
    def add_metric(self, 
                  compression_ratio: float,
                  reconstruction_loss: float,
                  processing_time: float,
                  memory_saved: float):
        self.compression_ratios.append(compression_ratio)
        self.reconstruction_losses.append(reconstruction_loss)
        self.processing_times.append(processing_time)
        self.memory_savings.append(memory_saved)
        
    def get_average_metrics(self) -> Dict:
        if not self.compression_ratios:
            return {}
            
        return {
            "avg_compression_ratio": np.mean(self.compression_ratios),
            "avg_reconstruction_loss": np.mean(self.reconstruction_losses),
            "avg_processing_time": np.mean(self.processing_times),
            "avg_memory_saved": np.mean(self.memory_savings),
            "total_samples": len(self.compression_ratios)
        }

class ArchitectureEvolution:
    def __init__(self):
        self.architecture_history = []
        self.performance_metrics = defaultdict(list)
        self.best_architectures = {}
        
    def add_architecture_result(self, 
                              component: str,
                              architecture: Dict,
                              performance: Dict):
        self.architecture_history.append({
            "component": component,
            "architecture": architecture,
            "performance": performance,
            "timestamp": time.time()
        })
        
        # Track performance metrics
        for metric, value in performance.items():
            self.performance_metrics[f"{component}_{metric}"].append(value)
            
        # Update best architecture if better
        current_best = self.best_architectures.get(component, {}).get("performance", {}).get("val_accuracy", 0)
        if performance.get("val_accuracy", 0) > current_best:
            self.best_architectures[component] = {
                "architecture": architecture,
                "performance": performance
            }
            
    def get_architecture_trend(self, component: str) -> Dict:
        """Analyze architecture evolution trend for a component."""
        metrics = {}
        for metric in ["val_accuracy", "flops", "params"]:
            key = f"{component}_{metric}"
            if key in self.performance_metrics:
                values = self.performance_metrics[key]
                if len(values) >= 2:
                    trend = np.polyfit(range(len(values)), values, 1)[0]
                    metrics[f"{metric}_trend"] = trend
                    
        return metrics

class SelfEvolutionController:
    """Controls and coordinates system-wide evolution."""
    
    def __init__(self,
                memory_system,
                retrieval_system,
                debate_system,
                config: Optional[Dict] = None):
        self.memory_system = memory_system
        self.retrieval_system = retrieval_system
        self.debate_system = debate_system
        
        self.config = config or {}
        self.metrics = EvolutionMetrics()
        self.evolution_state = "idle"
        
        # Component optimizers
        self.memory_optimizer = None
        self.retrieval_optimizer = None
        self.debate_optimizer = None
        
        self.compression_performance = CompressionPerformance()
        self.architecture_evolution = ArchitectureEvolution()
        
        # Initialize NAS components
        self.memory_nas = EnhancedNeuralArchitectureEvolution(
            input_shape=self.config.get("memory_input_shape", (768,)),
            population_size=self.config.get("nas_population_size", 10),
            evolution_cycles=self.config.get("nas_evolution_cycles", 5)
        )
        
        self.retrieval_nas = EnhancedNeuralArchitectureEvolution(
            input_shape=self.config.get("retrieval_input_shape", (768,)),
            population_size=self.config.get("nas_population_size", 10),
            evolution_cycles=self.config.get("nas_evolution_cycles", 5)
        )
        
        self._initialize_optimizers()
        
    def _initialize_optimizers(self):
        """Initialize multi-objective Bayesian optimizers for each component."""
        # Memory system optimization parameters and objectives
        memory_params = {
            "compression_ratio": (0.1, 0.5),
            "cache_size": (100, 10000),
            "pruning_threshold": (0.1, 0.9)
        }
        self.memory_optimizer = MultiObjectiveBayesianOptimizer(
            parameter_ranges=memory_params,
            n_objectives=4,  # accuracy, efficiency, adaptability, compression
            weights=[0.3, 0.3, 0.2, 0.2]
        )
        
        # Retrieval system optimization parameters and objectives
        retrieval_params = {
            "sparse_weight": (0.2, 0.8),
            "top_k": (3, 20),
            "min_similarity": (0.1, 0.5)
        }
        self.retrieval_optimizer = MultiObjectiveBayesianOptimizer(
            parameter_ranges=retrieval_params,
            n_objectives=3,  # accuracy, efficiency, adaptability
            weights=[0.5, 0.3, 0.2]
        )
        
        # Debate system optimization parameters and objectives
        debate_params = {
            "num_rounds": (2, 10),
            "consensus_threshold": (0.6, 0.9),
            "diversity_weight": (0.1, 0.5)
        }
        self.debate_optimizer = MultiObjectiveBayesianOptimizer(
            parameter_ranges=debate_params,
            n_objectives=3,  # accuracy, efficiency, adaptability
            weights=[0.4, 0.3, 0.3]
        )
        
    def trigger_evolution_cycle(self) -> Dict:
        """Trigger a system-wide evolution cycle with error recovery."""
        logger.info("Starting system evolution cycle")
        self.evolution_state = "active"
        
        cycle_results = {
            "status": "error",
            "timestamp": time.time(),
            "components": {}
        }

        try:
            # Validate components first
            validations = self.validate_system_components()
            if not all(validations.values()):
                invalid = [k for k, v in validations.items() if not v]
                raise ValueError(f"System validation failed for: {invalid}")

            # Collect current metrics
            metrics = self._collect_system_metrics()
            
            # Optimize each component with error handling
            components = ["memory", "retrieval", "debate"]
            
            for component in components:
                try:
                    optimizer_func = getattr(self, f"_optimize_{component}")
                    results = optimizer_func()
                    cycle_results["components"][component] = {
                        "status": "success",
                        "results": results
                    }
                except Exception as e:
                    error_info = self.handle_evolution_error(e, component)
                    cycle_results["components"][component] = {
                        "status": "error",
                        "error_info": error_info
                    }

            # Update evolution history
            self.metrics.evolution_history.append(cycle_results)
            
            # Check if any components succeeded
            successful = any(comp["status"] == "success" for comp in cycle_results["components"].values())
            if successful:
                cycle_results["status"] = "partial_success" if "error" in str(cycle_results) else "success"
            
            self.evolution_state = "idle"
            return cycle_results

        except Exception as e:
            logger.error(f"Evolution cycle failed: {str(e)}")
            self.evolution_state = "error"
            return {
                "status": "error",
                "error": str(e),
                "timestamp": time.time()
            }

    def handle_evolution_error(self, error: Exception, component: str) -> Dict:
        """Handle and log evolution errors with appropriate recovery actions."""
        error_info = {
            "component": component,
            "error": str(error),
            "timestamp": time.time(),
            "recovery_action": None
        }

        try:
            if "out of memory" in str(error):
                self._cleanup_component_memory(component)
                error_info["recovery_action"] = "memory_cleanup"
            elif "convergence" in str(error):
                self._reset_component_config(component)
                error_info["recovery_action"] = "reset_config"
            else:
                self._restart_component(component)
        
        # Get next set of parameters to try
        params = self.memory_optimizer.suggest_params_multi()
        
        # Apply parameters
        self.memory_system.update_configuration(params)
        
        # Record compression performance
        compression_stats = self.memory_system.get_compression_statistics()
        self.compression_performance.add_metric(
            compression_ratio=compression_stats["compression_ratio"],
            reconstruction_loss=compression_stats["reconstruction_loss"],
            processing_time=compression_stats["processing_time"],
            memory_saved=compression_stats["memory_saved"]
        )
        
        # Evaluate new performance with compression metrics
        new_metrics = self._collect_system_metrics()["memory"]
        compression_avg = self.compression_performance.get_average_metrics()
        
        # Update optimizer with multiple objectives
        objective_values = [
            new_metrics.accuracy,
            new_metrics.efficiency,
            new_metrics.adaptability,
            compression_avg.get("avg_compression_ratio", 0.0)
        ]
        
        self.memory_optimizer.update_multi(params, objective_values)
        
        results = {
            "old_metrics": current_metrics,
            "new_metrics": new_metrics,
            "compression_metrics": compression_avg,
            "parameters": params,
            "pareto_front": self.memory_optimizer.get_pareto_front()
        }
        
        # Run neural architecture search if data is available
        if hasattr(self.memory_system, "get_training_data"):
            train_data, val_data = self.memory_system.get_training_data()
            
            # Create data loaders
            train_loader = torch.utils.data.DataLoader(
                train_data, batch_size=32, shuffle=True
            )
            val_loader = torch.utils.data.DataLoader(
                val_data, batch_size=32, shuffle=False
            )
            
            # Run architecture search
            best_architecture = self.memory_nas.run_evolution(
                train_loader=train_loader,
                val_loader=val_loader
            )
            
            # Record architecture evolution results
            self.architecture_evolution.add_architecture_result(
                component="memory",
                architecture=best_architecture,
                performance={
                    "val_accuracy": best_architecture["val_accuracy"],
                    "flops": best_architecture["flops"],
                    "params": best_architecture["params"]
                }
            )
            
            # Update results with architecture information
            results["architecture"] = {
                "config": best_architecture,
                "performance": self.architecture_evolution.get_architecture_trend("memory")
            }
            
        return results
        
    def _optimize_memory(self) -> Dict:
        """Optimize memory system using multi-objective optimization."""
        # Get current metrics before optimization
        current_metrics = self.metrics.get_component_trend("memory")
        
        # Get next set of parameters to try
        params = self.memory_optimizer.suggest_params_multi()
        
        # Apply parameters
        self.memory_system.update_configuration(params)
        
        # Record compression performance
        compression_stats = self.memory_system.get_compression_statistics()
        self.compression_performance.add_metric(
            compression_ratio=compression_stats["compression_ratio"],
            reconstruction_loss=compression_stats["reconstruction_loss"],
            processing_time=compression_stats["processing_time"],
            memory_saved=compression_stats["memory_saved"]
        )
        
        # Evaluate new performance with compression metrics
        new_metrics = self._collect_system_metrics()["memory"]
        compression_avg = self.compression_performance.get_average_metrics()
        
        # Update optimizer with multiple objectives
        objective_values = [
            new_metrics.accuracy,
            new_metrics.efficiency,
            new_metrics.adaptability,
            compression_avg.get("avg_compression_ratio", 0.0)
        ]
        
        self.memory_optimizer.update_multi(params, objective_values)
        
        results = {
            "old_metrics": current_metrics,
            "new_metrics": new_metrics,
            "compression_metrics": compression_avg,
            "parameters": params,
            "pareto_front": self.memory_optimizer.get_pareto_front()
        }
        
        # Run neural architecture search if data is available
        if hasattr(self.memory_system, "get_training_data"):
            train_data, val_data = self.memory_system.get_training_data()
            
            # Create data loaders
            train_loader = torch.utils.data.DataLoader(
                train_data, batch_size=32, shuffle=True
            )
            val_loader = torch.utils.data.DataLoader(
                val_data, batch_size=32, shuffle=False
            )
            
            # Run architecture search
            best_architecture = self.memory_nas.run_evolution(
                train_loader=train_loader,
                val_loader=val_loader
            )
            
            # Record architecture evolution results
            self.architecture_evolution.add_architecture_result(
                component="memory",
                architecture=best_architecture,
                performance={
                    "val_accuracy": best_architecture["val_accuracy"],
                    "flops": best_architecture["flops"],
                    "params": best_architecture["params"]
                }
            )
            
            # Update results with architecture information
            results["architecture"] = {
                "config": best_architecture,
                "performance": self.architecture_evolution.get_architecture_trend("memory")
            }
            
        return results
        
    def _optimize_retrieval(self) -> Dict:
        """Optimize retrieval system using multi-objective optimization."""
        current_metrics = self.metrics.get_component_trend("retrieval")
        
        # Get next set of parameters to try
        params = self.retrieval_optimizer.suggest_params_multi()
        
        # Apply parameters
        self.retrieval_system.update_configuration(params)
        
        # Evaluate new performance
        new_metrics = self._collect_system_metrics()["retrieval"]
        
        # Update optimizer with multiple objectives
        objective_values = [
            new_metrics.accuracy,
            new_metrics.efficiency,
            new_metrics.adaptability
        ]
        self.retrieval_optimizer.update_multi(params, objective_values)
        
        results = {
            "old_metrics": current_metrics,
            "new_metrics": new_metrics,
            "parameters": params,
            "pareto_front": self.retrieval_optimizer.get_pareto_front()
        }
        
        # Run neural architecture search if data is available
        if hasattr(self.retrieval_system, "get_training_data"):
            train_data, val_data = self.retrieval_system.get_training_data()
            
            train_loader = torch.utils.data.DataLoader(
                train_data, batch_size=32, shuffle=True
            )
            val_loader = torch.utils.data.DataLoader(
                val_data, batch_size=32, shuffle=False
            )
            
            best_architecture = self.retrieval_nas.run_evolution(
                train_loader=train_loader,
                val_loader=val_loader
            )
            
            self.architecture_evolution.add_architecture_result(
                component="retrieval",
                architecture=best_architecture,
                performance={
                    "val_accuracy": best_architecture["val_accuracy"],
                    "flops": best_architecture["flops"],
                    "params": best_architecture["params"]
                }
            )
            
            results["architecture"] = {
                "config": best_architecture,
                "performance": self.architecture_evolution.get_architecture_trend("retrieval")
            }
            
        return results
        
    def _optimize_debate(self) -> Dict:
        """Optimize debate system using multi-objective optimization."""
        current_metrics = self.metrics.get_component_trend("debate")
        
        # Get next set of parameters to try
        params = self.debate_optimizer.suggest_params_multi()
        
        # Apply parameters
        self.debate_system.update_configuration(params)
        
        # Evaluate new performance
        new_metrics = self._collect_system_metrics()["debate"]
        
        # Update optimizer with multiple objectives
        objective_values = [
            new_metrics.accuracy,
            new_metrics.efficiency,
            new_metrics.adaptability
        ]
        self.debate_optimizer.update_multi(params, objective_values)
        
        return {
            "old_metrics": current_metrics,
            "new_metrics": new_metrics,
            "parameters": params,
            "pareto_front": self.debate_optimizer.get_pareto_front()
        }
        
    def _analyze_improvements(self, cycle_results: Dict) -> Dict:
        """Analyze improvements from evolution cycle."""
        improvements = {}
        
        for component in ["memory", "retrieval", "debate"]:
            old_metrics = cycle_results[component]["old_metrics"]
            new_metrics = cycle_results[component]["new_metrics"]
            
            if not old_metrics:  # First evolution cycle
                continue
                
            improvements[component] = {
                "accuracy_change": new_metrics.accuracy - old_metrics["accuracy_trend"],
                "efficiency_change": new_metrics.efficiency - old_metrics["efficiency_trend"],
                "adaptability_change": new_metrics.adaptability - old_metrics["adaptability_trend"]
            }
            
        return improvements
        
    def get_evolution_status(self) -> Dict:
        """Get current evolution status and metrics."""
        return {
            "state": self.evolution_state,
            "last_cycle": self.metrics.evolution_history[-1] if self.metrics.evolution_history else None,
            "component_trends": {
                "memory": self.metrics.get_component_trend("memory"),
                "retrieval": self.metrics.get_component_trend("retrieval"),
                "debate": self.metrics.get_component_trend("debate")
            }
        }
        
    def get_compression_status(self) -> Dict:
        """Get detailed compression performance metrics."""
        return {
            "current_performance": self.compression_performance.get_average_metrics(),
            "optimization_trend": self._analyze_compression_trend()
        }
        
    def _analyze_compression_trend(self) -> Dict:
        """Analyze compression performance trend."""
        if not self.compression_performance.compression_ratios:
            return {}
            
        # Get last 10 measurements
        recent_ratios = self.compression_performance.compression_ratios[-10:]
        recent_losses = self.compression_performance.reconstruction_losses[-10:]
        
        return {
            "compression_trend": np.polyfit(range(len(recent_ratios)), recent_ratios, 1)[0],
            "quality_trend": np.polyfit(range(len(recent_losses)), recent_losses, 1)[0],
            "samples_analyzed": len(recent_ratios)
        }
        
    def get_architecture_status(self) -> Dict:
        """Get status of architecture evolution."""
        return {
            "best_architectures": self.architecture_evolution.best_architectures,
            "trends": {
                "memory": self.architecture_evolution.get_architecture_trend("memory"),
                "retrieval": self.architecture_evolution.get_architecture_trend("retrieval")
            },
            "history_length": len(self.architecture_evolution.architecture_history)
        }

    def save_state(self, path: str):
        """Save evolution state and metrics."""
        state = {
            "metrics": self.metrics,
            "memory_optimizer": self.memory_optimizer.get_state(),
            "retrieval_optimizer": self.retrieval_optimizer.get_state(),
            "debate_optimizer": self.debate_optimizer.get_state(),
            "architecture_evolution": self.architecture_evolution,
            "config": self.config
        }
        torch.save(state, path)
        
    def load_state(self, path: str):
        """Load evolution state and metrics."""
        state = torch.load(path)
        self.metrics = state["metrics"]
        self.memory_optimizer.load_state(state["memory_optimizer"])
        self.retrieval_optimizer.load_state(state["retrieval_optimizer"])
        self.debate_optimizer.load_state(state["debate_optimizer"])
        self.architecture_evolution = state["architecture_evolution"]
        self.config = state["config"]

    def validate_system_components(self) -> Dict[str, bool]:
        """Validate that all required system components are properly initialized."""
        validations = {
            "memory": self._validate_memory_system(),
            "retrieval": self._validate_retrieval_system(),
            "debate": self._validate_debate_system(),
            "optimizers": self._validate_optimizers()
        }
        return validations
        
    def _validate_memory_system(self) -> bool:
        """Validate memory system configuration and capabilities."""
        try:
            # Check required methods
            required_methods = [
                "get_statistics",
                "get_compression_statistics",
                "update_configuration"
            ]
            
            for method in required_methods:
                if not hasattr(self.memory_system, method):
                    logger.error(f"Memory system missing required method: {method}")
                    return False
                    
            # Validate compression capabilities
            compression_stats = self.memory_system.get_compression_statistics()
            required_stats = ["compression_ratio", "reconstruction_loss", 
                            "processing_time", "memory_saved"]
                            
            for stat in required_stats:
                if stat not in compression_stats:
                    logger.error(f"Memory system missing compression stat: {stat}")
                    return False
                    
            return True
            
        except Exception as e:
            logger.error(f"Memory system validation failed: {str(e)}")
            return False
            
    def _validate_retrieval_system(self) -> bool:
        """Validate retrieval system configuration and capabilities."""
        try:
            # Check required methods
            required_methods = [
                "get_statistics",
                "update_configuration"
            ]
            
            for method in required_methods:
                if not hasattr(self.retrieval_system, method):
                    logger.error(f"Retrieval system missing required method: {method}")
                    return False
                    
            # Validate retrieval capabilities
            stats = self.retrieval_system.get_statistics()
            required_stats = ["search_accuracy", "avg_search_time", "learning_rate"]
            
            for stat in required_stats:
                if stat not in stats:
                    logger.error(f"Retrieval system missing stat: {stat}")
                    return False
                    
            return True
            
        except Exception as e:
            logger.error(f"Retrieval system validation failed: {str(e)}")
            return False
            
    def _validate_debate_system(self) -> bool:
        """Validate debate system configuration and capabilities."""
        try:
            # Check required methods
            required_methods = [
                "get_statistics",
                "update_configuration"
            ]
            
            for method in required_methods:
                if not hasattr(self.debate_system, method):
                    logger.error(f"Debate system missing required method: {method}")
                    return False
                    
            # Validate debate capabilities
            stats = self.debate_system.get_statistics()
            required_stats = ["consensus_rate", "avg_resolution_time", "innovation_rate"]
            
            for stat in required_stats:
                if stat not in stats:
                    logger.error(f"Debate system missing stat: {stat}")
                    return False
                    
            return True
            
        except Exception as e:
            logger.error(f"Debate system validation failed: {str(e)}")
            return False
            
    def _validate_optimizers(self) -> bool:
        """Validate optimizer configurations."""
        try:
            optimizers = [
                (self.memory_optimizer, "Memory"),
                (self.retrieval_optimizer, "Retrieval"),
                (self.debate_optimizer, "Debate")
            ]
            
            for optimizer, name in optimizers:
                if optimizer is None:
                    logger.error(f"{name} optimizer not initialized")
                    return False
                    
                if not hasattr(optimizer, "suggest_params_multi"):
                    logger.error(f"{name} optimizer missing multi-objective capabilities")
                    return False
                    
            return True
            
        except Exception as e:
            logger.error(f"Optimizer validation failed: {str(e)}")
            return False
            
    def handle_evolution_error(self, error: Exception, component: str) -> Dict:
        """Handle and log evolution errors with appropriate recovery actions."""
        error_info = {
            "component": component,
            "error": str(error),
            "timestamp": time.time(),
            "recovery_action": None
        }
        
        try:
            if isinstance(error, ValueError):
                # Configuration error
                error_info["recovery_action"] = "reset_config"
                self._reset_component_config(component)
            elif isinstance(error, RuntimeError):
                # Runtime error
                error_info["recovery_action"] = "restart_component"
                self._restart_component(component)
            else:
                # Unknown error
                error_info["recovery_action"] = "log_only"
                
            logger.error(f"Evolution error in {component}: {str(error)}")
            logger.info(f"Recovery action: {error_info['recovery_action']}")
            
            # Add to error history
            self.metrics.optimization_history[component].append(error_info)
            
        except Exception as recovery_error:
            logger.error(f"Error recovery failed: {str(recovery_error)}")
            error_info["recovery_failed"] = True
            
        return error_info
        
    def _reset_component_config(self, component: str):
        """Reset component configuration to default values."""
        default_configs = {
            "memory": {
                "compression_ratio": 0.3,
                "cache_size": 1000,
                "pruning_threshold": 0.5
            },
            "retrieval": {
                "sparse_weight": 0.5,
                "top_k": 10,
                "min_similarity": 0.3
            },
            "debate": {
                "num_rounds": 5,
                "consensus_threshold": 0.7,
                "diversity_weight": 0.3
            }
        }
        
        if component in default_configs:
            system = getattr(self, f"{component}_system")
            system.update_configuration(default_configs[component])
            logger.info(f"Reset {component} configuration to defaults")
            
    def _restart_component(self, component: str):
        """Attempt to restart a component after runtime error."""
        try:
            system = getattr(self, f"{component}_system")
            if hasattr(system, "restart"):
                system.restart()
                logger.info(f"Successfully restarted {component} system")
            else:
                logger.warning(f"No restart method available for {component} system")
                
        except Exception as e:
            logger.error(f"Failed to restart {component} system: {str(e)}")
            raise
