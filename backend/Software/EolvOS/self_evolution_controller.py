"""
Self-Evolution Controller

This module implements the controller for self-evolution in the EvolvOS system,
enabling autonomous improvement through evolution cycles and feedback loops.
"""

import os
import sys
import time
import json
import logging
import random
import uuid
from typing import Dict, List, Any, Optional, Tuple, Callable, Set
import numpy as np
import traceback

# Add new imports for LLM-guided evolution
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from collections import deque

logger = logging.getLogger("EvolvOS.SelfEvolution")

class EvolutionMetrics:
    """Tracks metrics related to system evolution."""
    
    def __init__(self):
        """Initialize evolution metrics."""
        self.cycles_completed = 0
        self.improvements_applied = 0
        self.rejected_improvements = 0
        self.performance_history = []
        self.cycle_durations = []
        self.start_time = time.time()
        
    def record_cycle(self, metrics: Dict[str, float], duration: float, improvements: int):
        """
        Record metrics for an evolution cycle.
        
        Args:
            metrics: Performance metrics from the cycle
            duration: Duration of the cycle in seconds
            improvements: Number of improvements applied
        """
        self.cycles_completed += 1
        self.improvements_applied += improvements
        self.performance_history.append(metrics)
        self.cycle_durations.append(duration)
        
    def record_rejected(self, count: int = 1):
        """
        Record rejected improvements.
        
        Args:
            count: Number of rejected improvements
        """
        self.rejected_improvements += count
        
    def get_summary(self) -> Dict:
        """
        Get a summary of evolution metrics.
        
        Returns:
            Dictionary with evolution metrics
        """
        total_time = time.time() - self.start_time
        
        # Calculate improvement rate (improvements per hour)
        hours_elapsed = total_time / 3600
        improvement_rate = self.improvements_applied / max(1, hours_elapsed)
        
        # Calculate average cycle duration
        avg_duration = sum(self.cycle_durations) / max(1, len(self.cycle_durations))
        
        # Calculate performance trends if we have history
        performance_trends = {}
        if len(self.performance_history) > 1:
            for metric in self.performance_history[0].keys():
                values = [cycle_metrics.get(metric, 0) for cycle_metrics in self.performance_history]
                if len(values) > 1:
                    # Simple linear trend (positive is improving)
                    trend = values[-1] - values[0]
                    performance_trends[metric] = trend
        
        return {
            "cycles_completed": self.cycles_completed,
            "improvements_applied": self.improvements_applied,
            "rejected_improvements": self.rejected_improvements,
            "total_runtime_seconds": total_time,
            "improvement_rate_per_hour": improvement_rate,
            "average_cycle_duration": avg_duration,
            "performance_trends": performance_trends,
            "last_metrics": self.performance_history[-1] if self.performance_history else {}
        }
        
    def to_dict(self) -> Dict:
        """Convert metrics to dictionary for serialization."""
        return {
            "cycles_completed": self.cycles_completed,
            "improvements_applied": self.improvements_applied,
            "rejected_improvements": self.rejected_improvements,
            "performance_history": self.performance_history,
            "cycle_durations": self.cycle_durations,
            "start_time": self.start_time
        }
        
    @classmethod
    def from_dict(cls, data: Dict) -> 'EvolutionMetrics':
        """Create metrics object from dictionary."""
        metrics = cls()
        metrics.cycles_completed = data.get("cycles_completed", 0)
        metrics.improvements_applied = data.get("improvements_applied", 0)
        metrics.rejected_improvements = data.get("rejected_improvements", 0)
        metrics.performance_history = data.get("performance_history", [])
        metrics.cycle_durations = data.get("cycle_durations", [])
        metrics.start_time = data.get("start_time", time.time())
        return metrics

class Improvement:
    """Represents a potential system improvement."""
    
    def __init__(self, 
                 component: str, 
                 description: str,
                 implementation: Optional[str] = None,
                 estimated_impact: Optional[Dict[str, float]] = None):
        """
        Initialize an improvement.
        
        Args:
            component: System component to improve
            description: Description of the improvement
            implementation: Implementation details (code, config, etc.)
            estimated_impact: Estimated impact on metrics
        """
        self.id = str(uuid.uuid4())
        self.component = component
        self.description = description
        self.implementation = implementation
        self.estimated_impact = estimated_impact or {}
        self.creation_time = time.time()
        self.applied = False
        self.applied_time = None
        self.actual_impact = {}
        self.rejected = False
        self.rejection_reason = None
        
    def apply(self):
        """Mark the improvement as applied."""
        self.applied = True
        self.applied_time = time.time()
        
    def reject(self, reason: str):
        """
        Mark the improvement as rejected.
        
        Args:
            reason: Reason for rejection
        """
        self.rejected = True
        self.rejection_reason = reason
        
    def record_impact(self, metrics: Dict[str, float]):
        """
        Record actual impact of the improvement.
        
        Args:
            metrics: Measured impact on metrics
        """
        self.actual_impact = metrics
        
    def to_dict(self) -> Dict:
        """Convert improvement to dictionary for serialization."""
        return {
            "id": self.id,
            "component": self.component,
            "description": self.description,
            "implementation": self.implementation,
            "estimated_impact": self.estimated_impact,
            "creation_time": self.creation_time,
            "applied": self.applied,
            "applied_time": self.applied_time,
            "actual_impact": self.actual_impact,
            "rejected": self.rejected,
            "rejection_reason": self.rejection_reason
        }
        
    @classmethod
    def from_dict(cls, data: Dict) -> 'Improvement':
        """Create improvement object from dictionary."""
        improvement = cls(
            component=data.get("component", "unknown"),
            description=data.get("description", ""),
            implementation=data.get("implementation"),
            estimated_impact=data.get("estimated_impact", {})
        )
        improvement.id = data.get("id", improvement.id)
        improvement.creation_time = data.get("creation_time", improvement.creation_time)
        improvement.applied = data.get("applied", False)
        improvement.applied_time = data.get("applied_time")
        improvement.actual_impact = data.get("actual_impact", {})
        improvement.rejected = data.get("rejected", False)
        improvement.rejection_reason = data.get("rejection_reason")
        return improvement

class SystemEvaluator:
    """Evaluates system performance for evolution."""
    
    def __init__(self, evaluation_metrics: Optional[List[str]] = None):
        """
        Initialize the system evaluator.
        
        Args:
            evaluation_metrics: List of metrics to evaluate
        """
        self.metrics = evaluation_metrics or [
            "accuracy", "efficiency", "adaptability", "robustness", "memory_usage"
        ]
        
    def evaluate(self, system: Any) -> Dict[str, float]:
        """
        Evaluate system performance.
        
        Args:
            system: System to evaluate
            
        Returns:
            Dictionary of performance metrics
        """
        metrics = {}
        
        # Get system status
        status = system.get_status()
        
        # Memory efficiency metrics
        if "memory" in status:
            memory_stats = status["memory"]
            total_items = memory_stats.get("total_items", 0)
            volatile_items = memory_stats.get("volatile_items", 0)
            
            # Calculate memory usage ratio (lower is better)
            memory_usage = volatile_items / max(1, total_items)
            metrics["memory_usage"] = memory_usage
            
            # Estimate retrieval efficiency based on memory distribution
            retrieval_efficiency = 1.0 - (memory_usage * 0.5)  # Simple heuristic
            metrics["efficiency"] = retrieval_efficiency
        
        # Simulate accuracy based on system components
        components = status.get("system", {}).get("components", [])
        
        # More components generally means more capability
        component_count = len(components)
        capability_score = min(1.0, component_count / 10)
        metrics["adaptability"] = capability_score
        
        # Accuracy depends on retrieval system and memory
        has_retrieval = "retrieval" in components
        has_memory = "memory" in components
        has_entity_graph = "entity_graph" in components
        
        # Better accuracy with more sophisticated components
        base_accuracy = 0.7
        if has_retrieval:
            base_accuracy += 0.1
        if has_memory:
            base_accuracy += 0.1
        if has_entity_graph:
            base_accuracy += 0.1
            
        metrics["accuracy"] = base_accuracy
        
        # Robustness - placeholder for more sophisticated evaluation
        metrics["robustness"] = 0.5 + random.random() * 0.2
        
        # Add random noise to make it more realistic
        for key in metrics:
            # Add small random fluctuation
            metrics[key] += (random.random() - 0.5) * 0.05
            # Ensure values are in [0, 1]
            metrics[key] = max(0.0, min(1.0, metrics[key]))
            
        return metrics
        
    def evaluate_improvement(self, 
                            system: Any, 
                            improvement: Improvement) -> Dict[str, float]:
        """
        Estimate impact of an improvement.
        
        Args:
            system: Current system state
            improvement: Proposed improvement
            
        Returns:
            Dictionary of estimated metric changes
        """
        baseline = self.evaluate(system)
        impact = {}
        
        # Simple heuristic for estimating impact based on improvement component
        component = improvement.component.lower()
        
        if component == "memory":
            # Memory improvements typically affect efficiency and memory usage
            impact["memory_usage"] = random.uniform(0.05, 0.15)  # Reduction
            impact["efficiency"] = random.uniform(0.03, 0.08)  # Improvement
            
        elif component == "retrieval":
            # Retrieval improvements affect accuracy and efficiency
            impact["accuracy"] = random.uniform(0.03, 0.1)
            impact["efficiency"] = random.uniform(0.02, 0.07)
            
        elif component == "optimization":
            # General optimizations improve efficiency
            impact["efficiency"] = random.uniform(0.05, 0.12)
            impact["memory_usage"] = random.uniform(0.02, 0.08)
            
        elif component == "architecture":
            # Architecture improvements affect adaptability and robustness
            impact["adaptability"] = random.uniform(0.05, 0.15)
            impact["robustness"] = random.uniform(0.03, 0.1)
            
        elif component == "integration":
            # Integration improvements affect overall adaptability
            impact["adaptability"] = random.uniform(0.04, 0.12)
            
        else:
            # Unknown components have small random improvements
            for metric in self.metrics:
                impact[metric] = random.uniform(0.01, 0.05)
                
        # Ensure realistic improvements
        for metric in impact:
            # Convert reductions to negative values
            if metric == "memory_usage":
                impact[metric] = -impact[metric]
                
            # Ensure improvements don't exceed reasonable bounds
            if metric in baseline:
                if baseline[metric] + impact[metric] > 1.0:
                    impact[metric] = 1.0 - baseline[metric]
                if baseline[metric] + impact[metric] < 0.0:
                    impact[metric] = -baseline[metric]
                    
        return impact

class ImprovementGenerator:
    """Generates potential system improvements."""
    
    def __init__(self):
        """Initialize the improvement generator."""
        self.component_templates = {
            "memory": [
                "Implement {technique} for more efficient memory storage",
                "Optimize memory retrieval using {algorithm}",
                "Enhance entity extraction with {method}",
                "Improve memory compression ratio with {approach}",
                "Add {feature} to memory system for better recall"
            ],
            "retrieval": [
                "Implement {technique} for more accurate retrieval",
                "Optimize query processing with {algorithm}",
                "Add support for {feature} in search results",
                "Enhance ranking algorithm with {method}",
                "Implement {approach} for better semantic matching"
            ],
            "optimization": [
                "Apply {technique} to reduce computational overhead",
                "Implement {algorithm} for faster processing",
                "Use {method} to optimize resource utilization",
                "Apply {approach} to improve system responsiveness",
                "Implement {feature} for better scalability"
            ],
            "architecture": [
                "Refactor {component} for better modularity",
                "Implement {pattern} to improve system flexibility",
                "Restructure {component} using {architecture}",
                "Add {feature} to improve system extensibility",
                "Implement {technique} for better component integration"
            ]
        }
        
        self.techniques = {
            "memory": [
                "hierarchical indexing", "content-based hashing", 
                "delta encoding", "adaptive caching", "bloom filters",
                "prefix trees", "sparse matrix compression"
            ],
            "retrieval": [
                "bi-encoder matching", "approximate nearest neighbors",
                "query expansion", "relevance feedback", "lexical pruning",
                "hybrid ranking", "contextual reranking"
            ],
            "optimization": [
                "lazy evaluation", "memoization", "vectorized operations",
                "batched processing", "asynchronous execution",
                "parallel computation", "incremental updates"
            ],
            "architecture": [
                "mediator pattern", "observer pattern", "dependency injection",
                "microservices", "event sourcing", "CQRS pattern",
                "hexagonal architecture", "layered design"
            ]
        }
        
    def generate_description(self, component: str) -> str:
        """
        Generate an improvement description.
        
        Args:
            component: System component to improve
            
        Returns:
            Improvement description
        """
        # Get templates for component
        templates = self.component_templates.get(
            component, self.component_templates["architecture"]
        )
        
        # Get techniques for component
        techniques = self.techniques.get(
            component, self.techniques["optimization"]
        )
        
        # Select random template and technique
        template = random.choice(templates)
        technique = random.choice(techniques)
        
        # Fill template
        placeholders = ["technique", "algorithm", "method", "approach", "feature", 
                      "component", "pattern", "architecture"]
        
        for placeholder in placeholders:
            if "{" + placeholder + "}" in template:
                template = template.replace("{" + placeholder + "}", technique)
                
        return template
        
    def generate_improvement(self, 
                          system: Any, 
                          evaluator: SystemEvaluator) -> Improvement:
        """
        Generate a potential system improvement.
        
        Args:
            system: Current system state
            evaluator: System evaluator
            
        Returns:
            Generated improvement
        """
        # Select a component to improve
        components = ["memory", "retrieval", "optimization", "architecture"]
        weights = [0.3, 0.3, 0.2, 0.2]  # Higher weights for memory and retrieval
        component = random.choices(components, weights=weights, k=1)[0]
        
        # Generate improvement description
        description = self.generate_description(component)
        
        # Generate placeholder implementation
        implementation = f"# TODO: Implement {description}\n"
        
        # Create improvement
        improvement = Improvement(
            component=component,
            description=description,
            implementation=implementation
        )
        
        # Estimate impact
        impact = evaluator.evaluate_improvement(system, improvement)
        improvement.estimated_impact = impact
        
        return improvement
        
    def generate_improvements(self, 
                           system: Any,
                           evaluator: SystemEvaluator,
                           count: int = 3) -> List[Improvement]:
        """
        Generate multiple potential improvements.
        
        Args:
            system: Current system state
            evaluator: System evaluator
            count: Number of improvements to generate
            
        Returns:
            List of generated improvements
        """
        improvements = []
        
        for _ in range(count):
            improvement = self.generate_improvement(system, evaluator)
            improvements.append(improvement)
            
        return improvements

class ImprovementSelector:
    """Selects the best improvements to apply."""
    
    def __init__(self):
        """Initialize the improvement selector."""
        self.metric_weights = {
            "accuracy": 0.3,
            "efficiency": 0.25,
            "adaptability": 0.2,
            "robustness": 0.15,
            "memory_usage": 0.1
        }
        
    def calculate_score(self, improvement: Improvement) -> float:
        """
        Calculate a score for an improvement.
        
        Args:
            improvement: Improvement to score
            
        Returns:
            Improvement score
        """
        score = 0.0
        impact = improvement.estimated_impact
        
        for metric, weight in self.metric_weights.items():
            if metric in impact:
                # For memory_usage, lower is better
                metric_impact = impact[metric]
                if metric == "memory_usage":
                    metric_impact = -metric_impact
                    
                score += metric_impact * weight
                
        return score
        
    def rank_improvements(self, 
                        improvements: List[Improvement]) -> List[Tuple[Improvement, float]]:
        """
        Rank improvements by score.
        
        Args:
            improvements: List of improvements to rank
            
        Returns:
            List of (improvement, score) tuples, sorted by score
        """
        scored_improvements = []
        
        for improvement in improvements:
            score = self.calculate_score(improvement)
            scored_improvements.append((improvement, score))
            
        # Sort by score (descending)
        scored_improvements.sort(key=lambda x: x[1], reverse=True)
        
        return scored_improvements
        
    def select_improvements(self, 
                         improvements: List[Improvement],
                         max_count: int = 1) -> List[Improvement]:
        """
        Select the best improvements to apply.
        
        Args:
            improvements: List of improvements to select from
            max_count: Maximum number of improvements to select
            
        Returns:
            List of selected improvements
        """
        # Rank improvements
        ranked_improvements = self.rank_improvements(improvements)
        
        # Select top improvements
        selected = []
        for improvement, score in ranked_improvements:
            if len(selected) >= max_count:
                break
                
            # Only select improvements with positive score
            if score > 0:
                selected.append(improvement)
                
        return selected

# Add a new class for LLM-guided evolution
class LLMGuidedEvolution:
    """
    Implements LLM-guided evolution with Evolution of Thought (EoT) technique.
    
    Based on research in "LLM Guided Evolution -- The Automation of Models Advancing Models" (2024),
    this approach uses LLMs to reflect on previous evolution attempts and guide future mutations.
    """
    
    def __init__(self, model_name: str = "meta-llama/Llama-3.2-3B", device: str = "cpu", 
                 memory_size: int = 10, temperature: float = 0.7):
        """
        Initialize the LLM-guided evolution system.
        
        Args:
            model_name: Name of the LLM model to use
            device: Device to run the model on ('cpu' or 'cuda')
            memory_size: Number of past mutations to remember
            temperature: Temperature for LLM generation (higher = more diverse)
        """
        self.model_name = model_name
        self.device = device
        self.memory_size = memory_size
        self.temperature = temperature
        self.evolution_memory = deque(maxlen=memory_size)
        
        # Load the model and tokenizer if available
        try:
            logger.info(f"Loading LLM model {model_name}...")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
            self.llm_available = True
            logger.info(f"LLM model loaded successfully")
        except Exception as e:
            logger.warning(f"Could not load LLM model: {e}")
            logger.warning("Falling back to template-based evolution")
            self.llm_available = False
    
    def record_mutation(self, component: str, mutation: str, metrics_before: Dict[str, float], 
                       metrics_after: Dict[str, float], success: bool):
        """
        Record a mutation and its impact for learning.
        
        Args:
            component: Component that was modified
            mutation: Description of the modification
            metrics_before: Performance metrics before modification
            metrics_after: Performance metrics after modification
            success: Whether the mutation was successful
        """
        impact = {}
        for key in metrics_after:
            if key in metrics_before:
                impact[key] = metrics_after[key] - metrics_before[key]
        
        self.evolution_memory.append({
            "component": component,
            "mutation": mutation,
            "impact": impact,
            "success": success,
            "timestamp": time.time()
        })
        
    def generate_mutation(self, component: str, current_code: str, 
                         target_metrics: List[str]) -> str:
        """
        Generate a mutation using the LLM with Evolution of Thought.
        
        Args:
            component: Component to modify
            current_code: Current implementation
            target_metrics: Metrics to improve
            
        Returns:
            Proposed mutation
        """
        if not self.llm_available:
            return self._template_based_mutation(component, current_code)
        
        # Build the prompt with evolution history
        prompt = self._build_eot_prompt(component, current_code, target_metrics)
        
        # Generate using the LLM
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=2000,
                temperature=self.temperature,
                top_p=0.9,
                repetition_penalty=1.1
            )
            
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract the mutation from the response
        mutation = self._extract_mutation(response)
        return mutation
    
    def _build_eot_prompt(self, component: str, current_code: str, target_metrics: List[str]) -> str:
        """
        Build a prompt for the LLM that includes Evolution of Thought.
        
        Args:
            component: Component to modify
            current_code: Current implementation
            target_metrics: Metrics to improve
            
        Returns:
            Prompt for the LLM
        """
        # Start with a system prompt
        prompt = f"""You are an expert AI evolutionary system that optimizes code.

TASK: You need to improve the following component: {component}

TARGET METRICS TO IMPROVE: {', '.join(target_metrics)}

CURRENT IMPLEMENTATION:
```python
{current_code}
```

"""

        # Add evolution history for Evolution of Thought
        if self.evolution_memory:
            prompt += "\nLEARNINGS FROM PREVIOUS MUTATIONS:\n"
            
            for idx, item in enumerate(self.evolution_memory):
                success_str = "SUCCESSFUL" if item["success"] else "UNSUCCESSFUL"
                impact_str = ", ".join([f"{k}: {v:+.4f}" for k, v in item["impact"].items()])
                
                prompt += f"{idx+1}. [{success_str}] Component: {item['component']}\n"
                prompt += f"   Impact: {impact_str}\n"
                prompt += f"   Mutation: {item['mutation']}\n\n"
                
            prompt += "ANALYSIS OF PREVIOUS MUTATIONS:\n"
            prompt += "1. What patterns do you observe in successful mutations?\n"
            prompt += "2. What should be avoided based on unsuccessful mutations?\n"
            prompt += "3. How can you build upon or combine successful strategies?\n\n"
        
        prompt += """INSTRUCTIONS:
1. Analyze the current implementation and identify areas for improvement
2. Consider learnings from previous mutations (if available)
3. Generate a specific, targeted improvement
4. Explain your reasoning
5. Provide the exact code that should replace the current implementation

YOUR EVOLVED IMPLEMENTATION:
```python
"""
        
        return prompt
    
    def _extract_mutation(self, response: str) -> str:
        """Extract the mutation code from the LLM response."""
        if "```python" in response and "```" in response:
            # Extract code between python code blocks
            code_start = response.find("```python") + 9
            code_end = response.find("```", code_start)
            return response[code_start:code_end].strip()
        
        # Fall back to a simple extraction approach
        lines = response.split('\n')
        in_code_block = False
        code_lines = []
        
        for line in lines:
            if line.strip() == "```python" or line.strip() == "```":
                in_code_block = not in_code_block
                continue
                
            if in_code_block:
                code_lines.append(line)
                
        return "\n".join(code_lines)
    
    def _template_based_mutation(self, component: str, current_code: str) -> str:
        """
        Fallback mutation generator when LLM is not available.
        
        Args:
            component: Component to modify
            current_code: Current implementation
            
        Returns:
            Proposed mutation
        """
        # Very basic template-based mutation for fallback
        return current_code

class SelfEvolutionController:
    """
    Controller for system self-evolution.
    
    This class manages the evolution process, including generating and
    applying improvements, evaluating system performance, and tracking
    evolution metrics.
    """
    
    def __init__(self, system: Any, model_name: str = "meta-llama/Llama-3.2-3B"):
        """
        Initialize the self-evolution controller.
        
        Args:
            system: System to evolve
            model_name: Name of the LLM model to use for guided evolution
        """
        self.system = system
        self.metrics = EvolutionMetrics()
        self.evaluator = SystemEvaluator()
        self.generator = ImprovementGenerator()
        self.selector = ImprovementSelector()
        self.improvements = []
        self.applied_improvements = []
        self.rejected_improvements = []
        
        # Add the LLM-guided evolution component
        self.llm_evolution = LLMGuidedEvolution(model_name=model_name)
        
        # Try to detect device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        logger.info(f"Self-evolution controller initialized (using {self.device})")
        
    def run_evolution_cycle(self, 
                         max_improvements: int = 1,
                         generation_count: int = 5,
                         use_llm_guidance: bool = True) -> Dict:
        """
        Run a single evolution cycle.
        
        Args:
            max_improvements: Maximum number of improvements to apply
            generation_count: Number of candidate improvements to generate
            use_llm_guidance: Whether to use LLM-guided evolution
            
        Returns:
            Dictionary with cycle results
        """
        cycle_start = time.time()
        
        try:
            logger.info("Starting evolution cycle")
            
            # Step 1: Evaluate current system
            current_metrics = self.evaluator.evaluate(self.system)
            logger.info(f"Current system metrics: {current_metrics}")
            
            # Step 2: Generate improvements
            if use_llm_guidance and hasattr(self, 'llm_evolution'):
                logger.info("Using LLM-guided evolution to generate improvements")
                
                # Get code for each component
                components = self._get_system_components()
                
                candidate_improvements = []
                for component_name, component_code in components.items():
                    # Generate an improvement with LLM guidance
                    mutation = self.llm_evolution.generate_mutation(
                        component=component_name,
                        current_code=component_code,
                        target_metrics=list(current_metrics.keys())
                    )
                    
                    improvement = Improvement(
                        component=component_name,
                        description=f"LLM-guided improvement for {component_name}",
                        implementation=mutation
                    )
                    
                    candidate_improvements.append(improvement)
            else:
                # Fall back to original improvement generation logic
                logger.info(f"Generating {generation_count} candidate improvements")
                candidate_improvements = self.generator.generate_improvements(
                    self.system, 
                    self.evaluator,
                    count=generation_count
                )
            
            # Step 3: Select improvements to apply
            logger.info(f"Selecting up to {max_improvements} improvements to apply")
            selected_improvements = self.selector.select_improvements(
                candidate_improvements,
                max_count=max_improvements
            )
            
            # Step 4: Apply improvements
            improvements_applied = 0
            for improvement in selected_improvements:
                logger.info(f"Applying improvement: {improvement.description}")
                
                # Record metrics before applying
                before_metrics = self.evaluator.evaluate(self.system)
                
                # Apply the improvement
                try:
                    # This would implement the actual code changes
                    # For prototype, we just mark it as applied
                    improvement.apply()
                    improvements_applied += 1
                    self.applied_improvements.append(improvement)
                    
                    # Evaluate impact
                    after_metrics = self.evaluator.evaluate(self.system)
                    improvement.record_impact(after_metrics)
                    
                    # Record for Evolution of Thought
                    if hasattr(self, 'llm_evolution'):
                        self.llm_evolution.record_mutation(
                            component=improvement.component,
                            mutation=improvement.implementation,
                            metrics_before=before_metrics,
                            metrics_after=after_metrics,
                            success=True  # Assume success for now
                        )
                        
                    logger.info(f"Improvement applied successfully")
                except Exception as e:
                    logger.error(f"Error applying improvement: {e}")
                    improvement.reject(f"Error during application: {e}")
                    self.rejected_improvements.append(improvement)
                    
                    # Record failed attempt for Evolution of Thought
                    if hasattr(self, 'llm_evolution'):
                        self.llm_evolution.record_mutation(
                            component=improvement.component,
                            mutation=improvement.implementation,
                            metrics_before=before_metrics,
                            metrics_after=before_metrics,  # No change
                            success=False
                        )
            
            # Step 5: Record cycle metrics
            cycle_duration = time.time() - cycle_start
            final_metrics = self.evaluator.evaluate(self.system)
            
            self.metrics.record_cycle(
                metrics=final_metrics,
                duration=cycle_duration,
                improvements=improvements_applied
            )
            
            logger.info(f"Evolution cycle completed in {cycle_duration:.2f} seconds")
            logger.info(f"Applied {improvements_applied} improvements")
            
            return {
                "cycle_duration": cycle_duration,
                "improvements_applied": improvements_applied,
                "metrics_before": current_metrics,
                "metrics_after": final_metrics
            }
            
        except Exception as e:
            cycle_duration = time.time() - cycle_start
            logger.error(f"Error during evolution cycle: {e}")
            logger.error(traceback.format_exc())
            
            return {
                "cycle_duration": cycle_duration,
                "error": str(e),
                "success": False
            }
    
    def _get_system_components(self) -> Dict[str, str]:
        """
        Get the code for each system component.
        
        In a real implementation, this would extract actual code.
        For the prototype, we return dummy code.
        
        Returns:
            Dictionary mapping component names to code
        """
        # This would be implemented to extract actual code from components
        # For prototype, return dummy implementations
        return {
            "memory": "class Memory:\n    def __init__(self):\n        self.data = {}\n",
            "retrieval": "class Retrieval:\n    def search(self, query):\n        return []\n",
            "evolution": "class Evolution:\n    def evolve(self):\n        pass\n"
        }
        
    def run_evolution_cycles(self, 
                          cycles: int = 3,
                          max_improvements_per_cycle: int = 1) -> Dict:
        """
        Run multiple evolution cycles.
        
        Args:
            cycles: Number of evolution cycles to run
            max_improvements_per_cycle: Maximum improvements per cycle
            
        Returns:
            Dictionary with evolution results
        """
        results = []
        
        for _ in range(cycles):
            cycle_result = self.run_evolution_cycle(
                max_improvements=max_improvements_per_cycle
            )
            results.append(cycle_result)
            
        # Get overall metrics
        summary = self.metrics.get_summary()
        
        return {
            "cycles": results,
            "summary": summary
        }
        
    def get_applied_improvements(self) -> List[Improvement]:
        """
        Get list of applied improvements.
        
        Returns:
            List of applied improvements
        """
        return [imp for imp in self.improvements if imp.applied]
        
    def get_rejected_improvements(self) -> List[Improvement]:
        """
        Get list of rejected improvements.
        
        Returns:
            List of rejected improvements
        """
        return [imp for imp in self.improvements if imp.rejected]
        
    def get_improvement_history(self) -> List[Dict]:
        """
        Get history of all improvements.
        
        Returns:
            List of improvement dictionaries
        """
        return [imp.to_dict() for imp in self.improvements]
        
    def save_state(self, path: str):
        """
        Save controller state to file.
        
        Args:
            path: Path to save state
        """
        state = {
            "metrics": self.metrics.to_dict(),
            "improvements": [imp.to_dict() for imp in self.improvements],
            "applied_improvements": self.applied_improvements
        }
        
        with open(path, 'w') as f:
            json.dump(state, f, indent=2)
            
        logger.info(f"Saved evolution state to {path}")
        
    def load_state(self, path: str) -> bool:
        """
        Load controller state from file.
        
        Args:
            path: Path to load state from
            
        Returns:
            True if loaded successfully, False otherwise
        """
        try:
            with open(path, 'r') as f:
                state = json.load(f)
                
            # Load metrics
            if "metrics" in state:
                self.metrics = EvolutionMetrics.from_dict(state["metrics"])
                
            # Load improvements
            if "improvements" in state:
                self.improvements = [
                    Improvement.from_dict(imp_data)
                    for imp_data in state["improvements"]
                ]
                
            # Load applied improvements
            if "applied_improvements" in state:
                self.applied_improvements = state["applied_improvements"]
                
            logger.info(f"Loaded evolution state from {path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load evolution state: {e}")
            logger.error(traceback.format_exc())
            return False

# Example usage
def example_usage():
    """Demonstrate usage of the self-evolution controller."""
    # Create a mock system
    class MockSystem:
        def __init__(self):
            self.components = {
                "memory": {"count": 100},
                "retrieval": {"queries": 0},
                "entity_graph": {"entities": 0}
            }
            
        def get_status(self):
            return {
                "system": {
                    "components": list(self.components.keys())
                },
                "memory": {
                    "volatile_items": 50,
                    "archival_items": 50,
                    "total_items": 100
                }
            }
    
    # Create controller
    system = MockSystem()
    controller = SelfEvolutionController(system)
    
    # Run evolution cycles
    results = controller.run_evolution_cycles(cycles=3, max_improvements_per_cycle=2)
    
    print("\nEvolution Results:")
    print(f"Cycles completed: {results['summary']['cycles_completed']}")
    print(f"Improvements applied: {results['summary']['improvements_applied']}")
    print(f"Average cycle duration: {results['summary']['average_cycle_duration']:.2f}s")
    
    # Print performance trends
    print("\nPerformance Trends:")
    for metric, trend in results['summary']['performance_trends'].items():
        direction = "improved" if trend > 0 else "declined"
        print(f"  {metric}: {direction} by {abs(trend):.4f}")
    
    # Print applied improvements
    print("\nApplied Improvements:")
    for improvement in controller.get_applied_improvements():
        print(f"  - {improvement.description}")
        
    return controller, results

if __name__ == "__main__":
    example_usage() 