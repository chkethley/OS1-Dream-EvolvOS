"""
Neural Architecture Evolution

This module implements neural architecture evolution for EvolvOS, enabling the system
to automatically discover and optimize neural network architectures through evolutionary algorithms.
"""

import os
import sys
import time
import json
import logging
import random
import uuid
import copy
from typing import Dict, List, Any, Optional, Tuple, Callable, Set
import numpy as np

logger = logging.getLogger("EvolvOS.NeuralEvolution")

class ModelBlock:
    """Represents a modular block in a neural network architecture."""
    
    def __init__(self, 
                 block_type: str,
                 config: Dict[str, Any],
                 name: Optional[str] = None):
        """
        Initialize a model block.
        
        Args:
            block_type: Type of neural network block (conv, linear, attention, etc.)
            config: Configuration parameters for the block
            name: Optional name for the block
        """
        self.id = str(uuid.uuid4())[:8]
        self.block_type = block_type
        self.config = config
        self.name = name or f"{block_type}_{self.id}"
        self.inputs = []  # IDs of input blocks
        self.outputs = []  # IDs of output blocks
        
    def to_dict(self) -> Dict:
        """Convert block to dictionary representation."""
        return {
            "id": self.id,
            "type": self.block_type,
            "name": self.name,
            "config": self.config,
            "inputs": self.inputs,
            "outputs": self.outputs
        }
        
    @classmethod
    def from_dict(cls, data: Dict) -> 'ModelBlock':
        """Create block from dictionary representation."""
        block = cls(
            block_type=data["type"],
            config=data["config"],
            name=data["name"]
        )
        block.id = data["id"]
        block.inputs = data["inputs"]
        block.outputs = data["outputs"]
        return block
    
    def mutate(self, mutation_rate: float = 0.2) -> 'ModelBlock':
        """
        Create a mutated copy of this block.
        
        Args:
            mutation_rate: Probability of mutating each parameter
            
        Returns:
            A new ModelBlock with mutated parameters
        """
        new_config = copy.deepcopy(self.config)
        
        # For each parameter in the config
        for key, value in self.config.items():
            # Decide whether to mutate this parameter
            if random.random() < mutation_rate:
                if isinstance(value, int):
                    # For integers, adjust by a small amount
                    new_config[key] = max(1, value + random.randint(-2, 2))
                elif isinstance(value, float):
                    # For floats, adjust by a percentage
                    factor = 1.0 + (random.random() - 0.5) * 0.4  # ±20%
                    new_config[key] = value * factor
                elif isinstance(value, bool):
                    # For booleans, flip with some probability
                    new_config[key] = not value
                elif isinstance(value, str) and key == "activation":
                    # For activation functions, select from common options
                    options = ["relu", "gelu", "silu", "tanh", "sigmoid"]
                    new_config[key] = random.choice(options)
        
        # Create a new block with the mutated config
        mutated = ModelBlock(
            block_type=self.block_type,
            config=new_config,
            name=f"{self.name}_mutated"
        )
        mutated.inputs = self.inputs.copy()
        mutated.outputs = self.outputs.copy()
        
        return mutated
    
    def generate_code(self) -> str:
        """
        Generate PyTorch code for this block.
        
        Returns:
            Python code string for implementing this block
        """
        code = ""
        
        if self.block_type == "linear":
            in_features = self.config.get("in_features", 128)
            out_features = self.config.get("out_features", 128)
            bias = self.config.get("bias", True)
            activation = self.config.get("activation", "relu")
            
            code += f"# {self.name}: Linear layer\n"
            code += f"self.{self.name} = nn.Linear({in_features}, {out_features}, bias={bias})\n"
            
            if activation:
                code += f"self.{self.name}_act = nn.{activation.capitalize()}()\n"
                
        elif self.block_type == "conv":
            in_channels = self.config.get("in_channels", 32)
            out_channels = self.config.get("out_channels", 64)
            kernel_size = self.config.get("kernel_size", 3)
            stride = self.config.get("stride", 1)
            padding = self.config.get("padding", 1)
            
            code += f"# {self.name}: Convolutional layer\n"
            code += (f"self.{self.name} = nn.Conv2d({in_channels}, {out_channels}, "
                    f"kernel_size={kernel_size}, stride={stride}, padding={padding})\n")
            
        elif self.block_type == "attention":
            dim = self.config.get("dim", 128)
            heads = self.config.get("heads", 4)
            dim_head = self.config.get("dim_head", 32)
            
            code += f"# {self.name}: Multi-head attention\n"
            code += f"self.{self.name} = MultiHeadAttention(dim={dim}, heads={heads}, dim_head={dim_head})\n"
            
        elif self.block_type == "dropout":
            rate = self.config.get("rate", 0.1)
            code += f"# {self.name}: Dropout layer\n"
            code += f"self.{self.name} = nn.Dropout(p={rate})\n"
            
        return code
    
    def __str__(self) -> str:
        """String representation of the block."""
        return f"{self.name} ({self.block_type}): {self.config}"

class ModelArchitecture:
    """Represents a complete neural network architecture."""
    
    def __init__(self, name: Optional[str] = None):
        """
        Initialize a model architecture.
        
        Args:
            name: Optional name for the architecture
        """
        self.id = str(uuid.uuid4())[:8]
        self.name = name or f"Model_{self.id}"
        self.blocks: Dict[str, ModelBlock] = {}  # Dict of blocks by ID
        self.input_blocks: List[str] = []  # IDs of input blocks
        self.output_blocks: List[str] = []  # IDs of output blocks
        self.fitness: Optional[float] = None  # Evaluated fitness
        self.eval_metrics: Dict[str, float] = {}  # Detailed evaluation metrics
        
    def add_block(self, block: ModelBlock) -> str:
        """
        Add a block to the architecture.
        
        Args:
            block: The block to add
            
        Returns:
            ID of the added block
        """
        if block.id in self.blocks:
            # Create a new ID if this ID is already used
            block.id = str(uuid.uuid4())[:8]
            block.name = f"{block.block_type}_{block.id}"
            
        self.blocks[block.id] = block
        return block.id
    
    def connect_blocks(self, source_id: str, target_id: str):
        """
        Connect two blocks in the architecture.
        
        Args:
            source_id: ID of the source block
            target_id: ID of the target block
        """
        if source_id not in self.blocks or target_id not in self.blocks:
            raise ValueError(f"Cannot connect: blocks {source_id} or {target_id} not found")
        
        # Update the outputs of the source block
        if target_id not in self.blocks[source_id].outputs:
            self.blocks[source_id].outputs.append(target_id)
            
        # Update the inputs of the target block
        if source_id not in self.blocks[target_id].inputs:
            self.blocks[target_id].inputs.append(source_id)
    
    def set_as_input(self, block_id: str):
        """Mark a block as an input to the model."""
        if block_id not in self.blocks:
            raise ValueError(f"Block {block_id} not found")
        
        if block_id not in self.input_blocks:
            self.input_blocks.append(block_id)
    
    def set_as_output(self, block_id: str):
        """Mark a block as an output from the model."""
        if block_id not in self.blocks:
            raise ValueError(f"Block {block_id} not found")
        
        if block_id not in self.output_blocks:
            self.output_blocks.append(block_id)
    
    def get_block_order(self) -> List[str]:
        """
        Determine a valid order to process blocks for forward pass.
        
        Returns:
            List of block IDs in processing order
        """
        # Implementation of topological sort
        visited = set()
        temp = set()
        order = []
        
        def visit(block_id):
            if block_id in temp:
                raise ValueError("Cycle detected in architecture")
            if block_id in visited:
                return
            
            temp.add(block_id)
            
            # Visit all dependencies (inputs) first
            for input_id in self.blocks[block_id].inputs:
                visit(input_id)
                
            temp.remove(block_id)
            visited.add(block_id)
            order.append(block_id)
        
        # Start with input blocks
        for input_id in self.input_blocks:
            if input_id not in visited:
                visit(input_id)
                
        # Process any remaining blocks
        for block_id in self.blocks:
            if block_id not in visited:
                visit(block_id)
                
        return order
    
    def validate(self) -> bool:
        """
        Validate the architecture for consistency.
        
        Returns:
            True if the architecture is valid
        """
        try:
            # Check that we have input and output blocks
            if not self.input_blocks:
                logger.error("No input blocks defined")
                return False
                
            if not self.output_blocks:
                logger.error("No output blocks defined")
                return False
                
            # Check for cycles
            self.get_block_order()
            
            # Check that all referenced blocks exist
            for block_id, block in self.blocks.items():
                for input_id in block.inputs:
                    if input_id not in self.blocks:
                        logger.error(f"Block {block_id} references non-existent input {input_id}")
                        return False
                        
                for output_id in block.outputs:
                    if output_id not in self.blocks:
                        logger.error(f"Block {block_id} references non-existent output {output_id}")
                        return False
            
            return True
            
        except Exception as e:
            logger.error(f"Validation error: {e}")
            return False
    
    def generate_code(self) -> str:
        """
        Generate PyTorch code for this architecture.
        
        Returns:
            Python code string for implementing this model
        """
        if not self.validate():
            raise ValueError("Cannot generate code for invalid architecture")
            
        # Get blocks in topological order
        block_order = self.get_block_order()
        
        model_code = f"""
# Auto-generated model: {self.name}
import torch
import torch.nn as nn
import torch.nn.functional as F

class {self.name.replace('-', '_')}(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Layer definitions
"""
        
        # Add block definitions
        for block_id in block_order:
            block = self.blocks[block_id]
            model_code += "        " + block.generate_code().replace("\n", "\n        ") + "\n"
            
        # Create forward method
        model_code += """
    def forward(self, x):
        # Store intermediate activations
        activations = {}
"""
        
        # Add forward pass
        for block_id in block_order:
            block = self.blocks[block_id]
            
            # Handle input blocks
            if block_id in self.input_blocks:
                model_code += f"        activations['{block_id}'] = x\n"
                continue
                
            # Determine inputs for this block
            inputs = block.inputs
            
            if not inputs:
                # Skip blocks with no inputs (except input blocks which were handled above)
                continue
                
            if len(inputs) == 1:
                # Single input
                input_expr = f"activations['{inputs[0]}']"
            else:
                # Multiple inputs, concatenate
                input_expr = f"torch.cat([activations['{input_id}'] for input_id in {inputs}], dim=1)"
                
            # Apply block
            model_code += f"        # Apply {block.name}\n"
            if block.block_type == "linear" and block.config.get("activation"):
                model_code += (f"        activations['{block_id}'] = self.{block.name}_act("
                              f"self.{block.name}({input_expr}))\n")
            else:
                model_code += f"        activations['{block_id}'] = self.{block.name}({input_expr})\n"
                
        # Return output
        if len(self.output_blocks) == 1:
            model_code += f"\n        return activations['{self.output_blocks[0]}']\n"
        else:
            out_list = ", ".join([f"activations['{block_id}']" for block_id in self.output_blocks])
            model_code += f"\n        return {out_list}\n"
            
        return model_code
    
    def to_dict(self) -> Dict:
        """Convert architecture to dictionary representation."""
        return {
            "id": self.id,
            "name": self.name,
            "blocks": {bid: block.to_dict() for bid, block in self.blocks.items()},
            "input_blocks": self.input_blocks,
            "output_blocks": self.output_blocks,
            "fitness": self.fitness,
            "eval_metrics": self.eval_metrics
        }
        
    @classmethod
    def from_dict(cls, data: Dict) -> 'ModelArchitecture':
        """Create architecture from dictionary representation."""
        arch = cls(name=data.get("name"))
        arch.id = data["id"]
        
        # Load blocks
        for bid, block_data in data["blocks"].items():
            arch.blocks[bid] = ModelBlock.from_dict(block_data)
            
        arch.input_blocks = data["input_blocks"]
        arch.output_blocks = data["output_blocks"]
        arch.fitness = data.get("fitness")
        arch.eval_metrics = data.get("eval_metrics", {})
        
        return arch
    
    def mutate(self, mutation_rate: float = 0.3) -> 'ModelArchitecture':
        """
        Create a mutated copy of this architecture.
        
        Args:
            mutation_rate: Probability of mutations
            
        Returns:
            A new mutated ModelArchitecture
        """
        mutated = ModelArchitecture(name=f"{self.name}_mutated")
        
        # Copy and potentially mutate existing blocks
        for bid, block in self.blocks.items():
            if random.random() < mutation_rate:
                # Mutate this block
                mutated_block = block.mutate()
                mutated.blocks[bid] = mutated_block
            else:
                # Keep the original block
                mutated.blocks[bid] = copy.deepcopy(block)
                
        # Potentially add a new block
        if random.random() < mutation_rate:
            new_block = self._generate_random_block()
            new_id = mutated.add_block(new_block)
            
            # Connect to a random existing block
            if mutated.blocks:
                existing_id = random.choice(list(mutated.blocks.keys()))
                if random.random() < 0.5:
                    # New block comes after existing block
                    mutated.connect_blocks(existing_id, new_id)
                else:
                    # New block comes before existing block
                    mutated.connect_blocks(new_id, existing_id)
        
        # Potentially remove a block (but not input/output blocks)
        removable_blocks = [bid for bid in self.blocks 
                          if bid not in self.input_blocks and bid not in self.output_blocks]
        if removable_blocks and random.random() < mutation_rate:
            block_to_remove = random.choice(removable_blocks)
            
            # Connect the inputs of this block to its outputs
            block = mutated.blocks[block_to_remove]
            for input_id in block.inputs:
                for output_id in block.outputs:
                    mutated.connect_blocks(input_id, output_id)
                    
            # Remove the block
            mutated.blocks.pop(block_to_remove)
            
        # Copy input and output blocks
        mutated.input_blocks = self.input_blocks.copy()
        mutated.output_blocks = self.output_blocks.copy()
        
        return mutated
    
    def _generate_random_block(self) -> ModelBlock:
        """Generate a random block for mutation operations."""
        block_types = ["linear", "dropout", "attention"]
        block_type = random.choice(block_types)
        
        if block_type == "linear":
            config = {
                "in_features": random.choice([64, 128, 256, 512]),
                "out_features": random.choice([64, 128, 256, 512]),
                "bias": random.choice([True, False]),
                "activation": random.choice(["relu", "gelu", "tanh", None])
            }
        elif block_type == "dropout":
            config = {
                "rate": random.uniform(0.1, 0.5)
            }
        elif block_type == "attention":
            dim = random.choice([64, 128, 256])
            config = {
                "dim": dim,
                "heads": random.choice([2, 4, 8]),
                "dim_head": dim // random.choice([2, 4])
            }
            
        return ModelBlock(block_type=block_type, config=config)
    
    def __str__(self) -> str:
        """String representation of the architecture."""
        return (f"{self.name}: {len(self.blocks)} blocks, "
                f"{len(self.input_blocks)} inputs, {len(self.output_blocks)} outputs")

class ZeroCostEvaluator:
    """Evaluates model architectures using zero-cost proxies without training."""
    
    def __init__(self, input_shape: Tuple[int, ...]):
        """
        Initialize evaluator.
        
        Args:
            input_shape: Shape of input tensor (batch_size, ...)
        """
        self.input_shape = input_shape
        
    def evaluate(self, architecture: ModelArchitecture) -> Dict[str, float]:
        """
        Evaluate an architecture using zero-cost proxies.
        
        Args:
            architecture: Model architecture to evaluate
            
        Returns:
            Dictionary of evaluation metrics
        """
        try:
            # This is a simulated evaluation since implementing true zero-cost
            # metrics would require actual PyTorch models
            
            # In a real implementation, we would analyze:
            # - Parameter count / model size
            # - FLOPs / compute requirements
            # - NASWOT score (gradient-free complexity measure)
            # - Synflow score (connectivity proxy)
            
            metrics = {}
            
            # Count blocks by type
            block_types = {}
            for block in architecture.blocks.values():
                block_types[block.block_type] = block_types.get(block.block_type, 0) + 1
                
            # Estimate parameter count
            param_count = 0
            for block in architecture.blocks.values():
                if block.block_type == "linear":
                    in_feat = block.config.get("in_features", 128)
                    out_feat = block.config.get("out_features", 128)
                    has_bias = block.config.get("bias", True)
                    params = in_feat * out_feat
                    if has_bias:
                        params += out_feat
                    param_count += params
                elif block.block_type == "attention":
                    dim = block.config.get("dim", 128)
                    heads = block.config.get("heads", 4)
                    dim_head = block.config.get("dim_head", 32)
                    # Simplified estimate
                    params = dim * heads * dim_head * 4
                    param_count += params
                    
            metrics["parameter_count"] = param_count
            
            # Calculate model depth (longest path from input to output)
            # TODO: Implement path length calculation
            
            # Model complexity (a simple heuristic)
            complexity = len(architecture.blocks) * np.log(param_count + 1)
            metrics["complexity"] = complexity
            
            # Diversity of block types
            metrics["block_type_diversity"] = len(block_types) / max(1, len(architecture.blocks))
            
            # Connectivity (average number of connections per block)
            total_connections = sum(len(block.inputs) + len(block.outputs) 
                                   for block in architecture.blocks.values())
            metrics["connectivity"] = total_connections / max(1, len(architecture.blocks))
            
            # Calculate a fitness score from the metrics
            # This is a simplified version - in practice this would be more sophisticated
            fitness = (100000 / (param_count + 1)) * metrics["block_type_diversity"] * metrics["connectivity"]
            
            # Cap the fitness
            fitness = min(100, max(0, fitness))
            metrics["fitness"] = fitness
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error evaluating architecture: {e}")
            return {"fitness": 0, "error": str(e)}

class NeuralArchitectureEvolution:
    """Manages evolution of neural network architectures."""
    
    def __init__(self, 
                 input_shape: Tuple[int, ...],
                 population_size: int = 10,
                 evolution_cycles: int = 10):
        """
        Initialize the neural architecture evolution.
        
        Args:
            input_shape: Input tensor shape (batch_size, ...)
            population_size: Size of the population
            evolution_cycles: Number of evolution cycles to run
        """
        self.input_shape = input_shape
        self.population_size = population_size
        self.evolution_cycles = evolution_cycles
        self.population: List[ModelArchitecture] = []
        self.evaluator = ZeroCostEvaluator(input_shape)
        self.best_architecture: Optional[ModelArchitecture] = None
        self.best_fitness: float = 0
        self.history: List[Dict] = []
        
    def initialize_population(self):
        """Initialize a random population of architectures."""
        self.population = []
        
        for i in range(self.population_size):
            arch = self._generate_random_architecture(f"Model_{i}")
            self.population.append(arch)
            
    def _generate_random_architecture(self, name: str) -> ModelArchitecture:
        """
        Generate a random model architecture.
        
        Args:
            name: Name for the architecture
            
        Returns:
            A random ModelArchitecture
        """
        arch = ModelArchitecture(name=name)
        
        # Create input block
        input_block = ModelBlock(
            block_type="input",
            config={"shape": self.input_shape}
        )
        input_id = arch.add_block(input_block)
        arch.set_as_input(input_id)
        
        # Create some hidden layers
        num_hidden = random.randint(2, 5)
        hidden_ids = []
        
        prev_id = input_id
        for i in range(num_hidden):
            if random.random() < 0.7:
                # Linear layer
                hidden_block = ModelBlock(
                    block_type="linear",
                    config={
                        "in_features": random.choice([64, 128, 256]),
                        "out_features": random.choice([64, 128, 256]),
                        "bias": True,
                        "activation": random.choice(["relu", "gelu"])
                    }
                )
            else:
                # Attention layer
                dim = random.choice([64, 128, 256])
                hidden_block = ModelBlock(
                    block_type="attention",
                    config={
                        "dim": dim,
                        "heads": random.choice([2, 4, 8]),
                        "dim_head": dim // random.choice([2, 4])
                    }
                )
                
            hidden_id = arch.add_block(hidden_block)
            arch.connect_blocks(prev_id, hidden_id)
            hidden_ids.append(hidden_id)
            
            # Sometimes add a dropout after the layer
            if random.random() < 0.3:
                dropout_block = ModelBlock(
                    block_type="dropout",
                    config={"rate": random.uniform(0.1, 0.3)}
                )
                dropout_id = arch.add_block(dropout_block)
                arch.connect_blocks(hidden_id, dropout_id)
                prev_id = dropout_id
            else:
                prev_id = hidden_id
        
        # Create output block
        output_block = ModelBlock(
            block_type="linear",
            config={
                "in_features": random.choice([64, 128, 256]),
                "out_features": random.choice([10, 1]), # Target size
                "bias": True,
                "activation": None
            }
        )
        output_id = arch.add_block(output_block)
        arch.connect_blocks(prev_id, output_id)
        arch.set_as_output(output_id)
        
        return arch
    
    def evaluate_population(self):
        """Evaluate all architectures in the population."""
        for arch in self.population:
            metrics = self.evaluator.evaluate(arch)
            arch.fitness = metrics["fitness"]
            arch.eval_metrics = metrics
            
            if arch.fitness > self.best_fitness:
                self.best_fitness = arch.fitness
                self.best_architecture = arch
                
    def select_parents(self, count: int = 2) -> List[ModelArchitecture]:
        """
        Select parents for reproduction using tournament selection.
        
        Args:
            count: Number of parents to select
            
        Returns:
            List of selected parent architectures
        """
        parents = []
        
        for _ in range(count):
            # Tournament selection (pick 3 random candidates, choose the best)
            tournament_size = min(3, len(self.population))
            candidates = random.sample(self.population, tournament_size)
            candidates.sort(key=lambda x: x.fitness or 0, reverse=True)
            parents.append(candidates[0])
            
        return parents
    
    def crossover(self, parents: List[ModelArchitecture]) -> ModelArchitecture:
        """
        Create a child architecture by combining elements from parents.
        
        Args:
            parents: List of parent architectures
            
        Returns:
            A new child architecture
        """
        if len(parents) < 2:
            # Not enough parents, return a copy of the first one
            return copy.deepcopy(parents[0])
        
        parent1, parent2 = parents[0], parents[1]
        child = ModelArchitecture(name=f"Child_{parent1.name}_{parent2.name}")
        
        # Start with input blocks from parent1
        for block_id in parent1.input_blocks:
            block = copy.deepcopy(parent1.blocks[block_id])
            child.add_block(block)
            child.set_as_input(block.id)
            
        # Copy blocks randomly from either parent
        # This is a simplified crossover - in practice, we'd need to ensure
        # that the resulting architecture makes sense
        non_input_blocks1 = [bid for bid in parent1.blocks if bid not in parent1.input_blocks]
        non_input_blocks2 = [bid for bid in parent2.blocks if bid not in parent2.input_blocks]
        
        # Take some blocks from each parent
        blocks_to_copy1 = random.sample(non_input_blocks1, min(len(non_input_blocks1), 
                                                              len(non_input_blocks1) // 2))
        blocks_to_copy2 = random.sample(non_input_blocks2, min(len(non_input_blocks2),
                                                              len(non_input_blocks2) // 2))
        
        # Copy selected blocks from parent1
        for block_id in blocks_to_copy1:
            block = copy.deepcopy(parent1.blocks[block_id])
            child.add_block(block)
            
        # Copy selected blocks from parent2
        for block_id in blocks_to_copy2:
            block = copy.deepcopy(parent2.blocks[block_id])
            child.add_block(block)
            
        # Copy connections from both parents (if both blocks exist in child)
        for parent in [parent1, parent2]:
            for block_id, block in parent.blocks.items():
                if block_id in child.blocks:
                    for output_id in block.outputs:
                        if output_id in child.blocks:
                            child.connect_blocks(block_id, output_id)
                            
        # Add an output block
        output_block = ModelBlock(
            block_type="linear",
            config={
                "in_features": 128,  # This might need adjustment
                "out_features": 10,  # This might need adjustment
                "bias": True,
                "activation": None
            }
        )
        output_id = child.add_block(output_block)
        child.set_as_output(output_id)
        
        # Connect a random block to the output
        if child.blocks:
            random_id = random.choice(list(child.blocks.keys()))
            if random_id != output_id:  # Avoid self-connection
                child.connect_blocks(random_id, output_id)
                
        return child
    
    def evolve_generation(self):
        """Evolve the population by one generation."""
        # Evaluate current population
        self.evaluate_population()
        
        # Sort population by fitness
        self.population.sort(key=lambda x: x.fitness or 0, reverse=True)
        
        # Keep track of best architecture
        if self.population[0].fitness > self.best_fitness:
            self.best_fitness = self.population[0].fitness
            self.best_architecture = self.population[0]
            
        # Record history
        avg_fitness = sum(arch.fitness or 0 for arch in self.population) / len(self.population)
        history_entry = {
            "generation": len(self.history),
            "best_fitness": self.best_fitness,
            "avg_fitness": avg_fitness,
            "population_size": len(self.population)
        }
        self.history.append(history_entry)
        
        # Create new population
        new_population = []
        
        # Elitism: Keep the best architecture
        new_population.append(self.population[0])
        
        # Fill the rest with children
        while len(new_population) < self.population_size:
            # Select parents
            parents = self.select_parents(2)
            
            # Create child through crossover
            child = self.crossover(parents)
            
            # Mutate child
            mutated_child = child.mutate()
            
            # Add to new population
            new_population.append(mutated_child)
            
        # Replace population
        self.population = new_population
        
    def run_evolution(self):
        """Run the complete evolution process."""
        # Initialize population
        self.initialize_population()
        
        # Run evolution cycles
        for cycle in range(self.evolution_cycles):
            self.evolve_generation()
            
            # Log progress
            logger.info(f"Evolution cycle {cycle}: "
                      f"Best fitness = {self.best_fitness:.4f}, "
                      f"Avg fitness = {self.history[-1]['avg_fitness']:.4f}")
            
        # Final evaluation
        self.evaluate_population()
        
        return self.best_architecture
    
    def get_top_architectures(self, count: int = 3) -> List[ModelArchitecture]:
        """
        Get the top performing architectures.
        
        Args:
            count: Number of top architectures to return
            
        Returns:
            List of top architectures
        """
        # Sort by fitness
        sorted_pop = sorted(self.population, key=lambda x: x.fitness or 0, reverse=True)
        return sorted_pop[:count]
    
    def save_evolution_state(self, path: str):
        """
        Save the current state of evolution.
        
        Args:
            path: Path to save state
        """
        state = {
            "input_shape": self.input_shape,
            "population_size": self.population_size,
            "evolution_cycles": self.evolution_cycles,
            "best_fitness": self.best_fitness,
            "history": self.history,
            "population": [arch.to_dict() for arch in self.population],
            "best_architecture": self.best_architecture.to_dict() if self.best_architecture else None
        }
        
        with open(path, 'w') as f:
            json.dump(state, f, indent=2)
            
    def load_evolution_state(self, path: str) -> bool:
        """
        Load evolution state from file.
        
        Args:
            path: Path to load state from
            
        Returns:
            True if loaded successfully
        """
        try:
            with open(path, 'r') as f:
                state = json.load(f)
                
            self.input_shape = tuple(state["input_shape"])
            self.population_size = state["population_size"]
            self.evolution_cycles = state["evolution_cycles"]
            self.best_fitness = state["best_fitness"]
            self.history = state["history"]
            
            # Load population
            self.population = []
            for arch_data in state["population"]:
                arch = ModelArchitecture.from_dict(arch_data)
                self.population.append(arch)
                
            # Load best architecture
            if state["best_architecture"]:
                self.best_architecture = ModelArchitecture.from_dict(state["best_architecture"])
                
            return True
                
        except Exception as e:
            logger.error(f"Error loading evolution state: {e}")
            return False

def example_usage():
    """Example usage of the neural architecture evolution."""
    
    # Define input shape (batch_size, channels, height, width)
    input_shape = (32, 3, 32, 32)  # For example, CIFAR-10 images
    
    # Create evolution manager
    evolver = NeuralArchitectureEvolution(
        input_shape=input_shape,
        population_size=20,
        evolution_cycles=5
    )
    
    # Run evolution
    best_arch = evolver.run_evolution()
    
    print(f"Best architecture: {best_arch}")
    print(f"Fitness: {best_arch.fitness}")
    print(f"Metrics: {best_arch.eval_metrics}")
    
    # Generate code for the best architecture
    model_code = best_arch.generate_code()
    print("\nGenerated model code:")
    print(model_code)
    
    # Save evolution state
    evolver.save_evolution_state("evolution_state.json")
    
if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Run example
    example_usage() 