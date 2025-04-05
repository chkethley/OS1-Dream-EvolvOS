"""
OS1 Integration Layer

This module provides the main integration layer for the OS1 system, coordinating between
memory, retrieval, compression, and evolution components.
"""

import logging
import torch
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass

from .enhanced_memory import EnhancedMemoryInterface
from .advanced_retrieval import AdvancedRetrieval, ContrastiveSPLADEEncoder
from .neural_compressor import AdaptiveCompressor
from .self_evolution_controller import SelfEvolutionController
from .enhanced_vsa import EnhancedVSACommunicationBus
from .vector_symbolic_architecture import VSACommunicationBus
from .debate_system import DebateSystem, DebateConfig

logger = logging.getLogger("OS1.Integration")

@dataclass
class OS1Config:
    """Configuration for OS1 system."""
    memory_size: int = 10000
    embedding_dim: int = 768
    vocab_size: int = 30000
    compression_ratio: float = 0.3
    batch_size: int = 32
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_debate_rounds: int = 5
    consensus_threshold: float = 0.8
    diversity_weight: float = 0.3

class OS1Integration:
    """Main integration layer for OS1 system."""
    
    def __init__(self, config: Optional[OS1Config] = None):
        """Initialize OS1 integration layer."""
        self.config = config or OS1Config()
        
        # Initialize VSA communication
        self.vsa_bus = EnhancedVSACommunicationBus()
        
        # Initialize memory components
        self.memory_interface = self._init_memory()
        
        # Initialize retrieval
        self.retrieval = self._init_retrieval()
        
        # Initialize compression
        self.compressor = self._init_compression()
        
        # Initialize debate system
        self.debate = self._init_debate()
        
        # Initialize evolution controller
        self.evolution = self._init_evolution()
        
        # Initialize cleanup scheduling
        self.cleanup_interval = 3600  # 1 hour
        self.last_cleanup = time.time()
        self.cleanup_scheduled = False
        
        logger.info("OS1 Integration layer initialized")

    def _init_memory(self) -> EnhancedMemoryInterface:
        """Initialize memory subsystem."""
        retrieval = AdvancedRetrieval(
            vocab_size=self.config.vocab_size,
            embedding_dim=self.config.embedding_dim,
            device=self.config.device
        )
        return EnhancedMemoryInterface(retrieval)

    def _init_retrieval(self) -> AdvancedRetrieval:
        """Initialize retrieval subsystem."""
        encoder = ContrastiveSPLADEEncoder(
            vocab_size=self.config.vocab_size,
            hidden_dim=self.config.embedding_dim,
            output_dim=self.config.embedding_dim,
            device=self.config.device
        )
        return AdvancedRetrieval(
            vocab_size=self.config.vocab_size,
            embedding_dim=self.config.embedding_dim,
            device=self.config.device
        )

    def _init_compression(self) -> AdaptiveCompressor:
        """Initialize compression subsystem."""
        return AdaptiveCompressor(
            input_dim=self.config.embedding_dim,
            min_compression_ratio=self.config.compression_ratio / 2,
            max_compression_ratio=self.config.compression_ratio * 2
        )

    def _init_debate(self) -> DebateSystem:
        """Initialize debate system."""
        debate_config = DebateConfig(
            num_rounds=self.config.num_debate_rounds,
            consensus_threshold=self.config.consensus_threshold,
            diversity_weight=self.config.diversity_weight
        )
        return DebateSystem(config=debate_config)

    def _init_evolution(self) -> SelfEvolutionController:
        """Initialize evolution controller."""
        config = {
            "memory_input_shape": (self.config.embedding_dim,),
            "retrieval_input_shape": (self.config.embedding_dim,),
            "nas_population_size": 10,
            "nas_evolution_cycles": 5,
            "batch_size": self.config.batch_size,
            "device": self.config.device
        }
        return SelfEvolutionController(
            memory_system=self.memory_interface,
            retrieval_system=self.retrieval,
            debate_system=self.debate,  # Add debate system
            config=config
        )

    def _schedule_cleanup(self):
        """Schedule periodic cleanup of memory resources."""
        current_time = time.time()
        if not self.cleanup_scheduled and (current_time - self.last_cleanup) > self.cleanup_interval:
            self._perform_cleanup()
            self.last_cleanup = current_time
            
    def _perform_cleanup(self):
        """Perform thorough cleanup of memory resources."""
        try:
            # Clean up memory system
            if hasattr(self.memory_interface, 'cleanup'):
                self.memory_interface.cleanup()
                
            # Clean up retrieval system
            if hasattr(self.retrieval, 'cleanup'):
                self.retrieval.cleanup()
                
            # Clean up compressor
            if hasattr(self.compressor, 'cleanup'):
                self.compressor.cleanup()
                
            # Clean up GPU memory if available
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            logger.info("Successfully performed system-wide cleanup")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")
        finally:
            self.cleanup_scheduled = False

    def store(self, content: str, metadata: Optional[Dict] = None, tags: Optional[List[str]] = None) -> str:
        """Store content in memory with optional metadata and tags."""
        self._schedule_cleanup()  # Check if cleanup is needed
        
        # First compress if needed
        if len(content) > 1000:  # Threshold for compression
            embedding = self.retrieval.encoder.get_contrastive_embedding(content)
            compressed = self.compressor.compress(torch.tensor(embedding))
            metadata = metadata or {}
            metadata["compressed"] = True
            metadata["compression_id"] = id(compressed)

        # Store in memory
        memory_id = self.memory_interface.store(content, metadata=metadata, tags=tags)
        
        # Index for retrieval
        self.retrieval.index_content(content, memory_id, metadata)
        
        return memory_id

    def retrieve(self, 
                query: str, 
                strategy: str = "hybrid", 
                top_k: int = 5,
                provide_feedback: bool = True) -> List[Dict]:
        """
        Retrieve relevant content using specified strategy.
        
        Args:
            query: Search query
            strategy: Search strategy (hybrid, sparse, dense)
            top_k: Number of results to return
            provide_feedback: Whether to provide feedback for evolution
            
        Returns:
            List of retrieval results
        """
        self._schedule_cleanup()  # Check if cleanup is needed
        results = self.memory_interface.search(query, top_k=top_k, strategy=strategy)
        
        if provide_feedback:
            # Provide feedback to evolution system
            for result in results:
                self.evolution.handle_retrieval_feedback(
                    query=query,
                    memory_id=result["memory_id"],
                    score=result["score"]
                )
        
        return results

    def trigger_evolution(self) -> Dict:
        """Trigger system evolution cycle."""
        try:
            evolution_results = self.evolution.trigger_evolution_cycle()
            
            # Apply any architecture updates
            if "memory" in evolution_results.get("components", {}):
                memory_results = evolution_results["components"]["memory"]
                if memory_results.get("architecture"):
                    self._update_memory_architecture(memory_results["architecture"])
                    
            if "retrieval" in evolution_results.get("components", {}):
                retrieval_results = evolution_results["components"]["retrieval"]
                if retrieval_results.get("architecture"):
                    self._update_retrieval_architecture(retrieval_results["architecture"])
            
            return evolution_results
            
        except Exception as e:
            logger.error(f"Evolution cycle failed: {str(e)}")
            return {"status": "error", "error": str(e)}

    def _update_memory_architecture(self, architecture: Dict):
        """Update memory system architecture."""
        try:
            self.memory_interface.update_architecture(architecture)
            logger.info("Memory architecture updated successfully")
        except Exception as e:
            logger.error(f"Failed to update memory architecture: {str(e)}")

    def _update_retrieval_architecture(self, architecture: Dict):
        """Update retrieval system architecture."""
        try:
            self.retrieval.update_architecture(architecture)
            logger.info("Retrieval architecture updated successfully")
        except Exception as e:
            logger.error(f"Failed to update retrieval architecture: {str(e)}")

    def start_debate(self, topic: str, context: Optional[List[str]] = None) -> str:
        """Start a new debate on a topic."""
        if context is None:
            # Retrieve relevant context from memory
            results = self.retrieve(topic, strategy="hybrid", top_k=3)
            context = [r["content"] for r in results]
            
        debate_id = self.debate.start_debate(topic, context)
        logger.info(f"Started debate {debate_id} on topic: {topic}")
        return debate_id

    def get_debate_summary(self, debate_id: str) -> Dict:
        """Get summary of debate progress and conclusions."""
        return self.debate.get_debate_summary(debate_id)

    def contribute_to_debate(self, 
                           debate_id: str,
                           claim: str,
                           evidence: List[str],
                           agent_id: str = "debate",
                           stance: str = "support") -> bool:
        """Add a contribution to an ongoing debate."""
        return self.debate.contribute_argument(
            debate_id=debate_id,
            agent_id=agent_id,
            claim=claim,
            evidence=evidence,
            stance=stance
        )

    def save_state(self, path: str):
        """Save system state."""
        state = {
            "memory": self.memory_interface.get_statistics(),
            "retrieval": self.retrieval.get_statistics(),
            "evolution": self.evolution.get_evolution_status(),
            "debate": self.debate.get_statistics(),
            "config": self.config.__dict__
        }
        torch.save(state, path)
        logger.info(f"System state saved to {path}")

    def load_state(self, path: str):
        """Load system state."""
        try:
            state = torch.load(path)
            self.config.__dict__.update(state["config"])
            self.evolution.load_state(state["evolution"])
            logger.info(f"System state loaded from {path}")
        except Exception as e:
            logger.error(f"Failed to load state: {str(e)}")

    def get_system_status(self) -> Dict:
        """Get overall system status."""
        return {
            "memory": self.memory_interface.get_statistics(),
            "retrieval": self.retrieval.get_statistics(),
            "debate": self.debate.get_statistics(),
            "evolution": self.evolution.get_evolution_status(),
            "compression": {
                "ratio": self.config.compression_ratio,
                "compressor_status": "initialized" if self.compressor else "not_initialized"
            }
        }

    def __del__(self):
        """Ensure cleanup on deletion."""
        self._perform_cleanup()