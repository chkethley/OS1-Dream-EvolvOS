#!/usr/bin/env python3
"""
EvolvOS: Recursive Self-Evolving AI System
Main Entry Point

This script serves as the main entry point for the EvolvOS system,
integrating all components into a cohesive self-evolving AI framework.
"""

import os
import sys
import argparse
import json
import logging
from typing import Dict, List, Any, Optional, Tuple
import time
import traceback

# Import system components
try:
    from enhanced_memory import HierarchicalMemory, EnhancedMemory, EntityRelationshipGraph
    from splade_retrieval import SpladeEncoder, SpladeIndex, EnhancedRetrieval
    # Import as needed when files are created
    # from neural_compressor import NeuralCompressor, DynamicMemoryPool
    # from vector_symbolic_architecture import VSAEncoder
    # from bayesian_optimizer import BayesianOptimizer, EvolvOSOptimizer
    # from self_evolution_controller import SelfEvolutionController
except ImportError as e:
    print(f"Error importing components: {e}")
    print("Make sure all required modules are in the same directory or PYTHONPATH.")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("evolvos.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("EvolvOS")

class EvolvOS:
    """
    Main class for the EvolvOS system.
    
    This class integrates all components of the self-evolving AI system
    and provides a unified interface for interaction.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the EvolvOS system.
        
        Args:
            config_path: Path to configuration file (optional)
        """
        self.components = {}
        self.config = self._load_config(config_path)
        self.initialized = False
        
        # System metadata
        self.version = "0.1.0"
        self.start_time = time.time()
        
        logger.info(f"Initializing EvolvOS v{self.version}")
        
    def _load_config(self, config_path: Optional[str]) -> Dict:
        """
        Load configuration from file or use defaults.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Configuration dictionary
        """
        default_config = {
            "memory": {
                "volatile_capacity": 1000,
                "archival_enabled": True,
                "entity_tracking": True
            },
            "retrieval": {
                "splade_model": "naver/splade-cocondenser-ensembledistil",
                "index_path": "splade_index",
                "default_strategy": "hybrid"
            },
            "evolution": {
                "auto_evolve": False,
                "evolution_cycles": 3,
                "evaluation_metrics": ["accuracy", "efficiency", "adaptability"]
            },
            "system": {
                "log_level": "INFO",
                "debug_mode": False,
                "save_state_interval": 3600  # seconds
            }
        }
        
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    user_config = json.load(f)
                    # Deep merge configs
                    config = self._merge_configs(default_config, user_config)
                    logger.info(f"Loaded configuration from {config_path}")
                    return config
            except Exception as e:
                logger.error(f"Error loading config from {config_path}: {e}")
                logger.warning("Falling back to default configuration")
                return default_config
        else:
            if config_path:
                logger.warning(f"Config file {config_path} not found, using defaults")
            return default_config
            
    def _merge_configs(self, default: Dict, user: Dict) -> Dict:
        """
        Recursively merge two configuration dictionaries.
        
        Args:
            default: Default configuration
            user: User configuration (overrides defaults)
            
        Returns:
            Merged configuration
        """
        result = default.copy()
        
        for key, value in user.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_configs(result[key], value)
            else:
                result[key] = value
                
        return result
        
    def initialize(self):
        """Initialize all system components based on configuration."""
        try:
            # Initialize memory system
            logger.info("Initializing memory system...")
            memory_config = self.config["memory"]
            
            self.components["memory"] = EnhancedMemory(
                volatile_capacity=memory_config["volatile_capacity"],
                with_archival=memory_config["archival_enabled"]
            )
            
            if memory_config["entity_tracking"]:
                self.components["entity_graph"] = EntityRelationshipGraph()
                
            # Initialize retrieval system
            logger.info("Initializing retrieval system...")
            retrieval_config = self.config["retrieval"]
            
            self.components["splade_encoder"] = SpladeEncoder(
                model_name_or_path=retrieval_config["splade_model"]
            )
            
            self.components["splade_index"] = SpladeIndex(
                index_path=retrieval_config["index_path"],
                encoder=self.components["splade_encoder"],
                create_if_missing=True
            )
            
            self.components["retrieval"] = EnhancedRetrieval(
                index=self.components["splade_index"],
                default_strategy=retrieval_config["default_strategy"]
            )
            
            # Initialize additional components when implemented
            # TODO: Add neural compression when implemented
            # TODO: Add VSA encoder when implemented
            # TODO: Add Bayesian optimizer when implemented
            # TODO: Add self-evolution controller when implemented
            
            self.initialized = True
            logger.info("EvolvOS system initialization complete")
            
        except Exception as e:
            logger.error(f"Error during initialization: {e}")
            logger.error(traceback.format_exc())
            raise
            
    def store_memory(self, content: str, metadata: Optional[Dict] = None) -> str:
        """
        Store content in the memory system.
        
        Args:
            content: Content to store
            metadata: Additional metadata (optional)
            
        Returns:
            Memory ID
        """
        if not self.initialized:
            self.initialize()
            
        memory = self.components["memory"]
        memory_id = memory.store(content, metadata or {})
        
        # Extract and store entities if entity tracking is enabled
        if "entity_graph" in self.components:
            entities = self._extract_entities(content)
            entity_graph = self.components["entity_graph"]
            
            for entity in entities:
                entity_graph.add_entity(entity["name"], entity["type"])
                entity_graph.link_entity_to_memory(entity["name"], memory_id)
                
        return memory_id
        
    def _extract_entities(self, content: str) -> List[Dict]:
        """
        Extract entities from content.
        
        This is a placeholder for a more sophisticated entity extraction system.
        
        Args:
            content: Text content
            
        Returns:
            List of entity dictionaries
        """
        # Placeholder implementation
        # In a real system, this would use NER or other techniques
        entities = []
        common_entity_types = {
            "algorithm": ["algorithm", "method", "technique", "approach"],
            "concept": ["concept", "principle", "theory", "framework"],
            "system": ["system", "architecture", "platform", "infrastructure"],
            "component": ["component", "module", "function", "class"]
        }
        
        words = content.split()
        for i, word in enumerate(words):
            if len(word) > 5 and word[0].isupper():  # Simple heuristic
                # Try to determine entity type
                context = " ".join(words[max(0, i-3):min(len(words), i+4)])
                entity_type = "unknown"
                
                for type_name, type_markers in common_entity_types.items():
                    if any(marker in context.lower() for marker in type_markers):
                        entity_type = type_name
                        break
                        
                entities.append({
                    "name": word,
                    "type": entity_type
                })
                
        return entities
        
    def retrieve(self, query: str, strategy: Optional[str] = None, top_k: int = 5) -> List[Dict]:
        """
        Retrieve information from memory.
        
        Args:
            query: Search query
            strategy: Retrieval strategy ('keyword', 'semantic', 'hybrid')
            top_k: Number of results to return
            
        Returns:
            List of retrieved items
        """
        if not self.initialized:
            self.initialize()
            
        memory = self.components["memory"]
        retrieval_config = self.config["retrieval"]
        actual_strategy = strategy or retrieval_config["default_strategy"]
        
        return memory.search(query, strategy=actual_strategy, top_k=top_k)
        
    def evolve_system(self, target_component: Optional[str] = None):
        """
        Trigger system evolution.
        
        Args:
            target_component: Specific component to evolve (optional)
        """
        if not self.initialized:
            self.initialize()
            
        # This is a placeholder for when the self-evolution controller is implemented
        logger.info(f"System evolution triggered for component: {target_component or 'all'}")
        logger.warning("Self-evolution controller not yet implemented")
        
    def save_state(self, path: Optional[str] = None):
        """
        Save the current system state.
        
        Args:
            path: Directory to save state (optional)
        """
        if not self.initialized:
            return
            
        save_dir = path or "evolvos_state"
        os.makedirs(save_dir, exist_ok=True)
        
        # Save memory state
        if "memory" in self.components:
            memory_path = os.path.join(save_dir, "memory_state.json")
            self.components["memory"].save(memory_path)
            
        # Save entity graph if available
        if "entity_graph" in self.components:
            graph_path = os.path.join(save_dir, "entity_graph.json")
            self.components["entity_graph"].save(graph_path)
            
        # Save index if available
        if "splade_index" in self.components:
            index_path = os.path.join(save_dir, "splade_index")
            self.components["splade_index"].save(index_path)
            
        # Save system metadata
        meta_path = os.path.join(save_dir, "system_meta.json")
        with open(meta_path, 'w') as f:
            json.dump({
                "version": self.version,
                "uptime_seconds": time.time() - self.start_time,
                "config": self.config,
                "components": list(self.components.keys()),
                "timestamp": time.time()
            }, f, indent=2)
            
        logger.info(f"System state saved to {save_dir}")
        
    def load_state(self, path: str):
        """
        Load system state from saved files.
        
        Args:
            path: Directory with saved state
        """
        if not os.path.exists(path):
            logger.error(f"State directory {path} does not exist")
            return False
            
        try:
            # First load system metadata to check compatibility
            meta_path = os.path.join(path, "system_meta.json")
            if os.path.exists(meta_path):
                with open(meta_path, 'r') as f:
                    metadata = json.load(f)
                    
                if metadata.get("version") != self.version:
                    logger.warning(f"Version mismatch: saved={metadata.get('version')}, current={self.version}")
                    
                # Update config with saved config
                self.config = self._merge_configs(self.config, metadata.get("config", {}))
                
            # Initialize the system
            if not self.initialized:
                self.initialize()
                
            # Load memory state
            memory_path = os.path.join(path, "memory_state.json")
            if os.path.exists(memory_path) and "memory" in self.components:
                self.components["memory"].load(memory_path)
                
            # Load entity graph if available
            graph_path = os.path.join(path, "entity_graph.json")
            if os.path.exists(graph_path) and "entity_graph" in self.components:
                self.components["entity_graph"].load(graph_path)
                
            # Load index if available
            index_path = os.path.join(path, "splade_index")
            if os.path.exists(index_path) and "splade_index" in self.components:
                self.components["splade_index"].load(index_path)
                
            logger.info(f"System state loaded from {path}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading state from {path}: {e}")
            logger.error(traceback.format_exc())
            return False
            
    def get_status(self) -> Dict:
        """
        Get the current system status.
        
        Returns:
            Status dictionary
        """
        status = {
            "system": {
                "version": self.version,
                "uptime_seconds": time.time() - self.start_time,
                "initialized": self.initialized,
                "components": list(self.components.keys())
            }
        }
        
        # Add memory stats if available
        if "memory" in self.components:
            memory = self.components["memory"]
            status["memory"] = {
                "volatile_items": memory.volatile_memory.count(),
                "archival_items": memory.archival_memory.count() if memory.archival_memory else 0,
                "total_items": memory.count()
            }
            
        # Add entity stats if available
        if "entity_graph" in self.components:
            entity_graph = self.components["entity_graph"]
            status["entities"] = {
                "count": entity_graph.count_entities(),
                "types": entity_graph.get_entity_types()
            }
            
        return status
        
    def run_demo(self):
        """Run a demonstration of the system's capabilities."""
        if not self.initialized:
            self.initialize()
            
        logger.info("Running EvolvOS demonstration...")
        
        # Store some sample memories
        memory_ids = []
        sample_texts = [
            "The Recursive Self-Improvement methodology enables AI systems to modify and enhance their own algorithms.",
            "Hierarchical memory systems combine fast volatile memory with slower but more permanent archival storage.",
            "SPLADE is a technique for sparse retrieval that utilizes learned token weights for efficient search.",
            "The Vector-Symbolic Architecture allows for compositional representations in high-dimensional spaces.",
            "Neural compression agents can reduce memory footprint while preserving semantic information.",
            "Bayesian optimization provides an efficient approach for tuning hyperparameters in complex systems.",
            "Multi-agent debate frameworks enable diverse perspectives and more robust decision-making."
        ]
        
        print("\nStoring sample memories...")
        for i, text in enumerate(sample_texts):
            memory_id = self.store_memory(text, {"source": "demo", "importance": i / len(sample_texts)})
            memory_ids.append(memory_id)
            print(f"  Stored memory {i+1}: {text[:50]}... (ID: {memory_id})")
            
        # Demonstrate retrieval
        print("\nDemonstrating memory retrieval:")
        queries = [
            "self-improvement in AI",
            "memory systems",
            "efficient search techniques",
            "neural compression"
        ]
        
        for query in queries:
            print(f"\nQuery: {query}")
            results = self.retrieve(query, top_k=2)
            
            for i, result in enumerate(results):
                print(f"  Result {i+1} (score: {result['score']:.2f}):")
                print(f"  {result['content']}")
                
        # Print system status
        print("\nSystem Status:")
        status = self.get_status()
        for section, data in status.items():
            print(f"  {section.capitalize()}:")
            for key, value in data.items():
                print(f"    {key}: {value}")
                
        print("\nDemo complete!")

def main():
    """Main entry point for the EvolvOS CLI."""
    parser = argparse.ArgumentParser(description="EvolvOS: Recursive Self-Evolving AI System")
    
    # Main commands
    commands = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Initialize command
    init_cmd = commands.add_parser("init", help="Initialize the system")
    init_cmd.add_argument("--config", help="Path to configuration file")
    
    # Run command
    run_cmd = commands.add_parser("run", help="Run the system")
    run_cmd.add_argument("--config", help="Path to configuration file")
    run_cmd.add_argument("--state", help="Load state from directory")
    
    # Demo command
    demo_cmd = commands.add_parser("demo", help="Run a demonstration")
    demo_cmd.add_argument("--config", help="Path to configuration file")
    
    # Store command
    store_cmd = commands.add_parser("store", help="Store content in memory")
    store_cmd.add_argument("--content", help="Content to store")
    store_cmd.add_argument("--file", help="File containing content to store")
    store_cmd.add_argument("--config", help="Path to configuration file")
    
    # Retrieve command
    retrieve_cmd = commands.add_parser("retrieve", help="Retrieve from memory")
    retrieve_cmd.add_argument("query", help="Query to search for")
    retrieve_cmd.add_argument("--strategy", choices=["keyword", "semantic", "hybrid"], help="Retrieval strategy")
    retrieve_cmd.add_argument("--top-k", type=int, default=5, help="Number of results to return")
    retrieve_cmd.add_argument("--config", help="Path to configuration file")
    
    # Status command
    status_cmd = commands.add_parser("status", help="Show system status")
    status_cmd.add_argument("--config", help="Path to configuration file")
    
    # Save/load commands
    save_cmd = commands.add_parser("save", help="Save system state")
    save_cmd.add_argument("--path", help="Directory to save state")
    save_cmd.add_argument("--config", help="Path to configuration file")
    
    load_cmd = commands.add_parser("load", help="Load system state")
    load_cmd.add_argument("path", help="Directory with saved state")
    load_cmd.add_argument("--config", help="Path to configuration file")
    
    # Evolve command
    evolve_cmd = commands.add_parser("evolve", help="Trigger system evolution")
    evolve_cmd.add_argument("--component", help="Specific component to evolve")
    evolve_cmd.add_argument("--config", help="Path to configuration file")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Create system instance
    system = EvolvOS(config_path=getattr(args, "config", None))
    
    # Execute command
    if args.command == "init":
        system.initialize()
        print("System initialized successfully")
        
    elif args.command == "run":
        if hasattr(args, "state") and args.state:
            system.load_state(args.state)
        system.initialize()
        print("System running... (Press Ctrl+C to exit)")
        try:
            # Simple event loop
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nSaving state before exit...")
            system.save_state()
            print("Exiting")
            
    elif args.command == "demo":
        system.run_demo()
        
    elif args.command == "store":
        content = None
        if hasattr(args, "content") and args.content:
            content = args.content
        elif hasattr(args, "file") and args.file:
            with open(args.file, 'r') as f:
                content = f.read()
        
        if content:
            memory_id = system.store_memory(content)
            print(f"Stored content with ID: {memory_id}")
        else:
            print("Error: No content provided")
            
    elif args.command == "retrieve":
        results = system.retrieve(
            args.query,
            strategy=getattr(args, "strategy", None),
            top_k=getattr(args, "top_k", 5)
        )
        
        print(f"Results for query: {args.query}")
        for i, result in enumerate(results):
            print(f"\nResult {i+1} (score: {result['score']:.2f}):")
            print(f"ID: {result['id']}")
            print(f"Content: {result['content']}")
            
    elif args.command == "status":
        status = system.get_status()
        print("System Status:")
        for section, data in status.items():
            print(f"\n{section.capitalize()}:")
            for key, value in data.items():
                print(f"  {key}: {value}")
                
    elif args.command == "save":
        path = getattr(args, "path", None)
        system.save_state(path)
        print(f"System state saved to {path or 'evolvos_state'}")
        
    elif args.command == "load":
        if hasattr(args, "path") and args.path:
            success = system.load_state(args.path)
            if success:
                print(f"System state loaded from {args.path}")
            else:
                print(f"Failed to load state from {args.path}")
        else:
            print("Error: No path provided")
            
    elif args.command == "evolve":
        component = getattr(args, "component", None)
        system.evolve_system(component)
        print(f"Evolution triggered for component: {component or 'all'}")
        
    else:
        parser.print_help()

if __name__ == "__main__":
    main() 