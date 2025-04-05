"""
Example usage of the OS1 integration layer.

This module demonstrates how to use the OS1 integration layer to coordinate
memory, retrieval, debate, and evolution components.
"""

import logging
import os
from typing import List, Dict

from ..os1_integration import OS1Integration, OS1Config

def run_example():
    """Run example usage of OS1 integration."""
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("OS1.Example")

    # Initialize with custom config
    config = OS1Config(
        memory_size=5000,
        embedding_dim=512,
        vocab_size=20000,
        compression_ratio=0.2,
        batch_size=16,
        num_debate_rounds=3,
        consensus_threshold=0.7,
        diversity_weight=0.4
    )
    
    os1 = OS1Integration(config)
    
    # Store some example content
    contents = [
        {
            "content": "Neural architecture search enables automated discovery of efficient model architectures",
            "tags": ["neural-networks", "architecture", "automation"]
        },
        {
            "content": "Vector symbolic architectures support compositional reasoning in neural systems",
            "tags": ["vsa", "reasoning", "composition"]
        },
        {
            "content": "Memory compression using neural networks can significantly reduce storage requirements",
            "tags": ["memory", "compression", "optimization"]
        },
        {
            "content": "Self-evolving systems can continuously improve their performance over time",
            "tags": ["evolution", "optimization", "learning"]
        }
    ]
    
    memory_ids = []
    for item in contents:
        memory_id = os1.store(
            content=item["content"],
            tags=item["tags"]
        )
        memory_ids.append(memory_id)
        logger.info(f"Stored content with ID: {memory_id}")
    
    # Try different retrieval strategies
    queries = [
        ("neural architecture", "sparse"),
        ("memory optimization", "dense"),
        ("self-improving systems", "hybrid")
    ]
    
    for query, strategy in queries:
        logger.info(f"\nRetrieval results for '{query}' using {strategy} strategy:")
        results = os1.retrieve(query, strategy=strategy, top_k=2)
        
        for i, result in enumerate(results):
            logger.info(f"{i+1}. Score: {result['score']:.4f}")
            logger.info(f"   Content: {result['content']}")
            if 'tags' in result.get('metadata', {}):
                logger.info(f"   Tags: {result['metadata']['tags']}")
    
    # Start a debate about system architecture
    logger.info("\nStarting debate about neural architectures...")
    debate_id = os1.start_debate(
        topic="optimal neural architectures",
        context=[
            "Neural architecture search can find efficient models",
            "Manual architecture design requires expert knowledge",
            "Evolution can optimize architectures automatically"
        ]
    )
    
    # Add arguments from different perspectives
    os1.contribute_to_debate(
        debate_id=debate_id,
        agent_id="debate",
        claim="Evolutionary search is optimal for architecture discovery",
        evidence=[
            "Can explore large search spaces efficiently",
            "Naturally handles multi-objective optimization"
        ],
        stance="support"
    )
    
    os1.contribute_to_debate(
        debate_id=debate_id,
        agent_id="critic",
        claim="Pure evolutionary search may miss important design principles",
        evidence=[
            "Expert knowledge can guide search effectively",
            "Some architectural patterns are known to work well"
        ],
        stance="oppose"
    )
    
    os1.contribute_to_debate(
        debate_id=debate_id,
        agent_id="synthesis",
        claim="Hybrid approaches combining evolution and expertise are most effective",
        evidence=[
            "Evolution can optimize within expert-defined constraints",
            "Expert knowledge can seed initial population"
        ],
        stance="support"
    )
    
    # Get debate summary
    debate_summary = os1.get_debate_summary(debate_id)
    logger.info("\nDebate Summary:")
    logger.info(f"Topic: {debate_summary.get('topic')}")
    logger.info(f"Participants: {debate_summary.get('participants')}")
    logger.info(f"Message count: {debate_summary.get('message_count')}")
    
    # Trigger evolution cycle
    logger.info("\nTriggering evolution cycle...")
    evolution_results = os1.trigger_evolution()
    
    if evolution_results.get("status") == "success":
        logger.info("Evolution cycle completed successfully")
        
        # Show improvements
        if "components" in evolution_results:
            for component, results in evolution_results["components"].items():
                if results.get("status") == "success":
                    improvements = results.get("improvements", {})
                    logger.info(f"\nImprovements for {component}:")
                    for metric, value in improvements.items():
                        logger.info(f"  {metric}: {value:+.4f}")
    
    # Save system state
    state_path = "os1_state.pt"
    os1.save_state(state_path)
    logger.info(f"\nSystem state saved to {state_path}")
    
    # Get system status
    status = os1.get_system_status()
    logger.info("\nSystem Status:")
    logger.info(f"Memory stats: {status['memory']}")
    logger.info(f"Compression ratio: {status['compression']['ratio']}")
    logger.info(f"Evolution state: {status['evolution'].get('state', 'unknown')}")
    logger.info(f"Active debates: {len(status['debate'].get('active_debates', []))}")

if __name__ == "__main__":
    run_example()