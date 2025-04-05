"""
Test suite for OS1 integration layer.
"""

import pytest
import torch
import logging
from typing import List, Dict
import tempfile
import os

from .os1_integration import OS1Integration, OS1Config

@pytest.fixture
def os1():
    """Create OS1 integration instance for testing."""
    try:
        config = OS1Config(
            memory_size=1000,
            embedding_dim=128,
            vocab_size=1000,
            compression_ratio=0.2,
            batch_size=8
        )
        instance = OS1Integration(config=config)
        yield instance
    except Exception as e:
        logging.error(f"Failed to create OS1 instance: {str(e)}")
        raise

def test_init(os1):
    """Test initialization of OS1 integration."""
    try:
        assert os1.memory_interface is not None
        assert os1.retrieval is not None
        assert os1.compressor is not None
        assert os1.evolution is not None
        assert os1.vsa_bus is not None
    except AssertionError as e:
        logging.error(f"Initialization test failed: {str(e)}")
        raise

def test_store_and_retrieve(os1):
    """Test storing and retrieving content."""
    try:
        content = "Test content for storage and retrieval"
        tags = ["test", "storage"]
        
        # Store content
        memory_id = os1.store(content, tags=tags)
        assert memory_id is not None
        
        # Retrieve using different strategies
        strategies = ["sparse", "dense", "hybrid"]
        for strategy in strategies:
            results = os1.retrieve("test storage", strategy=strategy, top_k=1)
            assert len(results) > 0
            assert results[0]["content"] == content
            assert "score" in results[0]
    except Exception as e:
        logging.error(f"Store and retrieve test failed: {str(e)}")
        raise

def test_compression(os1):
    """Test content compression."""
    try:
        long_content = "Long test content " * 100  # Create content > 1000 chars
        
        # Store with compression
        memory_id = os1.store(long_content)
        assert memory_id is not None
        
        # Retrieve and verify
        results = os1.retrieve("long test", top_k=1)
        assert len(results) > 0
        assert results[0]["content"] == long_content
        assert results[0].get("metadata", {}).get("compressed") is True
    except Exception as e:
        logging.error(f"Compression test failed: {str(e)}")
        raise

def test_evolution_cycle(os1):
    """Test evolution cycle."""
    try:
        # Store some content first
        contents = [
            "First test content for evolution",
            "Second test content for system improvement",
            "Third test content about neural architectures"
        ]
        
        for content in contents:
            os1.store(content)
        
        # Perform some retrievals to generate feedback
        queries = ["test content", "neural", "system"]
        for query in queries:
            os1.retrieve(query, provide_feedback=True)
        
        # Trigger evolution
        results = os1.trigger_evolution()
        assert results is not None
        assert "status" in results
        
        if results["status"] == "success":
            assert "components" in results
            assert len(results["components"]) > 0
    except Exception as e:
        logging.error(f"Evolution cycle test failed: {str(e)}")
        raise

def test_system_status(os1):
    """Test system status reporting."""
    try:
        status = os1.get_system_status()
        
        assert "memory" in status
        assert "retrieval" in status
        assert "evolution" in status
        assert "compression" in status
        
        assert isinstance(status["compression"]["ratio"], float)
        assert status["compression"]["compressor_status"] == "initialized"
    except Exception as e:
        logging.error(f"System status test failed: {str(e)}")
        raise

def test_state_save_load(os1):
    """Test saving and loading system state."""
    try:
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as tmp:
            tmp_path = tmp.name
            
        try:
            # Store some content
            content = "Test content for state saving"
            memory_id = os1.store(content)
            
            # Save state
            os1.save_state(tmp_path)
            assert os.path.exists(tmp_path)
            
            # Create new instance and load state
            new_os1 = OS1Integration()
            new_os1.load_state(tmp_path)
            
            # Verify config was loaded
            assert new_os1.config.embedding_dim == os1.config.embedding_dim
            
            # Verify retrieval still works
            results = new_os1.retrieve("test content", top_k=1)
            assert len(results) > 0
        finally:
            # Cleanup
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    except Exception as e:
        logging.error(f"State save/load test failed: {str(e)}")
        raise

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    pytest.main([__file__])