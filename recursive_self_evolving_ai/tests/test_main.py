#!/usr/bin/env python3
"""
Tests for the EvolvOS main system.
"""

import os
import sys
import unittest
import tempfile
import json
from unittest.mock import patch, MagicMock

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the main module
try:
    from main import EvolvOS
except ImportError:
    print("Could not import EvolvOS. Make sure you're running tests from the project root.")
    sys.exit(1)

class TestEvolvOS(unittest.TestCase):
    """Test cases for the EvolvOS main system."""
    
    def setUp(self):
        """Set up test environment before each test."""
        # Create a temporary config file
        self.config_file = tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.json')
        config = {
            "memory": {
                "volatile_capacity": 10,
                "archival_enabled": True,
                "entity_tracking": True
            },
            "retrieval": {
                "splade_model": "test-model",
                "index_path": "test-index",
                "default_strategy": "hybrid"
            },
            "system": {
                "log_level": "ERROR"
            }
        }
        json.dump(config, self.config_file)
        self.config_file.close()
        
    def tearDown(self):
        """Clean up after each test."""
        os.unlink(self.config_file.name)
        
    @patch('main.EnhancedMemory')
    @patch('main.EntityRelationshipGraph')
    @patch('main.SpladeEncoder')
    @patch('main.SpladeIndex')
    @patch('main.EnhancedRetrieval')
    def test_initialization(self, mock_retrieval, mock_index, mock_encoder, 
                          mock_entity_graph, mock_memory):
        """Test that the system initializes correctly."""
        # Mock objects
        mock_memory_instance = MagicMock()
        mock_memory.return_value = mock_memory_instance
        
        mock_entity_graph_instance = MagicMock()
        mock_entity_graph.return_value = mock_entity_graph_instance
        
        mock_encoder_instance = MagicMock()
        mock_encoder.return_value = mock_encoder_instance
        
        mock_index_instance = MagicMock()
        mock_index.return_value = mock_index_instance
        
        mock_retrieval_instance = MagicMock()
        mock_retrieval.return_value = mock_retrieval_instance
        
        # Create EvolvOS instance
        system = EvolvOS(config_path=self.config_file.name)
        
        # Initialize the system
        system.initialize()
        
        # Check that the components were created
        self.assertTrue(system.initialized)
        self.assertIn("memory", system.components)
        self.assertIn("entity_graph", system.components)
        self.assertIn("splade_encoder", system.components)
        self.assertIn("splade_index", system.components)
        self.assertIn("retrieval", system.components)
        
        # Check that the constructors were called with the right arguments
        mock_memory.assert_called_once_with(
            volatile_capacity=10,
            with_archival=True
        )
        
        mock_entity_graph.assert_called_once()
        
        mock_encoder.assert_called_once_with(
            model_name_or_path="test-model"
        )
        
        mock_index.assert_called_once_with(
            index_path="test-index",
            encoder=mock_encoder_instance,
            create_if_missing=True
        )
        
        mock_retrieval.assert_called_once_with(
            index=mock_index_instance,
            default_strategy="hybrid"
        )
        
    @patch('main.EnhancedMemory')
    @patch('main.EntityRelationshipGraph')
    @patch('main.SpladeEncoder')
    @patch('main.SpladeIndex')
    @patch('main.EnhancedRetrieval')
    def test_store_memory(self, mock_retrieval, mock_index, mock_encoder, 
                        mock_entity_graph, mock_memory):
        """Test storing content in memory."""
        # Mock objects
        mock_memory_instance = MagicMock()
        mock_memory_instance.store.return_value = "memory123"
        mock_memory.return_value = mock_memory_instance
        
        mock_entity_graph_instance = MagicMock()
        mock_entity_graph.return_value = mock_entity_graph_instance
        
        # Create and initialize EvolvOS instance
        system = EvolvOS(config_path=self.config_file.name)
        system.components["memory"] = mock_memory_instance
        system.components["entity_graph"] = mock_entity_graph_instance
        system.initialized = True
        
        # Store content
        content = "Test content with Entity mentioned"
        metadata = {"source": "test"}
        
        # Mock _extract_entities to return a test entity
        system._extract_entities = MagicMock(return_value=[
            {"name": "Entity", "type": "concept"}
        ])
        
        # Store the memory
        memory_id = system.store_memory(content, metadata)
        
        # Check results
        self.assertEqual(memory_id, "memory123")
        mock_memory_instance.store.assert_called_once_with(content, metadata)
        
        # Check entity handling
        system._extract_entities.assert_called_once_with(content)
        mock_entity_graph_instance.add_entity.assert_called_once_with("Entity", "concept")
        mock_entity_graph_instance.link_entity_to_memory.assert_called_once_with(
            "Entity", "memory123"
        )
        
    @patch('main.EnhancedMemory')
    @patch('main.SpladeEncoder')
    @patch('main.SpladeIndex')
    @patch('main.EnhancedRetrieval')
    def test_retrieve(self, mock_retrieval, mock_index, mock_encoder, mock_memory):
        """Test retrieving content from memory."""
        # Mock objects
        mock_memory_instance = MagicMock()
        mock_results = [
            {"id": "memory1", "content": "Result 1", "score": 0.9},
            {"id": "memory2", "content": "Result 2", "score": 0.8}
        ]
        mock_memory_instance.search.return_value = mock_results
        mock_memory.return_value = mock_memory_instance
        
        # Create and initialize EvolvOS instance
        system = EvolvOS(config_path=self.config_file.name)
        system.components["memory"] = mock_memory_instance
        system.initialized = True
        
        # Retrieve content
        query = "test query"
        results = system.retrieve(query, strategy="semantic", top_k=2)
        
        # Check results
        self.assertEqual(results, mock_results)
        mock_memory_instance.search.assert_called_once_with(
            query, strategy="semantic", top_k=2
        )
        
    def test_config_loading(self):
        """Test that configuration is loaded correctly."""
        system = EvolvOS(config_path=self.config_file.name)
        
        # Check that config was loaded
        self.assertEqual(system.config["memory"]["volatile_capacity"], 10)
        self.assertEqual(system.config["retrieval"]["splade_model"], "test-model")
        self.assertEqual(system.config["system"]["log_level"], "ERROR")
        
    def test_config_merging(self):
        """Test that configuration merging works correctly."""
        default_config = {
            "section1": {
                "param1": "default1",
                "param2": "default2"
            },
            "section2": {
                "param3": "default3"
            }
        }
        
        user_config = {
            "section1": {
                "param1": "user1"
            },
            "section3": {
                "param4": "user4"
            }
        }
        
        system = EvolvOS()
        merged = system._merge_configs(default_config, user_config)
        
        # Check merged config
        self.assertEqual(merged["section1"]["param1"], "user1")  # Overridden
        self.assertEqual(merged["section1"]["param2"], "default2")  # Preserved
        self.assertEqual(merged["section2"]["param3"], "default3")  # Preserved
        self.assertEqual(merged["section3"]["param4"], "user4")  # Added
        
if __name__ == "__main__":
    unittest.main() 