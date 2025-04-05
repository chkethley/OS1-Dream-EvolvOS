"""
Enhanced Hierarchical Memory System for OS1

This module extends the base HierarchicalMemory from the self-evolving-prototype
with advanced retrieval capabilities using SPLADE sparse vectors.
"""

import time
import uuid
from typing import Dict, List, Any, Optional, Tuple
from collections import OrderedDict
import sys
import os
import importlib.util
import torch
import glob
import logging

logger = logging.getLogger(__name__)

# Load the self_evolving_prototype module from the file
spec = importlib.util.spec_from_file_location("self_evolving_prototype", 
                                             os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                                        "self-evolving-prototype.py"))
self_evolving_prototype = importlib.util.module_from_spec(spec)
spec.loader.exec_module(self_evolving_prototype)

# Import the necessary classes
HierarchicalMemory = self_evolving_prototype.HierarchicalMemory
VolatileMemory = self_evolving_prototype.VolatileMemory
CompressedMemory = self_evolving_prototype.CompressedMemory

# Import SPLADE retrieval
from splade_retrieval import EnhancedRetrieval

class EntityNode:
    """Node in entity relationship graph."""
    
    def __init__(self, entity_id: str, name: str, entity_type: str):
        self.id = entity_id
        self.name = name
        self.type = entity_type
        self.properties = {}
        self.relationships = {}  # {relation_type -> [target_entity_id]}
        self.memory_references = []  # Memory IDs related to this entity

    def add_relationship(self, relation_type: str, target_entity_id: str):
        """Add a relationship to another entity."""
        if relation_type not in self.relationships:
            self.relationships[relation_type] = []
        
        if target_entity_id not in self.relationships[relation_type]:
            self.relationships[relation_type].append(target_entity_id)

    def add_memory_reference(self, memory_id: str):
        """Add a reference to a memory item."""
        if memory_id not in self.memory_references:
            self.memory_references.append(memory_id)

    def to_dict(self) -> Dict:
        """Convert to dictionary representation."""
        return {
            "id": self.id,
            "name": self.name,
            "type": self.type,
            "properties": self.properties,
            "relationships": self.relationships,
            "memory_references": self.memory_references
        }

class EntityRelationshipGraph:
    """Object-centric memory structure representing entities and their relationships."""
    
    def __init__(self):
        self.entities = {}  # {entity_id -> EntityNode}
    
    def create_entity(self, name: str, entity_type: str) -> str:
        """Create a new entity and return its ID."""
        entity_id = str(uuid.uuid4())
        self.entities[entity_id] = EntityNode(entity_id, name, entity_type)
        return entity_id
    
    def add_relationship(self, source_id: str, relation_type: str, target_id: str) -> bool:
        """Add a relationship between two entities."""
        if source_id not in self.entities or target_id not in self.entities:
            return False
        
        self.entities[source_id].add_relationship(relation_type, target_id)
        return True
    
    def associate_memory(self, entity_id: str, memory_id: str) -> bool:
        """Associate a memory item with an entity."""
        if entity_id not in self.entities:
            return False
        
        self.entities[entity_id].add_memory_reference(memory_id)
        return True
    
    def get_entity_by_name(self, name: str, entity_type: Optional[str] = None) -> List[EntityNode]:
        """Find entities by name and optional type."""
        results = []
        
        for entity in self.entities.values():
            if entity.name.lower() == name.lower():
                if entity_type is None or entity.type == entity_type:
                    results.append(entity)
                    
        return results
    
    def get_related_entities(self, entity_id: str, relation_type: Optional[str] = None) -> List[EntityNode]:
        """Get entities related to the specified entity."""
        if entity_id not in self.entities:
            return []
        
        entity = self.entities[entity_id]
        related_ids = []
        
        if relation_type is not None:
            # Get entities with specific relationship
            related_ids = entity.relationships.get(relation_type, [])
        else:
            # Get all related entities
            for ids in entity.relationships.values():
                related_ids.extend(ids)
        
        # Return unique entities
        return [self.entities[e_id] for e_id in set(related_ids) if e_id in self.entities]
    
    def get_entity_memories(self, entity_id: str) -> List[str]:
        """Get memory IDs associated with an entity."""
        if entity_id not in self.entities:
            return []
        
        return self.entities[entity_id].memory_references

class ArchivalMemory:
    """Long-term structured storage organized hierarchically."""
    
    def __init__(self):
        self.archives = {}  # {archive_id -> archive}
        self.categories = {}  # {category_name -> [archive_id]}
        self.memory_location = {}  # {memory_id -> archive_id}
    
    def create_archive(self, name: str, category: str, description: str = "") -> str:
        """Create a new archive for storing related memories."""
        archive_id = str(uuid.uuid4())
        
        archive = {
            "id": archive_id,
            "name": name,
            "category": category,
            "description": description,
            "created_at": time.time(),
            "memories": [],
            "metadata": {}
        }
        
        self.archives[archive_id] = archive
        
        # Add to category index
        if category not in self.categories:
            self.categories[category] = []
        
        self.categories[category].append(archive_id)
        
        return archive_id
    
    def add_memory(self, archive_id: str, memory_id: str, metadata: Optional[Dict] = None) -> bool:
        """Add a memory to an archive."""
        if archive_id not in self.archives:
            return False
        
        # Add to archive
        self.archives[archive_id]["memories"].append({
            "memory_id": memory_id,
            "added_at": time.time(),
            "metadata": metadata or {}
        })
        
        # Update location index
        self.memory_location[memory_id] = archive_id
        
        return True
    
    def get_archive(self, archive_id: str) -> Optional[Dict]:
        """Get archive by ID."""
        return self.archives.get(archive_id)
    
    def find_archives_by_category(self, category: str) -> List[Dict]:
        """Find archives by category."""
        archive_ids = self.categories.get(category, [])
        return [self.archives[a_id] for a_id in archive_ids if a_id in self.archives]
    
    def find_archive_for_memory(self, memory_id: str) -> Optional[Dict]:
        """Find the archive containing a specific memory."""
        archive_id = self.memory_location.get(memory_id)
        if archive_id:
            return self.archives.get(archive_id)
        return None
    
    def search_archives(self, query: str) -> List[Dict]:
        """Simple keyword search across archives."""
        results = []
        
        query_terms = query.lower().split()
        for archive in self.archives.values():
            # Search in name, category, and description
            text = f"{archive['name']} {archive['category']} {archive['description']}".lower()
            
            match_score = sum(term in text for term in query_terms)
            if match_score > 0:
                results.append({
                    "archive": archive,
                    "score": match_score
                })
        
        # Sort by score
        results.sort(key=lambda x: x["score"], reverse=True)
        return [r["archive"] for r in results]

class EnhancedHierarchicalMemory(HierarchicalMemory):
    """Advanced hierarchical memory with enhanced retrieval and storage capabilities."""
    
    def __init__(self, volatile_size: int = 1000, archival_enabled: bool = True):
        """
        Initialize the enhanced hierarchical memory system.
        
        Args:
            volatile_size: Size of volatile memory cache
            archival_enabled: Whether to enable archival memory
        """
        # Initialize base components
        super().__init__(volatile_size=volatile_size)
        
        # Add advanced components
        self.enhanced_retrieval = EnhancedRetrieval(vocab_size=30000)
        self.entity_memory = EntityRelationshipGraph()
        
        # Add archival memory if enabled
        self.archival_enabled = archival_enabled
        if archival_enabled:
            self.archival_memory = ArchivalMemory()
        
        self.cleanup_scheduled = False
        self.last_cleanup = time.time()
        self.memory_monitor = MemoryMonitor()
    
    def cleanup(self):
        """Perform thorough cleanup of memory resources."""
        try:
            # Clear memory maps
            for memory_id in list(self.documents.keys()):
                self._cleanup_memory_maps(memory_id)
                
            # Clear GPU cache if available
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            # Clear internal caches
            self.cache.clear()
            self._clear_temp_storage()
            
            # Reset monitoring
            self.cleanup_scheduled = False
            self.last_cleanup = time.time()
            
            logger.info("Successfully cleaned up memory resources")
            
        except Exception as e:
            logger.error(f"Error during memory cleanup: {str(e)}")
            
    def _cleanup_memory_maps(self, memory_id: str):
        """Clean up memory maps for a specific memory ID."""
        try:
            mmap_path = f"mmap_{memory_id}.bin"
            if os.path.exists(mmap_path):
                os.remove(mmap_path)
        except Exception as e:
            logger.error(f"Error cleaning up memory maps for {memory_id}: {str(e)}")
            
    def _clear_temp_storage(self):
        """Clear temporary storage files."""
        try:
            temp_pattern = "temp_*.bin"
            for temp_file in glob.glob(temp_pattern):
                os.remove(temp_file)
        except Exception as e:
            logger.error(f"Error clearing temporary storage: {str(e)}")
            
    def __del__(self):
        """Ensure cleanup on deletion."""
        self.cleanup()
    
    def store(self, content: Any, metadata: Optional[Dict] = None) -> str:
        """
        Store content in memory and return memory ID.
        
        Extends base implementation with enhanced indexing.
        
        Args:
            content: Content to store
            metadata: Additional metadata
            
        Returns:
            Memory ID
        """
        # Use base implementation for initial storage
        memory_id = super().store(content, metadata)
        
        # Additionally index in enhanced retrieval if content is text
        if isinstance(content, str):
            self.enhanced_retrieval.index_content(content, memory_id, metadata)
            
            # Extract entities if metadata indicates entity extraction should be performed
            if metadata and metadata.get("extract_entities", False):
                self._extract_and_store_entities(content, memory_id)
        
        return memory_id
    
    def retrieve(self, memory_id: str) -> Optional[Dict]:
        """
        Retrieve content by memory ID.
        
        Args:
            memory_id: Memory ID to retrieve
            
        Returns:
            Memory entry if found, None otherwise
        """
        # Use base implementation
        return super().retrieve(memory_id)
    
    def archive_memory(self, memory_id: str, archive_name: str, category: str) -> bool:
        """
        Archive a memory item for long-term storage.
        
        Args:
            memory_id: Memory ID to archive
            archive_name: Name of the archive
            category: Category for the archive
            
        Returns:
            True if successful, False otherwise
        """
        if not self.archival_enabled:
            return False
            
        # Retrieve memory to check it exists
        memory = self.retrieve(memory_id)
        if not memory:
            return False
            
        # Find or create archive
        existing_archives = self.archival_memory.find_archives_by_category(category)
        archive_id = None
        
        for archive in existing_archives:
            if archive["name"] == archive_name:
                archive_id = archive["id"]
                break
                
        if not archive_id:
            # Create new archive
            archive_id = self.archival_memory.create_archive(
                name=archive_name,
                category=category,
                description=f"Auto-created archive for {archive_name}"
            )
            
        # Add memory to archive
        return self.archival_memory.add_memory(
            archive_id=archive_id,
            memory_id=memory_id,
            metadata=memory.get("metadata")
        )
    
    def search(self, query: str, strategy: str = "semantic", top_k: int = 5) -> List[Dict]:
        """
        Search memory using different strategies.
        
        Args:
            query: Search query
            strategy: Search strategy ('keyword', 'semantic', or 'hybrid')
            top_k: Number of results to return
            
        Returns:
            List of search results
        """
        if strategy == "keyword":
            # Call the parent's search method with the correct number of arguments
            # The parent class takes 2 arguments (self, query)
            return super().search(query)
            
        elif strategy == "semantic":
            # Use enhanced semantic retrieval
            return self.enhanced_retrieval.retrieve(query, top_k)
            
        elif strategy == "hybrid":
            # Combine results from both methods
            keyword_results = super().search(query)
            semantic_results = self.enhanced_retrieval.retrieve(query, top_k)
            
            # Take only the top_k keyword results
            keyword_results = keyword_results[:top_k]
            
            # Merge results, prioritizing items that appear in both
            memory_scores = {}
            
            # Add keyword results
            for result in keyword_results:
                memory_id = result["memory_id"]
                memory_scores[memory_id] = {
                    "memory_id": memory_id,
                    "keyword_score": result["score"],
                    "semantic_score": 0,
                    "combined_score": result["score"],
                    "content": result["content"],
                    "metadata": result["metadata"]
                }
                
            # Add semantic results
            for result in semantic_results:
                memory_id = result["memory_id"]
                if memory_id in memory_scores:
                    # Update existing entry
                    memory_scores[memory_id]["semantic_score"] = result["score"]
                    memory_scores[memory_id]["combined_score"] += result["score"]
                else:
                    # Get content from memory
                    memory = self.retrieve(memory_id)
                    content = memory["content"] if memory else "Unknown content"
                    
                    # Add new entry
                    memory_scores[memory_id] = {
                        "memory_id": memory_id,
                        "keyword_score": 0,
                        "semantic_score": result["score"],
                        "combined_score": result["score"],
                        "content": content,
                        "metadata": result["metadata"]
                    }
            
            # Sort by combined score and return top-k
            sorted_results = sorted(
                memory_scores.values(),
                key=lambda x: x["combined_score"],
                reverse=True
            )
            
            return sorted_results[:top_k]
        
        else:
            # Default to keyword search
            return super().search(query)
    
    def search_by_entity(self, entity_name: str, entity_type: Optional[str] = None) -> List[Dict]:
        """
        Search memories associated with a specific entity.
        
        Args:
            entity_name: Name of the entity
            entity_type: Optional entity type
            
        Returns:
            List of memories associated with the entity
        """
        # Find entities matching name and type
        entities = self.entity_memory.get_entity_by_name(entity_name, entity_type)
        
        if not entities:
            return []
            
        # Collect all associated memory IDs
        memory_ids = []
        for entity in entities:
            memory_ids.extend(entity.memory_references)
            
        # Retrieve memories
        results = []
        for memory_id in memory_ids:
            memory = self.retrieve(memory_id)
            if memory:
                results.append({
                    "memory_id": memory_id,
                    "content": memory["content"],
                    "metadata": memory["metadata"]
                })
                
        return results
    
    def _extract_and_store_entities(self, content: str, memory_id: str) -> None:
        """
        Extract entities from content and store in entity memory.
        
        This is a simplified implementation. In a real system, this would use
        a named entity recognition model.
        
        Args:
            content: Text content to extract entities from
            memory_id: Memory ID to associate with extracted entities
        """
        # Simplified entity extraction (placeholder)
        # In a real implementation, use NER model
        
        # For demonstration, extract simple patterns
        words = content.split()
        potential_entities = [w for w in words if len(w) > 4 and w[0].isupper()]
        
        for entity_name in potential_entities:
            # Check if entity exists
            existing = self.entity_memory.get_entity_by_name(entity_name)
            
            if existing:
                # Associate with existing entity
                entity_id = existing[0].id
            else:
                # Create new entity
                entity_id = self.entity_memory.create_entity(
                    name=entity_name,
                    entity_type="unknown"  # Type would be determined by NER
                )
                
            # Associate memory with entity
            self.entity_memory.associate_memory(entity_id, memory_id)
    
    def get_statistics(self) -> Dict:
        """
        Get statistics about the memory system.
        
        Returns:
            Dictionary with memory statistics
        """
        stats = {
            "volatile_memory": {
                "size": len(self.volatile_memory.cache),
                "capacity": self.volatile_memory.max_size,
                "usage_percentage": len(self.volatile_memory.cache) / self.volatile_memory.max_size * 100
            },
            "enhanced_retrieval": self.enhanced_retrieval.get_statistics(),
            "entity_memory": {
                "entity_count": len(self.entity_memory.entities)
            }
        }
        
        if self.archival_enabled:
            stats["archival_memory"] = {
                "archive_count": len(self.archival_memory.archives),
                "category_count": len(self.archival_memory.categories)
            }
            
        return stats

# Example usage
def example_usage():
    """Demonstrate usage of the enhanced hierarchical memory."""
    # Initialize memory system
    memory = EnhancedHierarchicalMemory(volatile_size=100, archival_enabled=True)
    
    # Store some content
    texts = [
        "SPLADE is a sparse lexical method for information retrieval that combines efficiency with semantic understanding.",
        "Neural compression techniques can significantly reduce memory usage while preserving meaning.",
        "Memory systems in AI need to balance speed, capacity, and retrieval accuracy.",
        "Hierarchical memory organization mimics human memory with different storage durations and access speeds.",
        "The Entity Relationship Graph provides object-centric memory representation for complex knowledge."
    ]
    
    memory_ids = []
    for i, text in enumerate(texts):
        metadata = {
            "source": "documentation",
            "topic": "memory_systems" if i < 3 else "advanced_concepts",
            "extract_entities": True
        }
        
        memory_id = memory.store(text, metadata)
        memory_ids.append(memory_id)
        
        # Archive some memories
        if i % 2 == 0:
            memory.archive_memory(
                memory_id=memory_id,
                archive_name="Technical Concepts",
                category="AI Architecture"
            )
    
    # Print statistics
    print("Memory System Statistics:")
    stats = memory.get_statistics()
    print(f"Volatile Memory: {stats['volatile_memory']['size']}/{stats['volatile_memory']['capacity']} items")
    print(f"Entity Count: {stats['entity_memory']['entity_count']}")
    if 'archival_memory' in stats:
        print(f"Archives: {stats['archival_memory']['archive_count']}")
    
    # Perform searches with different strategies
    query = "efficient memory retrieval"
    
    print("\nKeyword Search:")
    keyword_results = memory.search(query, strategy="keyword")
    for i, result in enumerate(keyword_results[:3]):
        print(f"{i+1}. {result['content']}")
    
    print("\nSemantic Search:")
    semantic_results = memory.search(query, strategy="semantic")
    for i, result in enumerate(semantic_results[:3]):
        # Look up original content
        memory_id = result["memory_id"]
        original = memory.retrieve(memory_id)
        content = original["content"] if original else "Unknown content"
        print(f"{i+1}. {content}")
    
    print("\nHybrid Search:")
    hybrid_results = memory.search(query, strategy="hybrid")
    for i, result in enumerate(hybrid_results[:3]):
        print(f"{i+1}. Score: {result['combined_score']:.4f}")
        print(f"   {result['content']}")
    
    # Entity-based search
    print("\nEntity Search:")
    entity_results = memory.search_by_entity("SPLADE")
    for i, result in enumerate(entity_results):
        print(f"{i+1}. {result['content']}")

if __name__ == "__main__":
    example_usage()