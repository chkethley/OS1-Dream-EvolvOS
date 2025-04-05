"""
Advanced SPLADE Retrieval System with Contrastive Learning

This module enhances the SPLADE retrieval system with contrastive learning,
attention mechanisms, and improved index structures for more efficient and
accurate memory retrieval.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Any, Optional, Union, Set
import uuid
import time
import logging
import faiss
from collections import defaultdict, OrderedDict
import json
import os
import pickle

logger = logging.getLogger("OS1.AdvancedRetrieval")

class ContrastiveSPLADEEncoder(nn.Module):
    """
    Advanced SPLADE encoder with contrastive learning capabilities.
    
    Uses a transformer backbone with a contrastive learning approach
    to create better sparse representations.
    """
    
    def __init__(self, 
                 vocab_size: int = 30000, 
                 hidden_dim: int = 768,
                 output_dim: int = 768,
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """
        Initialize the encoder.
        
        Args:
            vocab_size: Size of the vocabulary
            hidden_dim: Hidden dimension size
            output_dim: Output dimension size
            device: Device to run the model on
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.device = device
        
        # In a real implementation, this would load a pretrained transformer model
        # For simulation, we'll create a simplified model
        
        # Create embedding layer
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        
        # Create attention layers
        self.self_attention = nn.MultiheadAttention(hidden_dim, num_heads=8)
        
        # Create SPLADE layers
        self.query_projection = nn.Linear(hidden_dim, vocab_size)
        self.document_projection = nn.Linear(hidden_dim, vocab_size)
        
        # Create contrastive learning components
        self.contrastive_projection = nn.Linear(hidden_dim, output_dim)
        
        # Move to device
        self.to(device)
        
        logger.info(f"Initialized ContrastiveSPLADEEncoder with vocabulary size {vocab_size}")
    
    def tokenize(self, text: str) -> torch.Tensor:
        """
        Tokenize text (simulated).
        
        Args:
            text: Input text
            
        Returns:
            Tensor of token indices
        """
        # In a real implementation, this would use a proper tokenizer
        # For simulation, we'll create a simple hash-based tokenization
        
        words = text.lower().split()
        tokens = []
        
        for word in words:
            # Generate a deterministic token ID
            token_id = sum(ord(c) for c in word) % (self.vocab_size - 1) + 1
            tokens.append(token_id)
            
        # Convert to tensor
        return torch.tensor(tokens, device=self.device)
    
    def _attention_pooling(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Apply self-attention pooling.
        
        Args:
            embeddings: Token embeddings [seq_len, hidden_dim]
            
        Returns:
            Pooled representation [hidden_dim]
        """
        # Add batch dimension
        embeddings = embeddings.unsqueeze(1)  # [seq_len, 1, hidden_dim]
        
        # Apply self-attention
        attn_output, _ = self.self_attention(
            embeddings, embeddings, embeddings
        )
        
        # Remove batch dimension and mean pool
        attn_output = attn_output.squeeze(1)  # [seq_len, hidden_dim]
        pooled = torch.mean(attn_output, dim=0)  # [hidden_dim]
        
        return pooled
    
    def forward_dense(self, text: str) -> torch.Tensor:
        """
        Create dense representation for contrastive learning.
        
        Args:
            text: Input text
            
        Returns:
            Dense vector representation
        """
        # Tokenize
        tokens = self.tokenize(text)
        
        # Get embeddings
        embeddings = self.embedding(tokens)  # [seq_len, hidden_dim]
        
        # Apply attention pooling
        pooled = self._attention_pooling(embeddings)  # [hidden_dim]
        
        # Project to output dimension
        output = self.contrastive_projection(pooled)  # [output_dim]
        
        # Normalize
        output = F.normalize(output, p=2, dim=0)
        
        return output
    
    def forward_sparse(self, text: str, is_query: bool = False) -> Dict[int, float]:
        """
        Create sparse SPLADE representation.
        
        Args:
            text: Input text
            is_query: Whether this is a query (vs. document)
            
        Returns:
            Sparse vector representation as {token_id: weight} dict
        """
        # Tokenize
        tokens = self.tokenize(text)
        
        # Get embeddings
        embeddings = self.embedding(tokens)  # [seq_len, hidden_dim]
        
        # Apply attention pooling
        pooled = self._attention_pooling(embeddings)  # [hidden_dim]
        
        # Project to vocab space
        if is_query:
            logits = self.query_projection(pooled)  # [vocab_size]
        else:
            logits = self.document_projection(pooled)  # [vocab_size]
            
        # Apply SPLADE activation (log(1 + ReLU(x)))
        weights = torch.log1p(F.relu(logits))
        
        # Convert to sparse representation
        sparse_vector = {}
        
        # Get non-zero elements
        non_zero_indices = torch.nonzero(weights).squeeze(1)
        for idx in non_zero_indices:
            token_id = idx.item()
            weight = weights[idx].item()
            if weight > 0:
                sparse_vector[token_id] = weight
                
        return sparse_vector
    
    def encode(self, text: str, is_query: bool = False) -> Dict[int, float]:
        """
        Encode text into a SPLADE sparse vector.
        
        Args:
            text: Input text
            is_query: Whether this is a query (vs. document)
            
        Returns:
            Dictionary mapping token indices to weight values
        """
        with torch.no_grad():
            return self.forward_sparse(text, is_query)
    
    def get_contrastive_embedding(self, text: str) -> np.ndarray:
        """
        Get contrastive learning embedding.
        
        Args:
            text: Input text
            
        Returns:
            Dense embedding vector
        """
        with torch.no_grad():
            embedding = self.forward_dense(text)
            return embedding.cpu().numpy()
    
    def compute_contrastive_loss(self, 
                               anchor_texts: List[str], 
                               positive_texts: List[str],
                               negative_texts: List[str]) -> torch.Tensor:
        """
        Compute contrastive loss for model training.
        
        Args:
            anchor_texts: List of anchor texts
            positive_texts: List of positive examples
            negative_texts: List of negative examples
            
        Returns:
            Contrastive loss value
        """
        # Get embeddings
        anchor_embeddings = torch.stack([self.forward_dense(text) for text in anchor_texts])
        positive_embeddings = torch.stack([self.forward_dense(text) for text in positive_texts])
        negative_embeddings = torch.stack([self.forward_dense(text) for text in negative_texts])
        
        # Compute similarities
        positive_sim = torch.sum(anchor_embeddings * positive_embeddings, dim=1)
        negative_sim = torch.sum(anchor_embeddings * negative_embeddings, dim=1)
        
        # Compute contrastive loss
        loss = -torch.mean(torch.log(torch.exp(positive_sim) / (torch.exp(positive_sim) + torch.exp(negative_sim))))
        
        return loss

class MemoryMonitor:
    """
    Monitor system memory usage and detect potential leaks.
    """
    def __init__(self, threshold_mb: float = 100):
        self.threshold_mb = threshold_mb
        self.memory_samples = []
        self.sample_interval = 60  # seconds
        self.last_sample_time = 0
        
    def check_leaks(self) -> bool:
        """
        Check for memory leaks by analyzing memory usage patterns.
        
        Returns:
            True if leaks are detected, False otherwise.
        """
        current_time = time.time()
        if current_time - self.last_sample_time < self.sample_interval:
            return False
            
        # Get current memory usage
        if torch.cuda.is_available():
            current_memory = torch.cuda.memory_allocated() / (1024 * 1024)  # Convert to MB
        else:
            import psutil
            process = psutil.Process()
            current_memory = process.memory_info().rss / (1024 * 1024)  # Convert to MB
            
        # Add new sample
        self.memory_samples.append((current_time, current_memory))
        self.last_sample_time = current_time
        
        # Keep last hour of samples
        cutoff_time = current_time - 3600
        self.memory_samples = [(t, m) for t, m in self.memory_samples if t > cutoff_time]
        
        # Need at least 3 samples to detect leaks
        if len(self.memory_samples) < 3:
            return False
            
        # Check for consistent increase
        times, memories = zip(*self.memory_samples)
        
        # Calculate rate of change
        memory_growth = (memories[-1] - memories[0]) / (times[-1] - times[0])  # MB/s
        
        # Check if memory is growing faster than threshold
        is_leaking = memory_growth > (self.threshold_mb / 3600)  # Convert threshold to MB/s
        
        if is_leaking:
            logger.warning(f"Potential memory leak detected! Growth rate: {memory_growth:.2f} MB/s")
            
        return is_leaking
        
    def get_memory_stats(self) -> Dict[str, float]:
        """Get current memory statistics."""
        if not self.memory_samples:
            return {}
            
        current_memory = self.memory_samples[-1][1]
        peak_memory = max(m for _, m in self.memory_samples)
        avg_memory = sum(m for _, m in self.memory_samples) / len(self.memory_samples)
        
        return {
            "current_mb": current_memory,
            "peak_mb": peak_memory,
            "average_mb": avg_memory,
            "samples_count": len(self.memory_samples)
        }

class HybridIndex:
    """
    Hybrid index combining sparse SPLADE vectors and dense embeddings.
    
    Provides both sparse retrieval for high precision and dense retrieval
    for semantic matching.
    """
    
    def __init__(self, dimension: int = 768):
        """
        Initialize the hybrid index.
        
        Args:
            dimension: Dimension of dense embeddings
        """
        # Sparse index
        self.sparse_index = {}  # token_id -> {memory_id: weight}
        
        # Dense index (FAISS)
        self.faiss_index = faiss.IndexFlatIP(dimension)
        self.faiss_index_mapping = []  # List of memory_ids corresponding to FAISS index
        
        # Document store
        self.documents = {}  # memory_id -> metadata
        
        # Statistics
        self.indexed_count = 0
        self.last_modified = time.time()
        
        # Cleanup scheduling
        self._cleanup_scheduled = False
        self._last_cleanup = time.time()
        self._cleanup_interval = 3600  # 1 hour
        
        # Memory monitoring and validation
        self._memory_monitor = MemoryMonitor()
        self._last_validation = time.time()
        self._validation_interval = 300  # 5 minutes
        
    def _schedule_cleanup(self):
        """Schedule periodic cleanup of memory resources."""
        current_time = time.time()
        if not self._cleanup_scheduled and (current_time - self._last_cleanup) > self._cleanup_interval:
            self._cleanup_memory_resources()
            self._last_cleanup = current_time
            
    def _cleanup_memory_resources(self):
        """Clean up unused memory resources."""
        try:
            # Clean up invalid mappings
            valid_mappings = []
            for i, memory_id in enumerate(self.faiss_index_mapping):
                if memory_id is not None and memory_id in self.documents:
                    valid_mappings.append(memory_id)
                    
            # Rebuild FAISS index if needed
            if len(valid_mappings) < len(self.faiss_index_mapping):
                new_index = faiss.IndexFlatIP(self.faiss_index.d)
                for memory_id in valid_mappings:
                    vector = self._get_vector(memory_id)
                    if vector is not None:
                        new_index.add(np.array([vector], dtype=np.float32))
                self.faiss_index = new_index
                self.faiss_index_mapping = valid_mappings
                
            # Clean up unused mmap files
            for memory_id in list(self.documents.keys()):
                mmap_path = f"mmap_{memory_id}.bin"
                if not os.path.exists(mmap_path):
                    logger.warning(f"Missing mmap file for {memory_id}, removing from index")
                    self.remove_document(memory_id)
                    
            logger.info("Completed memory resource cleanup")
            
        except Exception as e:
            logger.error(f"Error during memory cleanup: {str(e)}")
            
    def _get_vector(self, memory_id: str) -> Optional[np.ndarray]:
        """Safely retrieve vector for memory ID."""
        try:
            mmap_path = f"mmap_{memory_id}.bin"
            if os.path.exists(mmap_path):
                vector_data = np.memmap(mmap_path, dtype='float32', mode='r',
                                      shape=(1, self.faiss_index.d))
                return vector_data[0]
        except Exception as e:
            logger.error(f"Error retrieving vector for {memory_id}: {str(e)}")
        return None
        
    def _validate_index(self) -> Dict[str, Any]:
        """Validate index integrity and return status."""
        status = {
            "is_valid": True,
            "errors": [],
            "repairs_made": []
        }
        
        try:
            # Check for memory leaks
            if self._memory_monitor.check_leaks():
                status["errors"].append("Memory leak detected")
                status["is_valid"] = False
                self._cleanup_memory_resources()
                status["repairs_made"].append("Memory cleanup performed")
            
            # Validate FAISS index
            valid_count = 0
            for i, memory_id in enumerate(self.faiss_index_mapping):
                if memory_id is not None:
                    if memory_id not in self.documents:
                        status["errors"].append(f"Orphaned FAISS entry: {memory_id}")
                        status["is_valid"] = False
                    else:
                        valid_count += 1
                        
            if valid_count != len(self.documents):
                status["errors"].append("FAISS index/document store mismatch")
                status["is_valid"] = False
                
            # Validate memory mapped files
            for memory_id in self.documents:
                mmap_path = f"mmap_{memory_id}.bin"
                if not os.path.exists(mmap_path):
                    status["errors"].append(f"Missing mmap file: {memory_id}")
                    status["is_valid"] = False
                    
            return status
            
        except Exception as e:
            status["is_valid"] = False
            status["errors"].append(f"Validation error: {str(e)}")
            return status
            
    def _periodic_validation(self):
        """Run periodic validation checks."""
        current_time = time.time()
        if current_time - self._last_validation > self._validation_interval:
            status = self._validate_index()
            if not status["is_valid"]:
                logger.warning(f"Index validation failed: {status['errors']}")
                if status["repairs_made"]:
                    logger.info(f"Repairs performed: {status['repairs_made']}")
            self._last_validation = current_time
            
    def index_document(self, 
                     memory_id: str, 
                     sparse_vector: Dict[int, float],
                     dense_vector: Optional[np.ndarray] = None,
                     metadata: Optional[Dict] = None) -> None:
        try:
            self._periodic_validation()  # Run validation check
            # Store document metadata
            self.documents[memory_id] = {
                "indexed_at": time.time(),
                "sparse_nnz": len(sparse_vector),
                "metadata": metadata or {}
            }
            
            # Index sparse vector
            for token_id, weight in sparse_vector.items():
                if token_id not in self.sparse_index:
                    self.sparse_index[token_id] = {}
                    
                self.sparse_index[token_id][memory_id] = weight
                
            # Index dense vector if provided
            if dense_vector is not None:
                self.faiss_index_mapping.append(memory_id)
                self.faiss_index.add(np.array([dense_vector], dtype=np.float32))
                
            # Update statistics
            self.indexed_count += 1
            self.last_modified = time.time()
            
            self._schedule_cleanup()
        except Exception as e:
            logger.error(f"Error indexing document {memory_id}: {str(e)}")
            raise
        
    def search_sparse(self, 
                     query_vector: Dict[int, float], 
                     top_k: int = 10) -> List[Dict]:
        """
        Search using SPLADE sparse vectors.
        
        Args:
            query_vector: SPLADE sparse vector for the query
            top_k: Number of top results to return
            
        Returns:
            List of search results
        """
        # Calculate scores
        scores = {}
        
        # Calculate dot product between query vector and documents
        for token_id, query_weight in query_vector.items():
            if token_id in self.sparse_index:
                for memory_id, doc_weight in self.sparse_index[token_id].items():
                    if memory_id not in scores:
                        scores[memory_id] = 0.0
                        
                    # Dot product contribution
                    scores[memory_id] += query_weight * doc_weight
                    
        # Sort by score
        top_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        # Format results
        results = []
        for memory_id, score in top_results:
            results.append({
                "memory_id": memory_id,
                "score": score,
                "search_type": "sparse",
                "metadata": self.documents.get(memory_id, {}).get("metadata", {})
            })
            
        return results
    
    def search_dense(self, 
                    query_vector: np.ndarray, 
                    top_k: int = 10) -> List[Dict]:
        """
        Search using dense embedding vectors.
        
        Args:
            query_vector: Dense query vector
            top_k: Number of top results to return
            
        Returns:
            List of search results
        """
        if len(self.faiss_index_mapping) == 0:
            return []
            
        # Search FAISS index
        k = min(top_k, len(self.faiss_index_mapping))
        scores, indices = self.faiss_index.search(np.array([query_vector], dtype=np.float32), k)
        
        # Format results
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < 0 or idx >= len(self.faiss_index_mapping):
                continue
                
            memory_id = self.faiss_index_mapping[idx]
            score = scores[0][i]
            
            results.append({
                "memory_id": memory_id,
                "score": float(score),
                "search_type": "dense",
                "metadata": self.documents.get(memory_id, {}).get("metadata", {})
            })
            
        return results
    
    def search_hybrid(self, 
                     sparse_vector: Dict[int, float],
                     dense_vector: np.ndarray,
                     top_k: int = 10,
                     sparse_weight: float = 0.5) -> List[Dict]:
        """
        Perform hybrid search using both sparse and dense vectors.
        
        Args:
            sparse_vector: SPLADE sparse vector
            dense_vector: Dense embedding vector
            top_k: Number of top results to return
            sparse_weight: Weight for sparse results (0-1)
            
        Returns:
            List of hybrid search results
        """
        # Get sparse and dense results
        sparse_results = self.search_sparse(sparse_vector, top_k=top_k*2)
        dense_results = self.search_dense(dense_vector, top_k=top_k*2)
        
        # Combine results
        combined_scores = {}
        
        # Normalize sparse scores
        sparse_max = max([r["score"] for r in sparse_results]) if sparse_results else 1.0
        
        # Normalize dense scores
        dense_max = max([r["score"] for r in dense_results]) if dense_results else 1.0
        
        # Add sparse scores
        for result in sparse_results:
            memory_id = result["memory_id"]
            score = result["score"] / sparse_max
            combined_scores[memory_id] = score * sparse_weight
            
        # Add dense scores
        for result in dense_results:
            memory_id = result["memory_id"]
            score = result["score"] / dense_max
            
            if memory_id in combined_scores:
                combined_scores[memory_id] += score * (1 - sparse_weight)
            else:
                combined_scores[memory_id] = score * (1 - sparse_weight)
                
        # Sort by score
        top_results = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        # Format results
        results = []
        for memory_id, score in top_results:
            results.append({
                "memory_id": memory_id,
                "score": score,
                "search_type": "hybrid",
                "metadata": self.documents.get(memory_id, {}).get("metadata", {})
            })
            
        return results
    
    def remove_document(self, memory_id: str) -> bool:
        """
        Remove a document from the index.
        
        Args:
            memory_id: Memory ID
            
        Returns:
            True if successful, False otherwise
        """
        if memory_id not in self.documents:
            return False
            
        # Remove from document store
        self.documents.pop(memory_id)
        
        # Remove from sparse index
        for token_id in list(self.sparse_index.keys()):
            if memory_id in self.sparse_index[token_id]:
                self.sparse_index[token_id].pop(memory_id)
                
                # Clean up empty entries
                if not self.sparse_index[token_id]:
                    self.sparse_index.pop(token_id)
                    
        # Remove from dense index
        if memory_id in self.faiss_index_mapping:
            # This is a simplification - in a real implementation, we would need to rebuild the FAISS index
            # For now, we'll just mark the entry as invalid by setting it to None
            idx = self.faiss_index_mapping.index(memory_id)
            self.faiss_index_mapping[idx] = None
            
        # Update statistics
        self.indexed_count -= 1
        self.last_modified = time.time()
        
        return True
    
    def get_statistics(self) -> Dict:
        """
        Get index statistics.
        
        Returns:
            Dictionary of statistics
        """
        return {
            "document_count": len(self.documents),
            "sparse_token_count": len(self.sparse_index),
            "dense_vector_count": len([x for x in self.faiss_index_mapping if x is not None]),
            "last_modified": self.last_modified,
            "index_size_estimate_mb": self._estimate_size_mb()
        }
        
    def _estimate_size_mb(self) -> float:
        """Estimate the size of the index in megabytes."""
        # Count sparse index entries
        sparse_entries = sum(len(docs) for docs in self.sparse_index.values())
        
        # Estimate size
        estimated_bytes = (
            # Size of sparse index
            sparse_entries * 12 +  # key + value (4 bytes each) + overhead
            
            # Size of dense index
            (len(self.faiss_index_mapping) * self.faiss_index.d * 4) +  # Each float is 4 bytes
            
            # Size of document store
            sum(128 + len(json.dumps(d)) for d in self.documents.values())  # Rough estimate
        )
        
        return estimated_bytes / (1024 * 1024)
    
    def save(self, path: str) -> None:
        """
        Save the index to disk.
        
        Args:
            path: Directory to save to
        """
        os.makedirs(path, exist_ok=True)
        
        # Save sparse index
        with open(os.path.join(path, "sparse_index.pkl"), "wb") as f:
            pickle.dump(self.sparse_index, f)
            
        # Save document store
        with open(os.path.join(path, "documents.pkl"), "wb") as f:
            pickle.dump(self.documents, f)
            
        # Save FAISS index
        faiss.write_index(self.faiss_index, os.path.join(path, "dense_index.faiss"))
        
        # Save mapping
        with open(os.path.join(path, "faiss_mapping.pkl"), "wb") as f:
            pickle.dump(self.faiss_index_mapping, f)
            
        logger.info(f"Saved index to {path}")
        
    def load(self, path: str) -> None:
        """
        Load the index from disk.
        
        Args:
            path: Directory to load from
        """
        # Load sparse index
        with open(os.path.join(path, "sparse_index.pkl"), "rb") as f:
            self.sparse_index = pickle.load(f)
            
        # Load document store
        with open(os.path.join(path, "documents.pkl"), "rb") as f:
            self.documents = pickle.load(f)
            
        # Load FAISS index
        self.faiss_index = faiss.read_index(os.path.join(path, "dense_index.faiss"))
        
        # Load mapping
        with open(os.path.join(path, "faiss_mapping.pkl"), "rb") as f:
            self.faiss_index_mapping = pickle.load(f)
            
        # Update statistics
        self.indexed_count = len(self.documents)
        self.last_modified = time.time()
        
        logger.info(f"Loaded index from {path}")

class AdvancedRetrieval:
    """
    Advanced retrieval system for OS1 memory.
    
    Combines contrastive learning, SPLADE sparse vectors, and
    dense embeddings for more effective memory retrieval.
    """
    
    def __init__(self, 
                 vocab_size: int = 30000,
                 embedding_dim: int = 768,
                 device: Optional[str] = None,
                 cache_size: int = 10000):  # Increased cache size
        """
        Initialize the advanced retrieval system.
        
        Args:
            vocab_size: Vocabulary size
            embedding_dim: Embedding dimension
            device: Device to use
            cache_size: Maximum cache size
        """
        # Set device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        # Initialize encoder
        self.encoder = ContrastiveSPLADEEncoder(
            vocab_size=vocab_size,
            hidden_dim=embedding_dim,
            output_dim=embedding_dim,
            device=self.device
        )
        
        # Initialize hybrid index
        self.index = HybridIndex(dimension=embedding_dim)
        
        # Cache for recently accessed items
        self.cache = OrderedDict()
        self.max_cache_size = cache_size
        self.mmap_storage = {}  # Memory-mapped storage for large vectors
        
        logger.info(f"Initialized AdvancedRetrieval system with embedding dimension {embedding_dim}")
        
    def index_content(self, 
                    content: str, 
                    memory_id: str, 
                    metadata: Optional[Dict] = None) -> None:
        """
        Index content for retrieval.
        
        Args:
            content: Text content
            memory_id: Memory ID
            metadata: Additional metadata
        """
        # Skip non-string content
        if not isinstance(content, str):
            return
            
        # Encode content
        sparse_vector = self.encoder.encode(content)
        dense_vector = self.encoder.get_contrastive_embedding(content)
        
        # Index content
        self.index.index_document(
            memory_id=memory_id,
            sparse_vector=sparse_vector,
            dense_vector=dense_vector,
            metadata=metadata
        )
        
        # Enhanced caching with memory mapping for large vectors
        if len(content) > 1000:  # For large content
            mmap_path = f"mmap_{memory_id}.bin"
            vector_data = np.memmap(mmap_path, dtype='float32', mode='w+', 
                                  shape=(1, self.encoder.output_dim))
            vector_data[0] = dense_vector
            self.mmap_storage[memory_id] = mmap_path
        else:
            self.cache[memory_id] = {
                "content": content,
                "metadata": metadata or {},
                "vector": dense_vector
            }
        
        if len(self.cache) > self.max_cache_size:
            # Enhanced LRU eviction
            old_id, old_data = self.cache.popitem(last=False)
            if old_id not in self.mmap_storage:
                # Move to memory-mapped storage
                mmap_path = f"mmap_{old_id}.bin"
                vector_data = np.memmap(mmap_path, dtype='float32', mode='w+',
                                      shape=(1, self.encoder.output_dim))
                vector_data[0] = old_data["vector"]
                self.mmap_storage[old_id] = mmap_path
            
    def retrieve(self, 
               query: str, 
               strategy: str = "hybrid", 
               top_k: int = 5,
               sparse_weight: float = 0.5) -> List[Dict]:
        """
        Retrieve relevant content.
        
        Args:
            query: Query text
            strategy: Retrieval strategy (sparse, dense, hybrid)
            top_k: Number of results to return
            sparse_weight: Weight for sparse results in hybrid search
            
        Returns:
            List of retrieval results
        """
        # Encode query
        sparse_vector = self.encoder.encode(query, is_query=True)
        dense_vector = self.encoder.get_contrastive_embedding(query)
        
        # Perform search based on strategy
        if strategy == "sparse":
            results = self.index.search_sparse(sparse_vector, top_k=top_k)
        elif strategy == "dense":
            results = self.index.search_dense(dense_vector, top_k=top_k)
        else:  # hybrid
            results = self.index.search_hybrid(
                sparse_vector=sparse_vector,
                dense_vector=dense_vector,
                top_k=top_k,
                sparse_weight=sparse_weight
            )
            
        return results
    
    def remove_content(self, memory_id: str) -> bool:
        """
        Remove content from the index.
        
        Args:
            memory_id: Memory ID
            
        Returns:
            True if successful, False otherwise
        """
        # Remove from cache
        if memory_id in self.cache:
            self.cache.pop(memory_id)
            
        # Remove from memory-mapped storage
        if memory_id in self.mmap_storage:
            mmap_path = self.mmap_storage.pop(memory_id)
            if os.path.exists(mmap_path):
                os.remove(mmap_path)
            
        # Remove from index
        return self.index.remove_document(memory_id)
    
    def get_statistics(self) -> Dict:
        """
        Get retrieval system statistics.
        
        Returns:
            Dictionary of statistics
        """
        return {
            "encoder_vocab_size": self.encoder.vocab_size,
            "embedding_dimension": self.encoder.output_dim,
            "device": self.device,
            "cache_size": len(self.cache),
            "index_stats": self.index.get_statistics()
        }
    
    def train_with_feedback(self, 
                          positive_examples: List[Tuple[str, str]],
                          negative_examples: List[Tuple[str, str]],
                          num_epochs: int = 5,
                          learning_rate: float = 1e-4) -> Dict:
        """
        Train the encoder with feedback from usage.
        
        Args:
            positive_examples: List of (query, relevant_content) pairs
            negative_examples: List of (query, irrelevant_content) pairs
            num_epochs: Number of training epochs
            learning_rate: Learning rate
            
        Returns:
            Training statistics
        """
        # This is a simplified training loop
        # In a real implementation, we would use proper batching, validation, etc.
        
        # Create optimizer
        optimizer = torch.optim.Adam(self.encoder.parameters(), lr=learning_rate)
        
        # Training loop
        losses = []
        
        for epoch in range(num_epochs):
            total_loss = 0.0
            
            # Process all examples
            for i in range(0, len(positive_examples), 10):
                # Get batch
                batch_pos = positive_examples[i:i+10]
                batch_neg = negative_examples[i:i+10]
                
                # Ensure equal sizes
                batch_size = min(len(batch_pos), len(batch_neg))
                batch_pos = batch_pos[:batch_size]
                batch_neg = batch_neg[:batch_size]
                
                if batch_size == 0:
                    continue
                    
                # Extract queries and contents
                queries = [pair[0] for pair in batch_pos]
                pos_contents = [pair[1] for pair in batch_pos]
                neg_contents = [pair[1] for pair in batch_neg]
                
                # Compute loss
                optimizer.zero_grad()
                loss = self.encoder.compute_contrastive_loss(queries, pos_contents, neg_contents)
                loss.backward()
                optimizer.step()
                
                # Track loss
                total_loss += loss.item()
                
            # Compute average loss
            avg_loss = total_loss / max(1, len(positive_examples) // 10)
            losses.append(avg_loss)
            
            logger.info(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
            
        # Return statistics
        return {
            "num_epochs": num_epochs,
            "final_loss": losses[-1] if losses else 0.0,
            "loss_history": losses,
            "positive_examples": len(positive_examples),
            "negative_examples": len(negative_examples)
        }
    
    def generate_query_variations(self, query: str, num_variations: int = 3) -> List[str]:
        """
        Generate variations of a query for better retrieval.
        
        Args:
            query: Original query
            num_variations: Number of variations to generate
            
        Returns:
            List of query variations
        """
        # This is a simplified implementation
        # In a real system, this would use more sophisticated techniques
        
        variations = [query]
        
        # Add variations
        words = query.lower().split()
        
        if len(words) > 3:
            # Variation 1: Remove stopwords
            stopwords = {"the", "a", "an", "in", "on", "at", "of", "for", "with", "by"}
            filtered = [w for w in words if w not in stopwords]
            if filtered:
                variations.append(" ".join(filtered))
                
        if len(words) > 2:
            # Variation 2: Use only key words
            variations.append(" ".join(words[:2] + words[-1:]))
            
        # Fill remaining variations with the original query
        while len(variations) < num_variations:
            variations.append(query)
            
        return variations
    
    def save_model(self, path: str) -> None:
        """
        Save the encoder model.
        
        Args:
            path: Path to save to
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.encoder.state_dict(), path)
        logger.info(f"Saved encoder model to {path}")
        
    def load_model(self, path: str) -> None:
        """
        Load the encoder model.
        
        Args:
            path: Path to load from
        """
        self.encoder.load_state_dict(torch.load(path, map_location=self.device))
        logger.info(f"Loaded encoder model from {path}")

class RetrievalPipeline:
    """
    Advanced retrieval pipeline integrating various retrieval strategies.
    
    Combines query rewriting, query expansion, and multi-strategy retrieval
    for improved memory access.
    """
    
    def __init__(self, retrieval_system: AdvancedRetrieval):
        """
        Initialize the retrieval pipeline.
        
        Args:
            retrieval_system: Advanced retrieval system
        """
        self.retrieval = retrieval_system
        self.feedback_buffer = []
        self.max_feedback_buffer = 1000
        
    def process_query(self, 
                    query: str, 
                    top_k: int = 5,
                    use_variations: bool = True,
                    use_hybrid: bool = True) -> List[Dict]:
        """
        Process a query through the retrieval pipeline.
        
        Args:
            query: Query text
            top_k: Number of results to return
            use_variations: Whether to use query variations
            use_hybrid: Whether to use hybrid retrieval
            
        Returns:
            List of retrieval results
        """
        results = []
        
        # Generate query variations if enabled
        if use_variations:
            variations = self.retrieval.generate_query_variations(query)
        else:
            variations = [query]
            
        # Retrieve results for each variation
        for variation in variations:
            # Use hybrid retrieval if enabled, otherwise use sparse
            strategy = "hybrid" if use_hybrid else "sparse"
            
            # Retrieve results
            variation_results = self.retrieval.retrieve(
                query=variation,
                strategy=strategy,
                top_k=top_k
            )
            
            # Add to results
            results.extend(variation_results)
            
        # Remove duplicates
        seen_ids = set()
        unique_results = []
        
        for result in results:
            memory_id = result["memory_id"]
            if memory_id not in seen_ids:
                seen_ids.add(memory_id)
                unique_results.append(result)
                
        # Return top results
        return sorted(unique_results, key=lambda x: x["score"], reverse=True)[:top_k]
    
    def add_feedback(self, 
                   query: str, 
                   memory_id: str, 
                   is_relevant: bool) -> None:
        """
        Add retrieval feedback for learning.
        
        Args:
            query: Query text
            memory_id: Memory ID
            is_relevant: Whether the result was relevant
        """
        # Get content
        content = self.retrieval.cache.get(memory_id, {}).get("content", "")
        if not content:
            return
            
        # Add to feedback buffer
        feedback = {
            "query": query,
            "content": content,
            "memory_id": memory_id,
            "is_relevant": is_relevant,
            "timestamp": time.time()
        }
        
        self.feedback_buffer.append(feedback)
        
        # Trim buffer if needed
        if len(self.feedback_buffer) > self.max_feedback_buffer:
            self.feedback_buffer.pop(0)
            
    def train_from_feedback(self, min_feedback: int = 50) -> Dict:
        """
        Train the retrieval system from feedback.
        
        Args:
            min_feedback: Minimum feedback items needed for training
            
        Returns:
            Training statistics
        """
        if len(self.feedback_buffer) < min_feedback:
            return {"status": "insufficient_feedback", "count": len(self.feedback_buffer)}
            
        # Prepare training data
        positive_examples = []
        negative_examples = []
        
        for feedback in self.feedback_buffer:
            if feedback["is_relevant"]:
                positive_examples.append((feedback["query"], feedback["content"]))
            else:
                negative_examples.append((feedback["query"], feedback["content"]))
                
        # Ensure balanced datasets
        min_size = min(len(positive_examples), len(negative_examples))
        positive_examples = positive_examples[:min_size]
        negative_examples = negative_examples[:min_size]
        
        # Train the model
        stats = self.retrieval.train_with_feedback(
            positive_examples=positive_examples,
            negative_examples=negative_examples
        )
        
        # Clear buffer after training
        self.feedback_buffer = []
        
        return {
            "status": "success",
            "positive_count": len(positive_examples),
            "negative_count": len(negative_examples),
            "training_stats": stats
        }

class EnhancedMemoryInterface:
    """
    Enhanced interface for memory operations.
    
    Provides a high-level interface for storing, retrieving, and querying
    memory using advanced retrieval techniques.
    """
    
    def __init__(self, retrieval_system: AdvancedRetrieval):
        """
        Initialize the memory interface.
        
        Args:
            retrieval_system: Advanced retrieval system
        """
        self.retrieval = retrieval_system
        self.pipeline = RetrievalPipeline(retrieval_system)
        self.memory_metadata = {}
        
    def store(self, 
            content: str, 
            metadata: Optional[Dict] = None,
            tags: Optional[List[str]] = None) -> str:
        """
        Store content in memory.
        
        Args:
            content: Content to store
            metadata: Additional metadata
            tags: Content tags
            
        Returns:
            Memory ID
        """
        # Generate memory ID
        memory_id = str(uuid.uuid4())
        
        # Prepare metadata
        full_metadata = metadata or {}
        
        if tags:
            full_metadata["tags"] = tags
            
        full_metadata["stored_at"] = time.time()
        
        # Store in memory metadata
        self.memory_metadata[memory_id] = full_metadata
        
        # Index for retrieval
        self.retrieval.index_content(content, memory_id, full_metadata)
        
        return memory_id
    
    def retrieve(self, memory_id: str) -> Optional[Dict]:
        """
        Retrieve content by memory ID.
        
        Args:
            memory_id: Memory ID
            
        Returns:
            Content and metadata if found, None otherwise
        """
        # Check cache first
        if memory_id in self.retrieval.cache:
            return {
                "memory_id": memory_id,
                "content": self.retrieval.cache[memory_id]["content"],
                "metadata": self.retrieval.cache[memory_id]["metadata"]
            }
            
        # Not found
        return None
    
    def search(self, 
             query: str, 
             top_k: int = 5,
             strategy: str = "pipeline") -> List[Dict]:
        """
        Search memory.
        
        Args:
            query: Search query
            top_k: Number of results to return
            strategy: Search strategy (pipeline, sparse, dense, hybrid)
            
        Returns:
            List of search results
        """
        if strategy == "pipeline":
            # Use the retrieval pipeline
            results = self.pipeline.process_query(
                query=query,
                top_k=top_k
            )
        else:
            # Use the specified strategy directly
            results = self.retrieval.retrieve(
                query=query,
                strategy=strategy,
                top_k=top_k
            )
            
        # Fetch content for each result
        for result in results:
            memory_id = result["memory_id"]
            content_info = self.retrieve(memory_id)
            
            if content_info:
                result["content"] = content_info["content"]
                
        return results
    
    def search_by_tags(self, 
                     tags: List[str], 
                     match_all: bool = False,
                     top_k: int = 10) -> List[Dict]:
        """
        Search memory by tags.
        
        Args:
            tags: List of tags to search for
            match_all: Whether to require all tags (AND) or any tag (OR)
            top_k: Maximum number of results
            
        Returns:
            List of matching memory items
        """
        results = []
        
        # Search through metadata
        for memory_id, metadata in self.memory_metadata.items():
            item_tags = metadata.get("tags", [])
            
            if not item_tags:
                continue
                
            # Check if tags match
            if match_all:
                matches = all(tag in item_tags for tag in tags)
            else:
                matches = any(tag in item_tags for tag in tags)
                
            if matches:
                # Get content
                content_info = self.retrieve(memory_id)
                
                if content_info:
                    results.append({
                        "memory_id": memory_id,
                        "content": content_info["content"],
                        "metadata": metadata,
                        "score": 1.0  # No scoring for tag-based search
                    })
                    
        # Sort by recency (newest first)
        results.sort(key=lambda x: x["metadata"].get("stored_at", 0), reverse=True)
        
        # Return top results
        return results[:top_k]
    
    def provide_feedback(self, 
                       query: str, 
                       memory_id: str, 
                       is_relevant: bool) -> None:
        """
        Provide feedback on search results.
        
        Args:
            query: Search query
            memory_id: Memory ID
            is_relevant: Whether the result was relevant
        """
        self.pipeline.add_feedback(query, memory_id, is_relevant)
        
    def forget(self, memory_id: str) -> bool:
        """
        Remove content from memory.
        
        Args:
            memory_id: Memory ID
            
        Returns:
            True if successful, False otherwise
        """
        # Remove from metadata
        if memory_id in self.memory_metadata:
            self.memory_metadata.pop(memory_id)
            
        # Remove from retrieval system
        return self.retrieval.remove_content(memory_id)
    
    def get_statistics(self) -> Dict:
        """
        Get memory statistics.
        
        Returns:
            Dictionary of statistics
        """
        return {
            "total_memories": len(self.memory_metadata),
            "retrieval_stats": self.retrieval.get_statistics(),
            "feedback_count": len(self.pipeline.feedback_buffer)
        }

# Example usage
def example_usage():
    """Demonstrate usage of the advanced retrieval system."""
    # Initialize the retrieval system
    retrieval = AdvancedRetrieval(vocab_size=10000, embedding_dim=128)
    
    # Create memory interface
    memory = EnhancedMemoryInterface(retrieval)
    
    # Store some content
    documents = [
        {
            "content": "Enhanced memory systems in AI require efficient storage and retrieval mechanisms",
            "tags": ["memory", "ai", "storage"]
        },
        {
            "content": "SPLADE combines sparse lexical retrieval with transformer-based representations",
            "tags": ["retrieval", "splade", "transformers"]
        },
        {
            "content": "Contrastive learning improves representation quality by bringing similar items closer",
            "tags": ["learning", "contrastive", "representation"]
        },
        {
            "content": "Hybrid retrieval systems combine sparse and dense representations for better results",
            "tags": ["retrieval", "hybrid", "sparse", "dense"]
        },
        {
            "content": "Self-evolving AI systems can learn and improve without human intervention",
            "tags": ["ai", "evolution", "autonomous"]
        }
    ]
    
    for doc in documents:
        memory.store(doc["content"], tags=doc["tags"])
        
    # Print statistics
    print("Memory Statistics:")
    stats = memory.get_statistics()
    print(f"Total memories: {stats['total_memories']}")
    print(f"Index size estimate: {stats['retrieval_stats']['index_stats']['index_size_estimate_mb']:.2f} MB")
    
    # Perform a search
    print("\nSearch Results for 'efficient memory retrieval':")
    results = memory.search("efficient memory retrieval", top_k=3)
    
    for i, result in enumerate(results):
        print(f"{i+1}. Score: {result['score']:.4f}")
        print(f"   Content: {result['content']}")
        print(f"   Search type: {result['search_type']}")
        print()
        
    # Search by tags
    print("\nSearch Results for tags ['retrieval', 'sparse']:")
    tag_results = memory.search_by_tags(["retrieval", "sparse"], match_all=False)
    
    for i, result in enumerate(tag_results):
        print(f"{i+1}. Tags: {result['metadata']['tags']}")
        print(f"   Content: {result['content']}")
        print()
        
    # Provide feedback
    if results:
        memory.provide_feedback("efficient memory retrieval", results[0]["memory_id"], True)
        print("\nProvided positive feedback for top result")
        
    return memory

if __name__ == "__main__":
    example_usage()
