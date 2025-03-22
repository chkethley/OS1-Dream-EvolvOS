"""
Neural Compressor

This module implements neural compression agents for optimizing memory usage
in the OS1 component of the EvolvOS system.
"""

import numpy as np
import random
import time
import uuid
import json
from typing import Dict, List, Any, Optional, Tuple
import hashlib
import zlib

class CompressionMetrics:
    """Class for tracking compression metrics and performance."""
    
    def __init__(self):
        self.compression_ratios = []
        self.processing_times = []
        self.reconstruction_losses = []
        
    def add_metric(self, original_size: int, compressed_size: int, 
                 processing_time: float, reconstruction_loss: float = None):
        """Add compression metrics from a single operation."""
        compression_ratio = original_size / max(1, compressed_size)
        self.compression_ratios.append(compression_ratio)
        self.processing_times.append(processing_time)
        if reconstruction_loss is not None:
            self.reconstruction_losses.append(reconstruction_loss)
            
    def get_average_metrics(self) -> Dict:
        """Get average performance metrics."""
        avg_ratio = np.mean(self.compression_ratios) if self.compression_ratios else 0
        avg_time = np.mean(self.processing_times) if self.processing_times else 0
        avg_loss = np.mean(self.reconstruction_losses) if self.reconstruction_losses else 0
        
        return {
            "avg_compression_ratio": avg_ratio,
            "avg_processing_time_ms": avg_time * 1000,  # Convert to ms
            "avg_reconstruction_loss": avg_loss,
            "samples": len(self.compression_ratios)
        }

class NeuralCompressor:
    """
    Neural Compressor for memory optimization.
    
    This class implements neural compression techniques for efficient
    storage and retrieval of memory items.
    """
    
    def __init__(self, compression_level: int = 3):
        """
        Initialize the neural compressor.
        
        Args:
            compression_level: Level of compression (1-9), higher means more compression
        """
        self.compression_level = min(9, max(1, compression_level))
        self.metrics = CompressionMetrics()
        self.id = str(uuid.uuid4())
        self.created_at = time.time()
        
        # For prototype, simulate different compression methods
        self.compression_methods = {
            "text": self._compress_text,
            "numeric": self._compress_numeric,
            "mixed": self._compress_mixed,
            "semantic": self._compress_semantic
        }
        
        print(f"Initialized Neural Compressor with compression level {self.compression_level}")
        
    def compress(self, data: Any, data_type: str = None) -> Dict:
        """
        Compress data using the appropriate method.
        
        Args:
            data: Data to compress
            data_type: Type of data ("text", "numeric", "mixed", "semantic")
                      If None, type will be inferred
                      
        Returns:
            Dictionary with compressed data and metadata
        """
        start_time = time.time()
        
        # Get the size of the original data
        original_size = self._get_data_size(data)
        
        # Infer data type if not provided
        if data_type is None:
            data_type = self._infer_data_type(data)
            
        # Select compression method
        compression_method = self.compression_methods.get(data_type, self._compress_mixed)
        
        # Apply compression
        compressed_data, compression_metadata = compression_method(data)
        
        # Get size of compressed data
        compressed_size = self._get_data_size(compressed_data)
        
        # Calculate compression ratio and processing time
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Add to metrics
        reconstruction_loss = compression_metadata.get("reconstruction_loss", 0)
        self.metrics.add_metric(original_size, compressed_size, processing_time, reconstruction_loss)
        
        # Create result
        result = {
            "compression_id": str(uuid.uuid4()),
            "compressed_data": compressed_data,
            "original_size_bytes": original_size,
            "compressed_size_bytes": compressed_size,
            "compression_ratio": original_size / max(1, compressed_size),
            "data_type": data_type,
            "compression_level": self.compression_level,
            "timestamp": time.time(),
            "metadata": compression_metadata
        }
        
        return result
        
    def decompress(self, compressed_result: Dict) -> Any:
        """
        Decompress data from a compressed result.
        
        Args:
            compressed_result: Result from compress() method
            
        Returns:
            Decompressed data
        """
        start_time = time.time()
        
        # Extract compression details
        compressed_data = compressed_result["compressed_data"]
        data_type = compressed_result["data_type"]
        
        # Select decompression method
        if data_type == "text":
            decompressed_data = self._decompress_text(compressed_data, compressed_result["metadata"])
        elif data_type == "numeric":
            decompressed_data = self._decompress_numeric(compressed_data, compressed_result["metadata"])
        elif data_type == "semantic":
            decompressed_data = self._decompress_semantic(compressed_data, compressed_result["metadata"])
        else:  # mixed or unknown
            decompressed_data = self._decompress_mixed(compressed_data, compressed_result["metadata"])
            
        # Measure decompression time
        end_time = time.time()
        decompression_time = end_time - start_time
        
        # Return decompressed data
        return decompressed_data
        
    def batch_compress(self, items: List[Dict]) -> Dict:
        """
        Compress a batch of memory items.
        
        Args:
            items: List of memory items to compress
            
        Returns:
            Dictionary with compressed batch and metadata
        """
        # Process items by type
        grouped_items = self._group_by_type(items)
        
        # Compress each group
        compressed_groups = {}
        total_original_size = 0
        total_compressed_size = 0
        
        for data_type, group_items in grouped_items.items():
            # Extract content from items
            contents = [item.get("content", "") for item in group_items]
            
            # Compress group as batch
            if data_type == "semantic":
                # For semantic data, compress as batch
                compressed_batch, metadata = self._compress_semantic_batch(contents)
                compressed_groups[data_type] = {
                    "compressed_data": compressed_batch,
                    "metadata": metadata,
                    "item_count": len(group_items),
                    "item_ids": [item.get("id", i) for i, item in enumerate(group_items)]
                }
            else:
                # For other types, compress individually but store together
                compressed_items = []
                for item in group_items:
                    compressed_result = self.compress(item.get("content", ""), data_type)
                    total_original_size += compressed_result["original_size_bytes"]
                    total_compressed_size += compressed_result["compressed_size_bytes"]
                    compressed_items.append({
                        "item_id": item.get("id", str(uuid.uuid4())),
                        "compressed_result": compressed_result
                    })
                    
                compressed_groups[data_type] = {
                    "items": compressed_items,
                    "item_count": len(compressed_items)
                }
        
        # Create batch result
        batch_result = {
            "batch_id": str(uuid.uuid4()),
            "compressed_groups": compressed_groups,
            "total_items": sum(group["item_count"] for group in compressed_groups.values()),
            "timestamp": time.time(),
            "total_original_size_bytes": total_original_size,
            "total_compressed_size_bytes": total_compressed_size,
            "overall_compression_ratio": total_original_size / max(1, total_compressed_size)
        }
        
        return batch_result
        
    def decompress_batch(self, batch_result: Dict) -> List[Dict]:
        """
        Decompress a batch of compressed items.
        
        Args:
            batch_result: Result from batch_compress() method
            
        Returns:
            List of decompressed items
        """
        decompressed_items = []
        
        # Process each group
        for data_type, group_data in batch_result["compressed_groups"].items():
            if data_type == "semantic":
                # Decompress semantic batch
                contents = self._decompress_semantic_batch(
                    group_data["compressed_data"], 
                    group_data["metadata"]
                )
                
                # Reconstruct items
                for i, item_id in enumerate(group_data["item_ids"]):
                    if i < len(contents):
                        decompressed_items.append({
                            "id": item_id,
                            "content": contents[i],
                            "data_type": data_type
                        })
            else:
                # Decompress individual items
                for item_data in group_data["items"]:
                    decompressed_content = self.decompress(item_data["compressed_result"])
                    decompressed_items.append({
                        "id": item_data["item_id"],
                        "content": decompressed_content,
                        "data_type": data_type
                    })
                    
        return decompressed_items
        
    def get_metrics(self) -> Dict:
        """Get compression performance metrics."""
        return {
            "compressor_id": self.id,
            "compression_level": self.compression_level,
            "metrics": self.metrics.get_average_metrics(),
            "created_at": self.created_at,
            "uptime_seconds": time.time() - self.created_at
        }
        
    def _infer_data_type(self, data: Any) -> str:
        """Infer the type of data for compression."""
        if data is None:
            return "mixed"
            
        if isinstance(data, str):
            # Check if it's numeric string
            if data.replace(".", "").replace("-", "").isdigit():
                return "numeric"
            # Check if it's mostly text
            return "text"
            
        if isinstance(data, (int, float, bool, np.number)):
            return "numeric"
            
        if isinstance(data, (list, tuple)):
            # Check if list of numbers
            if all(isinstance(item, (int, float, np.number)) for item in data):
                return "numeric"
            # Check if list of strings
            if all(isinstance(item, str) for item in data):
                return "text"
                
        if isinstance(data, dict) and "vector" in data:
            return "semantic"
            
        # Default
        return "mixed"
        
    def _get_data_size(self, data: Any) -> int:
        """Get approximate size of data in bytes."""
        if data is None:
            return 0
            
        if isinstance(data, (str, bytes)):
            return len(data)
            
        if isinstance(data, (int, bool)):
            return 8
            
        if isinstance(data, float):
            return 8
            
        if isinstance(data, (list, tuple)):
            return sum(self._get_data_size(item) for item in data)
            
        if isinstance(data, dict):
            return sum(self._get_data_size(k) + self._get_data_size(v) for k, v in data.items())
            
        # Default
        try:
            return len(str(data))
        except:
            return 32  # arbitrary fallback
            
    def _compress_text(self, text: str) -> Tuple[bytes, Dict]:
        """Compress text data."""
        if not isinstance(text, str):
            text = str(text)
            
        # For prototype, use zlib compression
        compressed = zlib.compress(text.encode('utf-8'), self.compression_level)
        
        metadata = {
            "original_length": len(text),
            "method": "zlib",
            "content_hash": hashlib.md5(text.encode('utf-8')).hexdigest()
        }
        
        return compressed, metadata
        
    def _decompress_text(self, compressed_data: bytes, metadata: Dict) -> str:
        """Decompress text data."""
        # For prototype, use zlib decompression
        decompressed = zlib.decompress(compressed_data).decode('utf-8')
        
        return decompressed
        
    def _compress_numeric(self, data: Any) -> Tuple[bytes, Dict]:
        """Compress numeric data."""
        # Convert to numpy array if not already
        if not isinstance(data, np.ndarray):
            if isinstance(data, (list, tuple)):
                data = np.array(data, dtype=float)
            else:
                data = np.array([float(data)])
                
        # For prototype, use simple quantization and compression
        data_min = data.min()
        data_max = data.max()
        
        # Normalize to [0, 255] for byte representation
        if data_max > data_min:
            normalized = ((data - data_min) / (data_max - data_min) * 255).astype(np.uint8)
        else:
            normalized = np.zeros_like(data, dtype=np.uint8)
            
        # Compress with zlib
        compressed = zlib.compress(normalized.tobytes(), self.compression_level)
        
        metadata = {
            "data_min": float(data_min),
            "data_max": float(data_max),
            "shape": data.shape,
            "method": "quantization_zlib",
            "dtype": str(data.dtype)
        }
        
        return compressed, metadata
        
    def _decompress_numeric(self, compressed_data: bytes, metadata: Dict) -> np.ndarray:
        """Decompress numeric data."""
        # Extract metadata
        data_min = metadata["data_min"]
        data_max = metadata["data_max"]
        shape = metadata["shape"]
        
        # Decompress
        decompressed_bytes = zlib.decompress(compressed_data)
        
        # Reshape
        normalized = np.frombuffer(decompressed_bytes, dtype=np.uint8).reshape(shape)
        
        # Denormalize
        if data_max > data_min:
            denormalized = (normalized.astype(float) / 255) * (data_max - data_min) + data_min
        else:
            denormalized = np.ones(shape, dtype=float) * data_min
            
        return denormalized
        
    def _compress_mixed(self, data: Any) -> Tuple[bytes, Dict]:
        """Compress mixed data types."""
        # Convert to JSON string first
        try:
            json_str = json.dumps(data)
            compressed, metadata = self._compress_text(json_str)
            metadata["method"] = "json_zlib"
            return compressed, metadata
        except:
            # Fallback to string representation
            str_data = str(data)
            compressed, metadata = self._compress_text(str_data)
            metadata["method"] = "str_zlib"
            return compressed, metadata
            
    def _decompress_mixed(self, compressed_data: bytes, metadata: Dict) -> Any:
        """Decompress mixed data types."""
        # Decompress to string first
        decompressed_str = self._decompress_text(compressed_data, metadata)
        
        # Try to parse as JSON
        if metadata.get("method") == "json_zlib":
            try:
                return json.loads(decompressed_str)
            except:
                pass
                
        # Return as string
        return decompressed_str
        
    def _compress_semantic(self, data: Any) -> Tuple[bytes, Dict]:
        """
        Compress semantic data (vectors).
        
        For prototype, simulates neural compression of semantic vectors.
        """
        # Convert to vector if needed
        if isinstance(data, dict) and "vector" in data:
            vector = data["vector"]
            metadata_extra = {k: v for k, v in data.items() if k != "vector"}
        elif isinstance(data, (list, tuple, np.ndarray)):
            vector = np.array(data)
            metadata_extra = {}
        else:
            # Fallback to mixed compression
            return self._compress_mixed(data)
            
        # Convert to numpy array
        if not isinstance(vector, np.ndarray):
            vector = np.array(vector, dtype=float)
            
        # For prototype, use SVD-like compression (dimension reduction)
        original_dim = vector.size
        
        # Compression level determines how many dimensions to keep
        # Level 1: 90% of dimensions, Level 9: 10% of dimensions
        keep_ratio = 1.0 - (self.compression_level / 10)
        reduced_dim = max(1, int(original_dim * keep_ratio))
        
        # Simulate dimension reduction (just keep subset for prototype)
        if original_dim > reduced_dim:
            # In a real implementation, this would use PCA or autoencoder
            # For prototype, just keep top dimensions
            indices = np.argsort(np.abs(vector))[::-1][:reduced_dim]
            reduced_vector = np.zeros(reduced_dim, dtype=float)
            for i, idx in enumerate(sorted(indices)):
                reduced_vector[i] = vector[idx]
                
            # Quantize to further compress
            data_min = reduced_vector.min()
            data_max = reduced_vector.max()
            
            # Normalize to [0, 255] for byte representation
            if data_max > data_min:
                normalized = ((reduced_vector - data_min) / (data_max - data_min) * 255).astype(np.uint8)
            else:
                normalized = np.zeros_like(reduced_vector, dtype=np.uint8)
                
            # Compress with zlib
            compressed = zlib.compress(normalized.tobytes(), self.compression_level)
            
            # Estimate reconstruction loss
            reconstruction_loss = 1.0 - (reduced_dim / original_dim)
            
            metadata = {
                "original_dim": original_dim,
                "reduced_dim": reduced_dim,
                "indices": indices.tolist(),
                "data_min": float(data_min),
                "data_max": float(data_max),
                "method": "dimension_reduction_quantization",
                "reconstruction_loss": float(reconstruction_loss),
                **metadata_extra
            }
        else:
            # No reduction needed
            compressed, base_metadata = self._compress_numeric(vector)
            metadata = {
                **base_metadata,
                "original_dim": original_dim,
                "reduced_dim": original_dim,
                "method": "quantization_only",
                "reconstruction_loss": 0.0,
                **metadata_extra
            }
            
        return compressed, metadata
        
    def _decompress_semantic(self, compressed_data: bytes, metadata: Dict) -> np.ndarray:
        """Decompress semantic data."""
        method = metadata.get("method", "")
        
        if method == "dimension_reduction_quantization":
            # Extract metadata
            original_dim = metadata["original_dim"]
            reduced_dim = metadata["reduced_dim"]
            indices = metadata["indices"]
            data_min = metadata["data_min"]
            data_max = metadata["data_max"]
            
            # Decompress reduced vector
            decompressed_bytes = zlib.decompress(compressed_data)
            normalized = np.frombuffer(decompressed_bytes, dtype=np.uint8)
            
            # Denormalize
            if data_max > data_min:
                reduced_vector = (normalized.astype(float) / 255) * (data_max - data_min) + data_min
            else:
                reduced_vector = np.ones(reduced_dim, dtype=float) * data_min
                
            # Reconstruct full vector
            full_vector = np.zeros(original_dim, dtype=float)
            for i, idx in enumerate(indices):
                if i < len(reduced_vector):
                    full_vector[idx] = reduced_vector[i]
                    
            # If original was a dict with vector, reconstruct
            metadata_extra = {k: v for k, v in metadata.items() 
                            if k not in {"original_dim", "reduced_dim", "indices", 
                                         "data_min", "data_max", "method", 
                                         "reconstruction_loss"}}
                                         
            if metadata_extra:
                return {"vector": full_vector, **metadata_extra}
            else:
                return full_vector
                
        elif method == "quantization_only":
            # Use numeric decompression
            decompressed = self._decompress_numeric(compressed_data, metadata)
            
            # If original was a dict with vector, reconstruct
            metadata_extra = {k: v for k, v in metadata.items() 
                            if k not in {"data_min", "data_max", "shape", 
                                         "method", "dtype", "original_dim", 
                                         "reduced_dim", "reconstruction_loss"}}
                                         
            if metadata_extra:
                return {"vector": decompressed, **metadata_extra}
            else:
                return decompressed
        else:
            # Fallback to mixed decompression
            return self._decompress_mixed(compressed_data, metadata)
            
    def _compress_semantic_batch(self, vectors: List[Any]) -> Tuple[bytes, Dict]:
        """
        Compress a batch of semantic vectors.
        
        For prototype, uses a shared compression space.
        """
        # Process vectors to numpy arrays
        processed_vectors = []
        max_dim = 0
        
        for vec in vectors:
            if isinstance(vec, dict) and "vector" in vec:
                processed_vectors.append((np.array(vec["vector"]), {k: v for k, v in vec.items() if k != "vector"}))
                max_dim = max(max_dim, len(vec["vector"]))
            elif isinstance(vec, (list, tuple, np.ndarray)):
                processed_vectors.append((np.array(vec), {}))
                max_dim = max(max_dim, len(vec))
            else:
                # Skip non-vector data
                processed_vectors.append((np.array([0.0]), {}))
                max_dim = max(max_dim, 1)
                
        # For prototype, use a simple matrix of vectors
        # In a real implementation, this would use more sophisticated batch compression
        
        # Create matrix of vectors, padding as needed
        matrix = np.zeros((len(processed_vectors), max_dim))
        
        for i, (vec, _) in enumerate(processed_vectors):
            vec_len = len(vec)
            matrix[i, :vec_len] = vec
            
        # Compression level determines how many dimensions to keep
        keep_ratio = 1.0 - (self.compression_level / 10)
        reduced_dim = max(1, int(max_dim * keep_ratio))
        
        # Simulate dimension reduction for entire matrix
        if max_dim > reduced_dim:
            # In a real implementation, this would use batch SVD or autoencoder
            # For prototype, just keep top dimensions based on column variance
            col_var = np.var(matrix, axis=0)
            indices = np.argsort(col_var)[::-1][:reduced_dim]
            reduced_matrix = matrix[:, sorted(indices)]
        else:
            reduced_matrix = matrix
            indices = list(range(max_dim))
            
        # Quantize and compress
        matrix_min = reduced_matrix.min()
        matrix_max = reduced_matrix.max()
        
        if matrix_max > matrix_min:
            normalized = ((reduced_matrix - matrix_min) / (matrix_max - matrix_min) * 255).astype(np.uint8)
        else:
            normalized = np.zeros_like(reduced_matrix, dtype=np.uint8)
            
        # Compress with zlib
        compressed = zlib.compress(normalized.tobytes(), self.compression_level)
        
        # Calculate estimated reconstruction loss
        reconstruction_loss = 1.0 - (reduced_dim / max_dim) if max_dim > 0 else 0
        
        # Metadata includes extra data from each vector
        metadata = {
            "vector_count": len(processed_vectors),
            "original_dim": max_dim,
            "reduced_dim": reduced_dim,
            "matrix_shape": reduced_matrix.shape,
            "indices": indices,
            "matrix_min": float(matrix_min),
            "matrix_max": float(matrix_max),
            "method": "batch_dimension_reduction",
            "reconstruction_loss": float(reconstruction_loss),
            "vector_metadata": [meta for _, meta in processed_vectors]
        }
        
        return compressed, metadata
        
    def _decompress_semantic_batch(self, compressed_data: bytes, metadata: Dict) -> List[Dict]:
        """Decompress a batch of semantic vectors."""
        # Extract metadata
        vector_count = metadata["vector_count"]
        reduced_dim = metadata["reduced_dim"]
        original_dim = metadata["original_dim"]
        indices = metadata["indices"]
        matrix_shape = metadata["matrix_shape"]
        matrix_min = metadata["matrix_min"]
        matrix_max = metadata["matrix_max"]
        vector_metadata = metadata["vector_metadata"]
        
        # Decompress
        decompressed_bytes = zlib.decompress(compressed_data)
        normalized = np.frombuffer(decompressed_bytes, dtype=np.uint8).reshape(matrix_shape)
        
        # Denormalize
        if matrix_max > matrix_min:
            reduced_matrix = (normalized.astype(float) / 255) * (matrix_max - matrix_min) + matrix_min
        else:
            reduced_matrix = np.ones(matrix_shape, dtype=float) * matrix_min
            
        # Reconstruct full vectors
        result = []
        
        for i in range(vector_count):
            # Get reduced vector for this item
            reduced_vector = reduced_matrix[i]
            
            # Reconstruct full vector
            full_vector = np.zeros(original_dim)
            for j, idx in enumerate(indices):
                if j < len(reduced_vector):
                    full_vector[idx] = reduced_vector[j]
                    
            # Get original metadata if available
            extra_metadata = vector_metadata[i] if i < len(vector_metadata) else {}
            
            if extra_metadata:
                result.append({"vector": full_vector, **extra_metadata})
            else:
                result.append(full_vector)
                
        return result
        
    def _group_by_type(self, items: List[Dict]) -> Dict[str, List[Dict]]:
        """Group items by data type for batch compression."""
        grouped = {
            "text": [],
            "numeric": [],
            "mixed": [],
            "semantic": []
        }
        
        for item in items:
            content = item.get("content", "")
            data_type = self._infer_data_type(content)
            grouped[data_type].append(item)
            
        return grouped

class DynamicMemoryPool:
    """
    Dynamic memory pooling system that uses the neural compressor.
    Provides continuous learning and adaptation.
    """
    
    def __init__(self, max_uncompressed: int = 100, compression_level: int = 3):
        """
        Initialize the dynamic memory pool.
        
        Args:
            max_uncompressed: Maximum number of uncompressed items
            compression_level: Compression level (1-9)
        """
        self.max_uncompressed = max_uncompressed
        self.neural_compressor = NeuralCompressor(compression_level)
        
        # Storage
        self.uncompressed_pool = {}  # id -> memory item
        self.compressed_batches = []  # list of compressed batches
        
        # Stats
        self.item_count = 0
        self.compression_cycles = 0
        
    def store(self, content: Any, metadata: Optional[Dict] = None) -> str:
        """
        Store content in memory pool.
        
        Args:
            content: Content to store
            metadata: Optional metadata
            
        Returns:
            Memory item ID
        """
        item_id = str(uuid.uuid4())
        
        # Create memory item
        memory_item = {
            "id": item_id,
            "content": content,
            "metadata": metadata or {},
            "timestamp": time.time(),
            "access_count": 0
        }
        
        # Store in uncompressed pool
        self.uncompressed_pool[item_id] = memory_item
        self.item_count += 1
        
        # Check if compression needed
        if len(self.uncompressed_pool) > self.max_uncompressed:
            self._compress_oldest()
            
        return item_id
        
    def retrieve(self, item_id: str) -> Optional[Dict]:
        """
        Retrieve item by ID.
        
        Args:
            item_id: Item ID
            
        Returns:
            Memory item if found, None otherwise
        """
        # Check uncompressed pool first
        if item_id in self.uncompressed_pool:
            item = self.uncompressed_pool[item_id]
            item["access_count"] += 1
            return item
            
        # Search in compressed batches
        for batch in self.compressed_batches:
            for data_type, group_data in batch["compressed_groups"].items():
                # Check if item is in this group
                if data_type == "semantic":
                    if item_id in group_data["item_ids"]:
                        # Decompress the entire batch for simplicity
                        # In a real implementation, would decompress only the needed item
                        decompressed_items = self.neural_compressor.decompress_batch(batch)
                        
                        # Find the requested item
                        for item in decompressed_items:
                            if item["id"] == item_id:
                                # Move to uncompressed pool
                                self.uncompressed_pool[item_id] = {
                                    "id": item_id,
                                    "content": item["content"],
                                    "metadata": {"recovered_from_compression": True},
                                    "timestamp": time.time(),
                                    "access_count": 1
                                }
                                return self.uncompressed_pool[item_id]
                else:
                    for item_data in group_data["items"]:
                        if item_data["item_id"] == item_id:
                            # Decompress just this item
                            decompressed_content = self.neural_compressor.decompress(
                                item_data["compressed_result"]
                            )
                            
                            # Move to uncompressed pool
                            self.uncompressed_pool[item_id] = {
                                "id": item_id,
                                "content": decompressed_content,
                                "metadata": {"recovered_from_compression": True},
                                "timestamp": time.time(),
                                "access_count": 1
                            }
                            return self.uncompressed_pool[item_id]
                            
        # Not found
        return None
        
    def search_by_metadata(self, query: Dict) -> List[str]:
        """
        Search for items by metadata.
        
        Args:
            query: Metadata query dict
            
        Returns:
            List of matching item IDs
        """
        matching_ids = []
        
        # Search uncompressed pool
        for item_id, item in self.uncompressed_pool.items():
            if self._matches_metadata(item["metadata"], query):
                matching_ids.append(item_id)
                
        # Search compressed batches (just metadata, don't decompress)
        for batch in self.compressed_batches:
            for data_type, group_data in batch["compressed_groups"].items():
                if data_type == "semantic":
                    # Check semantic batch metadata
                    for i, item_id in enumerate(group_data["item_ids"]):
                        if i < len(group_data["metadata"]["vector_metadata"]):
                            metadata = group_data["metadata"]["vector_metadata"][i]
                            if self._matches_metadata(metadata, query):
                                matching_ids.append(item_id)
                else:
                    # Check individual item metadata
                    for item_data in group_data["items"]:
                        metadata = item_data["compressed_result"].get("metadata", {})
                        if self._matches_metadata(metadata, query):
                            matching_ids.append(item_data["item_id"])
                            
        return matching_ids
        
    def get_statistics(self) -> Dict:
        """Get memory pool statistics."""
        return {
            "uncompressed_items": len(self.uncompressed_pool),
            "compressed_batches": len(self.compressed_batches),
            "total_items": self.item_count,
            "compression_cycles": self.compression_cycles,
            "compression_stats": self.neural_compressor.get_metrics()
        }
        
    def _compress_oldest(self):
        """Compress oldest items in uncompressed pool."""
        # Sort by timestamp (oldest first)
        items_to_compress = sorted(
            self.uncompressed_pool.values(),
            key=lambda x: (x["access_count"], x["timestamp"])
        )
        
        # Take oldest half
        compress_count = max(1, len(items_to_compress) // 2)
        batch_items = items_to_compress[:compress_count]
        
        # Compress batch
        compressed_batch = self.neural_compressor.batch_compress(batch_items)
        self.compressed_batches.append(compressed_batch)
        
        # Remove compressed items from uncompressed pool
        for item in batch_items:
            self.uncompressed_pool.pop(item["id"], None)
            
        self.compression_cycles += 1
        
    def _matches_metadata(self, metadata: Dict, query: Dict) -> bool:
        """Check if metadata matches query."""
        for key, value in query.items():
            if key not in metadata:
                return False
                
            if metadata[key] != value:
                return False
                
        return True

# Example usage
def example_usage():
    """Demonstrate usage of the neural compressor."""
    # Initialize compressor
    compressor = NeuralCompressor(compression_level=5)
    
    # Text compression example
    text = "This is an example of neural compression for memory optimization in self-evolving AI systems."
    compressed_text = compressor.compress(text, "text")
    
    print(f"Text compression ratio: {compressed_text['compression_ratio']:.2f}x")
    
    # Decompress
    decompressed_text = compressor.decompress(compressed_text)
    print(f"Original text: {text}")
    print(f"Decompressed text: {decompressed_text}")
    print(f"Match: {text == decompressed_text}")
    
    # Numeric compression example
    data = np.random.random(100)
    compressed_numeric = compressor.compress(data, "numeric")
    
    print(f"\nNumeric compression ratio: {compressed_numeric['compression_ratio']:.2f}x")
    
    # Decompress
    decompressed_numeric = compressor.decompress(compressed_numeric)
    reconstruction_error = np.mean(np.abs(data - decompressed_numeric))
    print(f"Reconstruction error: {reconstruction_error:.6f}")
    
    # Memory pool example
    print("\nMemory Pool Example:")
    memory_pool = DynamicMemoryPool(max_uncompressed=10, compression_level=3)
    
    # Store items
    for i in range(20):
        item_type = "text" if i % 2 == 0 else "numeric"
        if item_type == "text":
            content = f"Memory item {i} with important information"
        else:
            content = np.random.random(50)
            
        item_id = memory_pool.store(content, {"type": item_type, "priority": i % 3})
        
        if i == 5:
            test_id = item_id
            
    # Retrieve item
    retrieved = memory_pool.retrieve(test_id)
    print(f"Retrieved item: {retrieved['content']}")
    
    # Get statistics
    stats = memory_pool.get_statistics()
    print(f"Memory pool stats: {stats}")
    
    return compressor, memory_pool

if __name__ == "__main__":
    example_usage() 