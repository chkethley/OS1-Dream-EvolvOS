"""
Vector Symbolic Architecture (VSA) for Agent Communication

This module implements Vector Symbolic Architecture for the Dream System multi-agent framework,
enabling more sophisticated agent communication and reasoning using high-dimensional vectors.
"""

import numpy as np
import random
from typing import Dict, List, Tuple, Any, Optional
import uuid
import hashlib

class VSAEncoder:
    """
    Vector Symbolic Architecture encoder for agent communication.
    
    Implements hyperdimensional computing techniques for symbolic manipulation
    and reasoning with distributed representations.
    """
    
    def __init__(self, dimension: int = 10000, seed: int = None):
        """
        Initialize the VSA encoder.
        
        Args:
            dimension: Dimensionality of the hypervectors
            seed: Random seed for reproducibility
        """
        self.dimension = dimension
        self.random_state = np.random.RandomState(seed)
        
        # Initialize symbol table for basic primitives
        self.symbols = {}
        
        # Initialize role vectors for binding operations
        self.roles = {}
        
        print(f"Initialized VSA encoder with {dimension}-dimensional vectors")
        
    def get_symbol(self, symbol_name: str) -> np.ndarray:
        """
        Get or create a symbol vector.
        
        Args:
            symbol_name: Name of the symbol
            
        Returns:
            Hypervector representing the symbol
        """
        if symbol_name in self.symbols:
            return self.symbols[symbol_name]
            
        # Create a new random hypervector
        # Using bipolar vectors {-1, 1} for better binding properties
        vector = self.random_state.choice([-1, 1], size=self.dimension)
        
        # Store for future use
        self.symbols[symbol_name] = vector
        
        return vector
        
    def get_role(self, role_name: str) -> np.ndarray:
        """
        Get or create a role vector for binding.
        
        Args:
            role_name: Name of the role
            
        Returns:
            Hypervector representing the role
        """
        if role_name in self.roles:
            return self.roles[role_name]
            
        # Create a new random hypervector
        vector = self.random_state.choice([-1, 1], size=self.dimension)
        
        # Store for future use
        self.roles[role_name] = vector
        
        return vector
        
    def bind(self, vector1: np.ndarray, vector2: np.ndarray) -> np.ndarray:
        """
        Bind two vectors using element-wise multiplication.
        
        Args:
            vector1: First vector
            vector2: Second vector
            
        Returns:
            Bound vector
        """
        return vector1 * vector2
        
    def unbind(self, bound_vector: np.ndarray, key_vector: np.ndarray) -> np.ndarray:
        """
        Unbind a vector using the key vector.
        
        Args:
            bound_vector: Bound vector
            key_vector: Key vector used for binding
            
        Returns:
            Unbound vector
        """
        # For bipolar vectors, unbinding is the same as binding
        return bound_vector * key_vector
        
    def bundle(self, vectors: List[np.ndarray]) -> np.ndarray:
        """
        Bundle vectors using element-wise addition.
        
        Args:
            vectors: List of vectors to bundle
            
        Returns:
            Bundled vector
        """
        if not vectors:
            return np.zeros(self.dimension)
            
        # Simple element-wise addition
        result = np.zeros(self.dimension)
        for vector in vectors:
            result += vector
            
        return result
        
    def normalize(self, vector: np.ndarray) -> np.ndarray:
        """
        Normalize a vector to have only {-1, 1} elements.
        
        Args:
            vector: Vector to normalize
            
        Returns:
            Normalized vector
        """
        # Convert to bipolar representation
        return np.sign(vector)
        
    def cleanup(self, vector: np.ndarray, candidates: List[np.ndarray]) -> np.ndarray:
        """
        Find the closest vector in the candidates list.
        
        Args:
            vector: Query vector
            candidates: List of candidate vectors
            
        Returns:
            The closest matching vector
        """
        if not candidates:
            return vector
            
        # Calculate cosine similarity with each candidate
        similarities = [self.similarity(vector, candidate) for candidate in candidates]
        
        # Return the most similar candidate
        best_index = np.argmax(similarities)
        return candidates[best_index]
        
    def similarity(self, vector1: np.ndarray, vector2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two vectors.
        
        Args:
            vector1: First vector
            vector2: Second vector
            
        Returns:
            Cosine similarity (-1 to 1)
        """
        return np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))

class VSAMessage:
    """Represents a message encoded using Vector Symbolic Architecture."""
    
    def __init__(self, encoder: VSAEncoder):
        self.id = str(uuid.uuid4())
        self.encoder = encoder
        self.vector = None
        self.content = {}
        
    def add_role_filler(self, role: str, filler: str) -> None:
        """
        Add a role-filler binding to the message.
        
        Args:
            role: Role name (e.g., "sender", "content", "action")
            filler: Filler value
        """
        # Get vectors
        role_vector = self.encoder.get_role(role)
        filler_vector = self.encoder.get_symbol(filler)
        
        # Bind them
        binding = self.encoder.bind(role_vector, filler_vector)
        
        # Store the content
        self.content[role] = filler
        
        # Add to the message vector
        if self.vector is None:
            self.vector = binding
        else:
            self.vector += binding
            
    def get_filler(self, role: str) -> Optional[str]:
        """
        Get the filler for a specific role.
        
        Args:
            role: Role name
            
        Returns:
            Filler value if found, None otherwise
        """
        return self.content.get(role)
    
    def decode(self, role: str) -> str:
        """
        Decode the filler for a specific role from the vector.
        
        Args:
            role: Role name
            
        Returns:
            Most likely filler for the role
        """
        if self.vector is None:
            return None
            
        # Get role vector
        role_vector = self.encoder.get_role(role)
        
        # Unbind
        unbound = self.encoder.unbind(self.vector, role_vector)
        
        # Find closest match
        candidates = [(name, self.encoder.get_symbol(name)) 
                     for name in self.encoder.symbols]
        
        # Calculate similarities
        similarities = [(name, self.encoder.similarity(unbound, vector)) 
                       for name, vector in candidates]
        
        # Return best match
        best_match = max(similarities, key=lambda x: x[1])
        return best_match[0]
        
    def to_dict(self) -> Dict:
        """Convert message to dictionary representation."""
        return {
            "id": self.id,
            "content": self.content,
            "vector_hash": hashlib.md5(str(self.vector).encode()).hexdigest()
        }

class VSACommunicationBus:
    """Communication bus for agents using Vector Symbolic Architecture."""
    
    def __init__(self, dimension: int = 10000):
        self.encoder = VSAEncoder(dimension=dimension)
        self.messages = []
        
    def create_message(self) -> VSAMessage:
        """Create a new message."""
        return VSAMessage(self.encoder)
        
    def publish(self, message: VSAMessage) -> str:
        """
        Publish a message to the communication bus.
        
        Args:
            message: VSA message to publish
            
        Returns:
            Message ID
        """
        self.messages.append(message)
        return message.id
        
    def query(self, query_vector: np.ndarray, top_k: int = 3) -> List[VSAMessage]:
        """
        Query for similar messages.
        
        Args:
            query_vector: Query vector
            top_k: Number of top results to return
            
        Returns:
            List of similar messages
        """
        if not self.messages:
            return []
            
        # Calculate similarities
        similarities = [(message, self.encoder.similarity(query_vector, message.vector)) 
                       for message in self.messages if message.vector is not None]
        
        # Sort by similarity
        sorted_messages = sorted(similarities, key=lambda x: x[1], reverse=True)
        
        # Return top-k
        return [message for message, _ in sorted_messages[:top_k]]
        
    def create_compositional_concept(self, concepts: List[str]) -> np.ndarray:
        """
        Create a compositional concept by bundling multiple concepts.
        
        Args:
            concepts: List of concept names
            
        Returns:
            Vector representing the compositional concept
        """
        # Get vectors for each concept
        vectors = [self.encoder.get_symbol(concept) for concept in concepts]
        
        # Bundle them
        return self.encoder.bundle(vectors)
        
    def create_relational_structure(self, relations: Dict[str, str]) -> np.ndarray:
        """
        Create a relational structure using role-filler bindings.
        
        Args:
            relations: Dictionary mapping roles to fillers
            
        Returns:
            Vector representing the relational structure
        """
        bindings = []
        
        for role, filler in relations.items():
            role_vector = self.encoder.get_role(role)
            filler_vector = self.encoder.get_symbol(filler)
            binding = self.encoder.bind(role_vector, filler_vector)
            bindings.append(binding)
            
        return self.encoder.bundle(bindings)
        
    def analogy(self, a: str, b: str, c: str) -> str:
        """
        Solve analogical reasoning (A is to B as C is to ?)
        
        Args:
            a: First term
            b: Second term
            c: Third term
            
        Returns:
            Fourth term completing the analogy
        """
        # Get vectors
        a_vector = self.encoder.get_symbol(a)
        b_vector = self.encoder.get_symbol(b)
        c_vector = self.encoder.get_symbol(c)
        
        # Calculate A:B::C:?
        # Use the transformation from A to B and apply to C
        # In VSA, this is often modeled as (B * ~A) * C
        # where ~A is the pseudo-inverse of A (for bipolar vectors, it's just A again)
        transformation = self.encoder.bind(b_vector, a_vector)  # B * A (since A * A = 1 for bipolar)
        target = self.encoder.bind(transformation, c_vector)
        
        # Find closest symbol
        candidates = [(name, self.encoder.get_symbol(name)) 
                     for name in self.encoder.symbols]
        
        # Calculate similarities
        similarities = [(name, self.encoder.similarity(target, vector)) 
                       for name, vector in candidates]
        
        # Return best match
        best_match = max(similarities, key=lambda x: x[1])
        return best_match[0]

# Example usage
def example_usage():
    """Demonstrate usage of the VSA system."""
    # Initialize communication bus
    bus = VSACommunicationBus(dimension=1000)
    
    # Create a message
    message = bus.create_message()
    message.add_role_filler("sender", "agent1")
    message.add_role_filler("receiver", "agent2")
    message.add_role_filler("content", "task_completion")
    message.add_role_filler("priority", "high")
    
    # Publish to bus
    bus.publish(message)
    
    # Create a query vector
    query = bus.encoder.bind(
        bus.encoder.get_role("content"),
        bus.encoder.get_symbol("task_completion")
    )
    
    # Query for similar messages
    results = bus.query(query)
    
    print("Query Results:")
    for msg in results:
        print(f"Message: {msg.to_dict()}")
        print(f"Decoded content: {msg.get_filler('content')}")
    
    # Demonstrate analogical reasoning
    result = bus.analogy("king", "queen", "man")
    print(f"\nAnalogy Result: king:queen::man:{result}")
    
    return bus

if __name__ == "__main__":
    example_usage() 