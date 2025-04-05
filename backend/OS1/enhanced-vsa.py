"""
Enhanced Vector Symbolic Architecture (VSA) with Hyperdimensional Computing

This module extends the basic VSA implementation to include advanced
hyperdimensional computing techniques, improved binding and bundling
operations, and efficient compositional reasoning.
"""

import numpy as np
import torch
import random
from typing import Dict, List, Tuple, Any, Optional, Union, Set
import uuid
import hashlib
import logging
from collections import defaultdict

logger = logging.getLogger("Dream.EnhancedVSA")

class HDVector:
    """
    Hyperdimensional Vector implementation supporting various HDC spaces.
    
    Supports multiple vector types:
    - Binary: {0, 1}
    - Bipolar: {-1, 1}
    - Complex: complex values on the unit circle
    - Holographic: real-valued vectors for holographic reduced representations
    """
    
    def __init__(self, 
                 data: Union[np.ndarray, torch.Tensor], 
                 vtype: str = "bipolar"):
        """
        Initialize hyperdimensional vector.
        
        Args:
            data: Vector data
            vtype: Vector type (binary, bipolar, complex, holographic)
        """
        if isinstance(data, torch.Tensor):
            self.data = data
            self.device = data.device
        else:
            self.data = torch.tensor(data)
            self.device = torch.device("cpu")
            
        self.vtype = vtype
        self.dimension = self.data.shape[0]
        
    @classmethod
    def random(cls, 
              dimension: int, 
              vtype: str = "bipolar", 
              device: torch.device = None) -> 'HDVector':
        """
        Create a random hyperdimensional vector.
        
        Args:
            dimension: Vector dimension
            vtype: Vector type
            device: Torch device
            
        Returns:
            Random hyperdimensional vector
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
        if vtype == "binary":
            data = torch.randint(0, 2, (dimension,), device=device).float()
        elif vtype == "bipolar":
            data = torch.randint(0, 2, (dimension,), device=device).float() * 2 - 1
        elif vtype == "complex":
            # Create complex vectors on the unit circle
            angles = torch.rand(dimension, device=device) * 2 * np.pi
            real = torch.cos(angles)
            imag = torch.sin(angles)
            data = torch.complex(real, imag)
        elif vtype == "holographic":
            # For holographic reduced representations
            data = torch.randn(dimension, device=device)
            # Normalize
            data = data / torch.norm(data)
        else:
            raise ValueError(f"Unknown vector type: {vtype}")
            
        return cls(data, vtype)
    
    def __add__(self, other: 'HDVector') -> 'HDVector':
        """Bundling operation (addition)."""
        if self.vtype != other.vtype:
            raise ValueError(f"Cannot add vectors of different types: {self.vtype} and {other.vtype}")
            
        return HDVector(self.data + other.data, self.vtype)
    
    def __mul__(self, other: Union['HDVector', float, int]) -> 'HDVector':
        """
        Binding operation (multiplication) or scalar multiplication.
        
        For HDVectors, performs element-wise multiplication (binding).
        For scalars, performs scalar multiplication.
        """
        if isinstance(other, HDVector):
            if self.vtype != other.vtype:
                raise ValueError(f"Cannot multiply vectors of different types: {self.vtype} and {other.vtype}")
                
            if self.vtype == "binary":
                # XOR for binary vectors
                return HDVector((self.data + other.data) % 2, self.vtype)
            else:
                # Element-wise multiplication for other types
                return HDVector(self.data * other.data, self.vtype)
        else:
            # Scalar multiplication
            return HDVector(self.data * other, self.vtype)
    
    def similarity(self, other: 'HDVector') -> float:
        """
        Calculate similarity between vectors.
        
        Depending on vector type, uses different similarity measures:
        - Binary/Bipolar: Cosine similarity
        - Complex: Phase correlation
        - Holographic: Cosine similarity
        
        Returns:
            Similarity score between -1 and 1
        """
        if self.vtype != other.vtype:
            raise ValueError(f"Cannot compare vectors of different types: {self.vtype} and {other.vtype}")
            
        if self.vtype == "binary":
            # Hamming similarity (ratio of matching bits)
            match = torch.sum(self.data == other.data).float()
            return (match / self.dimension).item() * 2 - 1  # Scale to [-1, 1]
        elif self.vtype == "bipolar":
            # Cosine similarity
            return torch.dot(self.data, other.data).item() / self.dimension
        elif self.vtype == "complex":
            # Phase correlation
            return torch.abs(torch.sum(self.data * other.data.conj())).item() / self.dimension
        else:  # holographic
            # Cosine similarity
            return torch.dot(self.data, other.data).item() / (torch.norm(self.data) * torch.norm(other.data)).item()
    
    def normalize(self) -> 'HDVector':
        """Normalize the vector."""
        if self.vtype == "binary":
            return self  # Binary vectors don't need normalization
        elif self.vtype == "bipolar":
            return HDVector(torch.sign(self.data), self.vtype)
        elif self.vtype == "complex":
            # Normalize to unit circle
            magnitude = torch.abs(self.data)
            return HDVector(self.data / torch.where(magnitude > 0, magnitude, 1e-10), self.vtype)
        else:  # holographic
            return HDVector(self.data / torch.norm(self.data), self.vtype)
    
    def to_device(self, device: torch.device) -> 'HDVector':
        """Move vector to a different device."""
        if self.device == device:
            return self
        return HDVector(self.data.to(device), self.vtype)
    
    def unbind(self, key: 'HDVector') -> 'HDVector':
        """
        Unbind operation (inverse of binding).
        
        Args:
            key: Binding key
            
        Returns:
            Unbound vector
        """
        if self.vtype != key.vtype:
            raise ValueError(f"Cannot unbind vectors of different types: {self.vtype} and {key.vtype}")
            
        if self.vtype == "binary":
            # XOR for binary vectors (same as binding)
            return HDVector((self.data + key.data) % 2, self.vtype)
        elif self.vtype == "bipolar":
            # Element-wise multiplication for bipolar (same as binding)
            return HDVector(self.data * key.data, self.vtype)
        elif self.vtype == "complex":
            # Complex conjugate multiplication
            return HDVector(self.data * key.data.conj(), self.vtype)
        else:  # holographic
            # Circular convolution for holographic vectors
            # Here we use a simplified approach with FFT/IFFT
            x_fft = torch.fft.fft(self.data)
            y_fft = torch.fft.fft(key.data)
            return HDVector(torch.fft.ifft(x_fft / y_fft).real, self.vtype)
    
    def to_binary(self) -> 'HDVector':
        """Convert to binary representation."""
        if self.vtype == "binary":
            return self
            
        if self.vtype == "bipolar":
            return HDVector((self.data + 1) / 2, "binary")
        else:
            # For complex and holographic, threshold at 0
            return HDVector((self.data.real > 0).float(), "binary")
    
    def to_bipolar(self) -> 'HDVector':
        """Convert to bipolar representation."""
        if self.vtype == "bipolar":
            return self
            
        if self.vtype == "binary":
            return HDVector(self.data * 2 - 1, "bipolar")
        else:
            # For complex and holographic, use sign
            return HDVector(torch.sign(self.data.real), "bipolar")

class EnhancedVSAEncoder:
    """
    Enhanced Vector Symbolic Architecture encoder with advanced HDC capabilities.
    """
    
    def __init__(self, 
                 dimension: int = 10000, 
                 vtype: str = "bipolar",
                 seed: Optional[int] = None,
                 device: Optional[torch.device] = None):
        """
        Initialize the enhanced VSA encoder.
        
        Args:
            dimension: Vector dimension
            vtype: Vector type (binary, bipolar, complex, holographic)
            seed: Random seed
            device: Torch device
        """
        self.dimension = dimension
        self.vtype = vtype
        
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
            
        # Set random seed
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
            
        # Initialize symbol table for basic primitives
        self.symbols = {}
        
        # Initialize role vectors for binding operations
        self.roles = {}
        
        # Initialize vector spaces for different concept types
        self.concept_spaces = {
            "entity": {},
            "relation": {},
            "action": {},
            "property": {},
            "time": {},
            "space": {}
        }
        
        # Initialize compositional structures
        self.compositions = {}
        
        logger.info(f"Initialized Enhanced VSA encoder with {dimension}-dimensional {vtype} vectors")
        
    def get_symbol(self, symbol_name: str) -> HDVector:
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
        vector = HDVector.random(self.dimension, self.vtype, self.device)
        
        # Store for future use
        self.symbols[symbol_name] = vector
        
        return vector
        
    def get_role(self, role_name: str) -> HDVector:
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
        vector = HDVector.random(self.dimension, self.vtype, self.device)
        
        # Store for future use
        self.roles[role_name] = vector
        
        return vector
    
    def get_concept(self, concept_name: str, concept_type: str = "entity") -> HDVector:
        """
        Get or create a concept vector within a specific concept space.
        
        Args:
            concept_name: Name of the concept
            concept_type: Type of concept (entity, relation, action, etc.)
            
        Returns:
            Hypervector representing the concept
        """
        if concept_type not in self.concept_spaces:
            raise ValueError(f"Unknown concept type: {concept_type}")
            
        concept_space = self.concept_spaces[concept_type]
        
        if concept_name in concept_space:
            return concept_space[concept_name]
            
        # Create a new random hypervector
        vector = HDVector.random(self.dimension, self.vtype, self.device)
        
        # Store for future use
        concept_space[concept_name] = vector
        
        return vector
        
    def bind(self, vector1: HDVector, vector2: HDVector) -> HDVector:
        """
        Bind two vectors.
        
        Args:
            vector1: First vector
            vector2: Second vector
            
        Returns:
            Bound vector
        """
        return vector1 * vector2
        
    def unbind(self, bound_vector: HDVector, key_vector: HDVector) -> HDVector:
        """
        Unbind a vector using the key vector.
        
        Args:
            bound_vector: Bound vector
            key_vector: Key vector used for binding
            
        Returns:
            Unbound vector
        """
        return bound_vector.unbind(key_vector)
        
    def bundle(self, vectors: List[HDVector]) -> HDVector:
        """
        Bundle vectors.
        
        Args:
            vectors: List of vectors to bundle
            
        Returns:
            Bundled vector
        """
        if not vectors:
            return HDVector.random(self.dimension, self.vtype, self.device) * 0
            
        # Add all vectors
        result = vectors[0]
        for vector in vectors[1:]:
            result = result + vector
            
        # Normalize
        return result.normalize()
        
    def create_relation(self, subject: str, relation: str, object: str) -> HDVector:
        """
        Create a relation triple as a compositional vector.
        
        Args:
            subject: Subject entity
            relation: Relation type
            object: Object entity
            
        Returns:
            Vector representation of the relation
        """
        # Get concept vectors
        subj_vec = self.get_concept(subject, "entity")
        rel_vec = self.get_concept(relation, "relation")
        obj_vec = self.get_concept(object, "entity")
        
        # Get role vectors
        subj_role = self.get_role("subject")
        rel_role = self.get_role("relation")
        obj_role = self.get_role("object")
        
        # Bind each concept to its role
        subj_binding = self.bind(subj_vec, subj_role)
        rel_binding = self.bind(rel_vec, rel_role)
        obj_binding = self.bind(obj_vec, obj_role)
        
        # Bundle all bindings
        triple = self.bundle([subj_binding, rel_binding, obj_binding])
        
        # Store in compositions
        triple_key = f"{subject}:{relation}:{object}"
        self.compositions[triple_key] = triple
        
        return triple
    
    def create_hierarchy(self, concept: str, is_a: str) -> HDVector:
        """
        Create a hierarchical relationship (is-a).
        
        Args:
            concept: The concept
            is_a: The parent concept
            
        Returns:
            Vector representation of the hierarchy
        """
        return self.create_relation(concept, "is_a", is_a)
    
    def create_holistic_scene(self, scene_elements: Dict[str, str]) -> HDVector:
        """
        Create a holistic scene representation from multiple elements.
        
        Args:
            scene_elements: Dictionary mapping roles to fillers
            
        Returns:
            Vector representation of the scene
        """
        bindings = []
        
        for role, filler in scene_elements.items():
            role_vec = self.get_role(role)
            filler_vec = self.get_symbol(filler)
            binding = self.bind(role_vec, filler_vec)
            bindings.append(binding)
            
        scene = self.bundle(bindings)
        return scene
    
    def query_scene(self, scene: HDVector, role: str) -> List[Tuple[str, float]]:
        """
        Query a scene representation for a specific role.
        
        Args:
            scene: Scene vector
            role: Role to query
            
        Returns:
            List of (symbol_name, similarity) tuples
        """
        role_vec = self.get_role(role)
        unbound = self.unbind(scene, role_vec)
        
        # Find closest symbols
        results = []
        for name, vector in self.symbols.items():
            sim = unbound.similarity(vector)
            results.append((name, sim))
            
        # Sort by similarity
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results
    
    def analogy(self, a: str, b: str, c: str) -> List[Tuple[str, float]]:
        """
        Solve analogy problems (A is to B as C is to ?).
        
        Args:
            a: First term
            b: Second term
            c: Third term
            
        Returns:
            List of (candidate, similarity) tuples for the fourth term
        """
        # Get vectors
        a_vec = self.get_symbol(a)
        b_vec = self.get_symbol(b)
        c_vec = self.get_symbol(c)
        
        # Calculate A:B::C:?
        # A is to B as C is to D means B - A + C = D
        if self.vtype == "binary":
            # For binary vectors, use XOR
            a_b_relation = (a_vec.data + b_vec.data) % 2
            target = HDVector((a_b_relation + c_vec.data) % 2, "binary")
        else:
            # For other types, use the appropriate operations
            # We unbind B from A to get the transformation, then apply to C
            transformation = self.unbind(b_vec, a_vec)
            target = self.bind(transformation, c_vec)
            
        # Find closest symbols
        results = []
        for name, vector in self.symbols.items():
            if name in [a, b, c]:  # Skip input terms
                continue
                
            sim = target.similarity(vector)
            results.append((name, sim))
            
        # Sort by similarity
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results
    
    def prototype(self, instances: List[str]) -> HDVector:
        """
        Create a prototype vector from multiple instances.
        
        Args:
            instances: List of instance names
            
        Returns:
            Prototype vector
        """
        vectors = [self.get_symbol(instance) for instance in instances]
        return self.bundle(vectors)
    
    def create_composite_concept(self, concepts: List[str], weights: Optional[List[float]] = None) -> HDVector:
        """
        Create a weighted composite concept from multiple concepts.
        
        Args:
            concepts: List of concept names
            weights: Optional list of weights
            
        Returns:
            Composite concept vector
        """
        if weights is None:
            weights = [1.0] * len(concepts)
            
        if len(concepts) != len(weights):
            raise ValueError("Number of concepts must match number of weights")
            
        vectors = []
        for concept, weight in zip(concepts, weights):
            vector = self.get_symbol(concept)
            vectors.append(vector * weight)
            
        return self.bundle(vectors)
    
    def hierarchical_inference(self, concept: str, hierarchy_depth: int = 3) -> List[str]:
        """
        Perform hierarchical inference to find the categories a concept belongs to.
        
        Args:
            concept: Concept name
            hierarchy_depth: Maximum depth of hierarchy to traverse
            
        Returns:
            List of inferred category memberships
        """
        # Start with the concept
        current = concept
        hierarchy = [current]
        
        # Traverse the hierarchy
        for _ in range(hierarchy_depth):
            # Look for is-a relationships
            triple_key = f"{current}:is_a:"
            matching_keys = [k for k in self.compositions.keys() if k.startswith(triple_key)]
            
            if not matching_keys:
                break
                
            # Get the parent category
            parent = matching_keys[0].split(":")[-1]
            hierarchy.append(parent)
            
            # Move up the hierarchy
            current = parent
            
        return hierarchy
    
    def analogical_inference(self, base_case: Tuple[str, str, str], target: str) -> str:
        """
        Perform analogical inference.
        
        Args:
            base_case: (entity, relation, value) tuple representing the base case
            target: Target entity
            
        Returns:
            Inferred value for the target
        """
        entity, relation, value = base_case
        
        # Create relation vector
        base_relation = self.create_relation(entity, relation, value)
        
        # Get entity vectors
        base_entity = self.get_concept(entity, "entity")
        target_entity = self.get_concept(target, "entity")
        
        # Perform analogical inference
        # Extract the relation pattern from the base case
        relation_pattern = self.unbind(base_relation, self.bind(base_entity, self.get_role("subject")))
        
        # Apply the pattern to the target
        target_binding = self.bind(target_entity, self.get_role("subject"))
        inference = self.bind(relation_pattern, target_binding)
        
        # Find the closest object
        results = []
        for obj in self.concept_spaces["entity"]:
            obj_vec = self.get_concept(obj, "entity")
            obj_binding = self.bind(obj_vec, self.get_role("object"))
            
            sim = obj_binding.similarity(inference)
            results.append((obj, sim))
            
        # Get the best match
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results[0][0] if results else None

class VSAMessage:
    """Enhanced message using Vector Symbolic Architecture."""
    
    def __init__(self, encoder: EnhancedVSAEncoder):
        """
        Initialize a VSA message.
        
        Args:
            encoder: VSA encoder
        """
        self.id = str(uuid.uuid4())
        self.encoder = encoder
        self.vector = None
        self.content = {}
        self.timestamp = None
        self.sender = None
        self.recipients = []
        
    def add_role_filler(self, role: str, filler: str) -> None:
        """
        Add a role-filler binding to the message.
        
        Args:
            role: Role name
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
            self.vector = self.vector + binding
            
        # Normalize
        self.vector = self.vector.normalize()
        
    def add_metadata(self, sender: str, recipients: List[str] = None, timestamp = None) -> None:
        """
        Add metadata to the message.
        
        Args:
            sender: Sender name
            recipients: List of recipient names
            timestamp: Message timestamp
        """
        self.sender = sender
        self.recipients = recipients or []
        self.timestamp = timestamp or time.time()
        
        # Add to vector representation
        self.add_role_filler("sender", sender)
        
        for i, recipient in enumerate(self.recipients):
            self.add_role_filler(f"recipient_{i}", recipient)
            
    def add_structured_content(self, structure_type: str, content: Dict[str, str]) -> None:
        """
        Add structured content to the message.
        
        Args:
            structure_type: Type of structure (e.g., 'argument', 'fact', 'question')
            content: Dictionary mapping roles to values
        """
        # Add structure type
        self.add_role_filler("structure_type", structure_type)
        
        # Add content
        for role, value in content.items():
            self.add_role_filler(f"{structure_type}_{role}", value)
    
    def get_filler(self, role: str) -> Optional[str]:
        """
        Get the filler for a specific role.
        
        Args:
            role: Role name
            
        Returns:
            Filler value if found, None otherwise
        """
        return self.content.get(role)
    
    def decode(self, role: str) -> List[Tuple[str, float]]:
        """
        Decode the filler for a specific role from the vector.
        
        Args:
            role: Role name
            
        Returns:
            List of (filler, similarity) tuples
        """
        if self.vector is None:
            return []
            
        # Get role vector
        role_vector = self.encoder.get_role(role)
        
        # Unbind
        unbound = self.encoder.unbind(self.vector, role_vector)
        
        # Find closest matches
        results = []
        for name, vector in self.encoder.symbols.items():
            sim = unbound.similarity(vector)
            if sim > 0.2:  # Threshold for meaningful similarity
                results.append((name, sim))
                
        # Sort by similarity
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results
    
    def to_dict(self) -> Dict:
        """Convert message to dictionary representation."""
        return {
            "id": self.id,
            "content": self.content,
            "sender": self.sender,
            "recipients": self.recipients,
            "timestamp": self.timestamp,
            "vector_hash": hashlib.md5(str(self.vector.data.tolist())).hexdigest() if self.vector is not None else None
        }
    
    def similarity(self, other: 'VSAMessage') -> float:
        """
        Calculate similarity with another message.
        
        Args:
            other: Other message
            
        Returns:
            Similarity score
        """
        if self.vector is None or other.vector is None:
            return 0.0
            
        return self.vector.similarity(other.vector)

class EnhancedVSACommunicationBus:
    """Enhanced communication bus for agents using Vector Symbolic Architecture."""
    
    def __init__(self, dimension: int = 10000, vtype: str = "bipolar"):
        """
        Initialize the communication bus.
        
        Args:
            dimension: Vector dimension
            vtype: Vector type
        """
        self.encoder = EnhancedVSAEncoder(dimension=dimension, vtype=vtype)
        self.messages = []
        self.agent_spaces = {}
        self.topic_spaces = {}
        self.active_dialogues = {}
        self.semantic_index = defaultdict(list)  # Maps concepts to messages
        
    def register_agent(self, agent_id: str) -> HDVector:
        """
        Register an agent with the communication bus.
        
        Args:
            agent_id: Agent identifier
            
        Returns:
            Agent's identity vector
        """
        if agent_id in self.agent_spaces:
            return self.agent_spaces[agent_id]
            
        # Create agent vector
        agent_vector = self.encoder.get_concept(agent_id, "entity")
        self.agent_spaces[agent_id] = agent_vector
        
        return agent_vector
    
    def register_topic(self, topic: str) -> HDVector:
        """
        Register a topic with the communication bus.
        
        Args:
            topic: Topic name
            
        Returns:
            Topic vector
        """
        if topic in self.topic_spaces:
            return self.topic_spaces[topic]
            
        # Create topic vector
        topic_vector = self.encoder.get_concept(topic, "entity")
        self.topic_spaces[topic] = topic_vector
        
        return topic_vector
    
    def create_message(self, sender: str, recipients: List[str] = None) -> VSAMessage:
        """
        Create a new message.
        
        Args:
            sender: Sender agent ID
            recipients: List of recipient agent IDs
            
        Returns:
            New message
        """
        # Register sender and recipients if needed
        self.register_agent(sender)
        
        if recipients:
            for recipient in recipients:
                self.register_agent(recipient)
                
        # Create message
        message = VSAMessage(self.encoder)
        message.add_metadata(sender, recipients)
        
        return message
    
    def create_argument(self, sender: str, claim: str, evidence: List[str], stance: str = "support") -> VSAMessage:
        """
        Create an argument message.
        
        Args:
            sender: Sender agent ID
            claim: Claim text
            evidence: List of evidence points
            stance: Stance (support or oppose)
            
        Returns:
            Argument message
        """
        message = self.create_message(sender)
        
        # Add argument structure
        message.add_structured_content("argument", {
            "claim": claim,
            "stance": stance
        })
        
        # Add evidence
        for i, point in enumerate(evidence):
            message.add_role_filler(f"argument_evidence_{i}", point)
            
        return message
    
    def create_question(self, sender: str, question: str, context: str = None) -> VSAMessage:
        """
        Create a question message.
        
        Args:
            sender: Sender agent ID
            question: Question text
            context: Optional context
            
        Returns:
            Question message
        """
        message = self.create_message(sender)
        
        # Add question structure
        content = {"question": question}
        if context:
            content["context"] = context
            
        message.add_structured_content("question", content)
        
        return message
    
    def create_answer(self, sender: str, question_id: str, answer: str, confidence: float = 1.0) -> VSAMessage:
        """
        Create an answer message.
        
        Args:
            sender: Sender agent ID
            question_id: ID of the question being answered
            answer: Answer text
            confidence: Confidence level (0-1)
            
        Returns:
            Answer message
        """
        message = self.create_message(sender)
        
        # Add answer structure
        message.add_structured_content("answer", {
            "question_id": question_id,
            "answer": answer,
            "confidence": str(confidence)
        })
        
        return message
        
    def publish(self, message: VSAMessage) -> str:
        """
        Publish a message to the communication bus.
        
        Args:
            message: VSA message to publish
            
        Returns:
            Message ID
        """
        # Add to message list
        self.messages.append(message)
        
        # Index for semantic search
        for role, filler in message.content.items():
            self.semantic_index[filler].append(message.id)
            
        return message.id
    
    def start_dialogue(self, topic: str, participants: List[str]) -> str:
        """
        Start a new dialogue.
        
        Args:
            topic: Dialogue topic
            participants: List of participant agent IDs
            
        Returns:
            Dialogue ID
        """
        dialogue_id = str(uuid.uuid4())
        
        # Register topic
        topic_vector = self.register_topic(topic)
        
        # Create dialogue
        dialogue = {
            "id": dialogue_id,
            "topic": topic,
            "topic_vector": topic_vector,
            "participants": participants,
            "messages": [],
            "state": "active",
            "created_at": time.time()
        }
        
        self.active_dialogues[dialogue_id] = dialogue
        
        return dialogue_id
    
    def add_to_dialogue(self, dialogue_id: str, message: VSAMessage) -> bool:
        """
        Add a message to a dialogue.
        
        Args:
            dialogue_id: Dialogue ID
            message: Message to add
            
        Returns:
            True if successful, False otherwise
        """
        if dialogue_id not in self.active_dialogues:
            return False
            
        dialogue = self.active_dialogues[dialogue_id]
        
        # Add dialogue reference to message
        message.add_role_filler("dialogue_id", dialogue_id)
        message.add_role_filler("dialogue_topic", dialogue["topic"])
        
        # Publish message
        message_id = self.publish(message)
        
        # Add to dialogue
        dialogue["messages"].append(message_id)
        
        return True
    
    def query(self, query_vector: HDVector, top_k: int = 3) -> List[VSAMessage]:
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
        similarities = [(message, message.vector.similarity(query_vector)) 
                       for message in self.messages 
                       if message.vector is not None]
        
        # Sort by similarity
        sorted_messages = sorted(similarities, key=lambda x: x[1], reverse=True)
        
        # Return top-k
        return [message for message, _ in sorted_messages[:top_k]]
    
    def search_by_content(self, content: str, top_k: int = 5) -> List[Tuple[VSAMessage, float]]:
        """
        Search messages by content.
        
        Args:
            content: Content to search for
            top_k: Number of top results to return
            
        Returns:
            List of (message, similarity) tuples
        """
        # Get content vector
        content_vector = self.encoder.get_symbol(content)
        
        # Search by vector similarity
        messages = self.query(content_vector, top_k=top_k)
        
        # Calculate similarities
        results = [(message, content_vector.similarity(message.vector)) for message in messages]
        
        # Sort by similarity
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results
    
    def search_by_concept(self, concept: str) -> List[VSAMessage]:
        """
        Search messages by indexed concept.
        
        Args:
            concept: Concept to search for
            
        Returns:
            List of messages containing the concept
        """
        if concept not in self.semantic_index:
            return []
            
        # Get message IDs
        message_ids = self.semantic_index[concept]
        
        # Get messages
        return [message for message in self.messages if message.id in message_ids]
    
    def get_dialogue_summary(self, dialogue_id: str) -> Dict:
        """
        Get a summary of a dialogue.
        
        Args:
            dialogue_id: Dialogue ID
            
        Returns:
            Dialogue summary
        """
        if dialogue_id not in self.active_dialogues:
            return {}
            
        dialogue = self.active_dialogues[dialogue_id]
        
        # Get messages
        message_objects = [msg for msg in self.messages if msg.id in dialogue["messages"]]
        
        # Create bundle of all message vectors
        if message_objects:
            vectors = [msg.vector for msg in message_objects if msg.vector is not None]
            if vectors:
                bundle = self.encoder.bundle(vectors)
            else:
                bundle = None
        else:
            bundle = None
            
        # Create summary
        summary = {
            "id": dialogue_id,
            "topic": dialogue["topic"],
            "participants": dialogue["participants"],
            "message_count": len(dialogue["messages"]),
            "created_at": dialogue["created_at"],
            "state": dialogue["state"],
            "semantic_bundle": bundle
        }
        
        return summary
    
    def find_similar_dialogues(self, dialogue_id: str, top_k: int = 3) -> List[Dict]:
        """
        Find dialogues similar to the given dialogue.
        
        Args:
            dialogue_id: Dialogue ID
            top_k: Number of top results to return
            
        Returns:
            List of similar dialogue summaries
        """
        if dialogue_id not in self.active_dialogues:
            return []
            
        # Get dialogue bundle
        summary = self.get_dialogue_summary(dialogue_id)
        bundle = summary["semantic_bundle"]
        
        if bundle is None:
            return []
            
        # Calculate similarities with other dialogues
        similarities = []
        for d_id, dialogue in self.active_dialogues.items():
            if d_id == dialogue_id:
                continue
                
            # Get bundle
            d_summary = self.get_dialogue_summary(d_id)
            d_bundle = d_summary["semantic_bundle"]
            
            if d_bundle is None:
                continue
                
            # Calculate similarity
            sim = bundle.similarity(d_bundle)
            similarities.append((d_summary, sim))
            
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return [summary for summary, _ in similarities[:top_k]]

# Example usage
def example_usage():
    """Demonstrate usage of the enhanced VSA system."""
    # Initialize communication bus
    bus = EnhancedVSACommunicationBus(dimension=1000, vtype="bipolar")
    
    # Create some agents
    agents = ["critic", "synthesizer", "pattern_recognizer", "debate_master"]
    for agent in agents:
        bus.register_agent(agent)
    
    # Start a dialogue
    dialogue_id = bus.start_dialogue("consciousness_in_ai", agents)
    
    # Create and publish messages
    msg1 = bus.create_argument(
        sender="critic",
        claim="Current AI systems lack true consciousness",
        evidence=["They don't have subjective experience", "They lack introspection abilities"],
        stance="support"
    )
    bus.add_to_dialogue(dialogue_id, msg1)
    
    msg2 = bus.create_argument(
        sender="pattern_recognizer",
        claim="Emergent consciousness may be possible",
        evidence=["Complex systems show emergent properties", "Consciousness may be an emergent phenomenon"],
        stance="oppose"
    )
    bus.add_to_dialogue(dialogue_id, msg2)
    
    # Create a query vector
    query = bus.encoder.get_symbol("consciousness")
    
    # Search for relevant messages
    results = bus.query(query)
    
    print("Query Results:")
    for msg in results:
        print(f"Message from {msg.sender}: {msg.get_filler('argument_claim')}")
    
    # Get dialogue summary
    summary = bus.get_dialogue_summary(dialogue_id)
    print(f"\nDialogue Summary: {summary['topic']} with {summary['message_count']} messages")
    
    return bus

if __name__ == "__main__":
    example_usage()
