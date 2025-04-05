"""
OS1: Memory and Operations Center for EvolvOS

This package provides the core memory, retrieval, and evolution components
for the EvolvOS self-evolving AI system.
"""

from .os1_integration import OS1Integration, OS1Config
from .enhanced_memory import EnhancedMemoryInterface
from .advanced_retrieval import AdvancedRetrieval
from .neural_compressor import AdaptiveCompressor
from .self_evolution_controller import SelfEvolutionController
from .enhanced_vsa import EnhancedVSACommunicationBus

__version__ = "0.1.0"
__all__ = [
    'OS1Integration',
    'OS1Config',
    'EnhancedMemoryInterface',
    'AdvancedRetrieval',
    'AdaptiveCompressor', 
    'SelfEvolutionController',
    'EnhancedVSACommunicationBus'
]