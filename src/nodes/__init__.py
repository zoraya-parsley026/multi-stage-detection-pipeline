"""
src.nodes
=========
Public API for all atomic pipeline nodes, the node factory, and node strategies.
"""

from src.nodes.base_node import PipelineNode
from src.nodes.node_factory import NodeFactory
from src.nodes.strategies import (
    BaseFilterNode,
    CropNode,
    LogicFilterNode,
    ModelClassificationNode,
    ModelDetectionNode,
    ModelImageClassificationNode,
    TaxonomyTranslationNode,
)

__all__ = [
    # Base class
    "PipelineNode",
    # Factory & Registry
    "NodeFactory",
    # Base strategy types
    "BaseFilterNode",
    # Concrete nodes
    "ModelDetectionNode",
    "ModelClassificationNode",
    "ModelImageClassificationNode",
    "CropNode",
    "TaxonomyTranslationNode",
    "LogicFilterNode",
]

