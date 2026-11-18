"""
src.nodes.strategies
====================
Concrete pipeline node strategies grouped by functional family:
  - inference.py: ModelDetectionNode, ModelClassificationNode
  - transforms.py: CropNode, TaxonomyTranslationNode
  - filters.py: BaseFilterNode, LogicFilterNode

Importing this package automatically registers all strategies with NodeFactory.
"""

from src.nodes.strategies.filters import (
    BaseFilterNode,
    LogicFilterNode,
)
from src.nodes.strategies.inference import (
    ModelClassificationNode,
    ModelDetectionNode,
    ModelImageClassificationNode,
)
from src.nodes.strategies.transforms import (
    CropNode,
    TaxonomyTranslationNode,
)

__all__ = [
    # Inference
    "ModelDetectionNode",
    "ModelClassificationNode",
    "ModelImageClassificationNode",
    # Transforms
    "CropNode",
    "TaxonomyTranslationNode",
    # Filters
    "BaseFilterNode",
    "LogicFilterNode",
]

