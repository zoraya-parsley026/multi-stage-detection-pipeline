"""Data ingestion package.

The implementation is split by responsibility:

* :mod:`src.data.data_loaders` exposes iteration and public data APIs.
* :mod:`src.data.datasets` contains dataset contracts and implementations.
* :mod:`src.data.parsers` contains annotation parsers.
"""

from .data_loaders import (
    AnnotationParserFactory,
    BaseAnnotationParser,
    BaseDataLoader,
    BaseDataset,
    DataLoader,
    DatasetFactory,
    DetectionDataset,
    YOLOLabelParser,
)

__all__ = [
    "BaseAnnotationParser", "BaseDataLoader", "BaseDataset", "DataLoader",
    "DatasetFactory", "DetectionDataset", "AnnotationParserFactory",
    "YOLOLabelParser",
]
