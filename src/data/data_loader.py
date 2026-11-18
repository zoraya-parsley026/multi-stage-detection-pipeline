"""Backward-compatible import for :mod:`src.data.data_loaders`."""

from .data_loaders.base_data_loader import BaseDataLoader, DataLoader

__all__ = ["BaseDataLoader", "DataLoader"]
