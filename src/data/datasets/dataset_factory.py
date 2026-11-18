from __future__ import annotations

from typing import Any

from .base_dataset import BaseDataset


class DatasetFactory:
    _registry: dict[str, type[BaseDataset]] = {}

    @classmethod
    def register(cls, dataset_type: str):
        def decorator(dataset_cls: type[BaseDataset]) -> type[BaseDataset]:
            key = dataset_type.lower()
            if key in cls._registry and cls._registry[key] is not dataset_cls:
                raise ValueError(f"dataset type '{dataset_type}' is already registered")
            cls._registry[key] = dataset_cls
            return dataset_cls
        return decorator

    @classmethod
    def create(cls, dataset_type: str, **kwargs: Any) -> BaseDataset:
        try:
            dataset_cls = cls._registry[dataset_type.lower()]
        except KeyError as exc:
            raise ValueError(f"dataset type '{dataset_type}' not found") from exc
        try:
            return dataset_cls(**kwargs)
        except TypeError as exc:
            raise ValueError(f"failed to instantiate {dataset_type}, {exc}") from exc

    @classmethod
    def list(cls) -> dict[str, type[BaseDataset]]:
        return dict(cls._registry)
