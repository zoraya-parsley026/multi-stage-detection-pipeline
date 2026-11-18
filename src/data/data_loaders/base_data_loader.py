from __future__ import annotations

import random
from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import Any


class BaseDataLoader(ABC):
    @abstractmethod
    def __len__(self) -> int: ...

    @abstractmethod
    def __iter__(self) -> Iterator[dict[str, Any]]: ...


class DataLoader(BaseDataLoader):
    
    def __init__(self, dataset: Any, shuffle: bool = False, seed: int | None = None) -> None:
        from ..datasets.base_dataset import BaseDataset
        if not isinstance(dataset, BaseDataset):
            raise TypeError(
                "DataLoader expects an instantiated BaseDataset, "
                f"got {type(dataset).__name__}."
            )
            
        self.dataset = dataset
        self.shuffle = shuffle
        self._rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.dataset)

    def __iter__(self) -> Iterator[dict[str, Any]]:
        indices = list(range(len(self.dataset)))
        if self.shuffle:
            self._rng.shuffle(indices)
        for index in indices:
            yield self.dataset[index]
