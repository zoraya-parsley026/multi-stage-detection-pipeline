from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseDataset(ABC):
    @abstractmethod
    def __len__(self) -> int: ...

    @abstractmethod
    def __getitem__(self, index: int) -> dict[str, Any]: ...
