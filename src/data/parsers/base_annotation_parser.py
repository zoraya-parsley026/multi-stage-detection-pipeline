from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path


class BaseAnnotationParser(ABC):
    @abstractmethod
    def parse(self, label_path: str | Path, original_shape: tuple[int, int]) -> list[dict] | None: ...
