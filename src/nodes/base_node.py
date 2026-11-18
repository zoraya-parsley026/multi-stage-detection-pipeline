from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.pipeline.pipeline_context import PipelineContext


class PipelineNode(ABC):
    name: str = "UnnamedNode"

    in_keys:  list[str] = []
    out_keys: list[str] = []

    # - - - #

    @abstractmethod
    def process(self, context: PipelineContext) -> None:
        ...

