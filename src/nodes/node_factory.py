from __future__ import annotations

import builtins
from typing import Any

from src.nodes.base_node import PipelineNode


class NodeFactory:
    _registry: dict[str, type[PipelineNode]] = {}

    # - - - #

    @classmethod
    def register(cls, node_type: str | builtins.list[str] | None = None):
        
        def decorator(node_cls: type[PipelineNode]) -> type[PipelineNode]:
            if isinstance(node_type, (list, tuple)):
                types = list(node_type)
            elif node_type:
                types = [node_type]
            else:
                types = [node_cls.__name__]

            if node_cls.__name__ not in types:
                types.append(node_cls.__name__)

            for t in types:
                t_lower = t.lower()
                for key in (t, t_lower):
                    if key in cls._registry and cls._registry[key] is not node_cls:
                        raise ValueError(f"node type '{key}' is already registered")
                    cls._registry[key] = node_cls

            return node_cls

        return decorator

    # - - - #

    @classmethod
    def create(cls, node_type: str, **kwargs: Any) -> PipelineNode:
        node_cls = cls.get(node_type)
        try:
            return node_cls(**kwargs)
        except TypeError as exc:
            raise ValueError(
                f"failed to instantiate {node_type}, {exc}"
            ) from exc

    # - - - #

    @classmethod
    def get(cls, node_type: str) -> type[PipelineNode]:
        if node_type in cls._registry:
            return cls._registry[node_type]

        node_type_lower = node_type.lower()
        if node_type_lower in cls._registry:
            return cls._registry[node_type_lower]

        raise ValueError(f"node type '{node_type}' not found")

    # - - - #

    @classmethod
    def list(cls) -> dict[str, type[PipelineNode]]:
        return dict(cls._registry)

    # - - - #

    @classmethod
    def _clear(cls) -> None:
        cls._registry.clear()



