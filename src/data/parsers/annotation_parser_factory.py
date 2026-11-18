from __future__ import annotations

from typing import Any

from .base_annotation_parser import BaseAnnotationParser


class AnnotationParserFactory:
    _registry: dict[str, type[BaseAnnotationParser]] = {}

    @classmethod
    def register(cls, parser_type: str):
        def decorator(parser_cls: type[BaseAnnotationParser]) -> type[BaseAnnotationParser]:
            if not issubclass(parser_cls, BaseAnnotationParser):
                raise TypeError("parser must subclass BaseAnnotationParser")
            key = parser_type.lower()
            if key in cls._registry and cls._registry[key] is not parser_cls:
                raise ValueError(f"parser type '{parser_type}' is already registered")
            cls._registry[key] = parser_cls
            return parser_cls
        return decorator

    @classmethod
    def create(cls, parser_type: str, **kwargs: Any) -> BaseAnnotationParser:
        try:
            return cls._registry[parser_type.lower()](**kwargs)
        except KeyError as exc:
            raise ValueError(f"parser type '{parser_type}' not found") from exc

    @classmethod
    def list(cls) -> dict[str, type[BaseAnnotationParser]]:
        return dict(cls._registry)
