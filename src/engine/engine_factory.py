from __future__ import annotations

from typing import Any

from src.engine.base_engine import BaseEngine


class EngineFactory:
    _registry: dict[str, type[BaseEngine]] = {}

    # - - - #

    @classmethod
    def register(cls, engine_type: str | list[str] | None = None):

        def decorator(engine_cls: type[BaseEngine]) -> type[BaseEngine]:
            if not issubclass(engine_cls, BaseEngine):
                raise TypeError("engine must subclass BaseEngine")

            if isinstance(engine_type, (list, tuple)):
                names = list(engine_type)
            elif engine_type:
                names = [engine_type]
            else:
                names = []
            
            names.append(engine_cls.__name__)
            keys = {key for name in names for key in (name, name.lower())}

            for key in keys:
                if key in cls._registry and cls._registry[key] is not engine_cls:
                    raise ValueError(f"engine type '{key}' is already registered")
            cls._registry.update({key: engine_cls for key in keys})
            return engine_cls

        return decorator

    # - - - #

    @classmethod
    def create(cls, engine_type: str, **kwargs: Any) -> BaseEngine:
        engine_cls = cls.get(engine_type)
        try:
            return engine_cls(**kwargs)
        except TypeError as exc:
            raise ValueError(
                f"failed to instantiate {engine_type}, {exc}"
            ) from exc

    # - - - #

    @classmethod
    def get(cls, engine_type: str) -> type[BaseEngine]:
        if engine_type in cls._registry:
            return cls._registry[engine_type]

        engine_type_lower = engine_type.lower()
        if engine_type_lower in cls._registry:
            return cls._registry[engine_type_lower]
        raise ValueError(f"engine type '{engine_type}' not found")

    # - - - #

    @classmethod
    def list(cls) -> dict[str, type[BaseEngine]]:
        return dict(cls._registry)

    # - - - #

    @classmethod
    def _clear(cls) -> None:
        cls._registry.clear()


