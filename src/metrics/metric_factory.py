from __future__ import annotations

import builtins
from typing import Any

from src.metrics.base_metric import BaseMetric, BaseSystemMetric


class MetricFactory:
    _registry: dict[str, type[BaseMetric | BaseSystemMetric]] = {}

    @classmethod
    def register(cls, metric_type: str | builtins.list[str] | None = None):

        def decorator(metric_cls: type[BaseMetric | BaseSystemMetric]) -> type[BaseMetric | BaseSystemMetric]:
            if isinstance(metric_type, (list, tuple)):
                types = list(metric_type)
            elif metric_type:
                types = [metric_type]
            else:
                types = [metric_cls.__name__]

            if metric_cls.__name__ not in types:
                types.append(metric_cls.__name__)

            for t in types:
                for key in (t, t.lower()):
                    if key in cls._registry and cls._registry[key] is not metric_cls:
                        raise ValueError(f"metric type '{key}' is already registered")
                    cls._registry[key] = metric_cls

            return metric_cls

        return decorator

    # - - - #

    @classmethod
    def create(cls, metric_type: str, **kwargs: Any) -> BaseMetric | BaseSystemMetric:

        metric_cls = cls.get(metric_type)
        try:
            return metric_cls(**kwargs)
        except TypeError as exc:
            raise ValueError(
                f"failed to instantiate {metric_type}, {exc}"
            ) from exc

    # - - - #

    @classmethod
    def create_suite(cls, metric_configs: builtins.list[str | dict[str, Any]]) -> builtins.list[BaseMetric | BaseSystemMetric]:

        suite: list[BaseMetric | BaseSystemMetric] = []
        for item in metric_configs:
            if isinstance(item, str):
                suite.append(cls.create(item))
            elif isinstance(item, dict):
                cfg = dict(item)
                name = cfg.pop("name") if "name" in cfg else cfg.pop("type")
                suite.append(cls.create(name, **cfg))
            else:
                raise TypeError(f"expected str or dict for metric config, got {type(item)}")
        return suite

    # - - - #

    @classmethod
    def get(cls, metric_type: str) -> type[BaseMetric | BaseSystemMetric]:
        if metric_type in cls._registry:
            return cls._registry[metric_type]

        metric_type_lower = metric_type.lower()
        if metric_type_lower in cls._registry:
            return cls._registry[metric_type_lower]

        raise ValueError(f"metric type '{metric_type}' not found")

    # - - - #

    @classmethod
    def list(cls) -> dict[str, type[BaseMetric | BaseSystemMetric]]:
        return dict(cls._registry)

    # - - - #

    @classmethod
    def _clear(cls) -> None:
        cls._registry.clear()


