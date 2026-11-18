from abc import ABC, abstractmethod

from src.metrics.matcher import BaseMatcher
from src.metrics.metric_result import MetricResult


class BaseMetric(ABC):
    @abstractmethod
    def compute(self, matcher: BaseMatcher) -> list[MetricResult]:
        raise NotImplementedError


class BaseSystemMetric(ABC):
    @abstractmethod
    def compute(self, records) -> list[MetricResult]:
        raise NotImplementedError

