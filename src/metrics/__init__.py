"""
src.metrics
===========
Public API for evaluation metrics, matchers, evaluators, and reporting.
"""

from src.metrics.base_metric import BaseMetric, BaseSystemMetric
from src.metrics.evaluator import EvaluationReport, Evaluator
from src.metrics.matcher import (
    BaseMatcher,
    ClassificationMatcher,
    ClassificationSummary,
    DetectionMatcher,
    DetectionSummary,
)
from src.metrics.metric_factory import MetricFactory
from src.metrics.metric_result import MetricResult
from src.metrics.reporting.plotter import MetricPlotter
from src.metrics.reporting.reporter import Reporter
from src.metrics.strategies import (
    ClassificationAccuracyMetric,
    ClassificationBalancedAccuracyMetric,
    ClassificationConfusionMatrixMetric,
    ClassificationCountMetric,
    ClassificationF1Metric,
    ClassificationPrecisionMetric,
    ClassificationRecallMetric,
    ClassificationTopKAccuracyMetric,
    DetectionAPMetric,
    DetectionBoxAccuracyMetric,
    DetectionBoxTopKAccuracyMetric,
    DetectionConfusionMatrixMetric,
    DetectionCountMetric,
    DetectionF1Metric,
    DetectionMacroF1Metric,
    DetectionMacroPrecisionMetric,
    DetectionMacroRecallMetric,
    DetectionPrecisionMetric,
    DetectionRecallMetric,
    FilterEfficiencyMetric,
    LatencyMetric,
)

__all__ = [
    "EvaluationReport",
    "ClassificationSummary",
    "ClassificationAccuracyMetric",
    "ClassificationBalancedAccuracyMetric",
    "ClassificationPrecisionMetric",
    "ClassificationRecallMetric",
    "ClassificationF1Metric",
    "ClassificationCountMetric",
    "ClassificationConfusionMatrixMetric",
    "ClassificationTopKAccuracyMetric",
    # Base Classes & States
    "BaseMetric",
    "BaseSystemMetric",
    "MetricResult",
    "BaseMatcher",
    "DetectionMatcher",
    "ClassificationMatcher",
    "DetectionSummary",
    # Evaluator & Reporting
    "Evaluator",
    "Reporter",
    "MetricPlotter",
    # Factory & Registry
    "MetricFactory",
    # Base Strategy Classes
    # Concrete Strategies
    "DetectionAPMetric",
    "DetectionPrecisionMetric",
    "DetectionRecallMetric",
    "DetectionF1Metric",
    "DetectionMacroPrecisionMetric",
    "DetectionMacroRecallMetric",
    "DetectionMacroF1Metric",
    "DetectionCountMetric",
    "DetectionBoxAccuracyMetric",
    "DetectionBoxTopKAccuracyMetric",
    "DetectionConfusionMatrixMetric",
    "LatencyMetric",
    "FilterEfficiencyMetric",
]
