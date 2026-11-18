"""
src.metrics.strategies
======================
Concrete metric strategies for detection, classification, confusion matrix, and system profiling.
Importing this package registers all metrics with MetricFactory.
"""

from src.metrics.matcher import ClassificationSummary
from src.metrics.strategies.classification_metrics import (
    ClassificationAccuracyMetric,
    ClassificationBalancedAccuracyMetric,
    ClassificationConfusionMatrixMetric,
    ClassificationCountMetric,
    ClassificationF1Metric,
    ClassificationPrecisionMetric,
    ClassificationRecallMetric,
    ClassificationTopKAccuracyMetric,
)
from src.metrics.strategies.detection_metrics import (
    DetectionAPMetric,
    DetectionBoxAccuracyMetric,
    DetectionBoxTopKAccuracyMetric,
    DetectionCountMetric,
    DetectionConfusionMatrixMetric,
    DetectionF1Metric,
    DetectionMacroF1Metric,
    DetectionMacroPrecisionMetric,
    DetectionMacroRecallMetric,
    DetectionPrecisionMetric,
    DetectionRecallMetric,
)
from src.metrics.strategies.system_metrics import FilterEfficiencyMetric, LatencyMetric

__all__ = [
    "ClassificationSummary",
    "ClassificationAccuracyMetric",
    "ClassificationBalancedAccuracyMetric",
    "ClassificationPrecisionMetric",
    "ClassificationRecallMetric",
    "ClassificationF1Metric",
    "ClassificationCountMetric",
    "ClassificationConfusionMatrixMetric",
    "ClassificationTopKAccuracyMetric",
    # Base Detection Contracts
    # Detection Standard & Macro
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
    # Confusion Matrix
    "DetectionConfusionMatrixMetric",
    # System Metrics
    "LatencyMetric",
    "FilterEfficiencyMetric",
]

