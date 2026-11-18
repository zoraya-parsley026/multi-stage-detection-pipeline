from __future__ import annotations

import math
from numbers import Integral, Real

from src.metrics.base_metric import BaseMetric
from src.metrics.matcher import ClassificationMatcher
from src.metrics.metric_factory import MetricFactory
from src.metrics.metric_result import MetricResult
from src.utils.validation import validate_class_id


def require_classification_matcher(matcher):
    if not isinstance(matcher, ClassificationMatcher):
        raise TypeError("Classification metrics require ClassificationMatcher.")
    return matcher


@MetricFactory.register(["classification_precision", "classification_precision_metric"])
class ClassificationPrecisionMetric(BaseMetric):
    metric_name = "precision"

    def __init__(self, average="macro", pos_label=1):
        if average not in ("macro", "micro", "weighted", "binary", None):
            raise ValueError("average must be macro, micro, weighted, binary, or None.")
        validate_class_id(pos_label)
        self.average = average
        self.pos_label = pos_label

    def compute(self, matcher):
        summary = require_classification_matcher(matcher).get_summary()
        value = summary.precision_score(self.average, self.pos_label)
        return [MetricResult(
            name=f"classification_{self.average or 'per_class'}_precision", value=value,
        )]


@MetricFactory.register(["classification_recall", "classification_recall_metric"])
class ClassificationRecallMetric(BaseMetric):
    metric_name = "recall"

    def __init__(self, average="macro", pos_label=1):
        if average not in ("macro", "micro", "weighted", "binary", None):
            raise ValueError("average must be macro, micro, weighted, binary, or None.")
        validate_class_id(pos_label)
        self.average = average
        self.pos_label = pos_label

    def compute(self, matcher):
        summary = require_classification_matcher(matcher).get_summary()
        value = summary.recall_score(self.average, self.pos_label)
        return [MetricResult(
            name=f"classification_{self.average or 'per_class'}_recall", value=value,
        )]


@MetricFactory.register(["classification_f1", "classification_f1_metric"])
class ClassificationF1Metric(BaseMetric):
    metric_name = "f1"

    def __init__(self, average="macro", pos_label=1):
        if average not in ("macro", "micro", "weighted", "binary", None):
            raise ValueError("average must be macro, micro, weighted, binary, or None.")
        validate_class_id(pos_label)
        self.average = average
        self.pos_label = pos_label

    def compute(self, matcher):
        summary = require_classification_matcher(matcher).get_summary()
        value = summary.f1_score(self.average, self.pos_label)
        return [MetricResult(
            name=f"classification_{self.average or 'per_class'}_f1", value=value,
        )]


@MetricFactory.register(["classification_counts", "classification_count_metric"])
class ClassificationCountMetric(BaseMetric):
    def compute(self, matcher):
        summary = require_classification_matcher(matcher).get_summary()
        return [
            MetricResult(name="classification_sample_count", value=summary.gt, format="{}"),
            MetricResult(name="classification_correct_count", value=summary.tp, format="{}"),
            MetricResult(
                name="classification_incorrect_count",
                value=summary.fp,
                format="{}",
                higher_is_better=False,
            ),
        ]


@MetricFactory.register(["classification_accuracy", "classification_accuracy_metric"])
class ClassificationAccuracyMetric(BaseMetric):
    def compute(self, matcher):
        summary = require_classification_matcher(matcher).get_summary()
        return [MetricResult(name="classification_accuracy", value=summary.accuracy)]


@MetricFactory.register(["classification_topk_accuracy", "classification_topk_accuracy_metric"])
class ClassificationTopKAccuracyMetric(BaseMetric):
    def __init__(self, k: int = 5) -> None:
        if isinstance(k, bool) or not isinstance(k, Integral) or k <= 0:
            raise ValueError("k must be a positive integer.")
        self.k = int(k)

    def compute(self, matcher):
        total = 0
        correct = 0
        for prediction, target in require_classification_matcher(matcher).iter_pairs():
            scores = prediction.get("class_scores")
            if not isinstance(scores, dict) or not scores:
                raise ValueError("Top-k accuracy requires a non-empty class_scores mapping.")
            if self.k > len(scores):
                raise ValueError("k cannot exceed the number of supplied class_scores.")
            for class_id, score in scores.items():
                validate_class_id(class_id)
                if isinstance(score, bool) or not isinstance(score, Real) or not math.isfinite(score):
                    raise ValueError("class_scores must contain finite numeric scores.")
            ranked_classes = sorted(scores, key=lambda class_id: scores[class_id], reverse=True)
            total += 1
            if target["class_id"] in ranked_classes[:self.k]:
                correct += 1
        accuracy = correct / total if total else 0.0
        return [MetricResult(name=f"classification_top{self.k}_accuracy", value=accuracy)]


@MetricFactory.register(["classification_confusion_matrix", "classification_confusion_matrix_metric"])
class ClassificationConfusionMatrixMetric(BaseMetric):
    """Rows are true classes, columns are predicted classes; omitted cells are zero."""

    def compute(self, matcher):
        summary = require_classification_matcher(matcher).get_summary()
        return [
            MetricResult(
                name="classification_confusion_matrix",
                value=summary.confusion_matrix,
                format="{}",
            )
        ]


@MetricFactory.register(["classification_balanced_accuracy", "classification_balanced_accuracy_metric"])
class ClassificationBalancedAccuracyMetric(BaseMetric):
    def compute(self, matcher):
        summary = require_classification_matcher(matcher).get_summary()
        return [MetricResult(name="classification_balanced_accuracy", value=summary.balanced_accuracy)]
