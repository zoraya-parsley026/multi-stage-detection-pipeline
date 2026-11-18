from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

from src.metrics.base_metric import BaseMetric, BaseSystemMetric
from src.metrics.matcher import BaseMatcher, DetectionMatcher
from src.metrics.metric_result import MetricResult

import numpy as np

def precision_recall_curve(outcomes, ground_truth_count):
    """Interpolated envelope and rectangle-area AP for confidence-ordered outcomes."""
    if not outcomes or not ground_truth_count:
        return np.array([0.0, 1.0]), np.array([0.0, 0.0]), 0.0
    tp = np.cumsum(np.asarray(outcomes, dtype=float))
    fp = np.cumsum(1 - np.asarray(outcomes, dtype=float))
    recall = np.concatenate(([0.0], tp / ground_truth_count, [1.0]))
    precision = np.concatenate(([1.0], tp / (tp + fp + 1e-16), [0.0]))
    precision = np.maximum.accumulate(precision[::-1])[::-1]
    changes = np.where(recall[1:] != recall[:-1])[0]
    ap = float(np.sum((recall[changes + 1] - recall[changes]) * precision[changes + 1]))
    return recall, precision, ap


@dataclass
class EvaluationReport:
    tag: str
    results: list[MetricResult]
    system_data: list[dict[str, Any]]
    confusion_matrix: dict = field(default_factory=dict)
    pr_curves: dict = field(default_factory=dict)


class Evaluator:
    def __init__(self, metrics: list[BaseMetric | BaseSystemMetric], tag="evaluator",
                 matcher: BaseMatcher | None = None):
        self.metrics = metrics
        self.tag = tag
        self.matcher = matcher if matcher is not None else DetectionMatcher()
        self.system_data = []

    def accumulate(self, predictions=None, ground_truth=None, baseline_detections=None,
                   batch_data=None, detections=None, **kwargs):
        predictions = predictions or []
        baseline = baseline_detections if baseline_detections is not None else detections or []
        self.matcher.accumulate(predictions, ground_truth)
        self.system_data.append({
            "latency": dict((batch_data or {}).get("latency", {})),
            "entering": len(baseline),
            "kept": len(predictions),
        })

    def clear(self):
        self.matcher.clear()
        self.system_data.clear()

    def compute(self) -> list[MetricResult]:
        results = [MetricResult(name="tag", value=self.tag, format="{}")]
        for metric in self.metrics:
            source = self.system_data if isinstance(metric, BaseSystemMetric) else self.matcher
            results.extend(metric.compute(source))
        names = [result.name for result in results]
        if len(set(names)) != len(names):
            raise ValueError("Metric result names must be unique within an evaluator.")
        return results

    def report(self, *, pr_thresholds=(0.5,), confusion_threshold=0.5) -> EvaluationReport:
        """Create a detached snapshot for tables, JSON, and plots."""
        results = self.compute()
        curves = {}
        if isinstance(self.matcher, DetectionMatcher):
            for threshold in pr_thresholds:
                predictions, gt_counts = self.matcher.get_ranked_outcomes(threshold)
                curves[threshold] = precision_recall_curve(
                    [matched for _, _, matched in predictions], sum(gt_counts.values()),
                )
            matrix = self.matcher.get_summary(confusion_threshold).confusion_matrix
        else:
            matrix = self.matcher.get_summary().confusion_matrix
        return EvaluationReport(
            tag=self.tag, results=deepcopy(results), system_data=deepcopy(self.system_data),
            confusion_matrix=deepcopy(matrix), pr_curves=curves,
        )

