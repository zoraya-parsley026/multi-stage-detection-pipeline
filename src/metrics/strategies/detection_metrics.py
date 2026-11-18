from __future__ import annotations

import numpy as np

from src.metrics.base_metric import BaseMetric
from src.metrics.matcher import DetectionMatcher
from src.metrics.metric_factory import MetricFactory
from src.metrics.metric_result import MetricResult


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

def require_detection_matcher(matcher):
    if not isinstance(matcher, DetectionMatcher):
        raise TypeError("Detection metrics require DetectionMatcher.")
    return matcher


@MetricFactory.register(["precision", "detection_precision", "detection_precision_metric"])
class DetectionPrecisionMetric(BaseMetric):
    def __init__(self, iou_threshold=0.5, iou_type=None):
        self.iou_threshold = float(iou_threshold)
        self.iou_type = iou_type

    def compute(self, matcher):
        summary = require_detection_matcher(matcher).get_summary(self.iou_threshold, self.iou_type)
        return [MetricResult(name="precision", value=summary.precision)]


@MetricFactory.register(["recall", "detection_recall", "detection_recall_metric"])
class DetectionRecallMetric(BaseMetric):
    def __init__(self, iou_threshold=0.5, iou_type=None):
        self.iou_threshold = float(iou_threshold)
        self.iou_type = iou_type

    def compute(self, matcher):
        summary = require_detection_matcher(matcher).get_summary(self.iou_threshold, self.iou_type)
        return [MetricResult(name="recall", value=summary.recall)]


@MetricFactory.register(["f1", "f1_score", "detection_f1", "detection_f1_metric"])
class DetectionF1Metric(BaseMetric):
    def __init__(self, iou_threshold=0.5, iou_type=None):
        self.iou_threshold = float(iou_threshold)
        self.iou_type = iou_type

    def compute(self, matcher):
        summary = require_detection_matcher(matcher).get_summary(self.iou_threshold, self.iou_type)
        return [MetricResult(name="f1_score", value=summary.f1)]


@MetricFactory.register(["macro_precision", "detection_macro_precision", "detection_macro_precision_metric"])
class DetectionMacroPrecisionMetric(BaseMetric):
    def __init__(self, iou_threshold=0.5, iou_type=None):
        self.iou_threshold = float(iou_threshold)
        self.iou_type = iou_type

    def compute(self, matcher):
        summary = require_detection_matcher(matcher).get_summary(self.iou_threshold, self.iou_type)
        return [MetricResult(name="macro_precision", value=summary.macro_precision)]


@MetricFactory.register(["macro_recall", "detection_macro_recall", "detection_macro_recall_metric"])
class DetectionMacroRecallMetric(BaseMetric):
    def __init__(self, iou_threshold=0.5, iou_type=None):
        self.iou_threshold = float(iou_threshold)
        self.iou_type = iou_type

    def compute(self, matcher):
        summary = require_detection_matcher(matcher).get_summary(self.iou_threshold, self.iou_type)
        return [MetricResult(name="macro_recall", value=summary.macro_recall)]


@MetricFactory.register(["macro_f1", "detection_macro_f1", "detection_macro_f1_metric"])
class DetectionMacroF1Metric(BaseMetric):
    def __init__(self, iou_threshold=0.5, iou_type=None):
        self.iou_threshold = float(iou_threshold)
        self.iou_type = iou_type

    def compute(self, matcher):
        summary = require_detection_matcher(matcher).get_summary(self.iou_threshold, self.iou_type)
        return [MetricResult(name="macro_f1", value=summary.macro_f1)]


@MetricFactory.register(["counts", "detection_counts", "detection_count_metric"])
class DetectionCountMetric(BaseMetric):
    def __init__(self, iou_threshold=0.5, iou_type=None):
        self.iou_threshold = float(iou_threshold)
        self.iou_type = iou_type

    def compute(self, matcher):
        summary = require_detection_matcher(matcher).get_summary(self.iou_threshold, self.iou_type)
        return [
            MetricResult(name="tp_count", value=summary.tp, format="{}"),
            MetricResult(name="fp_count", value=summary.fp, format="{}", higher_is_better=False),
            MetricResult(name="fn_count", value=summary.fn, format="{}", higher_is_better=False),
        ]


@MetricFactory.register(["map", "ap", "detection_ap", "detection_ap_metric"])
class DetectionAPMetric(BaseMetric):
    def __init__(self, iou_thresholds=None, iou_type=None):
        self.iou_thresholds = list(iou_thresholds) if iou_thresholds is not None else np.linspace(0.5, 0.95, 10).tolist()
        self.iou_type = iou_type

    def compute(self, matcher):
        matcher = require_detection_matcher(matcher)
        scores = {}
        for threshold in self.iou_thresholds:
            predictions, gt_counts = matcher.get_ranked_outcomes(threshold, self.iou_type)
            class_aps = []
            for class_id, count in gt_counts.items():
                outcomes = [matched for c, _, matched in predictions if c == class_id]
                _, _, ap = precision_recall_curve(outcomes, count)
                class_aps.append(ap)
            scores[threshold] = float(np.mean(class_aps)) if class_aps else 0.0
        results = [MetricResult(name="mAP", value=float(np.mean(list(scores.values()))) if scores else 0.0)]
        for threshold, score in scores.items():
            if np.isclose(threshold, 0.5):
                results.append(MetricResult(name="mAP@50", value=score))
                break
        return results


@MetricFactory.register(["box_accuracy", "detection_box_accuracy", "detection_box_accuracy_metric"])
class DetectionBoxAccuracyMetric(BaseMetric):
    """
    Measures classification accuracy ONLY for boxes that were correctly localized.
    Equation: (Correctly Labeled Boxes) / (All Correctly Localized Boxes)
    """
    def __init__(self, iou_threshold: float = 0.5) -> None:
        self.iou_threshold = float(iou_threshold)

    def compute(self, matcher):
        correct_labels = 0
        total_localized_boxes = 0

        for prediction, ground_truth_item in require_detection_matcher(matcher).iter_localized_pairs(self.iou_threshold):
            total_localized_boxes += 1
            if prediction["class_id"] == ground_truth_item["class_id"]:
                correct_labels += 1

        accuracy = correct_labels / total_localized_boxes if total_localized_boxes > 0 else 0.0
        return [MetricResult(name="box_classification_accuracy", value=accuracy)]


@MetricFactory.register(["box_topk_accuracy", "detection_box_topk_accuracy", "detection_box_topk_accuracy_metric"])
class DetectionBoxTopKAccuracyMetric(BaseMetric):
    """
    Measures if the ground truth class was in the Top-K predictions of a localized box.
    """
    def __init__(self, k: int = 5, iou_threshold: float = 0.5) -> None:
        self.k = k
        self.iou_threshold = float(iou_threshold)

    def compute(self, matcher):
        correct_in_top_k = 0
        total_localized_boxes = 0

        for prediction, ground_truth_item in require_detection_matcher(matcher).iter_localized_pairs(self.iou_threshold):
            total_localized_boxes += 1
            class_scores = prediction.get("class_scores", {prediction["class_id"]: prediction.get("confidence", 1.0)})
            top_k_classes = sorted(class_scores, key=class_scores.get, reverse=True)[:self.k]
            if ground_truth_item["class_id"] in top_k_classes:
                correct_in_top_k += 1

        accuracy = correct_in_top_k / total_localized_boxes if total_localized_boxes > 0 else 0.0
        return [MetricResult(name=f"box_top{self.k}_accuracy", value=accuracy)]


@MetricFactory.register(["confusion_matrix", "detection_confusion_matrix", "detection_confusion_matrix_metric"])
class DetectionConfusionMatrixMetric(BaseMetric):
    """Return the location-matched confusion matrix from ``DetectionSummary``."""

    def __init__(self, iou_threshold: float = 0.5) -> None:
        self.iou_threshold = float(iou_threshold)

    def compute(self, matcher):
        summary = require_detection_matcher(matcher).get_summary(self.iou_threshold)
        return [
            MetricResult(
                name="detection_confusion_matrix",
                value=summary.confusion_matrix,
                format="{}",
            )
        ]

