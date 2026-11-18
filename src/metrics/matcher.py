from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.utils.box_utils import box_ciou, box_diou, box_giou, box_iou
from src.utils.validation import validate_class_id

Detection = dict[str, Any]
ClassificationRecord = dict[str, Any]
ClassId = int | str
IoUFunction = Callable[[np.ndarray, np.ndarray], np.ndarray]
IOU_FUNCTIONS: dict[str, IoUFunction] = {"iou": box_iou, "giou": box_giou, "diou": box_diou, "ciou": box_ciou}


def sort_predictions(predictions: Iterable[Detection]) -> list[Detection]:
    return sorted(predictions, key=lambda prediction: prediction.get("confidence", 0.0), reverse=True)


def build_iou_matrix(predictions: list[Detection], ground_truth: list[Detection], iou_fn: IoUFunction, *, require_same_class: bool = False) -> np.ndarray:
    shape = (len(predictions), len(ground_truth))
    if not predictions or not ground_truth or not all("localization" in item for item in (*predictions, *ground_truth)):
        return np.full(shape, -np.inf, dtype=np.float32)

    prediction_boxes = np.asarray([item["localization"] for item in predictions], dtype=np.float32)
    ground_truth_boxes = np.asarray([item["localization"] for item in ground_truth], dtype=np.float32)
    ious = iou_fn(prediction_boxes, ground_truth_boxes)
    if not require_same_class:
        return ious

    prediction_classes = np.asarray([item["class_id"] for item in predictions])
    ground_truth_classes = np.asarray([item["class_id"] for item in ground_truth])
    return np.where(prediction_classes[:, None] == ground_truth_classes[None, :], ious, -np.inf)


def greedy_match(iou_matrix: np.ndarray, iou_threshold: float) -> list[int | None]:
    matched_ground_truth: set[int] = set()
    matches: list[int | None] = []
    for row in iou_matrix:
        candidates = [index for index in range(len(row)) if index not in matched_ground_truth]
        if not candidates:
            matches.append(None)
            continue
        best_index = max(candidates, key=lambda index: row[index])
        if float(row[best_index]) >= iou_threshold:
            matched_ground_truth.add(best_index)
            matches.append(best_index)
        else:
            matches.append(None)
    return matches


@dataclass
class DetectionSummary:
    iou_threshold: float
    tp: int = 0
    fp: int = 0
    fn: int = 0
    gt: int = 0
    tp_per_class: dict[Any, int] = field(default_factory=dict)
    fp_per_class: dict[Any, int] = field(default_factory=dict)
    fn_per_class: dict[Any, int] = field(default_factory=dict)
    gt_per_class: dict[Any, int] = field(default_factory=dict)
    all_classes: set[Any] = field(default_factory=set)
    confusion_matrix: dict[Any, dict[Any, int]] = field(default_factory=dict)

    # - - - #

    @property
    def precision(self) -> float:
        total = self.tp + self.fp
        return float(self.tp / (total + 1e-16)) if total > 0 else 0.0

    @property
    def recall(self) -> float:
        return float(self.tp / self.gt) if self.gt > 0 else 0.0

    @property
    def f1(self) -> float:
        precision = self.precision
        recall    = self.recall
        if precision + recall == 0:
            return 0.0
        return float(2 * precision * recall / (precision + recall + 1e-16))

    # - - - #

    @property
    def macro_precision(self) -> float:
        precisions = []
        for class_id in self.all_classes:
            true_positives    = self.tp_per_class.get(class_id, 0)
            false_positives   = self.fp_per_class.get(class_id, 0)
            total_predictions = true_positives + false_positives
            # ground_truth_count = self.gt_per_class.get(class_id, 0)

            precision = (
                true_positives / (total_predictions + 1e-16)
                if total_predictions > 0
                else 0.0
            )
            precisions.append(precision)

        return float(np.mean(precisions)) if precisions else 0.0

    @property
    def macro_recall(self) -> float:
        recalls = []
        for class_id in self.all_classes:
            true_positives     = self.tp_per_class.get(class_id, 0)
            # false_positives    = self.fp_per_class.get(class_id, 0)
            # total_predictions  = true_positives + false_positives
            ground_truth_count = self.gt_per_class.get(class_id, 0)

            recall = (
                true_positives / ground_truth_count 
                if ground_truth_count > 0 
                else 0.0
            )
            recalls.append(recall)

        return float(np.mean(recalls)) if recalls else 0.0

    @property
    def macro_f1(self) -> float:
        f1_scores = []
        for class_id in self.all_classes:
            true_positives     = self.tp_per_class.get(class_id, 0)
            false_positives    = self.fp_per_class.get(class_id, 0)
            ground_truth_count = self.gt_per_class.get(class_id, 0)
            total_predictions  = true_positives + false_positives

            precision = (
                true_positives / (total_predictions + 1e-16)
                if total_predictions > 0
                else 0.0
            )

            recall = (
                 true_positives / ground_truth_count 
                 if ground_truth_count > 0 
                 else 0.0
            )

            f1_score = (
                2 * precision * recall / (precision + recall + 1e-16)
                if precision + recall > 0
                else 0.0
            )
            f1_scores.append(f1_score)

        return float(np.mean(f1_scores)) if f1_scores else 0.0

    # - - - #


@dataclass
class ClassificationSummary:
    tp: int = 0
    fp: int = 0
    fn: int = 0
    gt: int = 0
    tp_per_class: dict[ClassId, int] = field(default_factory=dict)
    fp_per_class: dict[ClassId, int] = field(default_factory=dict)
    fn_per_class: dict[ClassId, int] = field(default_factory=dict)
    gt_per_class: dict[ClassId, int] = field(default_factory=dict)
    all_classes: set[ClassId] = field(default_factory=set)
    confusion_matrix: dict[Any, dict[Any, int]] = field(default_factory=dict)

    # - - - #

    @property
    def accuracy(self) -> float:
        return self.tp / self.gt if self.gt else 0.0

    # - - - #

    @property
    def micro_precision(self) -> float:
        total = self.tp + self.fp
        return self.tp / total if total else 0.0

    @property
    def micro_recall(self) -> float:
        total = self.tp + self.fn
        return self.tp / total if total else 0.0

    @property
    def micro_f1(self) -> float:
        total = 2 * self.tp + self.fp + self.fn
        return 2 * self.tp / total if total else 0.0

    # - - - #

    @property
    def macro_precision(self) -> float:
        return float(self.precision_score(average="macro"))

    @property
    def macro_recall(self) -> float:
        return float(self.recall_score(average="macro"))

    @property
    def macro_f1(self) -> float:
        return float(self.f1_score(average="macro"))

    # - - - #

    @property
    def balanced_accuracy(self) -> float:
        scores = [
            self.tp_per_class.get(c, 0) / count
            for c, count in self.gt_per_class.items() if count
        ]
        return float(np.mean(scores)) if scores else 0.0

    # - - - #

    def precision_score(
        self,
        average: str | None = "macro",
        pos_label: ClassId = 1,
    ) -> float | dict[ClassId, float]:
        scores: dict[ClassId, float] = {}
        for class_id in self.all_classes:
            tp = self.tp_per_class.get(class_id, 0)
            fp = self.fp_per_class.get(class_id, 0)
            total = tp + fp
            scores[class_id] = tp / total if total else 0.0
        return self._average_scores(scores, average, pos_label, self.micro_precision)

    def recall_score(
        self,
        average: str | None = "macro",
        pos_label: ClassId = 1,
    ) -> float | dict[ClassId, float]:
        scores = {
            class_id: self.tp_per_class.get(class_id, 0) / self.gt_per_class[class_id]
            if self.gt_per_class.get(class_id, 0)
            else 0.0
            for class_id in self.all_classes
        }
        return self._average_scores(scores, average, pos_label, self.micro_recall)

    def f1_score(
        self,
        average: str | None = "macro",
        pos_label: ClassId = 1,
    ) -> float | dict[ClassId, float]:
        scores: dict[ClassId, float] = {}
        for class_id in self.all_classes:
            tp = self.tp_per_class.get(class_id, 0)
            fp = self.fp_per_class.get(class_id, 0)
            fn = self.fn_per_class.get(class_id, 0)
            denominator = 2 * tp + fp + fn
            scores[class_id] = 2 * tp / denominator if denominator else 0.0
        return self._average_scores(scores, average, pos_label, self.micro_f1)

    # - - - #

    def _average_scores(
        self,
        scores: dict[ClassId, float],
        average: str | None,
        pos_label: ClassId,
        micro_score: float,
    ) -> float | dict[ClassId, float]:
        if average not in ("macro", "micro", "weighted", "binary", None):
            raise ValueError("average must be macro, micro, weighted, binary, or None.")
        validate_class_id(pos_label)

        if average is None:
            return scores
        if average == "binary":
            if len(self.all_classes) > 2:
                raise ValueError("Binary averaging requires at most two observed classes.")
            if len(self.all_classes) == 2 and pos_label not in self.all_classes:
                raise ValueError("pos_label must be one of the observed binary classes.")
            return scores.get(pos_label, 0.0)
        if average == "micro":
            return micro_score
        if average == "weighted":
            weighted_sum = sum(
                score * self.gt_per_class.get(class_id, 0)
                for class_id, score in scores.items()
            )
            return weighted_sum / self.gt if self.gt else 0.0
        return sum(scores.values()) / len(scores) if scores else 0.0


class BaseMatcher:
    def __init__(self) -> None:
        self.events: list[dict[str, list[Detection]]] = []

    def accumulate(self, predictions=None, ground_truth=None) -> None:
        self.events.append({
            "predictions": predictions or [],
            "ground_truth": ground_truth or [],
        })
        self.invalidate_cache()

    def clear(self) -> None:
        self.events.clear()
        self.invalidate_cache()

    def invalidate_cache(self) -> None:
        pass


@dataclass
class DetectionMatches:
    predictions: list[Detection]
    ground_truth: list[Detection]
    indices: list[int | None]


class DetectionMatcher(BaseMatcher):
    def __init__(self, iou_type: str = "iou", iou_fn: IoUFunction | None = None) -> None:
        super().__init__()
        self._summary_cache: dict[tuple, DetectionSummary] = {}
        self._matches_cache: dict[tuple, list[DetectionMatches]] = {}
        self.set_iou_metric(iou_type, iou_fn)

    def invalidate_cache(self) -> None:
        self._summary_cache.clear()
        self._matches_cache.clear()

    def set_iou_metric(self, iou_type="iou", iou_fn=None) -> None:
        clean_type = iou_type.strip().lower() if isinstance(iou_type, str) else "custom"
        if iou_fn is None and clean_type not in IOU_FUNCTIONS:
            raise ValueError(f"Unsupported iou_type '{iou_type}'.")
        self.iou_type = clean_type
        self.iou_fn = iou_fn if iou_fn is not None else IOU_FUNCTIONS[clean_type]
        self.invalidate_cache()

    def _resolve_iou(self, iou_type=None, iou_fn=None):
        if iou_fn is not None:
            return iou_fn
        if iou_type is None:
            return self.iou_fn
        name = iou_type.strip().lower()
        if name not in IOU_FUNCTIONS:
            raise ValueError(f"Unsupported iou_type '{iou_type}'.")
        return IOU_FUNCTIONS[name]

    def get_matches(self, iou_threshold=0.5, *, require_same_class=True,
                    iou_type=None, iou_fn=None) -> list[DetectionMatches]:
        threshold = float(iou_threshold)
        if not np.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError("iou_threshold must be finite and between 0 and 1.")
        fn = self._resolve_iou(iou_type, iou_fn)
        key = (threshold, id(fn), require_same_class)
        if key not in self._matches_cache:
            matches = []
            for event in self.events:
                predictions = sort_predictions(event["predictions"])
                ground_truth = event["ground_truth"]
                matrix = build_iou_matrix(
                    predictions, ground_truth, fn,
                    require_same_class=require_same_class,
                )
                matches.append(DetectionMatches(
                    predictions, ground_truth, greedy_match(matrix, threshold),
                ))
            self._matches_cache[key] = matches
        return self._matches_cache[key]

    def get_summary(self, iou_threshold=0.5, iou_type=None, iou_fn=None) -> DetectionSummary:
        fn = self._resolve_iou(iou_type, iou_fn)
        key = (float(iou_threshold), id(fn))
        if key not in self._summary_cache:
            summary = DetectionSummary(iou_threshold=float(iou_threshold))
            for event in self.get_matches(iou_threshold, iou_fn=fn):
                summary.gt += len(event.ground_truth)
                for target in event.ground_truth:
                    c = target["class_id"]
                    summary.gt_per_class[c] = summary.gt_per_class.get(c, 0) + 1
                    summary.all_classes.add(c)
                for prediction, index in zip(event.predictions, event.indices):
                    c = prediction["class_id"]
                    summary.all_classes.add(c)
                    if index is None:
                        summary.fp += 1
                        summary.fp_per_class[c] = summary.fp_per_class.get(c, 0) + 1
                    else:
                        summary.tp += 1
                        summary.tp_per_class[c] = summary.tp_per_class.get(c, 0) + 1
            summary.fn = summary.gt - summary.tp
            summary.fn_per_class = {
                c: summary.gt_per_class.get(c, 0) - summary.tp_per_class.get(c, 0)
                for c in summary.all_classes
            }
            summary.confusion_matrix = self.get_confusion_matrix(
                iou_threshold,
                iou_fn=fn,
            )
            self._summary_cache[key] = summary
        return self._summary_cache[key]

    def get_localization_matches(self, iou_threshold=0.5):
        return self.get_matches(iou_threshold, require_same_class=False)

    def iter_localized_pairs(self, iou_threshold=0.5):
        for event in self.get_localization_matches(iou_threshold):
            for prediction, index in zip(event.predictions, event.indices):
                if index is not None:
                    yield prediction, event.ground_truth[index]

    def get_ranked_outcomes(self, iou_threshold=0.5, iou_type=None):
        predictions = []
        gt_counts = {}
        for event in self.get_matches(iou_threshold, iou_type=iou_type):
            for target in event.ground_truth:
                c = target["class_id"]
                gt_counts[c] = gt_counts.get(c, 0) + 1
            for prediction, index in zip(event.predictions, event.indices):
                predictions.append((
                    prediction["class_id"], prediction.get("confidence", 0.0),
                    index is not None,
                ))
        predictions.sort(key=lambda item: item[1], reverse=True)
        return predictions, gt_counts

    def get_confusion_matrix(self, iou_threshold=0.5, iou_type=None, iou_fn=None):
        matrix = {}
        def add(true_class, predicted_class):
            row = matrix.setdefault(true_class, {})
            row[predicted_class] = row.get(predicted_class, 0) + 1
        for event in self.get_matches(
            iou_threshold,
            require_same_class=False,
            iou_type=iou_type,
            iou_fn=iou_fn,
        ):
            used = {i for i in event.indices if i is not None}
            for prediction, index in zip(event.predictions, event.indices):
                true_class = event.ground_truth[index]["class_id"] if index is not None else "Background"
                add(true_class, prediction["class_id"])
            for index, target in enumerate(event.ground_truth):
                if index not in used:
                    add(target["class_id"], "Background")
        return matrix


class ClassificationMatcher(BaseMatcher):
    def __init__(self) -> None:
        super().__init__()
        self._summary_cache: ClassificationSummary | None = None

    def clear(self) -> None:
        super().clear()
        self.invalidate_cache()

    def invalidate_cache(self) -> None:
        self._summary_cache = None

    def iter_pairs(self) -> Iterator[tuple[ClassificationRecord, ClassificationRecord]]:
        for event_index, event in enumerate(self.events):
            predictions = event.get("predictions") or []
            ground_truth = event.get("ground_truth") or []
            if len(predictions) != len(ground_truth):
                raise ValueError(
                    f"Classification event {event_index} requires one prediction per label; "
                    f"received {len(predictions)} predictions and {len(ground_truth)} labels."
                )

            for prediction, target in zip(predictions, ground_truth):
                for item in (prediction, target):
                    if not isinstance(item, dict) or "class_id" not in item:
                        raise ValueError("Each classification record must contain class_id.")
                    validate_class_id(item["class_id"])
                yield prediction, target

    def get_summary(self) -> ClassificationSummary:
        if self._summary_cache is None:
            summary = ClassificationSummary()
            for prediction, target in self.iter_pairs():
                predicted_class = prediction["class_id"]
                true_class = target["class_id"]
                summary.gt += 1
                summary.all_classes.update((predicted_class, true_class))
                summary.gt_per_class[true_class] = (
                    summary.gt_per_class.get(true_class, 0) + 1
                )
                row = summary.confusion_matrix.setdefault(true_class, {})
                row[predicted_class] = row.get(predicted_class, 0) + 1
                if predicted_class == true_class:
                    summary.tp += 1
                    summary.tp_per_class[true_class] = (
                        summary.tp_per_class.get(true_class, 0) + 1
                    )
                else:
                    summary.fp += 1
                    summary.fn += 1
                    summary.fp_per_class[predicted_class] = (
                        summary.fp_per_class.get(predicted_class, 0) + 1
                    )
                    summary.fn_per_class[true_class] = (
                        summary.fn_per_class.get(true_class, 0) + 1
                    )
            self._summary_cache = summary
        return self._summary_cache
