from __future__ import annotations

from src.metrics.base_metric import BaseSystemMetric
from src.metrics.metric_factory import MetricFactory
from src.metrics.metric_result import MetricResult

@MetricFactory.register(["latency", "system_latency", "latency_metric", "fps"])
class LatencyMetric(BaseSystemMetric):
    def __init__(self, detector_patterns: list[str] | None = None) -> None:
        self.detector_patterns = detector_patterns or ["detect", "yolo"]

    def compute(self, records) -> list[MetricResult]:
        total_detector_ms = 0.0
        total_validator_ms = 0.0
        total_ms = 0.0
        image_count = 0

        for batch_kwargs in records:
            lat = batch_kwargs["latency"]

            det_ms = lat.get("detector_ms")
            val_ms = lat.get("validator_ms")

            if det_ms is None or val_ms is None:
                fallback_det = 0.0
                fallback_val = 0.0
                for k, v in lat.items():
                    if k in ("total_ms", "detector_ms", "validator_ms") or not isinstance(v, (int, float)):
                        continue
                    k_lower = k.lower()
                    if any(pat in k_lower for pat in self.detector_patterns):
                        fallback_det += v
                    else:
                        fallback_val += v
                if det_ms is None:
                    det_ms = fallback_det
                if val_ms is None:
                    val_ms = fallback_val

            total_detector_ms += det_ms
            total_validator_ms += val_ms
            total_ms += lat.get("total_ms", 0.0)
            image_count += 1

        if image_count == 0:
            return []

        avg_ms = total_ms / image_count
        fps = 1000.0 / avg_ms if avg_ms > 0 else 0.0

        return [
            MetricResult(name="avg_latency_ms", value=avg_ms, higher_is_better=False),
            MetricResult(name="fps", value=fps),
            MetricResult(name="detector_ms", value=total_detector_ms / image_count, higher_is_better=False),
            MetricResult(name="validator_ms", value=total_validator_ms / image_count, higher_is_better=False),
        ]


@MetricFactory.register(["filter_efficiency", "efficiency", "filter_efficiency_metric"])
class FilterEfficiencyMetric(BaseSystemMetric):

    def compute(self, records) -> list[MetricResult]:
        total_entering = 0
        total_kept = 0

        for record in records:
            total_entering += record["entering"]
            total_kept += record["kept"]

        rejected = total_entering - total_kept
        retention_rate = (total_kept / total_entering) if total_entering > 0 else 0.0

        return [
            MetricResult(name="detections_entering", value=total_entering, format="{}"),
            MetricResult(name="detections_kept", value=total_kept, format="{}"),
            MetricResult(name="detections_rejected", value=rejected, format="{}"),
            MetricResult(name="retention_rate", value=retention_rate),
        ]

