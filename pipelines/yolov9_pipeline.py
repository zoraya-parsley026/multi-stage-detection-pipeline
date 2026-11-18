from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.data_loaders import DataLoader, DetectionDataset
from src.metrics.evaluator import Evaluator
from src.metrics.reporting.reporter import Reporter
from src.metrics.strategies.detection_metrics import (
    DetectionAPMetric,
    DetectionCountMetric,
    DetectionF1Metric,
    DetectionPrecisionMetric,
    DetectionRecallMetric,
)
from src.metrics.strategies.system_metrics import FilterEfficiencyMetric, LatencyMetric
from src.models.model_factory import ModelFactory
from src.nodes import ModelDetectionNode
from src.pipeline.observers import ContextObserver
from src.pipeline.pipeline import Pipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
REPORT_KEYS = ["mAP", "mAP@50", "precision", "recall", "f1_score", "fps"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a YOLOv9 ONNX detector.")
    parser.add_argument("--images", default="in_data/subset/images")
    parser.add_argument("--labels", default="in_data/subset/labels")
    parser.add_argument("--weights", default="weights/yolov9.onnx")
    parser.add_argument("--config", default="configuration/yolov9.yaml")
    parser.add_argument("--device", choices=["cpu", "cuda", "rocm"], default="cpu")
    parser.add_argument("--conf", type=float, default=None)
    parser.add_argument("--iou", type=float, default=None)
    parser.add_argument("--output", default="runs/results/yolov9.json")
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()

    detector = ModelFactory.create("yolov9", args.weights, args.config, device=args.device)
    if args.conf is not None:
        detector.config.postprocess["conf_thres"] = args.conf
    if args.iou is not None:
        detector.config.postprocess["iou_thres"] = args.iou

    evaluator = Evaluator(
        [
            DetectionAPMetric(), DetectionPrecisionMetric(), DetectionRecallMetric(),
            DetectionF1Metric(), DetectionCountMetric(), LatencyMetric(), FilterEfficiencyMetric(),
        ],
        tag="yolov9",
    )
    pipeline = Pipeline([ModelDetectionNode(detector, out_boxes_key="detections", node_name="YOLOv9")])
    pipeline.add_observer(ContextObserver(evaluator, target_key="detections", baseline_key="detections"))
    pipeline.run(
        DataLoader(DetectionDataset(args.images, args.labels)),
        progress=not args.no_progress,
    )

    report = evaluator.report()
    table = Reporter.compare([report], REPORT_KEYS)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {"display_keys": REPORT_KEYS, "evaluators": [{r.name: r.value for r in report.results}]},
            indent=2,
            default=str,
        ) + "\n",
        encoding="utf-8",
    )
    print(table)
    logger.info("Wrote results to %s", output)


if __name__ == "__main__":
    main()
