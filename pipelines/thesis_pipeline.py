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
from src.metrics.reporting import MetricPlotter
from src.metrics.reporting.reporter import Reporter
from src.metrics.strategies.detection_metrics import (
    DetectionAPMetric,
    DetectionConfusionMatrixMetric,
    DetectionCountMetric,
    DetectionF1Metric,
    DetectionPrecisionMetric,
    DetectionRecallMetric,
)
from src.metrics.strategies.system_metrics import FilterEfficiencyMetric, LatencyMetric
from src.models.model_factory import ModelFactory
from src.nodes import (
    CropNode,
    LogicFilterNode,
    ModelClassificationNode,
    ModelDetectionNode,
    TaxonomyTranslationNode,
)
from src.pipeline.observers import ContextObserver
from src.pipeline.pipeline import Pipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
REPORT_KEYS = ["mAP", "mAP@50", "precision", "recall", "f1_score", "fps", "ΔmAP", "ΔF1", "ΔFP", "Rel_Latency_Inc"]


def load_translation_map(path: str) -> dict[int, int]:
    source = Path(path)
    return {int(key): int(value) for key, value in json.loads(source.read_text(encoding="utf-8")).items()}


def create_evaluator(tag: str, *, confusion_matrix: bool = False) -> Evaluator:
    metrics = [
        DetectionAPMetric(), DetectionPrecisionMetric(), DetectionRecallMetric(),
        DetectionF1Metric(), DetectionCountMetric(), LatencyMetric(), FilterEfficiencyMetric(),
    ]
    if confusion_matrix:
        metrics.insert(5, DetectionConfusionMatrixMetric())
    return Evaluator(metrics, tag=tag)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the YOLOv7 and EfficientFormer cascade.")
    parser.add_argument("--images", default="in_data/subset/images")
    parser.add_argument("--labels", default="in_data/subset/labels")
    parser.add_argument("--detector-weights", default="weights/yolov7.onnx")
    parser.add_argument("--detector-config", default="configuration/yolov7.yaml")
    parser.add_argument("--classifier-weights", default="weights/efficientformer_l3.onnx")
    parser.add_argument("--classifier-config", default="configuration/efficientformer.yaml")
    parser.add_argument("--translation-map", default="configuration/class_translation.json")
    parser.add_argument("--device", choices=["cpu", "cuda", "rocm"], default="cpu")
    parser.add_argument("--conf", type=float, default=None)
    parser.add_argument("--iou", type=float, default=None)
    parser.add_argument("--output", default="runs/results/thesis.json")
    parser.add_argument("--plots-dir", default="runs/plots")
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()

    translation_map = load_translation_map(args.translation_map)
    dataloader = DataLoader(DetectionDataset(args.images, args.labels))
    detector = ModelFactory.create("yolov7", args.detector_weights, args.detector_config, device=args.device)
    if args.conf is not None:
        detector.config.postprocess["conf_thres"] = args.conf
    if args.iou is not None:
        detector.config.postprocess["iou_thres"] = args.iou
    classifier = ModelFactory.create("efficientformer", args.classifier_weights, args.classifier_config, device=args.device)

    baseline = create_evaluator("yolov7")
    cascade = create_evaluator("yolov7_efficientformer", confusion_matrix=True)
    baseline_pipeline = Pipeline([
        ModelDetectionNode(detector, out_boxes_key="detections", node_name="YOLOv7"),
    ])
    baseline_pipeline.add_observer(ContextObserver(baseline, target_key="detections", baseline_key="detections"))
    baseline_pipeline.run(dataloader, progress=not args.no_progress)

    pipeline = Pipeline([
        ModelDetectionNode(detector, out_boxes_key="detections", node_name="YOLOv7"),
        CropNode(in_boxes_key="detections", out_crops_key="crops", node_name="Cropper"),
        ModelClassificationNode(classifier, in_crops_key="crops", out_labels_key="imagenet_labels", node_name="EfficientFormer"),
        TaxonomyTranslationNode(translation_map, in_labels_key="imagenet_labels", out_labels_key="translated_labels"),
        LogicFilterNode(in_boxes_key="detections", in_labels_key="translated_labels", out_boxes_key="filtered_detections", node_name="LogicFilter"),
    ])
    pipeline.add_observer(ContextObserver(cascade, target_key="filtered_detections", baseline_key="detections"))
    pipeline.run(dataloader, progress=not args.no_progress)

    baseline_report = baseline.report()
    cascade_report = cascade.report()
    reports = [baseline_report, cascade_report]
    table = Reporter.compare(reports, REPORT_KEYS)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {"display_keys": REPORT_KEYS, "evaluators": [{r.name: r.value for r in report.results} for report in reports]},
            indent=2,
            default=str,
        ) + "\n",
        encoding="utf-8",
    )
    print(table)
    logger.info("Wrote results to %s", output)

    plotter = MetricPlotter(output_dir=args.plots_dir)
    plotter.plot_pipeline_comparison(reports, filename="thesis_comparison.png")
    plotter.plot_pr_curve(reports, filename="thesis_pr_curve.png")
    plotter.plot_filter_efficiency(cascade_report, filename="thesis_filter_efficiency.png")
    plotter.plot_latency_breakdown(cascade_report, filename="thesis_latency_breakdown.png")
    plotter.plot_confusion_matrix(cascade_report, filename="thesis_confusion_matrix.png")


if __name__ == "__main__":
    main()
