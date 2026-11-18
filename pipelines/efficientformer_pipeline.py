from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.data_loaders import DataLoader, DetectionDataset
from src.models.model_factory import ModelFactory
from src.nodes import ModelImageClassificationNode
from src.pipeline.pipeline import Pipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Classify complete images with EfficientFormer.")
    parser.add_argument("--images", default="in_data/subset/images")
    parser.add_argument("--labels", default=None)
    parser.add_argument("--weights", default="weights/efficientformer_l3.onnx")
    parser.add_argument("--config", default="configuration/efficientformer.yaml")
    parser.add_argument("--device", choices=["cpu", "cuda", "rocm"], default="cpu")
    parser.add_argument("--conf", type=float, default=None)
    parser.add_argument("--output", default="runs/results/efficientformer.json")
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()

    classifier = ModelFactory.create("efficientformer", args.weights, args.config, device=args.device)
    if args.conf is not None:
        classifier.config.postprocess["conf_thres"] = args.conf

    pipeline = Pipeline([ModelImageClassificationNode(classifier, out_predictions_key="classifications")])
    contexts = pipeline.run(
        DataLoader(DetectionDataset(args.images, args.labels)),
        progress=not args.no_progress,
    )

    predictions = [
        classifications[0]
        for context in contexts
        if (classifications := context.get("classifications", []))
    ]
    summary = {
        "tag": "efficientformer",
        "images_processed": len(contexts),
        "images_classified": len(predictions),
        "mean_confidence": (
            sum(prediction.get("confidence", 0.0) for prediction in predictions) / len(predictions)
            if predictions else 0.0
        ),
        "class_counts": dict(Counter(str(prediction.get("class_id", -1)) for prediction in predictions)),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(summary)
    logger.info("Wrote results to %s", output)


if __name__ == "__main__":
    main()
