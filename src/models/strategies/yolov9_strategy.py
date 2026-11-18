from __future__ import annotations

import numpy as np

from src.models.strategies.yolov7_strategy import (
    YOLONMSPostprocessStrategy,
    YOLOv7PreprocessStrategy,
)
from src.utils.box_utils import NonMaxSuppression, clip_boxes, scale_coords
from src.utils.validation import positive_integer


class YOLOv9PreprocessStrategy(YOLOv7PreprocessStrategy):
    """YOLOv9 uses the same letterbox and NCHW image preparation as YOLOv7."""


class YOLOv9PostprocessStrategy(YOLONMSPostprocessStrategy):
    """Convert YOLOv9 ONNX output into the shared detection contract."""

    def validate_config(self) -> None:
        super().validate_config()
        positive_integer(self.config.output.get("classes_count"), "output.classes_count")

    def process(self, raw_outputs: list[np.ndarray], routing_metadata: dict) -> list[dict]:
        self.validate_config()
        predictions = np.asarray(raw_outputs[0])
        class_count = self.config.output["classes_count"]
        channel_count = 4 + class_count

        if predictions.ndim != 3:
            raise ValueError(
                f"YOLOv9 output must have rank 3, received shape {predictions.shape}."
            )
        if predictions.shape[-1] == channel_count:
            candidates = predictions
        elif predictions.shape[1] == channel_count:
            candidates = np.transpose(predictions, (0, 2, 1))
        else:
            raise ValueError(
                "YOLOv9 output must contain four box channels followed by one score per class; "
                f"received shape {predictions.shape}."
            )

        objectness = np.ones((*candidates.shape[:2], 1), dtype=candidates.dtype)
        nms_input = np.concatenate((candidates[..., :4], objectness, candidates[..., 4:]), axis=-1)
        detections = NonMaxSuppression(
            conf_thres=self.config.postprocess.get("conf_thres", 0.25),
            iou_thres=self.config.postprocess.get("iou_thres", 0.45),
            max_det=self.config.postprocess.get("max_det", 300),
            coordinate_input_format="xywh",
        )(nms_input)[0]

        if len(detections) == 0:
            return []

        detections[:, :4] = scale_coords(
            routing_metadata["input_shape"],
            detections[:, :4],
            routing_metadata["original_shape"],
        )
        clip_boxes(detections[:, :4], routing_metadata["original_shape"])

        return [
            {
                "class_id": int(class_id),
                "confidence": float(confidence),
                "localization": [float(value) for value in xyxy],
            }
            for *xyxy, confidence, class_id in detections
        ]
