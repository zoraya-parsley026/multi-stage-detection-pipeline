"""
Inference Strategies
====================
Pipeline nodes that execute machine learning model inference.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any

from src.nodes.base_node import PipelineNode
from src.nodes.node_factory import NodeFactory
from src.pipeline.pipeline_context import PipelineContext

logger = logging.getLogger(__name__)


@NodeFactory.register(["detection", "model_detection"])
class ModelDetectionNode(PipelineNode):
    """
    Full-frame object detector node.

    Runs a detector model over the input image, assigns UUIDs to every
    returned detection, and optionally short-circuits the pipeline when no
    detections are found.

    Parameters
    ----------
    detector:
        Any object exposing a ``predict(image) -> list[Detection]`` interface.
    in_image_key:
        Context key for the input image (default: ``"image"``).
    out_boxes_key:
        Context key for the output detection list (default: ``"detections"``).
    node_name:
        Display name used in logs and the registry.
    abort_on_empty:
        If ``True`` (default), call ``context.abort()`` when 0 detections are
        found. Set to ``False`` when this node feeds into an ensemble so the
        pipeline is not short-circuited before fusion can occur.
    """

    def __init__(
        self,
        detector: Any,
        in_image_key: str = "image",
        out_boxes_key: str = "detections",
        node_name: str = "ModelDetectionNode",
        abort_on_empty: bool = True,
    ) -> None:
        self.detector = detector
        self.in_keys = [in_image_key]
        self.out_keys = [out_boxes_key]
        self.in_image_key = in_image_key
        self.out_boxes_key = out_boxes_key
        self.name = node_name
        self.abort_on_empty = abort_on_empty

    def process(self, context: PipelineContext) -> None:
        image = context.get(self.in_image_key)
        detections = self.detector.predict(image) or []

        # Assign UUIDs to detections for downstream tracking
        for det in detections:
            if "id" not in det:
                det["id"] = uuid.uuid4().hex

        context.set(self.out_boxes_key, detections)

        # Conditional short-circuit: No bounding boxes found
        if self.abort_on_empty and len(detections) == 0:
            logger.debug(
                f"{self.name}: 0 detections found. "
                "Short-circuiting pipeline execution."
            )
            context.abort()


@NodeFactory.register(["classification", "model_classification"])
class ModelClassificationNode(PipelineNode):
    """
    Performs batched classifier inference on cropped images.

    Parameters
    ----------
    classifier:
        Any object exposing a ``predict(crops: list) -> list[dict]`` interface.
    in_crops_key:
        Context key for the crops dict ``{id: np.ndarray}``
        (default: ``"crops"``).
    out_labels_key:
        Context key for the output labels dict ``{id: dict}``
        (default: ``"labels"``).
    node_name:
        Display name used in logs and the registry.
    """

    def __init__(
        self,
        classifier: Any,
        in_crops_key: str = "crops",
        out_labels_key: str = "labels",
        node_name: str = "ModelClassificationNode",
    ) -> None:
        self.classifier = classifier
        self.in_keys = [in_crops_key]
        self.out_keys = [out_labels_key]
        self.in_crops_key = in_crops_key
        self.out_labels_key = out_labels_key
        self.name = node_name

    def process(self, context: PipelineContext) -> None:
        crops_dict = context.get(self.in_crops_key)
        if not crops_dict:
            context.set(self.out_labels_key, {})
            context.abort()
            return

        ids = list(crops_dict.keys())
        crop_arrays = [crops_dict[i] for i in ids]

        results = self.classifier.predict(crop_arrays)
        labels_dict = {box_id: result for box_id, result in zip(ids, results)}
        context.set(self.out_labels_key, labels_dict)


@NodeFactory.register(["image_classification", "model_image_classification"])
class ModelImageClassificationNode(PipelineNode):
    """Run a classifier once over the complete input image."""

    def __init__(
        self,
        classifier: Any,
        in_image_key: str = "image",
        out_predictions_key: str = "classifications",
        node_name: str = "ModelImageClassificationNode",
    ) -> None:
        self.classifier = classifier
        self.in_keys = [in_image_key]
        self.out_keys = [out_predictions_key]
        self.in_image_key = in_image_key
        self.out_predictions_key = out_predictions_key
        self.name = node_name

    def process(self, context: PipelineContext) -> None:
        image = context.get(self.in_image_key)
        predictions = self.classifier.predict([image])
        context.set(self.out_predictions_key, predictions)

