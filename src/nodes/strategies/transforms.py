"""
Transformation Strategies
=========================
Pipeline nodes that perform spatial, coordinate, or taxonomy transformations.
"""

from __future__ import annotations

import logging

from src.nodes.base_node import PipelineNode
from src.nodes.node_factory import NodeFactory
from src.pipeline.pipeline_context import PipelineContext
from src.utils.box_utils import crop_roi

logger = logging.getLogger(__name__)


@NodeFactory.register(["crop", "crop_node"])
class CropNode(PipelineNode):
    """
    Crops regions of interest from the full image for each detection box.

    Parameters
    ----------
    in_image_key:
        Context key for the source image (default: ``"image"``).
    in_boxes_key:
        Context key for the detection list (default: ``"detections"``).
    out_crops_key:
        Context key for the output crops dict ``{id: np.ndarray}``
        (default: ``"crops"``).
    node_name:
        Display name used in logs and the registry.
    """

    def __init__(
        self,
        in_image_key: str = "image",
        in_boxes_key: str = "detections",
        out_crops_key: str = "crops",
        node_name: str = "CropNode",
    ) -> None:
        self.in_keys = [in_image_key, in_boxes_key]
        self.out_keys = [out_crops_key]
        self.in_image_key = in_image_key
        self.in_boxes_key = in_boxes_key
        self.out_crops_key = out_crops_key
        self.name = node_name

    def process(self, context: PipelineContext) -> None:
        image = context.get(self.in_image_key)
        detections = context.get(self.in_boxes_key) or []

        crops: dict[str, object] = {}
        for det in detections:
            try:
                crop = crop_roi(image, det["localization"])
                crops[det["id"]] = crop
            except ValueError as e:
                logger.warning(f"Skipping crop for detection {det.get('id')}: {e}")

        context.set(self.out_crops_key, crops)

        # Short-circuit early exit: No valid crops produced
        if len(crops) == 0:
            logger.debug(
                f"{self.name}: 0 valid crops produced. "
                "Short-circuiting pipeline execution."
            )
            context.abort()


@NodeFactory.register(["taxonomy", "taxonomy_translation"])
class TaxonomyTranslationNode(PipelineNode):
    """
    Translates prediction labels between disparate taxonomies
    (e.g., ImageNet 1000-class space → COCO 80-class space).

    Adheres strictly to the Single Responsibility Principle: it only
    performs label mapping, leaving filtering to :class:`LogicFilterNode`.

    Parameters
    ----------
    translation_map:
        Mapping of ``{source_class_id: target_class_id}``. Any class absent
        from the map is translated to ``-1`` (unknown).
    in_labels_key:
        Context key for the input labels dict (default: ``"imagenet_labels"``).
    out_labels_key:
        Context key for the translated labels dict (default: ``"coco_labels"``).
    node_name:
        Display name used in logs and the registry.
    """

    def __init__(
        self,
        translation_map: dict[int, int],
        in_labels_key: str = "imagenet_labels",
        out_labels_key: str = "coco_labels",
        node_name: str = "TaxonomyTranslationNode",
    ) -> None:
        self.translation_map = translation_map
        self.in_keys = [in_labels_key]
        self.out_keys = [out_labels_key]
        self.in_labels_key = in_labels_key
        self.out_labels_key = out_labels_key
        self.name = node_name

    def process(self, context: PipelineContext) -> None:
        labels_dict = context.get(self.in_labels_key) or {}
        translated: dict[str, dict] = {}
        for box_id, res in labels_dict.items():
            orig_id = res["class_id"]
            trans_id = self.translation_map.get(orig_id, -1)
            translated[box_id] = {
                "class_id": trans_id,
                "confidence": res.get("confidence", 0.0),
                "original_class_id": orig_id,
            }
        context.set(self.out_labels_key, translated)

