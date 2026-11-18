"""
Filter Strategies
=================
Pipeline nodes that filter, gate, or validate detections.
"""

from __future__ import annotations

import logging
from abc import abstractmethod
from typing import Any

from src.nodes.base_node import PipelineNode
from src.nodes.node_factory import NodeFactory
from src.pipeline.pipeline_context import PipelineContext

logger = logging.getLogger(__name__)


class BaseFilterNode(PipelineNode):
    """
    Abstract blueprint for detection filtering and gating nodes.

    Subclasses must implement :meth:`filter_detections` to define the gating
    logic. The base class handles reading detections from the context,
    invoking the filter, writing results back, and optional short-circuiting.

    Parameters
    ----------
    in_boxes_key:
        Context key for input detections (default: ``"detections"``).
    out_boxes_key:
        Context key for filtered detections (default: ``"filtered_detections"``).
    node_name:
        Display name used in logs and the registry.
    abort_on_empty:
        If ``True``, short-circuits the pipeline if 0 detections pass the filter.
    """

    def __init__(
        self,
        in_boxes_key: str = "detections",
        out_boxes_key: str = "filtered_detections",
        node_name: str = "BaseFilterNode",
        abort_on_empty: bool = False,
    ) -> None:
        self.in_keys = [in_boxes_key]
        self.out_keys = [out_boxes_key]
        self.in_boxes_key = in_boxes_key
        self.out_boxes_key = out_boxes_key
        self.name = node_name
        self.abort_on_empty = abort_on_empty

    def process(self, context: PipelineContext) -> None:
        detections = context.get(self.in_boxes_key) or []
        filtered = self.filter_detections(detections, context)
        context.set(self.out_boxes_key, filtered)

        if self.abort_on_empty and len(filtered) == 0:
            logger.debug(
                f"{self.name}: 0 detections passed filter. "
                "Short-circuiting pipeline execution."
            )
            context.abort()

    @abstractmethod
    def filter_detections(
        self,
        detections: list[dict[str, Any]],
        context: PipelineContext,
    ) -> list[dict[str, Any]]:
        """Filter the detections and return the subset that passes."""
        ...


@NodeFactory.register(["logic_filter", "logic_filter_node"])
class LogicFilterNode(BaseFilterNode):
    """
    Gating filter that compares detector classes against translated classifier
    classes and rejects detections where they disagree.

    Parameters
    ----------
    in_boxes_key:
        Context key for the raw detection list (default: ``"detections"``).
    in_labels_key:
        Context key for the translated labels dict
        (default: ``"translated_labels"``).
    out_boxes_key:
        Context key for the filtered detection list
        (default: ``"filtered_detections"``).
    node_name:
        Display name used in logs and the registry.
    abort_on_empty:
        Whether to short-circuit if no detections survive (default: ``False``).
    """

    def __init__(
        self,
        in_boxes_key: str = "detections",
        in_labels_key: str = "translated_labels",
        out_boxes_key: str = "filtered_detections",
        node_name: str = "LogicFilterNode",
        abort_on_empty: bool = False,
    ) -> None:
        super().__init__(
            in_boxes_key=in_boxes_key,
            out_boxes_key=out_boxes_key,
            node_name=node_name,
            abort_on_empty=abort_on_empty,
        )
        self.in_keys = [in_boxes_key, in_labels_key]
        self.in_labels_key = in_labels_key

    def filter_detections(
        self,
        detections: list[dict[str, Any]],
        context: PipelineContext,
    ) -> list[dict[str, Any]]:
        labels_dict = context.get(self.in_labels_key) or {}

        filtered = []
        for det in detections:
            det_id = det.get("id")
            if det_id not in labels_dict:
                continue

            res = labels_dict[det_id]
            if res["class_id"] == det["class_id"]:
                filtered.append(det)
            else:
                logger.debug(
                    f"Filtered detection {det_id}: "
                    f"detector class {det['class_id']} != "
                    f"classifier class {res['class_id']} "
                    f"(conf {res.get('confidence', 0.0):.2f})"
                )

        return filtered

