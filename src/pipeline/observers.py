import logging
from abc import ABC, abstractmethod
from typing import Any

from src.metrics.evaluator import Evaluator
from src.pipeline.pipeline_context import PipelineContext

logger = logging.getLogger(__name__)

class PipelineObserver(ABC):
    @abstractmethod
    def on_batch_complete(
        self,
        ground_truth: list[dict[str, Any]] | None,
        predictions:           list[dict[str, Any]],
        
        batch_data:                 dict[str, Any],
        
        context: PipelineContext
    ) -> None:
        pass

# - - - #

class ContextObserver(PipelineObserver):
    def __init__(
        self,
        evaluator: Evaluator,
        target_key: str,
        baseline_key: str | None = None
    ):
        self.evaluator = evaluator
        self.target_key = target_key
        self.baseline_key = baseline_key
    
    # - - - #

    def on_batch_complete(
        self,
        ground_truth: list[dict[str, Any]] | None = None,
        predictions:  list[dict[str, Any]] | None = None,
        
        batch_data:   dict[str, Any] | None  = None,

        context:      PipelineContext | None = None
    ) -> None:
        batch_data = batch_data or {}
        
        if predictions is None and context is not None:
            predictions = context.get(self.target_key, [])
            
        baseline_detections = []
        if context is not None and self.baseline_key is not None:
            baseline_detections = context.get(self.baseline_key, [])

        self.evaluator.accumulate(
            predictions=predictions or [],
            ground_truth=ground_truth if ground_truth is not None else batch_data.get("ground_truth"),
            batch_data=batch_data,
            detections=baseline_detections,
        )