from __future__ import annotations

import logging
import time
from collections.abc import Iterable
from typing import Any

try:
    from tqdm import tqdm as _tqdm
except ImportError:
    _tqdm = None  

from src.nodes.base_node import PipelineNode
from src.pipeline.observers import PipelineObserver
from src.pipeline.pipeline_context import PipelineContext

logger = logging.getLogger(__name__)

class TopologyError(Exception):
    pass

# - - - #

class Pipeline:
    def __init__(
        self,
        nodes: list[PipelineNode],
        initial_key: str = "image",
        sample_key : str = "data",
    ) -> None:
        if not nodes:
            raise ValueError("Pipeline must contain at least one node.")
        self.nodes:     list[PipelineNode]     = list(nodes)
        self.observers: list[PipelineObserver] = []
        
        self.initial_key = initial_key
        self.sample_key  = sample_key

        self._validate_topology()

    # - - - #

    def _validate_topology(self) -> None:
        available: set[str] = {self.initial_key}
        for idx, node in enumerate(self.nodes):
            for key in node.in_keys:
                if key not in available:
                    raise TopologyError(
                        f"Topology Error at Node {idx} ({node.name}): "
                        f"Required input key '{key}' is not available. "
                        f"Available keys: {available}"
                    )
            for key in node.out_keys:
                if key in available:
                    logger.debug("Node %r overwrites existing key %r.", node.name, key)
                available.add(key)

    def _compile(self) -> None:
        self._validate_topology()

    # - - - #

    def add_observer(self, observer: PipelineObserver) -> None:
        self.observers.append(observer)

    # - - - #

    @staticmethod
    def _execute_node(node: PipelineNode, context: PipelineContext) -> float:
        t0 = time.perf_counter()
        node.process(context)
        return (time.perf_counter() - t0) * 1_000.0

    # - - - #

    def _run_nodes(self, context: PipelineContext) -> tuple[dict[str, float], bool]:
        node_timings: dict[str, float] = {}

        for node in self.nodes:
            elapsed_ms = self._execute_node(node, context)
            node_timings[f"node_{node.name}_ms"] = elapsed_ms

            stage_tag = getattr(node, "stage_type", None)
            if stage_tag:
                node_timings[f"{stage_tag}_ms"] = (
                    node_timings.get(f"{stage_tag}_ms", 0.0) + elapsed_ms
                )

            if context.is_aborted:
                logger.debug(
                    "Short-circuit triggered at %r — skipping remaining nodes.",
                    node.name,
                )
                return node_timings, True

        return node_timings, False

    # - - - #

    @staticmethod
    def _build_latency(node_timings: dict[str, float], total_ms: float) -> dict[str, float]:
        latency: dict[str, float] = dict(node_timings)
        latency["total_ms"] = total_ms
        return latency

    def _notify_observers(self, sample: dict[str, Any], context: PipelineContext) -> None:
        for obs in self.observers:
            target_k = getattr(obs, "target_key", None)
            predictions = context.get(target_k) if target_k else None
            obs.on_batch_complete(
                ground_truth=sample.get("ground_truth"),
                predictions=predictions,
                batch_data=sample,
                context=context,
            )

    # - - - #

    def process_sample(self, sample: dict[str, Any]) -> PipelineContext:
        context = PipelineContext(
            initial_state={self.initial_key: sample.get(self.sample_key)}
        )

        t_start = time.perf_counter()
        node_timings, was_aborted = self._run_nodes(context)
        t_end = time.perf_counter()
        total_ms = (t_end - t_start) * 1000.0

        sample["latency"] = self._build_latency(node_timings, total_ms)
        self._notify_observers(sample, context)

        return context

    # - - - #

    def run(self, dataloader: Iterable[dict[str, Any]], *, progress: bool = True) -> list[PipelineContext]:
        logger.info("Starting sequential pipeline execution...")

        iterable = (
            _tqdm(dataloader, desc="Processing", unit="img")
            if progress and _tqdm is not None
            else dataloader
        )

        contexts = [self.process_sample(sample) for sample in iterable]

        logger.info("Pipeline execution complete.")
        return contexts