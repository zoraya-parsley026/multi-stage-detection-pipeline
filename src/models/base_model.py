import logging
from typing import Any

import numpy as np

from src.engine import BaseEngine, EngineFactory
from src.models.base_model_config import BaseModelConfig
from src.models.strategies.base_strategy import PostprocessStrategy, PreprocessStrategy

logger = logging.getLogger(__name__)

class BaseModel:
    def __init__(self, weights_path: str, config_path: str, device: str = "cpu", strategy_builder=None, *, engine_type: str = "onnx"):
        self.config_path  = config_path
        self.weights_path = weights_path

        self.device = device
        self.engine_type = engine_type
                
        self.config = None
        self.engine: BaseEngine | None = None
        
        self.strategy_builder = strategy_builder

        self.pre_strategy: PreprocessStrategy = None
        self.post_strategy: PostprocessStrategy = None
        self._supports_dynamic_batch: bool = True

    # - - - #

    def load(self):
        self.config = BaseModelConfig.from_yaml(self.config_path)

        self.engine = EngineFactory.create(
            self.engine_type, model_path=self.weights_path, device=self.device
        )
        self.engine.load()
        
        if self.strategy_builder:
            pre, post = self.strategy_builder(self)
            self.set_strategies(pre, post)
        
    # - - - #

    def set_strategies(self, pre_strategy: PreprocessStrategy, post_strategy: PostprocessStrategy):
        self.pre_strategy = pre_strategy
        self.post_strategy = post_strategy

    # - - - #

    def predict(self, data: Any) -> Any:
        if not self.pre_strategy:
            raise RuntimeError("pre_strategy")

        if not self.post_strategy:
            raise RuntimeError("post_strategy")

        # - - pre_
        engine_inputs, meta = self.pre_strategy.process(data)
        if not engine_inputs:
            return self.post_strategy.process([], meta)
        
        # - - infer_
        raw_outputs = self._infer(engine_inputs)
        
        # - - post_
        return self.post_strategy.process(raw_outputs, meta)

    # - - - #

    def _infer(self, engine_inputs: dict[str, np.ndarray]) -> list[np.ndarray]:
        batch_size = next(iter(engine_inputs.values())).shape[0]

        if (not self._supports_dynamic_batch) and (batch_size > 1):
            return self._infer_sequential(engine_inputs, batch_size)

        try:
            return self.engine.predict(engine_inputs)
        except Exception as exc:
            if batch_size > 1 and self._is_batch_shape_error(exc):
                self._supports_dynamic_batch = False
                
                logger.info("Falling back to sequential inference: %s", exc)
                return self._infer_sequential(engine_inputs, batch_size)
            raise exc

    def _infer_sequential(self, engine_inputs: dict[str, np.ndarray], batch_size: int) -> list[np.ndarray]:
        # - - infer 
        sample_outputs = []
        for i in range(batch_size):
            single_input = {k: v[i:i+1] for k, v in engine_inputs.items()}
            sample_outputs.append(self.engine.predict(single_input))

        # - - concat 
        concat_outputs = [
            np.concatenate([out[h] for out in sample_outputs], axis=0)
            for h in range(len(sample_outputs[0]))
        ]
        
        return concat_outputs

    # - - - #
    
    @staticmethod
    def _is_batch_shape_error(exc: Exception) -> bool:
        error_msg = str(exc)
        return (
            "INVALID_ARGUMENT" in error_msg
            or "dimension 0 must be" in error_msg
        )
