from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from src.models.base_model_config import BaseModelConfig


class PreprocessStrategy(ABC):
    def __init__(self, config: BaseModelConfig, engine_input_name: str):
        self.config = config
        self.engine_input_name = engine_input_name
        
    @abstractmethod
    def process(self, data: Any) -> tuple[dict[str, np.ndarray], dict]:
        ''' '''
        pass

    # - - - #

class PostprocessStrategy(ABC):
    def __init__(self, config: BaseModelConfig):
        self.config = config
        
    @abstractmethod
    def process(self, raw_outputs: list[np.ndarray], routing_metadata: dict) -> Any:
        ''' '''
        pass
