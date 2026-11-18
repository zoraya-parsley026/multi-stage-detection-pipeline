from collections.abc import Callable

from src.models.base_model import BaseModel
from src.models.strategies.base_strategy import PostprocessStrategy, PreprocessStrategy
from src.models.strategies.efficientformer_strategy import (
    ClassificationPostprocessStrategy,
    EfficientFormerPreprocessStrategy,
)
from src.models.strategies.yolov7_strategy import (
    YOLONMSPostprocessStrategy,
    YOLOv7PreprocessStrategy,
)
from src.models.strategies.yolov9_strategy import (
    YOLOv9PostprocessStrategy,
    YOLOv9PreprocessStrategy,
)

StrategyBuilder = Callable[[BaseModel], tuple[PreprocessStrategy, PostprocessStrategy]]


class ModelFactory:
    _registry: dict[str, StrategyBuilder] = {}

    # - - - #

    @classmethod
    def register(cls, model_type: str, builder: StrategyBuilder):
        model_type_lower = model_type.lower()

        if model_type_lower in cls._registry:
            raise ValueError(f"model type '{model_type}' is already registered")

        cls._registry[model_type_lower] = builder

    # - - - #

    @classmethod
    def create_unloaded(cls, model_type: str, weights_path: str, config_path: str, device: str = "cpu", *, engine_type: str = "onnx") -> BaseModel:
        builder = cls.get(model_type)

        model = BaseModel(weights_path, config_path, device, strategy_builder=builder, engine_type=engine_type)

        return model

    # - - - #

    @classmethod
    def create(cls, model_type: str, weights_path: str, config_path: str, device: str = "cpu", *, engine_type: str = "onnx") -> BaseModel:
        model = cls.create_unloaded(model_type, weights_path, config_path, device, engine_type=engine_type)

        model.load()
        
        return model

    # - - - #

    @classmethod
    def get(cls, model_type: str) -> StrategyBuilder:
        model_type_lower = model_type.lower()
        
        if model_type_lower not in cls._registry:
            raise ValueError(f"model type '{model_type}' not found")

        return cls._registry[model_type_lower]

    # - - - #

    @classmethod
    def list(cls):
        return list(cls._registry.keys())

    # - - - #

    @classmethod
    def _clear(cls) -> None:
        cls._registry.clear()

# Registration stays here so strategies do not import the factory.
ModelFactory.register("yolov7", lambda model: (
    YOLOv7PreprocessStrategy(model.config, model.engine.input_name),
    YOLONMSPostprocessStrategy(model.config),
))
ModelFactory.register("yolov9", lambda model: (
    YOLOv9PreprocessStrategy(model.config, model.engine.input_name),
    YOLOv9PostprocessStrategy(model.config),
))
ModelFactory.register("efficientformer", lambda model: (
    EfficientFormerPreprocessStrategy(model.config, model.engine.input_name),
    ClassificationPostprocessStrategy(model.config),
))
