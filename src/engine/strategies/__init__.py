"""Built-in inference strategies, registered through EngineFactory."""

from src.engine.strategies.onnx_strategy import ONNXEngine

__all__ = ["ONNXEngine"]
