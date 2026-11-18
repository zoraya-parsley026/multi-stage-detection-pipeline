from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort

from src.engine.base_engine import BaseEngine, TensorMetadata
from src.engine.engine_factory import EngineFactory

logger = logging.getLogger(__name__)

_DEVICE_PROVIDERS = {
    "cpu" : ("CPUExecutionProvider",),
    "cuda": ("CUDAExecutionProvider", "CPUExecutionProvider"),
    "rocm": ("ROCMExecutionProvider", "CPUExecutionProvider"),
}

_ONNX_DTYPES: dict[str, np.dtype[Any]] = {
    "tensor(bool)"   : np.dtype(np.bool_),
    
    "tensor(double)" : np.dtype(np.float64),
    
    "tensor(float)"  : np.dtype(np.float32),
    "tensor(float16)": np.dtype(np.float16),

    "tensor(int8)"   : np.dtype(np.int8),
    "tensor(int16)"  : np.dtype(np.int16),
    "tensor(int32)": np.dtype(np.int32),
    "tensor(int64)": np.dtype(np.int64),

    "tensor(uint8)": np.dtype(np.uint8),
    "tensor(uint16)": np.dtype(np.uint16),
    "tensor(uint32)": np.dtype(np.uint32),
    "tensor(uint64)": np.dtype(np.uint64),
}


@EngineFactory.register(["onnx", "onnxruntime"])

class ONNXEngine(BaseEngine):

    def __init__(self, model_path: str | Path, device: str = "cpu") -> None:
        super().__init__(model_path, device)
        self._session: ort.InferenceSession | None = None
        
        self._inputs: dict[str, TensorMetadata] = {}
        self._outputs: dict[str, TensorMetadata] = {}

    # - - - #

    def load(self) -> None:
        try:
            providers = _DEVICE_PROVIDERS[self.device]
        except KeyError as exc:
            raise ValueError(
                f"Unsupported device '{self.device}'. "
                f"Supported devices: {list(_DEVICE_PROVIDERS)}"
            ) from exc

        available = set(ort.get_available_providers())
        unavailable = [provider for provider in providers if provider not in available]

        if unavailable:
            raise RuntimeError(
                f"requested ONNX Runtime provider(s) unavailable: {unavailable}. "
                f"available: {sorted(available)}"
            )

        session = ort.InferenceSession(str(self._model_path), providers=list(providers))

        inputs = self._build_metadata(session.get_inputs(), "input")
        outputs = self._build_metadata(session.get_outputs(), "output")

        self._session = session

        self._inputs = inputs
        self._outputs = outputs

        logger.info(
            "loaded ONNX model %s; requested providers=%s; active providers=%s",
            self._model_path.name,
            providers,
            session.get_providers(),
        )


    def unload(self) -> None:
        self._session = None
        
        # clear in
        self._inputs.clear()
        # clear out
        self._outputs.clear()

    # - - - #

    def predict(self, inputs: dict[str, np.ndarray]) -> list[np.ndarray]:
        session = self._require_loaded()
        self.validate_inputs(inputs)
        return session.run(None, inputs)

    @property
    def input_metadata(self) -> dict[str, TensorMetadata]:
        self._require_loaded()
        return dict(self._inputs)

    @property
    def output_metadata(self) -> dict[str, TensorMetadata]:
        self._require_loaded()
        return dict(self._outputs)

    def _build_metadata(self, nodes: list[Any], direction: str) -> dict[str, TensorMetadata]:
        metadata: dict[str, TensorMetadata] = {}
        for node in nodes:
            native_type = node.type
            dtype = _ONNX_DTYPES.get(native_type)

            if dtype is None:
                raise ValueError(
                    f"Unsupported ONNX {direction} dtype '{native_type}' for '{node.name}'."
                )
            
            info = TensorMetadata(
                name=node.name,
                shape=tuple(node.shape),
                dtype=dtype,
                native_type=native_type,
            )
            metadata[info.name] = info
            
            logger.debug(
                "%s name=%s shape=%s type=%s dynamic=%s",
                direction,
                info.name,
                info.shape,
                info.native_type,
                info.dynamic,
            )
        return metadata

    def _require_loaded(self) -> ort.InferenceSession:
        if self._session is None:
            raise RuntimeError("engine not loaded")
        
        return self._session
