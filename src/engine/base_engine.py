from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class TensorMetadata:

    name: str
    shape: tuple[int | str | None, ...]
    dtype: np.dtype | None
    native_type: str

    @property
    def dynamic(self) -> bool:
        return any(dim is None or isinstance(dim, str) for dim in self.shape)


class BaseEngine(ABC):

    def __init__(self, model_path: str | Path, device: str = "cpu") -> None:
        self._model_path = Path(model_path)
        self._device = device

    # - - - #

    @abstractmethod
    def load(self) -> None:
        ...

    @abstractmethod
    def unload(self) -> None:
        ...

    # - - - #

    def validate_inputs(self, inputs: dict[str, np.ndarray]) -> None:
        if not isinstance(inputs, dict):
            raise TypeError(f"inputs must be a dict[str, np.ndarray], got {type(inputs).__name__}")

        metadata = self.input_metadata
        expected_names = set(metadata)
        provided_names = set(inputs)

        missing = expected_names - provided_names
        if missing:
            raise KeyError(f"Missing required model inputs: {sorted(missing)}")

        unexpected = provided_names - expected_names
        if unexpected:
            raise KeyError(f"Unexpected model inputs: {sorted(unexpected)}")

        for name, tensor in inputs.items():
            if not isinstance(tensor, np.ndarray):
                raise TypeError(f"Input '{name}' must be a numpy.ndarray, got {type(tensor).__name__}")

            expected = metadata[name]
            if expected.dtype is not None and tensor.dtype != expected.dtype:
                raise TypeError(
                    f"Input '{name}': expected dtype {expected.dtype.name}, got {tensor.dtype}."
                )

            if tensor.ndim != len(expected.shape):
                raise ValueError(
                    f"Input '{name}': expected rank {len(expected.shape)}, got {tensor.ndim}."
                )

            for axis, (actual, required) in enumerate(zip(tensor.shape, expected.shape)):
                if isinstance(required, (int, np.integer)) and actual != required:
                    raise ValueError(
                        f"Input '{name}': dimension {axis} must be {required}, got {actual}."
                    )

    # - - - #

    @abstractmethod
    def predict(self, inputs: dict[str, np.ndarray]) -> list[np.ndarray]:
        ...

    # - - - #

    @property
    @abstractmethod
    def input_metadata(self) -> dict[str, TensorMetadata]:
        ...

    @property
    @abstractmethod
    def output_metadata(self) -> dict[str, TensorMetadata]:
        ...

    # - - - #

    @property
    def input_names(self) -> list[str]:
        return list(self.input_metadata)

    @property
    def output_names(self) -> list[str]:
        return list(self.output_metadata)

    @property
    def input_name(self) -> str:
        return self.input_names[0]

    @property
    def output_name(self) -> str:
        return self.output_names[0]

    # - - - #

    @property
    def device(self) -> str:
        return self._device
