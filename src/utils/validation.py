"""Reusable checks for numeric values and image input configuration."""

import math
from numbers import Integral, Real

import numpy as np


def validate_class_id(value) -> None:
    if isinstance(value, bool) or not isinstance(value, (Integral, str)):
        raise ValueError("class_id must be an integer or string.")


def positive_integer(value, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer.")


def probability(value, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, Real) or not 0 <= value <= 1:
        raise ValueError(f"{name} must be a number between 0 and 1.")


def validate_image_dimensions(width, height) -> None:
    """Require positive image dimensions."""
    positive_integer(width, "input.width")
    positive_integer(height, "input.height")


def validate_image_channels(channels_order, channels_count) -> None:
    """Check the three-channel RGB/BGR contract."""
    if channels_order not in ("RGB", "BGR"):
        raise ValueError("input.channels_order must be RGB or BGR.")
    if channels_count != 3:
        raise ValueError("These image strategies require input.channels_count = 3.")


def validate_image_layout(layout) -> None:
    """Check the tensor layout supported by the current image strategies."""
    if layout != "NCHW":
        raise ValueError("These image strategies require input.format = NCHW.")


def validate_image_dtype(data_type) -> None:
    """Require a floating-point dtype for pixel scaling and normalization."""
    try:
        dtype = np.dtype(data_type)
    except (TypeError, ValueError) as exc:
        raise ValueError("input.data_type must be a floating-point NumPy dtype.") from exc
    if dtype.kind != "f":
        raise ValueError("input.data_type must be a floating-point NumPy dtype.")


def validate_normalization(normalize) -> None:
    """Validate optional per-channel mean/std settings; None disables them."""
    if normalize is None:
        return
    if not isinstance(normalize, dict):
        raise ValueError("input.normalize must be a mapping with mean and std.")
    for name in ("mean", "std"):
        values = normalize.get(name)
        if not isinstance(values, (list, tuple)) or len(values) != 3:
            raise ValueError(f"input.normalize.{name} must contain three numbers.")
        if any(isinstance(value, bool) or not isinstance(value, Real)
               or not math.isfinite(value) for value in values):
            raise ValueError(f"input.normalize.{name} must contain finite numbers.")
    if any(value <= 0 for value in normalize["std"]):
        raise ValueError("input.normalize.std must contain positive numbers.")


def validate_image_input(options: dict) -> None:
    """Compose the checks for the current three-channel NCHW image contract."""
    validate_image_dimensions(options.get("width"), options.get("height"))
    validate_image_channels(
        options.get("channels_order", "RGB"),
        options.get("channels_count", 3),
    )
    validate_image_layout(options.get("format", "NCHW"))
    validate_image_dtype(options.get("data_type", "float32"))
    validate_normalization(options.get("normalize"))
