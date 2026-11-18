import cv2
import numpy as np

from src.models.strategies.base_strategy import PostprocessStrategy, PreprocessStrategy
from src.utils.validation import probability, validate_image_input


class EfficientFormerPreprocessStrategy(PreprocessStrategy):
    def __init__(self, config, engine_input_name: str):
        super().__init__(config, engine_input_name)
        self.validate_config()

    def validate_config(self) -> None:
        validate_image_input(self.config.input)
        interpolation = self.config.preprocess.get("interpolation", "linear")
        if not isinstance(interpolation, str) or interpolation.lower() not in ("linear", "cubic", "nearest"):
            raise ValueError("preprocess.interpolation must be linear, cubic, or nearest.")

    def process(self, crops: list[np.ndarray]) -> tuple[dict[str, np.ndarray], dict]:
        """
        Receives N cropped images (already bounding-box sliced).
        Returns batched tensor [N, C, H, W] for single ONNX call.
        """
        self.validate_config()
        n_crops = len(crops)
        
        # Short-circuit logic if no crops are passed
        if n_crops == 0:
            return {}, {"n_crops": 0}

        batched_tensors = []
        target_size = (
            self.config.input["width"],
            self.config.input["height"],
        )
        
        interp_map = {
            "linear": cv2.INTER_LINEAR,
            "cubic": cv2.INTER_CUBIC,
            "nearest": cv2.INTER_NEAREST
        }
        interp = interp_map[self.config.preprocess.get("interpolation", "linear").lower()]

        for crop in crops:
            # 1. Resize to target dimension
            img = cv2.resize(crop, target_size, interpolation=interp)
            
            # 2. Convert to RGB (if required)
            if self.config.input.get("channels_order", "RGB") == "RGB":
                img = img[:, :, ::-1]

            # 3. HWC to CHW
            img = img.transpose(2, 0, 1)
            
            batched_tensors.append(img)
            
        # 4. Stack to [N, C, H, W] FIRST (Vectorized Optimization)
        final_tensor = np.stack(batched_tensors, axis=0)
        final_tensor = np.ascontiguousarray(final_tensor)
        final_tensor = final_tensor.astype(self.config.input.get("data_type", "float32"))

        # 5. Vectorized Normalization (Applied to the whole batch at once)
        final_tensor /= 255.0
        if self.config.input.get("normalize", None):
            mean = np.array(self.config.input["normalize"]["mean"], dtype=final_tensor.dtype).reshape(1, 3, 1, 1)
            std = np.array(self.config.input["normalize"]["std"], dtype=final_tensor.dtype).reshape(1, 3, 1, 1)
            final_tensor = (final_tensor - mean) / std

        tensors = {self.engine_input_name: final_tensor}
        routing_metadata = {"n_crops": n_crops}
        
        return tensors, routing_metadata


class ClassificationPostprocessStrategy(PostprocessStrategy):
    def __init__(self, config):
        super().__init__(config)
        self.validate_config()

    def validate_config(self) -> None:
        probability(self.config.postprocess.get("conf_thres", 0.25), "postprocess.conf_thres")

    def process(self, raw_outputs: list[np.ndarray], routing_metadata: dict) -> list[dict]:
        """
        Extracts Contract 3 data: list of {"class_id", "confidence"} for N crops.
        """
        self.validate_config()
        if routing_metadata["n_crops"] == 0:
            return []

        logits = raw_outputs[0] # [N, num_classes]

        # 1. Softmax to yield probabilities
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

        # 2. Extract max class and confidence
        class_ids = np.argmax(probs, axis=1)
        confidences = np.max(probs, axis=1)

        # 3. Thresholding (Reject low confidence crops as background)
        conf_mask = confidences < self.config.postprocess.get("conf_thres", 0.25)
        class_ids[conf_mask] = -1

        results = []
        for c_id, conf in zip(class_ids, confidences):
            results.append({
                "class_id": int(c_id),
                "confidence": float(conf)
            })

        return results
