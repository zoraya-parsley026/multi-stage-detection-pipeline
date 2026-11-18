import numpy as np

from src.models.strategies.base_strategy import PostprocessStrategy, PreprocessStrategy
from src.utils.box_utils import NonMaxSuppression, clip_boxes, letterbox, scale_coords
from src.utils.validation import positive_integer, probability, validate_image_input


class YOLOv7PreprocessStrategy(PreprocessStrategy):
    def __init__(self, config, engine_input_name: str):
        super().__init__(config, engine_input_name)
        self.validate_config()

    def validate_config(self) -> None:
        validate_image_input(self.config.input)
        positive_integer(self.config.preprocess.get("stride", 32), "preprocess.stride")
        if not isinstance(self.config.preprocess.get("letterbox_auto", True), bool):
            raise ValueError("preprocess.letterbox_auto must be a boolean.")

    def process(self, image: np.ndarray) -> tuple[dict[str, np.ndarray], dict]:
        self.validate_config()
        # 1. Letterbox
        img_lb, ratio, pad = letterbox(
            image, 
            new_shape=(
                self.config.input["height"],
                self.config.input["width"],
            ),
            auto=self.config.preprocess.get("letterbox_auto", True),
            stride=self.config.preprocess.get("stride", 32)
        )
        
        # 2. BGR to RGB
        img = img_lb[:, :, ::-1] if self.config.input.get("channels_order", "RGB") == "RGB" else img_lb
        
        # 3. HWC to CHW
        img = img.transpose(2, 0, 1)
        
        # 4. Contiguous array
        img = np.ascontiguousarray(img)
        
        # 5. Type cast
        img = img.astype(self.config.input.get("data_type", "float32"))
        
        # 6. Normalize
        img /= 255.0
        normalize = self.config.input.get("normalize")
        if normalize is not None:
            mean = np.asarray(normalize["mean"], dtype=img.dtype).reshape(3, 1, 1)
            std = np.asarray(normalize["std"], dtype=img.dtype).reshape(3, 1, 1)
            img = (img - mean) / std
            
        # 7. Add batch dim
        img = img[np.newaxis, :]

        tensors = {self.engine_input_name: img}
        routing_metadata = {
            "input_shape": img.shape[2:], # H_lb, W_lb
            "original_shape": image.shape[:2]
        }
        return tensors, routing_metadata

class YOLONMSPostprocessStrategy(PostprocessStrategy):
    def __init__(self, config):
        super().__init__(config)
        self.validate_config()

    def validate_config(self) -> None:
        probability(self.config.postprocess.get("conf_thres", 0.25), "postprocess.conf_thres")
        probability(self.config.postprocess.get("iou_thres", 0.45), "postprocess.iou_thres")
        positive_integer(self.config.postprocess.get("max_det", 300), "postprocess.max_det")

    def process(self, raw_outputs: list[np.ndarray], routing_metadata: dict) -> list[dict]:
        self.validate_config()
        # 1. NMS
        nms = NonMaxSuppression(
            conf_thres=self.config.postprocess.get("conf_thres", 0.25),
            iou_thres=self.config.postprocess.get("iou_thres", 0.45),
            max_det=self.config.postprocess.get("max_det", 300),
            coordinate_input_format="xywh"
        )
        
        # raw_outputs[0] -> (1, 25200, 85)
        detections = nms(raw_outputs[0])[0]  # Take batch 0 (output is list of batched detections)
        
        results = []
        if len(detections) == 0:
            return results

        # 2. Scale Coordinates
        detections[:, :4] = scale_coords(
            routing_metadata["input_shape"], 
            detections[:, :4], 
            routing_metadata["original_shape"]
        )
        
        # 3. Clip
        clip_boxes(detections[:, :4], routing_metadata["original_shape"])
        
        # 4. Format to Contract 2
        for *xyxy, conf, cls in detections:
            results.append({
                "class_id": int(cls),
                "confidence": float(conf),
                "localization": [float(c) for c in xyxy]
            })
            
        return results
