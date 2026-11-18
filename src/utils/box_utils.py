from typing import Any

import cv2
import numpy as np
import torch
import torchvision


def xywh2xyxy(x: np.ndarray) -> np.ndarray:
    y = np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2  
    y[..., 1] = x[..., 1] - x[..., 3] / 2  
    y[..., 2] = x[..., 0] + x[..., 2] / 2  
    y[..., 3] = x[..., 1] + x[..., 3] / 2  
    return y


def xyxy2xywh(x: np.ndarray) -> np.ndarray:
    y = np.copy(x)
    y[..., 0] = (x[..., 0] + x[..., 2]) / 2  
    y[..., 1] = (x[..., 1] + x[..., 3]) / 2  
    y[..., 2] =  x[..., 2] - x[..., 0]  
    y[..., 3] =  x[..., 3] - x[..., 1]  
    return y

# - - - #

def xywhn2xyxy(x: np.ndarray, w=640, h=640, padw=0, padh=0) -> np.ndarray:
    y = np.copy(x)
    y[..., 0] = w * (x[..., 0] - x[..., 2] / 2) + padw
    y[..., 1] = h * (x[..., 1] - x[..., 3] / 2) + padh
    y[..., 2] = w * (x[..., 0] + x[..., 2] / 2) + padw
    y[..., 3] = h * (x[..., 1] + x[..., 3] / 2) + padh
    return y


def xyxy2xywhn(x: np.ndarray, w=640, h=640, padw=0, padh=0) -> np.ndarray:
    if w <= 0 or h <= 0:
        raise ValueError("Image width and height must be positive.")
    y = np.copy(x)
    y[..., 0] = ((x[..., 0] + x[..., 2]) / 2 - padw) / w
    y[..., 1] = ((x[..., 1] + x[..., 3]) / 2 - padh) / h
    y[..., 2] =  (x[..., 2] - x[..., 0]) / w
    y[..., 3] =  (x[..., 3] - x[..., 1]) / h
    return y

# - - - #

def ltwh2xyxy(x: np.ndarray) -> np.ndarray:
    y = np.copy(x)
    y[..., 2] = x[..., 2] + x[..., 0]
    y[..., 3] = x[..., 3] + x[..., 1]
    return y


def xyxy2ltwh(x: np.ndarray) -> np.ndarray:
    y = np.copy(x)
    y[..., 2] = x[..., 2] - x[..., 0]
    y[..., 3] = x[..., 3] - x[..., 1]
    return y

# - - - #

def clip_coords(boxes, shape):
    boxes[..., [0, 2]] = boxes[..., [0, 2]].clip(0, shape[1])  # x1, x2
    boxes[..., [1, 3]] = boxes[..., [1, 3]].clip(0, shape[0])  # y1, y2

def clip_boxes(boxes, shape):
    clip_coords(boxes, shape)

def crop_roi(image: np.ndarray, localization: list[float]) -> np.ndarray:
    """
    Extracts a rectangular region of interest from the image.
    
    Args:
        image: Original BGR image array.
        localization: Bounding box as [x1, y1, x2, y2].
        
    Returns:
        The cropped image array.
        
    Raises:
        ValueError: If the resulting crop has a width or height <= 0.
    """
    x1, y1, x2, y2 = [int(v) for v in localization]
    if x2 - x1 <= 0 or y2 - y1 <= 0:
        raise ValueError(f"Crop area is zero for localization {localization}")
    return image[y1:y2, x1:x2].copy()

def crop_and_validate(
    image: np.ndarray,
    detections: list[dict],
    classifier: Any,
    class_translation_map: dict[int, int]
) -> list[dict]:
    """
    Helper function executing the 4-step cascading validation chain.
    """
    import uuid

    from src.nodes.strategies import (
        CropNode,
        LogicFilterNode,
        ModelClassificationNode,
        TaxonomyTranslationNode,
    )
    from src.pipeline.pipeline_context import PipelineContext
    
    for det in detections:
        if "id" not in det:
            det["id"] = uuid.uuid4().hex
            
    nodes = [
        CropNode(in_boxes_key="detections", out_crops_key="crops"),
        ModelClassificationNode(classifier, in_crops_key="crops", out_labels_key="imagenet_labels"),
        TaxonomyTranslationNode(class_translation_map, in_labels_key="imagenet_labels", out_labels_key="coco_labels"),
        LogicFilterNode(in_boxes_key="detections", in_labels_key="coco_labels", out_boxes_key="filtered")
    ]
    
    context = PipelineContext({"image": image, "detections": detections})
    for node in nodes:
        node.process(context)
        if context.is_aborted:
            break
            
    return context.get("filtered", [])

def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[..., [0, 2]] -= pad[0]  # x padding
    coords[..., [1, 3]] -= pad[1]  # y padding
    coords[..., [0, 2]] /= gain
    coords[..., [1, 3]] /= gain
    clip_coords(coords, img0_shape)
    return coords

def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)

def _as_box_matrix(boxes: np.ndarray, name: str) -> np.ndarray:
    """Normalize one box or a box collection to a float32 ``(N, 4)`` matrix."""
    result = np.asarray(boxes, dtype=np.float32)
    if result.ndim == 1:
        if result.size != 4:
            raise ValueError(f"{name} must contain four coordinates, got shape {result.shape}.")
        result = result.reshape(1, 4)
    elif result.ndim != 2 or result.shape[1] != 4:
        raise ValueError(f"{name} must have shape (4,) or (N, 4), got {result.shape}.")
    return result


# - - - #

def box_iou(box1: np.ndarray, box2: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    """Calculate pairwise intersection over union for ``xyxy`` boxes."""
    b1 = _as_box_matrix(box1, "box1")
    b2 = _as_box_matrix(box2, "box2")
    if len(b1) == 0 or len(b2) == 0:
        return np.zeros((len(b1), len(b2)), dtype=np.float32)

    intersection_width = np.maximum(
        0.0,
        np.minimum(b1[:, None, 2], b2[None, :, 2])
        - np.maximum(b1[:, None, 0], b2[None, :, 0]),
    )
    intersection_height = np.maximum(
        0.0,
        np.minimum(b1[:, None, 3], b2[None, :, 3])
        - np.maximum(b1[:, None, 1], b2[None, :, 1]),
    )
    intersection = intersection_width * intersection_height

    b1_width = np.maximum(0.0, b1[:, 2] - b1[:, 0])
    b1_height = np.maximum(0.0, b1[:, 3] - b1[:, 1])
    b2_width = np.maximum(0.0, b2[:, 2] - b2[:, 0])
    b2_height = np.maximum(0.0, b2[:, 3] - b2[:, 1])
    union = (
        b1_width[:, None] * b1_height[:, None]
        + b2_width[None, :] * b2_height[None, :]
        - intersection
    )

    return np.divide(
        intersection,
        union,
        out=np.zeros_like(intersection, dtype=np.float32),
        where=union > eps,
    )


def box_giou(box1: np.ndarray, box2: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    """Calculate pairwise generalized IoU for ``xyxy`` boxes."""
    b1 = _as_box_matrix(box1, "box1")
    b2 = _as_box_matrix(box2, "box2")
    if len(b1) == 0 or len(b2) == 0:
        return np.zeros((len(b1), len(b2)), dtype=np.float32)

    intersection_width = np.maximum(
        0.0,
        np.minimum(b1[:, None, 2], b2[None, :, 2])
        - np.maximum(b1[:, None, 0], b2[None, :, 0]),
    )
    intersection_height = np.maximum(
        0.0,
        np.minimum(b1[:, None, 3], b2[None, :, 3])
        - np.maximum(b1[:, None, 1], b2[None, :, 1]),
    )
    intersection = intersection_width * intersection_height

    b1_width = np.maximum(0.0, b1[:, 2] - b1[:, 0])
    b1_height = np.maximum(0.0, b1[:, 3] - b1[:, 1])
    b2_width = np.maximum(0.0, b2[:, 2] - b2[:, 0])
    b2_height = np.maximum(0.0, b2[:, 3] - b2[:, 1])
    union = (
        b1_width[:, None] * b1_height[:, None]
        + b2_width[None, :] * b2_height[None, :]
        - intersection
    )
    iou = np.divide(
        intersection,
        union,
        out=np.zeros_like(intersection, dtype=np.float32),
        where=union > eps,
    )

    enclosing_width = np.maximum(b1[:, None, 2], b2[None, :, 2]) - np.minimum(
        b1[:, None, 0], b2[None, :, 0]
    )
    enclosing_height = np.maximum(b1[:, None, 3], b2[None, :, 3]) - np.minimum(
        b1[:, None, 1], b2[None, :, 1]
    )
    enclosing_area = np.maximum(0.0, enclosing_width) * np.maximum(0.0, enclosing_height)
    uncovered_fraction = np.divide(
        enclosing_area - union,
        enclosing_area,
        out=np.zeros_like(enclosing_area, dtype=np.float32),
        where=enclosing_area > eps,
    )
    return iou - uncovered_fraction


def box_diou(box1: np.ndarray, box2: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    """Calculate pairwise distance IoU for ``xyxy`` boxes."""
    b1 = _as_box_matrix(box1, "box1")
    b2 = _as_box_matrix(box2, "box2")
    if len(b1) == 0 or len(b2) == 0:
        return np.zeros((len(b1), len(b2)), dtype=np.float32)

    intersection_width = np.maximum(
        0.0,
        np.minimum(b1[:, None, 2], b2[None, :, 2])
        - np.maximum(b1[:, None, 0], b2[None, :, 0]),
    )
    intersection_height = np.maximum(
        0.0,
        np.minimum(b1[:, None, 3], b2[None, :, 3])
        - np.maximum(b1[:, None, 1], b2[None, :, 1]),
    )
    intersection = intersection_width * intersection_height

    b1_width = np.maximum(0.0, b1[:, 2] - b1[:, 0])
    b1_height = np.maximum(0.0, b1[:, 3] - b1[:, 1])
    b2_width = np.maximum(0.0, b2[:, 2] - b2[:, 0])
    b2_height = np.maximum(0.0, b2[:, 3] - b2[:, 1])
    union = (
        b1_width[:, None] * b1_height[:, None]
        + b2_width[None, :] * b2_height[None, :]
        - intersection
    )
    iou = np.divide(
        intersection,
        union,
        out=np.zeros_like(intersection, dtype=np.float32),
        where=union > eps,
    )

    b1_center_x = (b1[:, 0] + b1[:, 2]) / 2.0
    b1_center_y = (b1[:, 1] + b1[:, 3]) / 2.0
    b2_center_x = (b2[:, 0] + b2[:, 2]) / 2.0
    b2_center_y = (b2[:, 1] + b2[:, 3]) / 2.0
    center_distance_squared = (
        (b1_center_x[:, None] - b2_center_x[None, :]) ** 2
        + (b1_center_y[:, None] - b2_center_y[None, :]) ** 2
    )

    enclosing_width = np.maximum(b1[:, None, 2], b2[None, :, 2]) - np.minimum(
        b1[:, None, 0], b2[None, :, 0]
    )
    enclosing_height = np.maximum(b1[:, None, 3], b2[None, :, 3]) - np.minimum(
        b1[:, None, 1], b2[None, :, 1]
    )
    enclosing_diagonal_squared = enclosing_width**2 + enclosing_height**2
    distance_penalty = np.divide(
        center_distance_squared,
        enclosing_diagonal_squared,
        out=np.zeros_like(center_distance_squared, dtype=np.float32),
        where=enclosing_diagonal_squared > eps,
    )
    return iou - distance_penalty


def box_ciou(box1: np.ndarray, box2: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    """Calculate pairwise complete IoU for ``xyxy`` boxes."""
    b1 = _as_box_matrix(box1, "box1")
    b2 = _as_box_matrix(box2, "box2")
    if len(b1) == 0 or len(b2) == 0:
        return np.zeros((len(b1), len(b2)), dtype=np.float32)

    intersection_width = np.maximum(
        0.0,
        np.minimum(b1[:, None, 2], b2[None, :, 2])
        - np.maximum(b1[:, None, 0], b2[None, :, 0]),
    )
    intersection_height = np.maximum(
        0.0,
        np.minimum(b1[:, None, 3], b2[None, :, 3])
        - np.maximum(b1[:, None, 1], b2[None, :, 1]),
    )
    intersection = intersection_width * intersection_height

    b1_width = np.maximum(0.0, b1[:, 2] - b1[:, 0])
    b1_height = np.maximum(0.0, b1[:, 3] - b1[:, 1])
    b2_width = np.maximum(0.0, b2[:, 2] - b2[:, 0])
    b2_height = np.maximum(0.0, b2[:, 3] - b2[:, 1])
    union = (
        b1_width[:, None] * b1_height[:, None]
        + b2_width[None, :] * b2_height[None, :]
        - intersection
    )
    iou = np.divide(
        intersection,
        union,
        out=np.zeros_like(intersection, dtype=np.float32),
        where=union > eps,
    )

    b1_center_x = (b1[:, 0] + b1[:, 2]) / 2.0
    b1_center_y = (b1[:, 1] + b1[:, 3]) / 2.0
    b2_center_x = (b2[:, 0] + b2[:, 2]) / 2.0
    b2_center_y = (b2[:, 1] + b2[:, 3]) / 2.0
    center_distance_squared = (
        (b1_center_x[:, None] - b2_center_x[None, :]) ** 2
        + (b1_center_y[:, None] - b2_center_y[None, :]) ** 2
    )

    enclosing_width = np.maximum(b1[:, None, 2], b2[None, :, 2]) - np.minimum(
        b1[:, None, 0], b2[None, :, 0]
    )
    enclosing_height = np.maximum(b1[:, None, 3], b2[None, :, 3]) - np.minimum(
        b1[:, None, 1], b2[None, :, 1]
    )
    enclosing_diagonal_squared = enclosing_width**2 + enclosing_height**2
    distance_penalty = np.divide(
        center_distance_squared,
        enclosing_diagonal_squared,
        out=np.zeros_like(center_distance_squared, dtype=np.float32),
        where=enclosing_diagonal_squared > eps,
    )

    aspect_difference = (
        np.arctan2(b2_width[None, :], b2_height[None, :])
        - np.arctan2(b1_width[:, None], b1_height[:, None])
    )
    aspect_penalty = (4.0 / np.pi**2) * aspect_difference**2
    aspect_weight = np.divide(
        aspect_penalty,
        1.0 - iou + aspect_penalty,
        out=np.zeros_like(aspect_penalty, dtype=np.float32),
        where=(1.0 - iou + aspect_penalty) > eps,
    )
    return iou - distance_penalty - aspect_weight * aspect_penalty

# - - - #

class NonMaxSuppression:
    def __init__(self, conf_thres=0.25, iou_thres=0.45, max_det=300, coordinate_input_format="xywh"):
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.max_det = max_det
        self.coordinate_input_format = coordinate_input_format

    def __call__(self, prediction: np.ndarray) -> list[np.ndarray]:
        """
        Runs Batched NMS.
        Note: Regardless of coordinate_input_format, the output is ALWAYS formatted
        as `xyxy` [x1, y1, x2, y2, conf, cls] for the detection contract.
        """
        # prediction shape is (batch, num_anchors, 85)
        output = [np.zeros((0, 6))] * prediction.shape[0]
        
        for xi, x in enumerate(prediction):
            # 1. Compute true scores: objectness * class_conf
            obj_conf = x[..., 4:5]
            cls_conf = x[..., 5:]
            scores = obj_conf * cls_conf
            
            # 2. Find best class and max score
            max_scores = np.max(scores, axis=-1)
            class_ids = np.argmax(scores, axis=-1)
            
            # 3. Filter by conf_thres
            mask = max_scores > self.conf_thres
            x_filtered = x[mask]
            scores_filtered = max_scores[mask]
            class_ids_filtered = class_ids[mask]
            
            if not x_filtered.shape[0]:
                continue
            
            # 4. Coordinate conversion
            boxes = x_filtered[:, :4]
            if self.coordinate_input_format == "xywh":
                boxes = xywh2xyxy(boxes)
                
            # 5. Batched NMS using torchvision (Tensor Bridge pattern)
            boxes_t = torch.from_numpy(np.ascontiguousarray(boxes)).float()
            scores_t = torch.from_numpy(np.ascontiguousarray(scores_filtered)).float()
            class_ids_t = torch.from_numpy(np.ascontiguousarray(class_ids_filtered)).float()
            
            max_wh = 4096  # classes are disjoint so add offset
            c = class_ids_t * max_wh
            boxes_t_shifted = boxes_t + c[:, None]
            
            keep = torchvision.ops.nms(boxes_t_shifted, scores_t, self.iou_thres)
            if keep.shape[0] > self.max_det:
                keep = keep[:self.max_det]
                
            keep_np = keep.numpy()
            
            # 6. Format to (N, 6) tensor
            out_boxes = boxes[keep_np]
            out_scores = scores_filtered[keep_np][:, None]
            out_cls = class_ids_filtered[keep_np][:, None].astype(float)
            
            output[xi] = np.concatenate((out_boxes, out_scores, out_cls), axis=1)
            
        return output
