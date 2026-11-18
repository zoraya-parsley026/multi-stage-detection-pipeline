from __future__ import annotations

import glob
from pathlib import Path

from ...parsers.annotation_parser_factory import AnnotationParserFactory
from ...parsers.base_annotation_parser import BaseAnnotationParser
from ...parsers.strategies import (
    YOLOLabelParser as YOLOLabelParser,  # register built-in parsers
)
from ..base_dataset import BaseDataset
from ..dataset_factory import DatasetFactory


@DatasetFactory.register("detection")
class DetectionDataset(BaseDataset):
    """Map-style image dataset with optional YOLO annotations."""

    VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

    def __init__(self, images_path: str | Path, labels_path: str | Path | None = None,
                 parser_type: str = "yolo", parser: BaseAnnotationParser | None = None) -> None:
        images_str = str(Path(images_path).absolute())
        self.labels_dir = Path(labels_path).absolute() if labels_path else None
        self.parser = parser or AnnotationParserFactory.create(parser_type)
        if "*" in images_str:
            raw_files = sorted(glob.glob(images_str, recursive=True))
            self.base_dir = Path(images_str.split("*")[0]).resolve()
        elif Path(images_str).is_dir():
            self.base_dir = Path(images_str).resolve()
            raw_files = sorted(p for p in self.base_dir.iterdir() if p.is_file())
        elif Path(images_str).is_file():
            self.base_dir = Path(images_str).resolve().parent
            raw_files = [images_str]
        else:
            raise FileNotFoundError(f"Path not found: {images_path}")
        self.image_files = [Path(f) for f in raw_files if Path(f).suffix.lower() in self.VALID_EXTENSIONS]

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx: int) -> dict:
        import cv2

        if idx < 0 or idx >= len(self.image_files):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self.image_files)}")
        img_path = self.image_files[idx]
        img = cv2.imread(str(img_path))
        if img is None:
            raise RuntimeError(f"Failed to read image at {img_path}")
        height, width = img.shape[:2]
        file_id = img_path.stem
        ground_truth = None
        if self.labels_dir:
            try:
                label_file = self.labels_dir / img_path.relative_to(self.base_dir).with_suffix(".txt")
            except ValueError:
                label_file = self.labels_dir / f"{file_id}.txt"
            ground_truth = self.parser.parse(label_file, (height, width))
        return {"data": img, "metadata": {"file_id": file_id,
                "original_shape": (height, width), "source_path": str(img_path)},
                "ground_truth": ground_truth}
