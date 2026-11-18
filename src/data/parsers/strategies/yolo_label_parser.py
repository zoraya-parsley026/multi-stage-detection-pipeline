import logging
from pathlib import Path

from ..annotation_parser_factory import AnnotationParserFactory
from ..base_annotation_parser import BaseAnnotationParser

logger = logging.getLogger(__name__)


@AnnotationParserFactory.register("yolo")
class YOLOLabelParser(BaseAnnotationParser):
    def parse(self, label_path: str | Path, original_shape: tuple[int, int]) -> list[dict] | None:
        path = Path(label_path)
        if not path.exists():
            logger.warning("Label file not found: %s", path)
            return None
        height, width = original_shape
        annotations = []
        for line in path.read_text(encoding="utf-8").splitlines():
            parts = line.strip().split()
            if not parts:
                continue
            if len(parts) < 5:
                logger.warning("Skipping malformed line in %s: %r", path, line)
                continue
            try:
                class_id = int(parts[0])
                xc, yc, box_w, box_h = (float(parts[i]) for i in range(1, 5))
            except ValueError:
                logger.warning("Non-numeric values in %s: %r", path, line)
                continue
            x1 = max(0.0, min(float(width), (xc - box_w / 2) * width))
            y1 = max(0.0, min(float(height), (yc - box_h / 2) * height))
            x2 = max(0.0, min(float(width), (xc + box_w / 2) * width))
            y2 = max(0.0, min(float(height), (yc + box_h / 2) * height))
            if x1 < x2 and y1 < y2:
                annotations.append({"class_id": class_id, "localization": [x1, y1, x2, y2]})
        return annotations
