"""
YOLO-based monster detector.

This module is optional at runtime. The rest of the bot can run without the
ultralytics package unless monster_detect.mode is set to "yolo".
"""

import os

from src.utils.logger import logger


class YoloMonsterDetector:
    """
    Detect monsters in a cropped ROI and return the bot's existing monster dicts.
    """
    def __init__(self, cfg):
        yolo_cfg = cfg["monster_detect"]["yolo"]
        self.model_path = yolo_cfg["model_path"]
        self.conf_thres = yolo_cfg["conf_thres"]
        self.iou_thres = yolo_cfg["iou_thres"]
        self.imgsz = yolo_cfg["imgsz"]
        self.device = yolo_cfg["device"]
        self.detect_interval = max(1, yolo_cfg["detect_interval"])
        self.max_detections = yolo_cfg["max_detections"]
        self.frame_idx = 0
        self.last_monsters = []

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"YOLO model not found: {self.model_path}. "
                "Train/export a model first, or switch monster_detect.mode back to contour_only."
            )

        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError(
                "monster_detect.mode is 'yolo', but ultralytics is not installed. "
                "Install it with: pip install ultralytics"
            ) from exc

        self.model = YOLO(self.model_path)
        logger.info(f"[YoloMonsterDetector] Loaded model: {self.model_path}")

    def detect(self, img_roi, offset):
        """
        Detect monsters in img_roi.

        Args:
            img_roi: BGR ROI from the current game frame.
            offset: (x0, y0) ROI top-left in full-frame coordinates.
        """
        self.frame_idx += 1
        if (self.frame_idx - 1) % self.detect_interval != 0:
            return self.last_monsters

        x0, y0 = offset
        results = self.model.predict(
            source=img_roi,
            imgsz=self.imgsz,
            conf=self.conf_thres,
            iou=self.iou_thres,
            device=self.device,
            verbose=False,
            max_det=self.max_detections,
        )

        monsters = []
        if not results:
            self.last_monsters = monsters
            return monsters

        result = results[0]
        if result.boxes is None:
            self.last_monsters = monsters
            return monsters

        for box in result.boxes:
            xyxy = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            x1, y1, x2, y2 = [int(round(v)) for v in xyxy]
            w = max(1, x2 - x1)
            h = max(1, y2 - y1)

            monsters.append({
                "name": "YOLO",
                "position": (x0 + x1, y0 + y1),
                "size": (h, w),
                "score": conf,
            })

        self.last_monsters = monsters
        return monsters
