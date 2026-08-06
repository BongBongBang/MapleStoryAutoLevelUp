import os
import time
import unittest
from pathlib import Path

import cv2

from src.engine.YoloMonsterDetector import YoloMonsterDetector

PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMAGE_EXTS = {".bmp", ".jpg", ".jpeg", ".png", ".webp"}


def find_default_model():
    candidates = [
        PROJECT_ROOT / "train" / "models" / "monster_yolo" / "best.pt",
        PROJECT_ROOT / "train" / "models" / "monster_yolo" / "train" / "weights" / "best.pt",
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]

def build_yolo_cfg(model_path):
    return {
        "monster_detect": {
            "yolo": {
                "model_path": str(model_path),
                "conf_thres": float(os.environ.get("YOLO_CONF", "0.25")),
                "iou_thres": float(os.environ.get("YOLO_IOU", "0.45")),
                "imgsz": int(os.environ.get("YOLO_IMGSZ", "416")),
                "device": os.environ.get("YOLO_DEVICE", "cpu"),
                "detect_interval": 1,
                "max_detections": int(os.environ.get("YOLO_MAX_DET", "20")),
            }
        }
    }


def draw_monsters(img, monsters):
    img_debug = img.copy()
    for monster in monsters:
        x, y = monster["position"]
        h, w = monster["size"]
        score = monster["score"]
        label = f"{monster['name']} {score:.2f}"

        cv2.rectangle(img_debug, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(
            img_debug,
            label,
            (x, max(18, y - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
    return img_debug

REPO_PATH = Path(__file__).resolve().parents[1]

class TestYOLO(unittest.TestCase):
    def test_yolo_inference(self):
        model_path = Path(find_default_model())
        if not model_path.exists():
            self.skipTest(f"YOLO model not found: {model_path}")

        print(f"{REPO_PATH}")

        image_path = REPO_PATH / "test.jpg"
        if image_path is None or not image_path.exists():
            self.skipTest("YOLO test image not found. Set YOLO_IMAGE or add images to train/datasets/monster_yolo/images/val")
        print(f"To test img: {image_path}")

        img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        self.assertIsNotNone(img, f"Failed to load image: {image_path}")

        detector = YoloMonsterDetector(build_yolo_cfg(model_path))
        start = time.perf_counter()
        monsters = detector.detect(img, offset=(0, 0))
        elapsed_ms = (time.perf_counter() - start) * 1000
        print(f"detect cost: {elapsed_ms:.2f} ms")

        output_dir = PROJECT_ROOT / "tests" / "output"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{image_path.stem}_yolo_detected.png"
        img_debug = draw_monsters(img, monsters)
        self.assertTrue(cv2.imwrite(str(output_path), img_debug), f"Failed to write output image: {output_path}")

        print(f"YOLO model: {model_path}")
        print(f"YOLO image: {image_path}")
        print(f"YOLO detections: {len(monsters)}")
        print(f"YOLO output: {output_path}")
        for monster in monsters:
            print(monster)

        cv2.imshow("YOLO Monster Detection", img_debug)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        self.assertIsInstance(monsters, list)


if __name__ == "__main__":
    unittest.main()
