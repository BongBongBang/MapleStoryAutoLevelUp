import os
import statistics
import time
import unittest
from pathlib import Path

import cv2
import numpy as np

from src.engine.YoloMonsterDetector import YoloMonsterDetector
from src.utils.common import load_image, load_yaml
from tests.test_yolo import build_yolo_cfg, find_default_model


PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMAGE_EXTS = {".bmp", ".jpg", ".jpeg", ".png", ".webp"}


def list_train_images():
    image_dir = PROJECT_ROOT / "train" / "datasets" / "monster_yolo" / "images" / "train"
    if not image_dir.exists():
        return []

    paths = [
        path for path in sorted(image_dir.iterdir())
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS
    ]
    limit = int(os.environ.get("DETECT_PERF_LIMIT", "50"))
    return paths[:limit]


def load_images(paths):
    images = []
    for path in paths:
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is not None:
            images.append((path, img))
    return images


def list_opencv_templates():
    template_dir = Path(os.environ.get(
        "OPENCV_TEMPLATE_DIR",
        PROJECT_ROOT / "monster" / "pig",
    ))
    template_pattern = os.environ.get("OPENCV_TEMPLATE_PATTERN", "pig_*.png")
    templates = sorted(template_dir.glob(template_pattern))
    return [path for path in templates if path.is_file()]


def percentile(values, percent):
    if not values:
        return 0.0
    sorted_values = sorted(values)
    idx = round((len(sorted_values) - 1) * percent / 100)
    return sorted_values[idx]


def print_perf_report(name, elapsed_ms, detections=None):
    avg_ms = statistics.mean(elapsed_ms)
    p50_ms = percentile(elapsed_ms, 50)
    p95_ms = percentile(elapsed_ms, 95)
    fps = 1000.0 / avg_ms if avg_ms > 0 else 0.0

    print(f"\n{name}")
    print(f"  samples: {len(elapsed_ms)}")
    print(f"  avg:     {avg_ms:.2f} ms")
    print(f"  p50:     {p50_ms:.2f} ms")
    print(f"  p95:     {p95_ms:.2f} ms")
    print(f"  fps:     {fps:.2f}")
    if detections is not None:
        print(f"  detections avg: {statistics.mean(detections):.2f}")


class DetectPerformanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        image_paths = list_train_images()
        if not image_paths:
            raise unittest.SkipTest(
                "No train images found under train/datasets/monster_yolo/images/train"
            )

        cls.images = load_images(image_paths)
        if not cls.images:
            raise unittest.SkipTest("No readable train images found.")

    def test_opencv_template_match_performance(self):
        cfg_path = PROJECT_ROOT / "config" / "config_default.yaml"
        cfg_yaml = load_yaml(cfg_path)
        blur = cfg_yaml["monster_detect"]["contour_blur"]
        diff_thres = cfg_yaml["monster_detect"]["diff_thres"]

        template_paths = list_opencv_templates()
        if not template_paths:
            self.skipTest("No OpenCV templates found. Set OPENCV_TEMPLATE_DIR or OPENCV_TEMPLATE_PATTERN.")

        templates = []
        for template_path in template_paths:
            cv_img = load_image(template_path)
            mask_pattern = np.all(cv_img == [0, 0, 0], axis=2).astype(np.uint8) * 255
            img_monster_blur = cv2.GaussianBlur(mask_pattern, (blur, blur), 0)
            templates.append((template_path, img_monster_blur))

        elapsed_ms = []
        detection_counts = []

        for _, img in self.images:
            start = time.perf_counter()
            img_frame_binary = np.all(img == [0, 0, 0], axis=2).astype(np.uint8) * 255
            img_frame_blur = cv2.GaussianBlur(img_frame_binary, (blur, blur), 0)
            num_matches = 0
            for _, img_monster_blur in templates:
                if (
                    img_monster_blur.shape[0] > img_frame_blur.shape[0]
                    or img_monster_blur.shape[1] > img_frame_blur.shape[1]
                ):
                    continue
                match_res = cv2.matchTemplate(
                    img_frame_blur,
                    img_monster_blur,
                    cv2.TM_SQDIFF_NORMED,
                )
                match_locations = np.where(match_res <= diff_thres)
                num_matches += len(match_locations[0])
            elapsed_ms.append((time.perf_counter() - start) * 1000)
            detection_counts.append(num_matches)

        print_perf_report("test_opencv template match performance", elapsed_ms, detection_counts)
        print(f"  template count: {len(templates)}")
        print(f"  template dir: {template_paths[0].parent}")
        print(f"  template pattern: {os.environ.get('OPENCV_TEMPLATE_PATTERN', 'pig_*.png')}")
        print(f"  blur: {blur}")
        print(f"  diff_thres: {diff_thres}")
        self.assertEqual(len(elapsed_ms), len(self.images))

    def test_yolo_monster_detect_performance(self):
        model_path = Path(os.environ.get("YOLO_MODEL", find_default_model()))
        if not model_path.exists():
            self.skipTest(f"YOLO model not found: {model_path}")

        detector = YoloMonsterDetector(build_yolo_cfg(model_path))

        warmup_count = min(3, len(self.images))
        for _, img in self.images[:warmup_count]:
            detector.detect(img, offset=(0, 0))

        elapsed_ms = []
        detection_counts = []
        for _, img in self.images:
            start = time.perf_counter()
            monsters = detector.detect(img, offset=(0, 0))
            elapsed_ms.append((time.perf_counter() - start) * 1000)
            detection_counts.append(len(monsters))

        print_perf_report("YOLO monster detect performance", elapsed_ms, detection_counts)
        print(f"  model: {model_path}")
        print(f"  device: {os.environ.get('YOLO_DEVICE', 'cpu')}")
        self.assertEqual(len(elapsed_ms), len(self.images))


if __name__ == "__main__":
    unittest.main()
