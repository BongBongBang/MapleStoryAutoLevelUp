import os
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from src.utils.common import (
    convert_lists_to_tuples,
    convert_tuples_to_lists,
    get_cfg_diff,
    get_iou,
    get_mask,
    get_player_location_on_minimap,
    load_yaml,
    nms,
    override_cfg,
    pad_to_size,
    to_opencv_hsv,
    to_standard_hsv,
)


class CommonConfigTests(unittest.TestCase):
  
    def test_get_player_location_on_minimap_with_local_debug_image(self):
        project_root = Path(__file__).resolve().parents[1]
        image_path = project_root / "minimaps" / "lost_time_1" / "map.png"
        minimap = cv2.imread(str(image_path), cv2.IMREAD_COLOR)

        self.assertIsNotNone(minimap, f"Failed to load test image: {image_path}")
        self.assertEqual(get_player_location_on_minimap(minimap), (32, 76))


if __name__ == "__main__":
    unittest.main()
