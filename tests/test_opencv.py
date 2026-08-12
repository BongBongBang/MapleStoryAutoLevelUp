import unittest
from pathlib import Path

import cv2
import numpy as np

from src.engine.HealthMonitor import HealthMonitor
from src.utils.common import load_image, load_yaml, get_bar_percent

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class OpenCVTest(unittest.TestCase):
    def test_numpy_all_mask_pattern(self):
        """
        把monster图二值化后, 进行高斯模糊 
        """
        img_path = PROJECT_ROOT / "monster" / "pig" / "pig_1.png"
        cv_img = load_image(img_path)
        mask_pattern = np.all(cv_img == [0, 0, 0], axis=2).astype(np.uint8) * 255
        cfg_path = PROJECT_ROOT / "config" / "config_default.yaml"
        cfg_yaml = load_yaml(cfg_path)
        blur = cfg_yaml["monster_detect"]["contour_blur"]
        img_monster_blur = cv2.GaussianBlur(mask_pattern, (blur, blur), 0)
        cv2.imshow("mask_template", img_monster_blur)

        img_frame_path = PROJECT_ROOT / "screenshot" / "2026-08-04_22-39-36_img_frame.png"
        img_frame = load_image(img_frame_path)
        img_frame_binary = np.all(img_frame == [0, 0, 0], axis=2).astype(np.uint8) * 255
        img_frame_blur = cv2.GaussianBlur(img_frame_binary, (blur, blur), 0)
        cv2.imshow("img_frame_blur", img_frame_blur)
        match_res = cv2.matchTemplate(img_frame_blur, img_monster_blur, cv2.TM_SQDIFF_NORMED)
        print(match_res)
        match_locations = np.where(match_res <= cfg_yaml["monster_detect"]["diff_thres"])

        for pt in zip(*match_locations[::-1]):
            print(pt)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def test_health_monitor2(self):
        cfg_path = PROJECT_ROOT / "config" / "config_default.yaml"
        cfg_yaml = load_yaml(cfg_path)
        health_monitor = HealthMonitor(cfg_yaml, None)

        img_path = PROJECT_ROOT / "test.png"
        img_frame = load_image(img_path)

        health_monitor.update_frame(img_frame)
        cv2.imshow("img_frame", img_frame)
        print(health_monitor.get_hp_mp_exp_percent())
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def test_hp_monitor(self):
        img_path = PROJECT_ROOT / "test.png"
        img_frame = load_image(img_path)
        img_frame_gray = cv2.cvtColor(img_frame, cv2.COLOR_BGR2GRAY)
        white_mask = cv2.inRange(img_frame_gray, 240, 255)
        cv2.imshow("white_mask", white_mask)
        contours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        loc_size_bars = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            # for game window resolution 752x1282, w/h == 7.5, w*h == 3630
            if 4 < w/h < 12 and 800 < w*h < 5000:
                loc_size_bars.append((x, y, w, h))
        # 复制原图，避免修改原始图像
        img_with_boxes = img_frame.copy()

        # 遍历每个尺寸条，绘制矩形框
        for i, (x, y, w, h) in enumerate(loc_size_bars):
            # 绘制绿色矩形框，线宽为2
            cv2.rectangle(img_with_boxes, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imshow("Detected Size Bars", img_with_boxes)

        # sort contours by x coordinate
        loc_size_bars = sorted(loc_size_bars, key=lambda bar: bar[0])

        bar_percents = []
        for x, y, w, h in loc_size_bars:
            bar_percents.append(get_bar_percent(img_frame[y:y+h, x:x+w]))
        print(bar_percents)
        cv2.waitKey(0)
        cv2.destroyAllWindows()



if __name__ == "__main__":
    unittest.main()
