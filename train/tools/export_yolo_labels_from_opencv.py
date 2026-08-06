"""
Generate YOLO pre-labels from the existing OpenCV monster templates.

Example:
    python train/tools/export_yolo_labels_from_opencv.py \
        --input-dir train/datasets/monster_yolo/images/raw \
        --output-label-dir train/datasets/monster_yolo/labels/raw \
        --map pig_shores \
        --mode contour_only \
        --preview-dir train/datasets/monster_yolo/previews/raw

The generated labels use a single class:
    0 monster
"""

import argparse
import glob
import os
import sys
from pathlib import Path

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.common import get_iou, get_mask, load_image, load_yaml, override_cfg


IMAGE_EXTS = {".bmp", ".jpg", ".jpeg", ".png", ".webp"}


def load_config(config_path, override_config_path):
    cfg = load_yaml(str(config_path))
    if override_config_path:
        cfg = override_cfg(cfg, load_yaml(str(override_config_path)))
    return cfg


def load_monster_templates(map_name):
    data = load_yaml(str(REPO_ROOT / "config" / "config_data.yaml"))
    if map_name not in data["map_mobs_mapping"]:
        raise ValueError(f"Unknown map: {map_name}")

    monsters_info = {}
    for monster_name in data["map_mobs_mapping"][map_name]:
        imgs = []
        pattern = REPO_ROOT / "monster" / monster_name / f"{monster_name}*.png"
        for file_path in sorted(glob.glob(str(pattern))):
            img = load_image(file_path)
            imgs.append((img, get_mask(img, (0, 255, 0))))

            img_flip = cv2.flip(img, 1)
            imgs.append((img_flip, get_mask(img_flip, (0, 255, 0))))

        if imgs:
            monsters_info[monster_name] = imgs

    if not monsters_info:
        raise RuntimeError(f"No monster templates loaded for map: {map_name}")

    return monsters_info


def prepare_templates(monsters_info, mode, blur):
    prepared = []
    for monster_name, monster_imgs in monsters_info.items():
        for img_monster, mask_monster in monster_imgs:
            item = {
                "name": monster_name,
                "img": img_monster,
                "mask": mask_monster,
                "height": img_monster.shape[0],
                "width": img_monster.shape[1],
            }
            if mode == "contour_only":
                mask_pattern = np.all(img_monster == [0, 0, 0], axis=2).astype(np.uint8) * 255
                item["contour_blur"] = cv2.GaussianBlur(mask_pattern, (blur, blur), 0)
            elif mode == "grayscale":
                item["gray"] = cv2.cvtColor(img_monster, cv2.COLOR_BGR2GRAY)
            prepared.append(item)
    return prepared


def parse_roi(value):
    if not value:
        return None
    parts = [int(v.strip()) for v in value.split(",")]
    if len(parts) != 4:
        raise ValueError("--roi must be x0,y0,x1,y1")
    x0, y0, x1, y1 = parts
    if x1 <= x0 or y1 <= y0:
        raise ValueError("--roi requires x1 > x0 and y1 > y0")
    return x0, y0, x1, y1


def clamp_roi(roi, image_shape):
    h, w = image_shape[:2]
    if roi is None:
        return 0, 0, w, h

    x0, y0, x1, y1 = roi
    return max(0, x0), max(0, y0), min(w, x1), min(h, y1)


def detect_template_free(img_roi, offset, min_area):
    x0, y0 = offset
    black_mask = np.all(img_roi == [0, 0, 0], axis=2).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))
    closed_mask = cv2.morphologyEx(black_mask, cv2.MORPH_CLOSE, kernel)
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(closed_mask, connectivity=8)

    monsters = []
    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        if area > min_area:
            monsters.append({
                "name": "template_free",
                "position": (x0 + x, y0 + y),
                "size": (h, w),
                "score": 1.0,
            })
    return monsters


def nms_height_width(monsters, iou_threshold):
    boxes = []
    for monster in monsters:
        x, y = monster["position"]
        h, w = monster["size"]
        boxes.append([x, y, x + w, y + h, monster["score"], monster])

    boxes.sort(key=lambda item: item[4], reverse=True)

    keep = []
    while boxes:
        best = boxes.pop(0)
        keep.append(best[5])
        boxes = [box for box in boxes if get_iou(best, box) < iou_threshold]

    return keep


def detect_with_templates(img_roi, offset, templates, mode, diff_thres, blur, max_candidates_per_template):
    x0, y0 = offset
    monsters = []

    roi_cache = {}
    if mode == "contour_only":
        mask_roi = np.all(img_roi == [0, 0, 0], axis=2).astype(np.uint8) * 255
        roi_cache["contour_blur"] = cv2.GaussianBlur(mask_roi, (blur, blur), 0)
    elif mode == "grayscale":
        roi_cache["gray"] = cv2.cvtColor(img_roi, cv2.COLOR_BGR2GRAY)

    for template in templates:
        h_temp = template["height"]
        w_temp = template["width"]

        if h_temp > img_roi.shape[0] or w_temp > img_roi.shape[1]:
            continue

        if mode == "contour_only":
            res = cv2.matchTemplate(
                roi_cache["contour_blur"],
                template["contour_blur"],
                cv2.TM_SQDIFF_NORMED,
            )
        elif mode == "grayscale":
            res = cv2.matchTemplate(
                roi_cache["gray"],
                template["gray"],
                cv2.TM_SQDIFF_NORMED,
                mask=template["mask"],
            )
        elif mode == "color":
            res = cv2.matchTemplate(
                img_roi,
                template["img"],
                cv2.TM_SQDIFF_NORMED,
                mask=template["mask"],
            )
        else:
            raise ValueError(f"Unsupported mode: {mode}")

        ys, xs = np.where(res <= diff_thres)
        if len(xs) > max_candidates_per_template:
            candidate_scores = res[ys, xs]
            best_idx = np.argpartition(candidate_scores, max_candidates_per_template - 1)[
                :max_candidates_per_template
            ]
            xs = xs[best_idx]
            ys = ys[best_idx]

        for x, y in zip(xs, ys):
            monsters.append({
                "name": template["name"],
                "position": (int(x0 + x), int(y0 + y)),
                "size": (h_temp, w_temp),
                "score": float(1.0 - res[y, x]),
            })

    return monsters


def add_hp_bar_detections(img_roi, offset, hp_bar_color, box_height, enabled):
    if not enabled:
        return []

    x0, y0 = offset
    mask = cv2.inRange(img_roi, np.array(hp_bar_color), np.array(hp_bar_color))
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    monsters = []
    for i in range(1, num_labels):
        x, y, _, _, area = stats[i]
        if area < 3:
            continue

        monsters.append({
            "name": "Health Bar",
            "position": (x0 + max(0, x), y0 + max(0, y + 10)),
            "size": (box_height, 70),
            "score": 1.0,
        })

    return monsters


def monster_to_yolo_line(monster, image_width, image_height, class_id):
    x, y = monster["position"]
    h, w = monster["size"]

    x = max(0, min(image_width - 1, x))
    y = max(0, min(image_height - 1, y))
    w = max(1, min(image_width - x, w))
    h = max(1, min(image_height - y, h))

    x_center = (x + w / 2) / image_width
    y_center = (y + h / 2) / image_height
    box_width = w / image_width
    box_height = h / image_height

    return f"{class_id} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}"


def draw_preview(img, monsters):
    preview = img.copy()
    for monster in monsters:
        x, y = monster["position"]
        h, w = monster["size"]
        cv2.rectangle(preview, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(
            preview,
            f"{monster['name']} {monster['score']:.2f}",
            (x, max(15, y - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )
    return preview


def list_images(input_dir):
    paths = []
    for path in Path(input_dir).iterdir():
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
            paths.append(path)
    return sorted(paths)


def main():
    parser = argparse.ArgumentParser(
        description="Generate YOLO pre-labels using the project's OpenCV monster detection."
    )
    parser.add_argument("--input-dir", required=True, help="Directory containing raw screenshots.")
    parser.add_argument("--output-label-dir", required=True, help="Directory for generated YOLO .txt labels.")
    parser.add_argument("--map", required=True, dest="map_name", help="Map name from config/config_data.yaml.")
    parser.add_argument(
        "--mode",
        default=None,
        choices=["contour_only", "grayscale", "color", "template_free"],
        help="OpenCV monster detection mode. Defaults to config value.",
    )
    parser.add_argument("--config", default="config/config_default.yaml", help="Base config path.")
    parser.add_argument("--override-config", default=None, help="Optional config override path.")
    parser.add_argument("--roi", default=None, help="Optional ROI as x0,y0,x1,y1. Defaults to full image.")
    parser.add_argument("--class-id", type=int, default=0, help="YOLO class id to write.")
    parser.add_argument("--diff-thres", type=float, default=None, help="Override monster_detect.diff_thres.")
    parser.add_argument("--nms-iou", type=float, default=0.4, help="NMS IoU threshold.")
    parser.add_argument(
        "--max-candidates-per-template",
        type=int,
        default=50,
        help="Limit raw template hits before NMS to avoid huge noisy label files.",
    )
    parser.add_argument("--preview-dir", default=None, help="Optional directory for images with detected boxes.")
    parser.add_argument(
        "--disable-hp-bar",
        action="store_true",
        help="Do not add existing enemy HP bar detections to the pre-labels.",
    )
    args = parser.parse_args()

    cfg = load_config(REPO_ROOT / args.config, REPO_ROOT / args.override_config if args.override_config else None)
    mode = args.mode or cfg["monster_detect"]["mode"]
    diff_thres = args.diff_thres
    if diff_thres is None:
        diff_thres = cfg["monster_detect"]["diff_thres"]

    roi = parse_roi(args.roi)
    input_paths = list_images(args.input_dir)
    if not input_paths:
        raise RuntimeError(f"No input images found in {args.input_dir}")

    output_label_dir = Path(args.output_label_dir)
    output_label_dir.mkdir(parents=True, exist_ok=True)

    preview_dir = Path(args.preview_dir) if args.preview_dir else None
    if preview_dir:
        preview_dir.mkdir(parents=True, exist_ok=True)

    monsters_info = load_monster_templates(args.map_name)
    min_template_height = min(img.shape[0] for imgs in monsters_info.values() for img, _ in imgs)
    templates = prepare_templates(
        monsters_info,
        mode,
        cfg["monster_detect"]["contour_blur"],
    )

    for idx, image_path in enumerate(input_paths, start=1):
        img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if img is None:
            print(f"Skip unreadable image: {image_path}")
            continue

        x0, y0, x1, y1 = clamp_roi(roi, img.shape)
        img_roi = img[y0:y1, x0:x1]

        if mode == "template_free":
            monsters = detect_template_free(img_roi, (x0, y0), min_area=1000)
        else:
            monsters = detect_with_templates(
                img_roi,
                (x0, y0),
                templates,
                mode,
                diff_thres,
                cfg["monster_detect"]["contour_blur"],
                args.max_candidates_per_template,
            )

        monsters = nms_height_width(monsters, iou_threshold=args.nms_iou)
        monsters.extend(add_hp_bar_detections(
            img_roi,
            (x0, y0),
            cfg["monster_detect"]["hp_bar_color"],
            min_template_height,
            cfg["monster_detect"]["with_enemy_hp_bar"] and not args.disable_hp_bar,
        ))
        monsters = nms_height_width(monsters, iou_threshold=args.nms_iou)

        label_path = output_label_dir / f"{image_path.stem}.txt"
        lines = [
            monster_to_yolo_line(monster, img.shape[1], img.shape[0], args.class_id)
            for monster in monsters
        ]
        label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

        if preview_dir:
            preview = draw_preview(img, monsters)
            cv2.imwrite(str(preview_dir / image_path.name), preview)

        print(f"[{idx}/{len(input_paths)}] {image_path.name}: {len(monsters)} labels")


if __name__ == "__main__":
    main()
