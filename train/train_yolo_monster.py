"""
Train a single-class YOLO monster detector.

Expected dataset layout:
    train/datasets/monster_yolo/
      images/
        train/
        val/
      labels/
        train/
        val/
      data.yaml

Example:
    python3 train/train_yolo_monster.py \
        --dataset train/datasets/monster_yolo \
        --model yolo11n.pt \
        --imgsz 416 \
        --epochs 80 \
        --device cuda \
        --export onnx
"""

import argparse
import shutil
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
IMAGE_EXTS = {".bmp", ".jpg", ".jpeg", ".png", ".webp"}


def count_files(path, suffixes=None):
    if not path.exists():
        return 0
    if suffixes is None:
        return sum(1 for item in path.iterdir() if item.is_file())
    return sum(1 for item in path.iterdir() if item.is_file() and item.suffix.lower() in suffixes)


def ensure_data_yaml(dataset_dir, class_name):
    import yaml

    data_yaml = dataset_dir / "data.yaml"
    data = {
        "path": str(dataset_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": {
            0: class_name,
        },
    }

    if data_yaml.exists():
        with data_yaml.open("r", encoding="utf-8") as f:
            existing = yaml.safe_load(f) or {}
        existing.setdefault("path", data["path"])
        existing.setdefault("train", data["train"])
        existing.setdefault("val", data["val"])
        existing.setdefault("names", data["names"])
        data = existing

    with data_yaml.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)

    return data_yaml


def validate_dataset(dataset_dir):
    required_dirs = [
        dataset_dir / "images" / "train",
        dataset_dir / "images" / "val",
        dataset_dir / "labels" / "train",
        dataset_dir / "labels" / "val",
    ]
    missing = [path for path in required_dirs if not path.exists()]
    if missing:
        lines = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(f"Missing required dataset directories:\n{lines}")

    train_images = count_files(dataset_dir / "images" / "train", IMAGE_EXTS)
    val_images = count_files(dataset_dir / "images" / "val", IMAGE_EXTS)
    train_labels = count_files(dataset_dir / "labels" / "train", {".txt"})
    val_labels = count_files(dataset_dir / "labels" / "val", {".txt"})

    if train_images == 0:
        raise RuntimeError(f"No training images found: {dataset_dir / 'images' / 'train'}")
    if val_images == 0:
        raise RuntimeError(f"No validation images found: {dataset_dir / 'images' / 'val'}")

    print("Dataset summary:")
    print(f"  train images: {train_images}")
    print(f"  train labels: {train_labels}")
    print(f"  val images:   {val_images}")
    print(f"  val labels:   {val_labels}")

    if train_labels < train_images:
        print("Warning: fewer train label files than images. Empty scenes should still have empty .txt files.")
    if val_labels < val_images:
        print("Warning: fewer val label files than images. Empty scenes should still have empty .txt files.")


def copy_best_weight(run_dir, output_dir):
    best_pt = run_dir / "weights" / "best.pt"
    if not best_pt.exists():
        print(f"Warning: best.pt not found under {run_dir / 'weights'}")
        return None

    output_dir.mkdir(parents=True, exist_ok=True)
    dest = output_dir / "best.pt"
    shutil.copy2(best_pt, dest)
    print(f"Copied best model to: {dest}")
    return dest


def train(args):
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError(
            "ultralytics is required for training. Install it with: pip install ultralytics"
        ) from exc

    dataset_dir = (REPO_ROOT / args.dataset).resolve() if not Path(args.dataset).is_absolute() else Path(args.dataset)
    output_dir = (REPO_ROOT / args.output_dir).resolve() if not Path(args.output_dir).is_absolute() else Path(args.output_dir)

    validate_dataset(dataset_dir)
    data_yaml = ensure_data_yaml(dataset_dir, args.class_name)

    model = YOLO(args.model)
    results = model.train(
        data=str(data_yaml),
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        device=args.device,
        project=str(output_dir),
        name=args.name,
        patience=args.patience,
        workers=args.workers,
        exist_ok=args.exist_ok,
    )

    run_dir = Path(results.save_dir)
    best_model = copy_best_weight(run_dir, output_dir)

    if args.export:
        export_model = YOLO(str(best_model or (run_dir / "weights" / "best.pt")))
        exported = export_model.export(format=args.export, imgsz=args.imgsz, device=args.device)
        exported_path = Path(exported)
        if exported_path.exists():
            dest = output_dir / exported_path.name
            shutil.copy2(exported_path, dest)
            print(f"Copied exported model to: {dest}")
        else:
            print(f"Exported model: {exported}")

    print(f"Training run directory: {run_dir}")


def main():
    parser = argparse.ArgumentParser(description="Train the MapleStory monster YOLO detector.")
    parser.add_argument("--dataset", default="train/datasets/monster_yolo", help="Dataset root directory.")
    parser.add_argument("--output-dir", default="train/models/monster_yolo", help="Where runs and copied weights go.")
    parser.add_argument("--name", default="train", help="Ultralytics run name under output-dir.")
    parser.add_argument("--class-name", default="monster", help="Single YOLO class name.")
    parser.add_argument("--model", default="yolo11n.pt", help="Base YOLO model or checkpoint.")
    parser.add_argument("--imgsz", type=int, default=416, help="Training image size.")
    parser.add_argument("--epochs", type=int, default=80, help="Number of training epochs.")
    parser.add_argument("--batch", type=int, default=16, help="Batch size. Use -1 for auto batch.")
    parser.add_argument("--device", default="cpu", help='Training device, e.g. "cpu", "cuda", "0", or "mps".')
    parser.add_argument("--patience", type=int, default=20, help="Early-stopping patience.")
    parser.add_argument("--workers", type=int, default=4, help="Data loader workers.")
    parser.add_argument("--exist-ok", action="store_true", help="Allow reusing an existing run directory.")
    parser.add_argument(
        "--export",
        default="",
        choices=["", "onnx", "engine", "coreml"],
        help="Optionally export best.pt after training.",
    )
    args = parser.parse_args()

    try:
        train(args)
    except Exception as exc:
        print(f"Training failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
