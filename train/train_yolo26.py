from __future__ import annotations

"""
Train the latest Ultralytics YOLO model (YOLO26 as of Jan 2026) on the SAME
Roboflow-style COCO dataset layout you used for DETR:

  data/
    train/ _annotations.coco.json + images...
    valid/ _annotations.coco.json + images...
    test/  _annotations.coco.json + images...

This script:
1) Reads your COCO JSON via your existing COCODataset class (for class names + id mapping).
2) Converts COCO annotations -> YOLO txt labels (cls cx cy w h, normalized 0..1).
3) Builds a YOLO-style dataset folder (images/ + labels/) using hardlinks when possible (fast).
4) Trains Ultralytics YOLO on it.

Install:
  pip install ultralytics

Run example:
  python train/train_yolo26.py --data-root data --img 640 --batch 16 --epochs 50 --num_workers 4

Notes:
- Uses YOLO26 by default: yolo26n.pt (nano). You can pass yolo26s.pt / yolo26m.pt / yolo26l.pt etc.
- If you change classes/order, delete the generated yolo dataset folder or pass --overwrite.
"""

import argparse
import os
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from coco_dataset import COCODataset  # your existing dataset class


def _resolve_image_path(data_root: Path, split_dir: Path, file_name: str) -> Path:
    """Match the logic in your COCODataset.__getitem__ for finding an image path."""
    p = split_dir / file_name
    if p.exists():
        return p
    # Some COCO exports include subfolders in file_name, or paths relative to root.
    p2 = data_root / file_name
    if p2.exists():
        return p2
    raise FileNotFoundError(f"Missing image for file_name='{file_name}'. Tried: {p} and {p2}")


def _safe_link_or_copy(src: Path, dst: Path) -> None:
    """
    Try to hardlink (fast + no extra disk usage). If that fails (different drive / permissions),
    fall back to copy.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        os.link(src, dst)  # hardlink
    except Exception:
        shutil.copy2(src, dst)


def coco_to_yolo_line(
    x1: float, y1: float, x2: float, y2: float, cls: int, W: int, H: int
) -> Optional[str]:
    """
    Convert XYXY (pixels) -> YOLO (cls cx cy w h), normalized.

    Returns None if box invalid.
    """
    ww = x2 - x1
    hh = y2 - y1
    if ww <= 0 or hh <= 0 or W <= 0 or H <= 0:
        return None
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    # Normalize
    cxn = cx / float(W)
    cyn = cy / float(H)
    wwn = ww / float(W)
    hhn = hh / float(H)
    # Clamp (just in case)
    cxn = min(max(cxn, 0.0), 1.0)
    cyn = min(max(cyn, 0.0), 1.0)
    wwn = min(max(wwn, 0.0), 1.0)
    hhn = min(max(hhn, 0.0), 1.0)
    return f"{cls} {cxn:.6f} {cyn:.6f} {wwn:.6f} {hhn:.6f}"


def export_split_to_yolo(
    ds: COCODataset,
    data_root: Path,
    split: str,
    yolo_root: Path,
    overwrite: bool = False,
) -> Tuple[int, int]:
    """
    Export one split (train/valid/test) into:
      yolo_root/images/<split>/...
      yolo_root/labels/<split>/...

    Returns: (num_images_exported, num_labels_written)
    """
    split_dir = data_root / split

    img_out_dir = yolo_root / "images" / split
    lbl_out_dir = yolo_root / "labels" / split
    if overwrite:
        shutil.rmtree(img_out_dir, ignore_errors=True)
        shutil.rmtree(lbl_out_dir, ignore_errors=True)

    img_out_dir.mkdir(parents=True, exist_ok=True)
    lbl_out_dir.mkdir(parents=True, exist_ok=True)

    num_imgs = 0
    num_lbls = 0

    # We avoid calling ds[i] to prevent opening images. Instead we use ds.images + ds.anns_by_image.
    # Those fields are created in your COCODataset __init__.
    for im in ds.images:
        image_id = int(im["id"])
        file_name = im["file_name"]
        W = int(im.get("width", 0))
        H = int(im.get("height", 0))

        src_img = _resolve_image_path(data_root, split_dir, file_name)

        # Make sure we preserve extension. For labels, YOLO uses same stem with .txt
        rel_name = Path(file_name).name  # flatten subfolders into filename; avoids nested dirs in yolo output
        dst_img = img_out_dir / rel_name
        _safe_link_or_copy(src_img, dst_img)

        # Build YOLO label lines
        lines: List[str] = []
        for a in ds.anns_by_image.get(image_id, []):
            orig_cat = int(a["category_id"])
            if orig_cat not in ds.id_to_contig:
                continue
            cls = int(ds.id_to_contig[orig_cat])

            x, y, w, h = a["bbox"]
            x = float(x); y = float(y); w = float(w); h = float(h)

            # Clip to image bounds (same idea as your dataset)
            x1 = max(0.0, x)
            y1 = max(0.0, y)
            x2 = min(float(W), x + w) if W > 0 else (x + w)
            y2 = min(float(H), y + h) if H > 0 else (y + h)

            # If width/height missing in JSON, you *can* still export but normalization will be wrong.
            # In that case, we raise so you fix the dataset export.
            if W <= 0 or H <= 0:
                raise ValueError(
                    f"Image width/height missing in COCO JSON for image_id={image_id}, file={file_name}. "
                    f"COCO needs width/height for YOLO normalization."
                )

            yolo = coco_to_yolo_line(x1, y1, x2, y2, cls, W, H)
            if yolo is not None:
                lines.append(yolo)

        # Write label file (even if empty -> no objects)
        dst_lbl = lbl_out_dir / (Path(rel_name).stem + ".txt")
        with open(dst_lbl, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + ("\n" if lines else ""))

        num_imgs += 1
        num_lbls += len(lines)

    return num_imgs, num_lbls


def write_dataset_yaml(yolo_root: Path, names: List[str], train_split: str, val_split: str, test_split: str) -> Path:
    """
    Create Ultralytics dataset.yaml.
    """
    yaml_path = yolo_root / "dataset.yaml"
    # Ultralytics expects:
    # path: <root>
    # train/val/test: relative to path
    # nc: number of classes
    # names: list
    content = [
        f"path: {yolo_root.as_posix()}",
        f"train: images/{train_split}",
        f"val: images/{val_split}",
        f"test: images/{test_split}",
        f"nc: {len(names)}",
        "names:",
    ]
    for n in names:
        # Quote strings safely
        nn = n.replace('"', '\\"')
        content.append(f'  - "{nn}"')
    yaml_path.write_text("\n".join(content) + "\n", encoding="utf-8")
    return yaml_path


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--data-root", default="data", help="Folder containing train/valid/test")
    ap.add_argument("--train-split", default="train")
    ap.add_argument("--val-split", default="valid")
    ap.add_argument("--test-split", default="test")

    # Latest Ultralytics YOLO model (as of Jan 2026): YOLO26
    ap.add_argument("--model", default="yolo26n.pt", help="Pretrained YOLO weights, e.g. yolo26n.pt, yolo26s.pt, ...")

    ap.add_argument("--outdir", default="runs_yolo26", help="Ultralytics run output directory (project)")
    ap.add_argument("--name", default="exp", help="Run name inside outdir")

    ap.add_argument("--img", type=int, default=640, help="imgsz")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--num_workers", type=int, default=4)

    ap.add_argument("--device", default=None, help="e.g. 0 or 'cpu'. Default: auto")
    ap.add_argument("--resume", action="store_true", help="Resume Ultralytics run from last.pt (outdir/name)")

    ap.add_argument("--yolo-ds", default="yolo_ds", help="Folder to create YOLO-format dataset (labels+images)")
    ap.add_argument("--overwrite", action="store_true", help="Rebuild YOLO dataset folder from scratch")

    args = ap.parse_args()

    data_root = Path(args.data_root)
    yolo_root = Path(args.yolo_ds)

    # 1) Load datasets (class mapping must be consistent across splits)
    train_ds = COCODataset(str(data_root), split=args.train_split)
    classes = train_ds.classes
    id_to_contig = train_ds.id_to_contig

    val_ds = COCODataset(str(data_root), split=args.val_split, classes=classes, id_to_contig=id_to_contig)
    test_ds = COCODataset(str(data_root), split=args.test_split, classes=classes, id_to_contig=id_to_contig)

    print("Classes:", classes)
    print("YOLO dataset root:", yolo_root.resolve())

    # 2) Export YOLO dataset
    yaml_path = yolo_root / "dataset.yaml"
    if args.overwrite or not yaml_path.exists():
        print("Exporting COCO -> YOLO labels...")
        yolo_root.mkdir(parents=True, exist_ok=True)

        n_img, n_lbl = export_split_to_yolo(train_ds, data_root, args.train_split, yolo_root, overwrite=args.overwrite)
        print(f"  train: images={n_img}, labels={n_lbl}")

        n_img, n_lbl = export_split_to_yolo(val_ds, data_root, args.val_split, yolo_root, overwrite=args.overwrite)
        print(f"  val:   images={n_img}, labels={n_lbl}")

        n_img, n_lbl = export_split_to_yolo(test_ds, data_root, args.test_split, yolo_root, overwrite=args.overwrite)
        print(f"  test:  images={n_img}, labels={n_lbl}")

        yaml_path = write_dataset_yaml(yolo_root, classes, args.train_split, args.val_split, args.test_split)
        print("Wrote:", yaml_path)
    else:
        print("Found existing YOLO dataset.yaml, skipping export:", yaml_path)

    # 3) Train with Ultralytics
    try:
        from ultralytics import YOLO
    except Exception as e:
        raise RuntimeError(
            "Ultralytics is not installed. Install with: pip install ultralytics"
        ) from e

    # If resuming, Ultralytics expects the *last.pt* weights path or resume=True with model pointing to last.pt.
    # We'll auto-locate last.pt if it exists.
    model_path = args.model
    if args.resume:
        last_pt = Path(args.outdir) / args.name / "weights" / "last.pt"
        # Ultralytics sometimes uses: outdir/name/weights/last.pt (project/name)
        # If not found, also try runs folder pattern: outdir/detect/name/weights/last.pt
        if not last_pt.exists():
            alt = Path(args.outdir) / "detect" / args.name / "weights" / "last.pt"
            if alt.exists():
                last_pt = alt
        if last_pt.exists():
            print("Resuming from:", last_pt)
            model_path = str(last_pt)
        else:
            print("⚠️  --resume set but last.pt not found, starting from:", args.model)

    model = YOLO(model_path)

    train_kwargs = dict(
        data=str(yaml_path),
        epochs=int(args.epochs),
        imgsz=int(args.img),
        batch=int(args.batch),
        workers=int(args.num_workers),
        project=str(args.outdir),
        name=str(args.name),
        exist_ok=True,   # don't error if the run folder already exists
        resume=bool(args.resume),
    )
    if args.device is not None:
        train_kwargs["device"] = args.device

    print("Training with args:", train_kwargs)
    model.train(**train_kwargs)

    # 4) Validate (prints COCO metrics including mAP50)
    print("Validating best model...")
    metrics = model.val(data=str(yaml_path), imgsz=int(args.img), batch=int(args.batch), workers=int(args.num_workers))
    # Ultralytics prints nicely; metrics object also contains values.
    print("Done.")


if __name__ == "__main__":
    main()
