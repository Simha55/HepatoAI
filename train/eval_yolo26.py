# train/eval_yolo26.py
from __future__ import annotations

"""
Evaluate an Ultralytics YOLO run trained using your train_yolo26.py pipeline.

What it does:
- Reuses your COCO->YOLO export logic (same class order / mapping via COCODataset).
- Finds the trained weights (best.pt / last.pt) from your --outdir/--name folder.
- Runs Ultralytics validation on the requested split (val/test/train).
- Prints: mAP@0.50, mAP@0.75, mAP@0.50:0.95, and Recall (where available).
- Optionally saves a few qualitative prediction images (Ultralytics saves inside the run folder).

Usage:
  python train/eval_yolo26.py --data-root data --split test --outdir runs_yolo26 --name exp --img 640 --batch 16

Notes:
- Ultralytics already computes COCO-style metrics (mAP50, mAP50-95, etc.) during val().
- This script ensures the YOLO dataset.yaml exists and is consistent with your COCO JSON mapping.
"""

import argparse
import os
import shutil
from pathlib import Path
from typing import List, Tuple, Optional

from coco_dataset import COCODataset


def _resolve_image_path(data_root: Path, split_dir: Path, file_name: str) -> Path:
    p = split_dir / file_name
    if p.exists():
        return p
    p2 = data_root / file_name
    if p2.exists():
        return p2
    raise FileNotFoundError(f"Missing image for file_name='{file_name}'. Tried: {p} and {p2}")


def _safe_link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except Exception:
        shutil.copy2(src, dst)


def coco_to_yolo_line(
    x1: float, y1: float, x2: float, y2: float, cls: int, W: int, H: int
) -> Optional[str]:
    ww = x2 - x1
    hh = y2 - y1
    if ww <= 0 or hh <= 0 or W <= 0 or H <= 0:
        return None
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0

    cxn = min(max(cx / float(W), 0.0), 1.0)
    cyn = min(max(cy / float(H), 0.0), 1.0)
    wwn = min(max(ww / float(W), 0.0), 1.0)
    hhn = min(max(hh / float(H), 0.0), 1.0)

    return f"{cls} {cxn:.6f} {cyn:.6f} {wwn:.6f} {hhn:.6f}"


def export_split_to_yolo(
    ds: COCODataset,
    data_root: Path,
    split: str,
    yolo_root: Path,
    overwrite: bool = False,
) -> Tuple[int, int]:
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

    for im in ds.images:
        image_id = int(im["id"])
        file_name = im["file_name"]
        W = int(im.get("width", 0))
        H = int(im.get("height", 0))
        if W <= 0 or H <= 0:
            raise ValueError(
                f"Missing width/height in COCO JSON for image_id={image_id}, file={file_name}. "
                f"YOLO export needs width/height."
            )

        src_img = _resolve_image_path(data_root, split_dir, file_name)
        rel_name = Path(file_name).name  # flatten
        dst_img = img_out_dir / rel_name
        _safe_link_or_copy(src_img, dst_img)

        lines: List[str] = []
        for a in ds.anns_by_image.get(image_id, []):
            orig_cat = int(a["category_id"])
            if orig_cat not in ds.id_to_contig:
                continue
            cls = int(ds.id_to_contig[orig_cat])

            x, y, w, h = a["bbox"]
            x = float(x); y = float(y); w = float(w); h = float(h)

            x1 = max(0.0, x)
            y1 = max(0.0, y)
            x2 = min(float(W), x + w)
            y2 = min(float(H), y + h)

            yolo = coco_to_yolo_line(x1, y1, x2, y2, cls, W, H)
            if yolo is not None:
                lines.append(yolo)

        dst_lbl = lbl_out_dir / (Path(rel_name).stem + ".txt")
        dst_lbl.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

        num_imgs += 1
        num_lbls += len(lines)

    return num_imgs, num_lbls


def write_dataset_yaml(
    yolo_root: Path, names: List[str], train_split: str, val_split: str, test_split: str
) -> Path:
    yaml_path = yolo_root / "dataset.yaml"
    content = [
        f"path: {yolo_root.as_posix()}",
        f"train: images/{train_split}",
        f"val: images/{val_split}",
        f"test: images/{test_split}",
        f"nc: {len(names)}",
        "names:",
    ]
    for n in names:
        nn = n.replace('"', '\\"')
        content.append(f'  - "{nn}"')
    yaml_path.write_text("\n".join(content) + "\n", encoding="utf-8")
    return yaml_path


def find_weights(outdir: Path, name: str, which: str) -> Path:
    """
    Your observed layout:
      runs/detect/runs_yolo26/exp/weights/best.pt

    So if you pass:
      --outdir runs/detect/runs_yolo26
      --name exp
    this should resolve:
      <outdir>/<name>/weights/<which>.pt

    But we'll also support a few other common Ultralytics layouts.
    """
    candidates = [
        # ✅ your layout (recommended)
        outdir / name / "weights" / f"{which}.pt",

        # If user accidentally passes only "runs" or "runs/detect"
        outdir / "detect" / name / "weights" / f"{which}.pt",
        outdir / "detect" / outdir.name / name / "weights" / f"{which}.pt",

        # Old / alternate nesting patterns
        outdir / outdir.name / name / "weights" / f"{which}.pt",
        outdir / "runs_yolo26" / name / "weights" / f"{which}.pt",
    ]

    for p in candidates:
        if p.exists():
            return p

    raise FileNotFoundError(
        f"Could not find {which}.pt. Tried:\n  " + "\n  ".join(str(c) for c in candidates)
    )



def _read_metric_attr(obj, names: List[str]) -> Optional[float]:
    for n in names:
        if hasattr(obj, n):
            v = getattr(obj, n)
            try:
                return float(v)
            except Exception:
                pass
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--train-split", default="train")
    ap.add_argument("--val-split", default="valid")
    ap.add_argument("--test-split", default="test")

    ap.add_argument("--split", default="test", choices=["train", "valid", "test"])
    ap.add_argument("--yolo-ds", default="yolo_ds", help="Folder containing YOLO-format dataset (labels+images)")
    ap.add_argument("--rebuild-yolo-ds", action="store_true", help="Force rebuild YOLO dataset folder")

    ap.add_argument(
    "--outdir",
    default="runs/detect/runs_yolo26",
    help="Ultralytics project folder used during training"
    )
    ap.add_argument("--name", default="exp", help="Run name used during training")

    ap.add_argument("--which", default="best", choices=["best", "last"], help="Evaluate best.pt or last.pt")

    ap.add_argument("--img", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--device", default=None, help="e.g. 0 or 'cpu'. Default: auto")

    ap.add_argument("--conf", type=float, default=None, help="Optional confidence threshold for evaluation")
    ap.add_argument("--iou", type=float, default=None, help="Optional IoU threshold for NMS during evaluation")

    ap.add_argument("--save-json", action="store_true", help="Ask Ultralytics to save COCO JSON results (if supported)")
    ap.add_argument("--plots", action="store_true", help="Ask Ultralytics to save plots (PR curves etc.)")
    ap.add_argument("--save-txt", action="store_true", help="Save predictions to txt (Ultralytics option)")
    ap.add_argument("--save-conf", action="store_true", help="Save confidences in txt (Ultralytics option)")

    args = ap.parse_args()

    data_root = Path(args.data_root)
    yolo_root = Path(args.yolo_ds)

    # 1) Ensure YOLO dataset.yaml exists and is consistent with your COCO mapping
    train_ds = COCODataset(str(data_root), split=args.train_split)
    classes = train_ds.classes
    id_to_contig = train_ds.id_to_contig

    val_ds = COCODataset(str(data_root), split=args.val_split, classes=classes, id_to_contig=id_to_contig)
    test_ds = COCODataset(str(data_root), split=args.test_split, classes=classes, id_to_contig=id_to_contig)

    yaml_path = yolo_root / "dataset.yaml"
    if args.rebuild_yolo_ds or not yaml_path.exists():
        print("Rebuilding YOLO dataset folder:", yolo_root.resolve())
        yolo_root.mkdir(parents=True, exist_ok=True)

        n_img, n_lbl = export_split_to_yolo(train_ds, data_root, args.train_split, yolo_root, overwrite=True)
        print(f"  train: images={n_img}, labels={n_lbl}")

        n_img, n_lbl = export_split_to_yolo(val_ds, data_root, args.val_split, yolo_root, overwrite=True)
        print(f"  val:   images={n_img}, labels={n_lbl}")

        n_img, n_lbl = export_split_to_yolo(test_ds, data_root, args.test_split, yolo_root, overwrite=True)
        print(f"  test:  images={n_img}, labels={n_lbl}")

        yaml_path = write_dataset_yaml(yolo_root, classes, args.train_split, args.val_split, args.test_split)
        print("Wrote:", yaml_path)
    else:
        print("Using existing:", yaml_path)

    # 2) Load YOLO weights from the training run folder
    outdir = Path(args.outdir)
    weights = find_weights(outdir, args.name, args.which)
    print("Evaluating weights:", weights)

    try:
        from ultralytics import YOLO
    except Exception as e:
        raise RuntimeError("Ultralytics not installed. Run: pip install ultralytics") from e

    model = YOLO(str(weights))

    # 3) Evaluate on requested split
    split_name = args.split  # must be train/val/test in Ultralytics
    ul_split = "val" if split_name == "valid" else split_name  # your split is "valid", ultralytics uses "val"

    val_kwargs = dict(
        data=str(yaml_path),
        imgsz=int(args.img),
        batch=int(args.batch),
        workers=int(args.num_workers),
        split=ul_split,
        plots=bool(args.plots),
        save_json=bool(args.save_json),
        save_txt=bool(args.save_txt),
        save_conf=bool(args.save_conf),
    )
    if args.device is not None:
        val_kwargs["device"] = args.device
    if args.conf is not None:
        val_kwargs["conf"] = float(args.conf)
    if args.iou is not None:
        val_kwargs["iou"] = float(args.iou)

    print("Ultralytics val args:", val_kwargs)
    metrics = model.val(**val_kwargs)

    # 4) Print key metrics in a stable way (Ultralytics versions differ slightly)
    # Most versions expose metrics.box.map, map50, map75, mar100 etc.
    box = getattr(metrics, "box", None)

    map5095 = None
    map50 = None
    map75 = None
    mar100 = None

    if box is not None:
        map5095 = _read_metric_attr(box, ["map", "map50_95", "map5095"])
        map50 = _read_metric_attr(box, ["map50"])
        map75 = _read_metric_attr(box, ["map75"])
        mar100 = _read_metric_attr(box, ["mar", "mar100"])
    else:
        # fallback: try on metrics directly
        map5095 = _read_metric_attr(metrics, ["map", "map50_95", "map5095"])
        map50 = _read_metric_attr(metrics, ["map50"])
        map75 = _read_metric_attr(metrics, ["map75"])
        mar100 = _read_metric_attr(metrics, ["mar", "mar100"])

    print("\n=== YOLO Evaluation Results ===")
    if map50 is not None:
        print(f"mAP@0.50          : {map50:.4f}")
    if map75 is not None:
        print(f"mAP@0.75          : {map75:.4f}")
    if map5095 is not None:
        print(f"mAP@0.50:0.95 (COCO): {map5095:.4f}")
    if mar100 is not None:
        print(f"mAR@100 (COCO)    : {mar100:.4f}")

    print("\nWhere results are saved:")
    # metrics.save_dir exists in many versions
    save_dir = getattr(metrics, "save_dir", None)
    if save_dir is not None:
        print("  ", save_dir)
    else:
        print("  (Ultralytics decides the save folder; check your runs directory.)")


if __name__ == "__main__":
    main()
