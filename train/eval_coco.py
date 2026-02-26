from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, List, Tuple, Dict, Optional

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")

# --- Conditional DETR import w/ fallback ---
try:
    from transformers import (
        ConditionalDetrForObjectDetection,
        ConditionalDetrImageProcessor,
    )
    _MODEL_CLS = "ConditionalDetrForObjectDetection"
    _PROC_CLS = "ConditionalDetrImageProcessor"
except Exception:
    from transformers import (
        ConditionalDetrForObjectDetection,
        DetrImageProcessor as ConditionalDetrImageProcessor,
    )
    _MODEL_CLS = "ConditionalDetrForObjectDetection"
    _PROC_CLS = "DetrImageProcessor (fallback)"

from torchmetrics.detection.mean_ap import MeanAveragePrecision

# Your COCO dataset class (same one used in training)
from coco_dataset import COCODataset


# -------------------------
# Collate (same as training)
# -------------------------
def collate_fn(processor, batch: List[Any], img_short: int, img_max: int):
    images = [b.image for b in batch]
    targets = [b.target for b in batch]

    enc = processor(
        images=images,
        annotations=targets,
        return_tensors="pt",
        do_pad=True,
        size={"shortest_edge": img_short, "longest_edge": img_max},
    )

    # torchmetrics wants GT in XYXY
    gt = []
    for b in batch:
        gt.append(
            {
                "boxes": torch.tensor(b.gt_boxes_xyxy, dtype=torch.float32),
                "labels": torch.tensor(b.gt_labels, dtype=torch.int64),
            }
        )

    return enc, gt, batch


# -------------------------
# Utilities
# -------------------------
@torch.no_grad()
def run_eval(
    model,
    processor,
    loader: DataLoader,
    device: torch.device,
    iou_thresholds: Optional[List[float]] = None,
    score_threshold: float = 0.0,
    max_dets: int = 100,
) -> Dict[str, float]:
    """
    Runs torchmetrics MeanAveragePrecision.
    - iou_thresholds:
        * [0.5] -> mAP50
        * [0.75] -> mAP75
        * None -> COCO-style 0.50:0.95 (default torchmetrics behavior)
    - score_threshold:
        Filtering applied in post_process_object_detection (not needed for true COCO AP, but useful sanity check).
    """
    # metric = MeanAveragePrecision(
    #     iou_type="bbox",
    #     iou_thresholds=iou_thresholds,
    #     max_detection_thresholds=[max_dets],
    # )
    metric = MeanAveragePrecision(iou_type="bbox", iou_thresholds=iou_thresholds)


    model.eval()

    for enc, gt, _raw in tqdm(loader, desc="eval", leave=False):
        pv = enc["pixel_values"].to(device, non_blocking=True)
        pm = enc["pixel_mask"].to(device, non_blocking=True)

        outputs = model(pixel_values=pv, pixel_mask=pm)

        # orig_size per image is inside enc["labels"]
        target_sizes = torch.stack([l["orig_size"] for l in enc["labels"]]).to(device)

        processed = processor.post_process_object_detection(
            outputs,
            target_sizes=target_sizes,
            threshold=float(score_threshold),
        )

        preds = []
        for p in processed:
            # Optional: ensure we cap detections to max_dets (torchmetrics already uses max_detection_thresholds,
            # but post_process returns all kept by threshold, so we also sort and clip for consistency)
            if p["scores"].numel() > 0:
                scores = p["scores"]
                idx = torch.argsort(scores, descending=True)
                idx = idx[:max_dets]
                boxes = p["boxes"][idx]
                labels = p["labels"][idx]
                scores = p["scores"][idx]
            else:
                boxes = p["boxes"]
                labels = p["labels"]
                scores = p["scores"]

            preds.append(
                {
                    "boxes": boxes.detach().cpu(),
                    "scores": scores.detach().cpu(),
                    "labels": labels.detach().cpu(),
                }
            )

        metric.update(preds, gt)

    res = metric.compute()

    out: Dict[str, float] = {}
    # torchmetrics returns tensors; convert to float
    for k, v in res.items():
        try:
            out[k] = float(v.item())
        except Exception:
            # map_per_class, mar_100_per_class, classes etc. are tensors; keep only scalar metrics
            pass
    return out


@torch.no_grad()
def save_qualitative(
    model,
    processor,
    device: torch.device,
    samples: List[Any],
    outdir: Path,
    id2label: Dict[int, str],
    limit: int = 8,
    score_threshold: float = 0.3,
):
    outdir.mkdir(parents=True, exist_ok=True)
    model.eval()
    saved = 0

    for s in samples:
        if saved >= limit:
            break

        img = s.image
        enc = processor(images=[img], return_tensors="pt", do_pad=True)
        pv = enc["pixel_values"].to(device)
        pm = enc["pixel_mask"].to(device)

        out = model(pixel_values=pv, pixel_mask=pm)

        target_sizes = torch.tensor([[img.height, img.width]], device=device)
        pred = processor.post_process_object_detection(
            out, target_sizes=target_sizes, threshold=float(score_threshold)
        )[0]

        plt.figure(figsize=(11, 8))
        plt.imshow(img)
        ax = plt.gca()

        boxes = pred["boxes"].detach().cpu()
        scores = pred["scores"].detach().cpu()
        labels = pred["labels"].detach().cpu()

        for box, score, lab in zip(boxes, scores, labels):
            x1, y1, x2, y2 = box.tolist()
            w = x2 - x1
            h = y2 - y1
            ax.add_patch(plt.Rectangle((x1, y1), w, h, fill=False, linewidth=2))
            name = id2label.get(int(lab), str(int(lab)))
            ax.text(
                x1,
                max(0, y1 - 6),
                f"{name} {float(score):.2f}",
                fontsize=10,
                bbox=dict(facecolor="white", alpha=0.6),
            )

        ax.axis("off")
        plt.tight_layout()
        fn = outdir / f"qual_{saved+1}.png"
        plt.savefig(fn)
        plt.close()
        saved += 1


# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--data-root", default="data", help="Folder containing train/valid/test")
    ap.add_argument("--split", default="test", choices=["train", "valid", "test"])
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--img", type=int, default=640)
    ap.add_argument("--img_max", type=int, default=1024)
    ap.add_argument("--num_workers", type=int, default=4)

    # Load from your trained output dir (HF save_pretrained folder)
    ap.add_argument("--outdir", required=True, help="Folder containing config.json + model weights")
    ap.add_argument("--device", default=None, help="e.g. cuda, cuda:0, cpu. Default: auto")

    # Score threshold for post_process (evaluation sanity option; set 0.0 for strict COCO-style)
    ap.add_argument("--score-threshold", type=float, default=0.0)

    # Extra outputs
    ap.add_argument("--save-qual", default="", help="If set, saves qualitative images to this folder")
    ap.add_argument("--qual-limit", type=int, default=8)
    ap.add_argument("--qual-threshold", type=float, default=0.3)
    ap.add_argument("--max-dets", type=int, default=100, help="Max detections per image for evaluation")

    args = ap.parse_args()

    outdir = Path(args.outdir)
    assert (outdir / "config.json").exists(), f"Missing config.json in {outdir}"
    # weights might be pytorch_model.bin or model.safetensors; HF handles it.

    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print("device:", device)
    print("loading model from:", outdir)
    processor = ConditionalDetrImageProcessor.from_pretrained(outdir)
    model = ConditionalDetrForObjectDetection.from_pretrained(outdir, ignore_mismatched_sizes=True).to(device)

    # Dataset (class mapping must match training; your COCODataset reads from split json)
    ds = COCODataset(args.data_root, split=args.split)
    classes = ds.classes
    print("split:", args.split)
    print("num_classes:", len(classes), "classes:", classes)
    print("model:", _MODEL_CLS, "| processor:", _PROC_CLS)
    print("model.num_queries:", getattr(model.config, "num_queries", "unknown"))

    id2label = {i: name for i, name in enumerate(classes)}

    loader = DataLoader(
        ds,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=lambda b: collate_fn(processor, b, args.img, args.img_max),
    )

    # Collect a few samples for qualitative
    samples_for_qual: List[Any] = []

    # We’ll run three evals:
    # 1) mAP@0.5 (map50)
    # 2) mAP@0.75 (map75)
    # 3) COCO mAP@0.50:0.95 (map)
    #
    # NOTE: torchmetrics uses key "map" for whatever thresholds you pass.
    # We'll rename outputs when printing.

    # mAP@0.5
    res50 = run_eval(
        model,
        processor,
        loader,
        device,
        iou_thresholds=[0.5],
        score_threshold=args.score_threshold,
        max_dets=args.max_dets,
    )

    # mAP@0.75
    res75 = run_eval(
        model,
        processor,
        loader,
        device,
        iou_thresholds=[0.75],
        score_threshold=args.score_threshold,
        max_dets=args.max_dets,
    )

    # COCO mAP@0.50:0.95
    rescoco = run_eval(
        model,
        processor,
        loader,
        device,
        iou_thresholds=None,  # default 0.50:0.95 step 0.05
        score_threshold=args.score_threshold,
        max_dets=args.max_dets,
    )

    # Print summary
    # For res50/res75, torchmetrics returns "map" but it corresponds to that single IoU threshold.
    print("\n=== Metrics (torchmetrics MeanAveragePrecision) ===")
    print(f"score_threshold used in post_process: {args.score_threshold}")
    print(f"max_dets per image: {args.max_dets}")
    print("-----------------------------------------------")
    print(f"mAP@0.50          : {res50.get('map', float('nan')):.4f}")
    print(f"mAP@0.75          : {res75.get('map', float('nan')):.4f}")
    print(f"mAP@0.50:0.95 (COCO): {rescoco.get('map', float('nan')):.4f}")
    # also print recall metrics if present
    if "mar_100" in rescoco:
        print(f"mAR@100 (COCO)    : {rescoco.get('mar_100', float('nan')):.4f}")

    # Optional: per-class AP (COCO thresholds)
    # torchmetrics includes map_per_class and classes tensors in compute(), but we didn’t return them in run_eval.
    # If you want per-class too, we can compute once more and print it, but keeping this script lean.

    # Save qualitative predictions
    if args.save_qual.strip():
        # get a few raw samples quickly: iterate loader until we have enough
        for _enc, _gt, raw in loader:
            samples_for_qual.extend(raw)
            if len(samples_for_qual) >= max(args.qual_limit, 8):
                break

        save_qualitative(
            model=model,
            processor=processor,
            device=device,
            samples=samples_for_qual,
            outdir=Path(args.save_qual),
            id2label=id2label,
            limit=args.qual_limit,
            score_threshold=args.qual_threshold,
        )
        print("\nSaved qualitative images to:", args.save_qual)

    print("\nDone.")


if __name__ == "__main__":
    main()
