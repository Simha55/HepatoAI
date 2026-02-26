from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import List, Any, Dict

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt
from transformers import ConditionalDetrConfig

matplotlib.use("Agg")

# Conditional DETR (with safe fallbacks if your transformers version uses DetrImageProcessor)
try:
    from transformers import ConditionalDetrForObjectDetection, ConditionalDetrImageProcessor
    _PROC_CLS = "ConditionalDetrImageProcessor"
except Exception:
    from transformers import ConditionalDetrForObjectDetection, DetrImageProcessor as ConditionalDetrImageProcessor
    _PROC_CLS = "DetrImageProcessor (fallback)"
from torchmetrics.detection.mean_ap import MeanAveragePrecision

from coco_dataset import COCODataset
from functools import partial


def collate_with_processor(batch, processor, img_short, img_max):
    return collate_fn(processor, batch, img_short, img_max)


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

    # torchmetrics ground truth wants XYXY + labels
    gt = []
    for b in batch:
        gt.append({
            "boxes": torch.tensor(b.gt_boxes_xyxy, dtype=torch.float32),
            "labels": torch.tensor(b.gt_labels, dtype=torch.int64),
        })
    return enc, gt


@torch.no_grad()
def eval_map50(model, processor, loader, device):
    metric = MeanAveragePrecision(iou_type="bbox", iou_thresholds=[0.5])
    model.eval()

    for enc, gt in tqdm(loader, desc="eval", leave=False):
        pv = enc["pixel_values"].to(device)
        pm = enc["pixel_mask"].to(device)

        outputs = model(pixel_values=pv, pixel_mask=pm)

        # enc["labels"] contains orig_size per image
        target_sizes = torch.stack([l["orig_size"] for l in enc["labels"]]).to(device)
        processed = processor.post_process_object_detection(
            outputs, target_sizes=target_sizes, threshold=0.0
        )

        preds = []
        for p in processed:
            preds.append({
                "boxes": p["boxes"].detach().cpu(),
                "scores": p["scores"].detach().cpu(),
                "labels": p["labels"].detach().cpu(),
            })

        metric.update(preds, gt)

    res = metric.compute()
    return float(res["map"].item())


def plot_curves(rows, outdir: Path):
    epochs = [r["epoch"] for r in rows]
    losses = [r["train_loss"] for r in rows]
    maps = [r["val_map50"] for r in rows]

    plt.figure()
    plt.plot(epochs, losses)
    plt.xlabel("epoch")
    plt.ylabel("train loss")
    plt.title("Train loss")
    plt.tight_layout()
    plt.savefig(outdir / "train_loss.png")
    plt.close()

    plt.figure()
    plt.plot(epochs, maps)
    plt.xlabel("epoch")
    plt.ylabel("val mAP@0.5")
    plt.title("Validation mAP@0.5")
    plt.tight_layout()
    plt.savefig(outdir / "val_map50.png")
    plt.close()


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--data-root", default="data", help="Folder containing train/valid/test")
    ap.add_argument("--train-split", default="train")
    ap.add_argument("--val-split", default="valid")
    ap.add_argument("--test-split", default="test")

    # ✅ different default outdir for Conditional DETR
    ap.add_argument("--outdir", default="weights/conditional_detr_coco_ft")
    # ✅ Conditional DETR checkpoint (COCO-pretrained)
    ap.add_argument("--model", default="microsoft/conditional-detr-resnet-50")

    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lr_backbone", type=float, default=1e-5)

    ap.add_argument("--img", type=int, default=640)
    ap.add_argument("--img_max", type=int, default=1024)

    ap.add_argument("--freeze_backbone_epochs", type=int, default=10)
    ap.add_argument("--unfreeze_mode", choices=["layer4", "all"], default="layer4")

    ap.add_argument("--num_workers", type=int, default=2)

    # ✅ Resume + force LR behavior
    ap.add_argument("--resume", action="store_true", help="Resume from outdir if it has a saved checkpoint")
    ap.add_argument(
        "--force_lr",
        action="store_true",
        help="When resuming, override checkpoint optimizer LRs with --lr/--lr_backbone. "
             "If not set, keep LRs from checkpoint."
    )
    ap.add_argument("--from_scratch", action="store_true", help="Initialize model with random weights (no pretrained).")
    ap.add_argument("--num_queries", type=int, default=300, help="Number of object queries (DETR queries).")


    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    # 1) Load datasets first so we know class count / mapping
    train_full = COCODataset(args.data_root, split=args.train_split)
    classes = train_full.classes
    id_to_contig = train_full.id_to_contig

    val_ds = COCODataset(args.data_root, split=args.val_split, classes=classes, id_to_contig=id_to_contig)
    test_ds = COCODataset(args.data_root, split=args.test_split, classes=classes, id_to_contig=id_to_contig)

    print("num_classes:", len(classes), "classes:", classes)

    # ✅ 2) Processor + Model (fresh vs resume)
    resume_ok = args.resume and (outdir / "config.json").exists() and (outdir / "preprocessor_config.json").exists()

    if resume_ok:
        print("Resuming from:", outdir)
        processor = ConditionalDetrImageProcessor.from_pretrained(outdir)
        model = ConditionalDetrForObjectDetection.from_pretrained(outdir).to(device)

        # overwrite label maps to match current dataset (safe)
        model.config.id2label = {i: name for i, name in enumerate(classes)}
        model.config.label2id = {name: i for i, name in model.config.id2label.items()}
    else:
        # Processor: you can still use pretrained processor even when model is scratch
        # (it only defines resize/normalize/pad params)
        processor = ConditionalDetrImageProcessor.from_pretrained(args.model)

        id2label = {i: name for i, name in enumerate(classes)}
        label2id = {name: i for i, name in id2label.items()}

        if args.from_scratch:
            # Build config from the base architecture, then random-init weights
            cfg = ConditionalDetrConfig.from_pretrained(
                args.model,
                num_labels=len(classes),
                id2label=id2label,
                label2id=label2id,
            )

            # ✅ reduce queries here
            cfg.num_queries = args.num_queries

            # random init (no pretrained weights)
            model = ConditionalDetrForObjectDetection(cfg).to(device)

            # (optional but nice) print to verify
            print(f"FROM SCRATCH ✅ num_queries={model.config.num_queries}")

        else:
            # ✅ pretrained init (your current behavior)
            model = ConditionalDetrForObjectDetection.from_pretrained(
                args.model,
                num_labels=len(classes),
                id2label=id2label,
                label2id=label2id,
                ignore_mismatched_sizes=True,
            ).to(device)

            # ✅ reduce queries even when starting from pretrained
            model.config.num_queries = args.num_queries
            print(f"PRETRAINED ✅ num_queries={model.config.num_queries}")

    print("processor:", _PROC_CLS)

    collate = partial(collate_with_processor, processor=processor, img_short=args.img, img_max=args.img_max)

    train_loader = DataLoader(
        train_full,
        batch_size=args.batch,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate,
    )

    def set_backbone(mode: str):
        for name, p in model.named_parameters():
            if "backbone" not in name:
                continue
            if mode == "frozen":
                p.requires_grad = False
            elif mode == "layer4":
                p.requires_grad = ("layer4" in name)
            elif mode == "all":
                p.requires_grad = True

    def build_optimizer():
        backbone_params, other_params = [], []
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            (backbone_params if "backbone" in name else other_params).append(p)

        return torch.optim.AdamW(
            [{"params": other_params, "lr": args.lr},
             {"params": backbone_params, "lr": args.lr_backbone}],
            weight_decay=1e-4,
        )

    rows: List[Dict[str, float]] = []
    best_map = -1.0
    start_epoch = 1

    hist_path = outdir / "history.csv"
    if args.resume and hist_path.exists():
        with open(hist_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            for r in reader:
                rows.append({
                    "epoch": int(float(r["epoch"])),
                    "train_loss": float(r["train_loss"]) if r["train_loss"] != "nan" else float("nan"),
                    "val_map50": float(r["val_map50"]),
                })
        if rows:
            best_map = max((r["val_map50"] for r in rows if r["epoch"] != 0), default=-1.0)
            start_epoch = rows[-1]["epoch"] + 1
        print(f"Loaded history: {len(rows)} rows. start_epoch={start_epoch}, best_map={best_map:.4f}")

    def write_history():
        with open(outdir / "history.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_map50"])
            w.writeheader()
            w.writerows(rows)
        plot_curves(rows, outdir)

    # Backbone freeze state when resuming
    if start_epoch > args.freeze_backbone_epochs:
        set_backbone(args.unfreeze_mode)
        print(f"Resume: backbone set to {args.unfreeze_mode}")
    else:
        set_backbone("frozen")
        print("Resume: backbone frozen")

    optimizer = build_optimizer()
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    # Load checkpoint on resume (optimizer + scaler + epoch)
    ckpt_path = outdir / "checkpoint.pt"
    if args.resume and ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device)
        optimizer.load_state_dict(ckpt["optimizer"])
        scaler.load_state_dict(ckpt["scaler"])

        if args.force_lr:
            optimizer.param_groups[0]["lr"] = args.lr
            optimizer.param_groups[1]["lr"] = args.lr_backbone
            print("Forced LRs:", [g["lr"] for g in optimizer.param_groups])
        else:
            print("Kept checkpoint LRs:", [g["lr"] for g in optimizer.param_groups])

        best_map = max(best_map, float(ckpt.get("best_map", -1.0)))
        start_epoch = max(start_epoch, int(ckpt["epoch"]) + 1)
        print(f"Loaded checkpoint.pt. start_epoch={start_epoch}, best_map={best_map:.4f}")

    # Don’t re-run baseline eval when resuming
    if not (args.resume and rows):
        baseline_map = eval_map50(model, processor, test_loader, device)
        rows.append({"epoch": 0, "train_loss": float("nan"), "val_map50": baseline_map})
        print(f"baseline test mAP@0.5 = {baseline_map:.4f}")
        write_history()
    else:
        write_history()

    for epoch in range(start_epoch, args.epochs + 1):
        if epoch == args.freeze_backbone_epochs + 1:
            set_backbone(args.unfreeze_mode)
            optimizer = build_optimizer()
            print(f"Unfroze backbone mode = {args.unfreeze_mode}")

        model.train()
        total_loss, steps = 0.0, 0
        pbar = tqdm(train_loader, desc=f"train e{epoch}", leave=False)

        for enc, _gt in pbar:
            pv = enc["pixel_values"].to(device, non_blocking=True)
            pm = enc["pixel_mask"].to(device, non_blocking=True)
            labels = [{k: v.to(device) for k, v in t.items()} for t in enc["labels"]]

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                out = model(pixel_values=pv, pixel_mask=pm, labels=labels)
                loss = out.loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += float(loss.item())
            steps += 1
            pbar.set_postfix(loss=float(loss.item()))

        train_loss = total_loss / max(1, steps)
        val_map50 = eval_map50(model, processor, val_loader, device)

        rows.append({"epoch": epoch, "train_loss": train_loss, "val_map50": val_map50})
        write_history()

        if val_map50 > best_map:
            best_map = val_map50
            model.save_pretrained(outdir)
            processor.save_pretrained(outdir)

        print(f"epoch={epoch} train_loss={train_loss:.4f} val_map50={val_map50:.4f} best={best_map:.4f}")

        ckpt = {
            "epoch": epoch,
            "best_map": best_map,
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
        }
        torch.save(ckpt, outdir / "checkpoint.pt")

    print("Reloading best checkpoint from:", outdir)
    model = ConditionalDetrForObjectDetection.from_pretrained(outdir).to(device)
    processor = ConditionalDetrImageProcessor.from_pretrained(outdir)

    final_test_map = eval_map50(model, processor, test_loader, device)
    print("FINAL test mAP@0.5:", final_test_map)

    print("Done. Saved best model to:", outdir)


if __name__ == "__main__":
    main()
