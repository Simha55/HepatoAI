from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import json

import torch
from torch.utils.data import Dataset
from PIL import Image


@dataclass
class CocoSample:
    image: Image.Image
    target: Dict[str, Any]          # COCO-like annotations for DetrImageProcessor
    gt_boxes_xyxy: List[List[float]]
    gt_labels: List[int]
    image_id: int
    image_path: str
    orig_size_hw: Tuple[int, int]   # (H, W)


class COCODataset(Dataset):
    """
    Expects Roboflow-style COCO layout:

      data/
        train/
          _annotations.coco.json
          <images...>
        valid/
          _annotations.coco.json
          <images...>
        test/
          _annotations.coco.json
          <images...>

    Produces:
      - target in COCO format for DetrImageProcessor: {"image_id": int, "annotations": [...]}
      - gt_* in XYXY for torchmetrics
    """

    def __init__(
        self,
        data_root: str,
        split: str,
        ann_path: Optional[str] = None,
        classes: Optional[List[str]] = None,
        id_to_contig: Optional[Dict[int, int]] = None,
    ) -> None:
        self.data_root = Path(data_root)
        self.split = split
        self.split_dir = self.data_root / split

        # Find annotation file
        candidates = []
        if ann_path:
            candidates.append(Path(ann_path))
        candidates += [
            self.split_dir / "_annotations.coco.json",
            self.split_dir / "annotations.coco.json",
            self.data_root / "_annotations.coco.json",
        ]
        self.ann_path = next((p for p in candidates if p.exists()), None)
        if self.ann_path is None:
            raise FileNotFoundError(
                f"Could not find COCO annotations for split='{split}'. Tried: "
                + ", ".join(str(p) for p in candidates)
            )

        with open(self.ann_path, "r") as f:
            coco = json.load(f)

        # Categories -> contiguous label ids [0..K-1]
        if classes is None or id_to_contig is None:
            cats = sorted(coco.get("categories", []), key=lambda c: int(c["id"]))
            cats = [c for c in cats if c["name"] != "diseases"]
            self.classes = [c["name"] for c in cats]
            orig_ids = [int(c["id"]) for c in cats]
            self.id_to_contig = {oid: i for i, oid in enumerate(orig_ids)}
        else:
            self.classes = classes
            self.id_to_contig = id_to_contig

        # Images index (sorted for deterministic ordering)
        images = coco.get("images", [])
        images = sorted(images, key=lambda im: int(im["id"]))
        self.images = images

        # Build annotations per image_id
        self.anns_by_image: Dict[int, List[dict]] = {}
        for a in coco.get("annotations", []):
            img_id = int(a["image_id"])
            self.anns_by_image.setdefault(img_id, []).append(a)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> CocoSample:
        im = self.images[idx]
        image_id = int(im["id"])
        file_name = im["file_name"]

        img_path = (self.split_dir / file_name)
        if not img_path.exists():
            # some COCO exports include subfolders in file_name
            img_path = (self.data_root / file_name)
        if not img_path.exists():
            raise FileNotFoundError(f"Missing image: {img_path}")

        image = Image.open(img_path).convert("RGB")
        W = int(im.get("width", image.size[0]))
        H = int(im.get("height", image.size[1]))

        anns_raw = self.anns_by_image.get(image_id, [])

        coco_anns: List[Dict[str, Any]] = []
        gt_boxes_xyxy: List[List[float]] = []
        gt_labels: List[int] = []

        for a in anns_raw:
            orig_cat = int(a["category_id"])
            if orig_cat not in self.id_to_contig:
                continue
            cat = int(self.id_to_contig[orig_cat])

            x, y, w, h = a["bbox"]
            x = float(x); y = float(y); w = float(w); h = float(h)

            # Clip to image bounds
            x1 = max(0.0, x)
            y1 = max(0.0, y)
            x2 = min(float(W), x + w)
            y2 = min(float(H), y + h)
            ww = x2 - x1
            hh = y2 - y1
            if ww <= 0 or hh <= 0:
                continue

            coco_anns.append({
                "bbox": [x1, y1, ww, hh],
                "category_id": cat,
                "area": float(ww * hh),
                "iscrowd": int(a.get("iscrowd", 0)),
            })

            gt_boxes_xyxy.append([x1, y1, x2, y2])
            gt_labels.append(cat)

        target = {"image_id": image_id, "annotations": coco_anns}

        return CocoSample(
            image=image,
            target=target,
            gt_boxes_xyxy=gt_boxes_xyxy,
            gt_labels=gt_labels,
            image_id=image_id,
            image_path=str(img_path),
            orig_size_hw=(H, W),
        )
