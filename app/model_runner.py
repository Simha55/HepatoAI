from __future__ import annotations

import time
import threading
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

import torch

try:
    import onnxruntime as ort
except Exception:  # pragma: no cover
    ort = None

from transformers import AutoModelForObjectDetection, AutoImageProcessor

from .config import settings
from .metrics import metrics

DISEASE_CLASSES = [x.strip() for x in settings.class_names.split(",") if x.strip()]

@dataclass
class InferenceOutput:
    detections: List[List[Dict[str, Any]]]
    backend: str
    latency_ms: float


class GPUHealth:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._unhealthy_until = 0.0

    def is_healthy(self) -> bool:
        with self._lock:
            return time.time() >= self._unhealthy_until

    def mark_unhealthy(self, seconds: float = 30.0) -> None:
        with self._lock:
            self._unhealthy_until = max(self._unhealthy_until, time.time() + seconds)


gpu_health = GPUHealth()


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - x.max(axis=axis, keepdims=True)
    ex = np.exp(x)
    return ex / ex.sum(axis=axis, keepdims=True)


def _cxcywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    cx, cy, w, h = boxes[..., 0], boxes[..., 1], boxes[..., 2], boxes[..., 3]
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    return np.stack([x1, y1, x2, y2], axis=-1)

import numpy as np

def _nms_xyxy(boxes: np.ndarray, scores: np.ndarray, iou_thresh: float) -> np.ndarray:
    """
    boxes: [N,4] in xyxy (pixel coordinates)
    scores: [N]
    returns: indices to keep (in descending score order)
    """
    if boxes.size == 0:
        return np.array([], dtype=np.int64)

    x1 = boxes[:, 0].astype(np.float32)
    y1 = boxes[:, 1].astype(np.float32)
    x2 = boxes[:, 2].astype(np.float32)
    y2 = boxes[:, 3].astype(np.float32)

    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = scores.argsort()[::-1].astype(np.int64)

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h

        union = areas[i] + areas[order[1:]] - inter + 1e-6
        iou = inter / union

        remain = np.where(iou <= iou_thresh)[0]
        order = order[remain + 1]

    return np.array(keep, dtype=np.int64)
def postprocess_raw_detr(
    logits: np.ndarray,
    pred_boxes: np.ndarray,
    orig_sizes,
    score_thresh: float,
    nms_iou: float = 0.45,
    max_dets: int = 10,
):
    probs = _softmax(logits, axis=-1)
    probs_fg = probs[..., :-1]              # drop no-object
    scores = probs_fg.max(axis=-1)          # [B,Q]
    labels = probs_fg.argmax(axis=-1)       # [B,Q]
    boxes_xyxy = _cxcywh_to_xyxy(pred_boxes)  # normalized xyxy

    B, Q, _ = logits.shape
    results = []

    for i in range(B):
        h, w = orig_sizes[i]

        # threshold
        keep = scores[i] >= score_thresh
        idx = np.where(keep)[0]
        if idx.size == 0:
            results.append([])
            continue

        # convert to pixel xyxy + clamp
        b = boxes_xyxy[i, idx].copy()
        b[:, 0] = np.clip(b[:, 0], 0.0, 1.0) * w
        b[:, 2] = np.clip(b[:, 2], 0.0, 1.0) * w
        b[:, 1] = np.clip(b[:, 1], 0.0, 1.0) * h
        b[:, 3] = np.clip(b[:, 3], 0.0, 1.0) * h

        s = scores[i, idx].astype(np.float32)
        l = labels[i, idx].astype(np.int64)

        # optional: pre top-K before NMS (faster/cleaner UI)
        if s.size > max_dets:
            top = np.argsort(-s)[:max_dets]
            b, s, l = b[top], s[top], l[top]

        # per-class NMS
        keep_global = []
        for cls in np.unique(l):
            cls_mask = (l == cls)
            cls_boxes = b[cls_mask]
            cls_scores = s[cls_mask]
            cls_keep = _nms_xyxy(cls_boxes, cls_scores, nms_iou)

            # map back to indices in b/s/l
            cls_indices = np.where(cls_mask)[0]
            keep_global.append(cls_indices[cls_keep])

        keep_global = np.concatenate(keep_global) if keep_global else np.array([], dtype=np.int64)

        # final sort by score
        keep_global = keep_global[np.argsort(-s[keep_global])]

        dets = []
        for k in keep_global:
            lab = int(l[k])
            dets.append({
                "label": lab,
                "label_name": DISEASE_CLASSES[lab] if lab < len(DISEASE_CLASSES) else str(lab),
                "score": float(s[k]),
                "box": [float(b[k,0]), float(b[k,1]), float(b[k,2]), float(b[k,3])],
            })

        results.append(dets)

    return results


class TorchRunner:
    """DETR Torch runner (CPU/GPU)"""

    def __init__(self, model_dir: str, device: torch.device) -> None:
        self.device = device
        self.processor = AutoImageProcessor.from_pretrained(model_dir)
        self.model = AutoModelForObjectDetection.from_pretrained(model_dir)
        # ✅ ensure label maps match your 4 classes
        self.model.config.id2label = {i: n for i, n in enumerate(DISEASE_CLASSES)}
        self.model.config.label2id = {n: i for i, n in enumerate(DISEASE_CLASSES)}

        self.model.to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def infer(self, images: List[Image.Image]) -> InferenceOutput:
        t0 = time.perf_counter()
        orig_sizes = [(img.height, img.width) for img in images]

        inputs = self.processor(
            images=images,
            return_tensors="pt",
            size={"shortest_edge": settings.image_short_side, "longest_edge": settings.image_max_side},
            pad_and_return_pixel_mask=True,
        )
        pv = inputs["pixel_values"].to(self.device, non_blocking=True)
        pm = inputs["pixel_mask"].to(self.device, non_blocking=True)

        out = self.model(pixel_values=pv, pixel_mask=pm)

        target_sizes = torch.tensor(orig_sizes, device=self.device)
        processed = self.processor.post_process_object_detection(
            out, target_sizes=target_sizes, threshold=settings.score_threshold
        )

        dets: List[List[Dict[str, Any]]] = []
        for p in processed:
            boxes = p["boxes"].detach().cpu().numpy()
            scores = p["scores"].detach().cpu().numpy()
            labels = p["labels"].detach().cpu().numpy()
            img_dets = []
            for b, s, l in zip(boxes, scores, labels):
                lab = int(l)
                name = DISEASE_CLASSES[lab] if lab < len(DISEASE_CLASSES) else str(lab)
                img_dets.append(
                    {"label": lab, "label_name": name, "score": float(s), "box": [float(x) for x in b.tolist()]}
                )
            dets.append(img_dets)

        dt = (time.perf_counter() - t0) * 1000.0
        return InferenceOutput(detections=dets, backend=f"detr_torch:{self.device.type}", latency_ms=dt)


class ORTRunner:
    """DETR ONNXRuntime runner"""

    def __init__(self, onnx_path: str, prefer_gpu: bool) -> None:
        if ort is None:
            raise RuntimeError("onnxruntime not installed")
        providers = ["CPUExecutionProvider"]
        if prefer_gpu and ort.get_device().lower() == "gpu":
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self.session = ort.InferenceSession(onnx_path, providers=providers)
        self.providers = providers
        self.input_names = [i.name for i in self.session.get_inputs()]
        self.has_mask = "pixel_mask" in self.input_names

    def infer(self, processor: AutoImageProcessor, images: List[Image.Image]) -> InferenceOutput:
        t0 = time.perf_counter()
        orig_sizes = [(img.height, img.width) for img in images]

        inputs = processor(
            images=images,
            return_tensors="pt",
            size={"shortest_edge": settings.image_short_side, "longest_edge": settings.image_max_side},
            pad_and_return_pixel_mask=True,
        )
        pv = inputs["pixel_values"].numpy().astype(np.float32)
        feed = {"pixel_values": pv}
        if self.has_mask:
            pm = inputs["pixel_mask"].numpy().astype(np.int64)
            feed["pixel_mask"] = pm

        outs = self.session.run(None, feed)
        logits = outs[0]
        pred_boxes = outs[1]

        dets = postprocess_raw_detr(logits, pred_boxes, orig_sizes, settings.score_threshold)

        dt = (time.perf_counter() - t0) * 1000.0
        backend = "detr_onnx:" + ("cuda" if "CUDAExecutionProvider" in self.providers else "cpu")
        return InferenceOutput(detections=dets, backend=backend, latency_ms=dt)


class UltralyticsYoloONNXRunner:
    """
    YOLO ONNX via Ultralytics.
    
    """

    def __init__(self, onnx_path: str) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as e:
            raise RuntimeError("ultralytics not installed. pip install ultralytics") from e

        self.model = YOLO(onnx_path)

        # Ensure class names are correct even if model metadata is missing/wrong
        try:
            self.model.names = {i: n for i, n in enumerate(DISEASE_CLASSES)}
        except Exception:
            pass

    def infer(self, images: List[Image.Image]) -> InferenceOutput:
        t0 = time.perf_counter()

        results = self.model.predict(
            images,
            imgsz=int(getattr(settings, "yolo_imgsz", 320)),
            conf=float(settings.score_threshold),
            iou=float(getattr(settings, "nms_iou_threshold", 0.45)),
            verbose=False,
        )

        dets: List[List[Dict[str, Any]]] = []
        for r in results:
            img_dets: List[Dict[str, Any]] = []
            if r.boxes is not None and len(r.boxes) > 0:
                xyxy = r.boxes.xyxy.detach().cpu().numpy()
                conf = r.boxes.conf.detach().cpu().numpy()
                cls = r.boxes.cls.detach().cpu().numpy().astype(int)

                for (x1, y1, x2, y2), s, c in zip(xyxy, conf, cls):
                    name = DISEASE_CLASSES[c] if c < len(DISEASE_CLASSES) else str(c)
                    if int(c) == 0:
                        continue
                    img_dets.append(
                        {
                            "label": int(c),
                            "label_name": name,
                            "score": float(s),
                            "box": [float(x1), float(y1), float(x2), float(y2)],
                        }
                    )
            dets.append(img_dets)

        dt = (time.perf_counter() - t0) * 1000.0
        return InferenceOutput(detections=dets, backend="yolo_ultralytics_onnx", latency_ms=dt)


class InferenceEngine:
    """
    Supports two model types:
      - DETR (torch or onnxruntime)
      - YOLO (onnx via ultralytics)
    Select with settings.model_type: "detr" or "yolo"
    """

    def __init__(self) -> None:
        self.model_type = getattr(settings, "model_type", "detr").lower()

        # YOLO path
        self.yolo: Optional[UltralyticsYoloONNXRunner] = None
        if self.model_type == "yolo":
            self.yolo = UltralyticsYoloONNXRunner(settings.onnx_path)
            return

        # DETR path
        self.torch_cpu = TorchRunner(settings.model_dir, torch.device("cpu"))

        self.torch_gpu: Optional[TorchRunner] = None
        if settings.prefer_gpu and torch.cuda.is_available():
            self.torch_gpu = TorchRunner(settings.model_dir, torch.device("cuda"))

        self.ort: Optional[ORTRunner] = None
        self.processor = AutoImageProcessor.from_pretrained(settings.model_dir)
        if settings.use_onnx:
            self.ort = ORTRunner(settings.onnx_path, prefer_gpu=settings.prefer_gpu)

    def infer(self, images: List[Image.Image]) -> InferenceOutput:
        metrics.inc("requests_total", 1)

        # YOLO mode
        if self.model_type == "yolo":
            assert self.yolo is not None
            out = self.yolo.infer(images)
            metrics.observe_latency_ms(out.latency_ms)
            return out

        # DETR ONNX (preferred if enabled and GPU healthy)
        if self.ort and (not settings.prefer_gpu or gpu_health.is_healthy()):
            try:
                out = self.ort.infer(self.processor, images)
                metrics.observe_latency_ms(out.latency_ms)
                return out
            except Exception as e:
                metrics.inc("onnx_failures_total", 1)
                msg = str(e).lower()
                if "cuda" in msg or "cudnn" in msg or "out of memory" in msg:
                    gpu_health.mark_unhealthy(30.0)
                    metrics.inc("gpu_unhealthy_trips_total", 1)

        # DETR Torch GPU
        if self.torch_gpu and gpu_health.is_healthy():
            try:
                out = self.torch_gpu.infer(images)
                metrics.observe_latency_ms(out.latency_ms)
                return out
            except RuntimeError as e:
                metrics.inc("torch_gpu_failures_total", 1)
                msg = str(e).lower()
                if "out of memory" in msg or "cuda" in msg:
                    gpu_health.mark_unhealthy(30.0)
                    metrics.inc("gpu_unhealthy_trips_total", 1)

        # DETR Torch CPU fallback
        metrics.inc("cpu_fallback_total", 1)
        out = self.torch_cpu.infer(images)
        metrics.observe_latency_ms(out.latency_ms)
        return out