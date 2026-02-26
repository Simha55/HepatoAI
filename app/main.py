from __future__ import annotations

import asyncio
import hashlib
import io
import os
import time
import uuid
from typing import List, Optional
import random
from fastapi import FastAPI, File, UploadFile, HTTPException, Header
from fastapi.responses import PlainTextResponse, Response
from PIL import Image

from .config import settings
from .metrics import metrics
from .model_runner import InferenceEngine, gpu_health
from .batcher import MicroBatcher
from . import db
from .ui import router as ui_router
import logging
logger = logging.getLogger("doclens")
app = FastAPI(title="DocLens (DETR on VOC)", version="1.0")

engine = InferenceEngine()
batcher = MicroBatcher(engine)

app.include_router(ui_router)

@app.on_event("startup")
async def _startup() -> None:
    os.makedirs(settings.image_store_dir, exist_ok=True)
    db.init_db()
    await batcher.start()
    metrics.set_gauge("service_up", 1.0)

@app.on_event("shutdown")
async def _shutdown() -> None:
    await batcher.stop()
    metrics.set_gauge("service_up", 0.0)

@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "gpu_healthy": gpu_health.is_healthy(),
        "queue_size": batcher.queue.qsize(),
        "use_onnx": settings.use_onnx,
        "model_dir": settings.model_dir,
    }

@app.get("/metrics", response_class=PlainTextResponse)
def get_metrics() -> str:
    return metrics.snapshot()

@app.post("/detect")
async def detect(file: UploadFile = File(...)) -> dict:
    t0 = time.perf_counter()

    if file.content_type not in ("image/jpeg", "image/png", "image/webp"):
        metrics.inc("bad_content_type_total", 1)
        raise HTTPException(status_code=415, detail="Unsupported image type")

    try:
        data = await file.read()
        img = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception:
        metrics.inc("bad_image_total", 1)
        raise HTTPException(status_code=400, detail="Invalid image")

    try:
        out = await asyncio.wait_for(batcher.submit(img), timeout=settings.request_timeout_s)
    except asyncio.TimeoutError:
        metrics.inc("request_timeout_total", 1)
        raise HTTPException(status_code=504, detail="Request timed out")
    except RuntimeError as e:
        if str(e) == "busy":
            raise HTTPException(status_code=503, detail="Server busy")
        raise
    except Exception:
        metrics.inc("request_failures_total", 1)
        logger.exception("Inference failed")  # <-- prints stacktrace to terminal
        raise HTTPException(status_code=500, detail=str(e))

    total_ms = (time.perf_counter() - t0) * 1000.0
    metrics.observe_latency_ms(total_ms)
    metrics.inc("responses_total", 1)

    return {
        "backend": out.backend,
        "batch_latency_ms": out.latency_ms,
        "total_latency_ms": total_ms,
        "image_size": {"w": img.width, "h": img.height},
        "detections": out.detections[0],
    }

def _sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

async def _process_one_image(asset_id: str, image_id: str, path: str) -> None:
    try:
        img = Image.open(path).convert("RGB")

        delay = 0.05
        for attempt in range(8):  # ~ bounded waiting
            try:
                out = await asyncio.wait_for(
                    batcher.submit(img),
                    timeout=max(10.0, settings.request_timeout_s),
                )
                db.set_image_result(image_id, "done", out.backend, out.latency_ms, out.detections[0])
                db.update_asset_progress(asset_id, processed_delta=1, failed_delta=0)
                return
            except RuntimeError as e:
                if str(e) != "busy":
                    raise
                # queue full -> wait + retry
                await asyncio.sleep(delay + random.random() * 0.02)
                delay = min(delay * 2, 1.0)  # exponential backoff up to 1s

        # If still busy after retries, mark failed (or "queued_retry" if you add status)
        raise RuntimeError("busy_after_retries")

    except Exception:
        metrics.inc("asset_image_failures_total", 1)
        db.set_image_result(image_id, "failed", "n/a", 0.0, [])
        db.update_asset_progress(asset_id, processed_delta=0, failed_delta=1)

@app.post("/v1/assets")
async def create_asset(
    files: List[UploadFile] = File(...),
    idempotency_key: Optional[str] = Header(default=None, alias="Idempotency-Key"),
) -> dict:
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")

    if idempotency_key:
        existing = db.get_asset_by_idempotency(idempotency_key)
        if existing:
            return {"asset_id": existing["asset_id"], "status": existing["status"], "idempotency_reused": True}

    asset_id = str(uuid.uuid4())
    now = time.time()
    db.insert_asset(asset_id, created_at=now, expected_images=len(files), idempotency_key=idempotency_key)

    for f in files:
        if f.content_type not in ("image/jpeg", "image/png", "image/webp"):
            continue

        data = await f.read()
        sha = _sha256_bytes(data)
        try:
            img = Image.open(io.BytesIO(data))
            w, h = img.width, img.height
        except Exception:
            continue

        image_id = str(uuid.uuid4())
        asset_dir = os.path.join(settings.image_store_dir, asset_id)
        os.makedirs(asset_dir, exist_ok=True)
        filename = f.filename or f"{image_id}.jpg"
        path = os.path.join(asset_dir, filename)
        with open(path, "wb") as out:
            out.write(data)

        db.insert_image(image_id, asset_id, filename, path, sha, w, h, created_at=time.time())
        asyncio.create_task(_process_one_image(asset_id, image_id, path))

    return {"asset_id": asset_id, "status": "pending"}

@app.get("/v1/assets/{asset_id}")
def get_asset(asset_id: str) -> dict:
    a = db.get_asset(asset_id)
    if not a:
        raise HTTPException(status_code=404, detail="Asset not found")
    imgs = db.list_images_for_asset(asset_id)
    return {
        "asset_id": asset_id,
        "status": a["status"],
        "expected_images": a["expected_images"],
        "processed_images": a["processed_images"],
        "failed_images": a["failed_images"],
        "images": imgs,
    }

@app.get("/v1/assets/{asset_id}/report")
def asset_report(asset_id: str) -> dict:
    a = db.get_asset(asset_id)
    if not a:
        raise HTTPException(status_code=404, detail="Asset not found")
    imgs = db.list_images_for_asset(asset_id)

    counts = {}
    conf_sum = {}
    conf_cnt = {}

    for im in imgs:
        dets = im.get("detections") or []
        for d in dets:
            name = d.get("label_name") or str(d.get("label"))
            counts[name] = counts.get(name, 0) + 1
            conf_sum[name] = conf_sum.get(name, 0.0) + float(d.get("score", 0.0))
            conf_cnt[name] = conf_cnt.get(name, 0) + 1

    avg_conf = {k: (conf_sum[k] / conf_cnt[k]) if conf_cnt.get(k) else 0.0 for k in counts.keys()}

    return {
        "asset_id": asset_id,
        "status": a["status"],
        "counts_by_label": counts,
        "avg_confidence_by_label": avg_conf,
    }

@app.get("/v1/assets/{asset_id}/export")
def export_asset(asset_id: str, format: str = "json") -> Response:
    a = db.get_asset(asset_id)
    if not a:
        raise HTTPException(status_code=404, detail="Asset not found")
    imgs = db.list_images_for_asset(asset_id)

    if format == "json":
        import json
        body = json.dumps({"asset": a, "images": imgs}, indent=2)
        return Response(content=body, media_type="application/json")

    if format == "csv":
        import csv
        import io as _io
        buf = _io.StringIO()
        w = csv.writer(buf)
        w.writerow(["asset_id","image_id","filename","backend","latency_ms","label","label_name","score","x1","y1","x2","y2"])
        for im in imgs:
            dets = im.get("detections") or []
            for d in dets:
                x1,y1,x2,y2 = d["box"]
                w.writerow([asset_id, im["image_id"], im["filename"], im.get("backend"), im.get("latency_ms"),
                            d.get("label"), d.get("label_name"), d.get("score"), x1,y1,x2,y2])
        return Response(content=buf.getvalue(), media_type="text/csv")

    raise HTTPException(status_code=400, detail="format must be json or csv")
