# AssetLens (DETR on VOC) — 4-day interview project

## What you get
- Fine-tune DETR on VOC2007 (transfer learning) + baseline evaluation + curves
- FastAPI backend with:
  - `/detect` (sync) used by a simple upload UI at `/`
  - `/v1/assets` workflow (upload many photos → stored detections → report/export)
  - micro-batching, backpressure, timeouts, metrics, GPU health breaker, CPU fallback
- ONNX export + ONNX Runtime toggle for latency comparisons
- HTTP benchmark script to print p50/p95/RPS

## Setup
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## VOC2007
Place dataset like:
```
data/VOCdevkit/VOC2007/JPEGImages
data/VOCdevkit/VOC2007/Annotations
data/VOCdevkit/VOC2007/ImageSets/Main
```

Set:
```bash
export VOC_ROOT=./data/VOCdevkit/VOC2007
```

## Day 1 baseline
```bash
python train/eval_voc.py --model facebook/detr-resnet-50 --split test
```

## Day 1–2 fine-tune
```bash
python train/finetune_voc.py --outdir weights/detr_voc_ft --epochs 30 --batch 2 --img 640 --freeze_backbone_epochs 10
```

Evaluate best checkpoint on test:
```bash
python train/eval_voc.py --model weights/detr_voc_ft --split test
```

## Export ONNX
```bash
python scripts/export_onnx.py --model-dir weights/detr_voc_ft --onnx-out weights/detr_voc_ft/model.onnx --img 640
```

## Run backend + UI
```bash
export MODEL_DIR=weights/detr_voc_ft
export ONNX_PATH=weights/detr_voc_ft/model.onnx
export USE_ONNX=0  # set 1 to use ONNX Runtime
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Open:
- http://127.0.0.1:8000
- http://127.0.0.1:8000/docs

## Benchmark
```bash
python scripts/bench_http.py --url http://127.0.0.1:8000/detect --image path/to/sample.jpg --concurrency 20 --requests 200
```
