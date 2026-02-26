# ---- Model selection ----
$env:MODEL_TYPE="yolo"
$env:MODEL_DIR="weights/conditional_detr_coco_ft"
$env:USE_ONNX="1"                  # enable ONNX path usage
$env:ONNX_PATH="weights/yolo/yolo.onnx"
$env:YOLO_IMGSZ="320"
$env:SCORE_THRESHOLD="0.25"
$env:NMS_IOU_THRESHOLD="0.45"
$env:CLASS_NAMES="diseases,ballooning,fibrosis,inflammation,steatosis"

# ---- Microbatching ----
$env:MAX_BATCH_SIZE="32"            # set 1 if ONNX isn't dynamic batch
$env:MAX_WAIT_MS="10"
$env:QUEUE_CAPACITY="256"
$env:REQUEST_TIMEOUT_S="10"

# ---- Optional GPU preference ----
$env:PREFER_GPU="1"

# ---- Start server ----
uvicorn app.main:app --host 0.0.0.0 --port 8000