from dataclasses import dataclass
import os


@dataclass(frozen=True)
class Settings:
    # Which model pipeline to use: "detr" or "yolo"
    model_type: str = os.getenv("MODEL_TYPE", "detr").lower()

    # Shared / default paths (meaning depends on model_type)
    # - detr: model_dir is HF dir, onnx_path is detr onnx
    # - yolo: onnx_path is yolo onnx
    model_dir: str = os.getenv("MODEL_DIR", "weights/detr_voc_ft")
    onnx_path: str = os.getenv("ONNX_PATH", "weights/detr_voc_ft/model.onnx")

    # Runtime backend toggles
    # In DETR mode: if use_onnx=1, ORT path is used (preferred when GPU healthy)
    # In YOLO mode: Ultralytics ONNX is used regardless; this flag can be ignored safely
    use_onnx: bool = os.getenv("USE_ONNX", "0") == "1"
    prefer_gpu: bool = os.getenv("PREFER_GPU", "1") == "1"

    # ✅ Class names for your dataset (used for label_name in API + reports)
    # Order must match training class IDs: 0..3
    class_names: str = os.getenv(
        "CLASS_NAMES",
        "ballooning,fibrosis,inflammation,steatosis",
    )

    # DETR preprocessing (only used when MODEL_TYPE=detr)
    image_short_side: int = int(os.getenv("IMAGE_SHORT_SIDE", "320"))
    image_max_side: int = int(os.getenv("IMAGE_MAX_SIDE", "1024"))

    # YOLO preprocessing (only used when MODEL_TYPE=yolo)
    # Must match your YOLO export/eval imgsz (you used 320)
    yolo_imgsz: int = int(os.getenv("YOLO_IMGSZ", "320"))
    nms_iou_threshold: float = float(os.getenv("NMS_IOU_THRESHOLD", "0.45"))

    # Shared detection threshold
    # Note: YOLO commonly uses 0.25; DETR often uses 0.5. You may tune per mode.
    score_threshold: float = float(os.getenv("SCORE_THRESHOLD", "0.5"))

    # Microbatching
    max_batch_size: int = int(os.getenv("MAX_BATCH_SIZE", "8"))
    max_wait_ms: int = int(os.getenv("MAX_WAIT_MS", "10"))

    queue_capacity: int = int(os.getenv("QUEUE_CAPACITY", "256"))
    request_timeout_s: float = float(os.getenv("REQUEST_TIMEOUT_S", "2.5"))

    # Storage (assets pipeline)
    sqlite_path: str = os.getenv("SQLITE_PATH", "data_db/assetlens.db")
    image_store_dir: str = os.getenv("IMAGE_STORE_DIR", "data_db/images")


settings = Settings()