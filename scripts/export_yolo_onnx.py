# export_yolo_onnx.py
from __future__ import annotations

import argparse
from pathlib import Path

def main() -> None:
    ap = argparse.ArgumentParser(description="Export a trained Ultralytics YOLO .pt model to ONNX")
    ap.add_argument("--weights", required=True, help="Path to YOLO weights .pt (e.g., runs/detect/exp/weights/best.pt)")
    ap.add_argument("--onnx-out", required=True, help="Output ONNX path (e.g., weights/yolo/yolo.onnx)")
    ap.add_argument("--imgsz", type=int, default=640, help="Export image size (square). Common: 320/640")
    ap.add_argument("--opset", type=int, default=17, help="ONNX opset version (17 is a good default)")
    ap.add_argument("--dynamic", action="store_true", help="Dynamic batch/height/width (more flexible, sometimes a bit slower)")
    ap.add_argument("--half", action="store_true", help="Export FP16 (best when deploying on GPU; requires CUDA for export)")
    ap.add_argument("--simplify", action="store_true", help="Run onnxsim to simplify graph (requires onnxsim)")
    ap.add_argument("--device", default="cpu", help="Export device: cpu, 0, 1, ... (gpu id). Example: --device 0")
    args = ap.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError as e:
        raise SystemExit(
            "Ultralytics not installed. Install with:\n"
            "  pip install ultralytics onnx onnxruntime\n"
            "Optional (for --simplify):\n"
            "  pip install onnxsim\n"
        ) from e

    weights = Path(args.weights)
    if not weights.exists():
        raise SystemExit(f"weights not found: {weights}")

    out_path = Path(args.onnx_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Ultralytics decides the output name automatically (usually best.onnx next to weights),
    # so we export to a temp folder then move/rename to --onnx-out.
    tmp_dir = out_path.parent / "_tmp_export"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(weights))

    exported = model.export(
        format="onnx",
        imgsz=args.imgsz,
        opset=args.opset,
        dynamic=args.dynamic,
        half=args.half,
        simplify=args.simplify,
        device=args.device,
        # Ultralytics exports into the same directory as weights by default;
        # setting 'project'/'name' keeps it controlled.
        project=str(tmp_dir),
        name="yolo_onnx",
    )

    # Ultralytics returns the exported path as a string (or similar).
    exported_path = Path(str(exported))
    if not exported_path.exists():
        # Fallback: search the tmp directory for an .onnx
        candidates = list(tmp_dir.rglob("*.onnx"))
        if not candidates:
            raise SystemExit("Export completed but ONNX file not found. Check Ultralytics logs/output.")
        exported_path = candidates[0]

    # Move/rename to requested location
    if out_path.exists():
        out_path.unlink()
    exported_path.replace(out_path)

    print(f"Exported YOLO ONNX: {out_path}")

    # Quick sanity check
    try:
        import onnx
        onnx.checker.check_model(onnx.load(str(out_path)))
        print("ONNX check: OK")
    except Exception as e:
        print(f"ONNX check: WARNING ({e})")
        print("You can still try loading it in onnxruntime, but verify inference parity.")

if __name__ == "__main__":
    main()
