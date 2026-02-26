import argparse
from pathlib import Path
import torch
from transformers import ConditionalDetrForObjectDetection

class Wrapper(torch.nn.Module):
    def __init__(self, model: ConditionalDetrForObjectDetection):
        super().__init__()
        self.model = model

    def forward(self, pixel_values, pixel_mask):
        out = self.model(pixel_values=pixel_values, pixel_mask=pixel_mask)
        return out.logits, out.pred_boxes

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--onnx-out", required=True)
    ap.add_argument("--img", type=int, default=640)
    args = ap.parse_args()

    model = ConditionalDetrForObjectDetection.from_pretrained(args.model_dir)
    model.eval()

    dummy = torch.randn(1, 3, args.img, args.img, dtype=torch.float32)
    dummy_mask = torch.ones(1, args.img, args.img, dtype=torch.int64)

    wrapper = Wrapper(model)

    out_path = Path(args.onnx_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        wrapper,
        (dummy, dummy_mask),
        str(out_path),
        opset_version=17,
        input_names=["pixel_values", "pixel_mask"],
        output_names=["logits", "pred_boxes"],
        dynamic_axes={
            "pixel_values": {0: "batch", 2: "height", 3: "width"},
            "pixel_mask": {0: "batch", 1: "height", 2: "width"},
            "logits": {0: "batch"},
            "pred_boxes": {0: "batch"},
        },
    )
    print("Exported ONNX:", out_path)

if __name__ == "__main__":
    main()
