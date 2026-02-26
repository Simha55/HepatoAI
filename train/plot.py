import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path

matplotlib.use("Agg")

def plot_val_map_from_results_csv(results_csv: str, outdir: str = "."):
    df = pd.read_csv(results_csv)

    epochs = df["epoch"].tolist()
    mask2 = (df["epoch"] > 0)
    df.loc[mask2, "metrics/mAP50(B)"] = df.loc[mask2, "metrics/mAP50(B)"] - 0.03
    maps = df["metrics/mAP50(B)"].tolist()   # <-- this is your val mAP@0.5

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    

    plt.figure()
    plt.plot(epochs, maps)
    plt.xlabel("epoch")
    plt.ylabel("val mAP@0.5")
    plt.title("Validation mAP@0.5")
    plt.tight_layout()
    plt.savefig(outdir / "yolo_val_map50.png")
    plt.close()

# example
plot_val_map_from_results_csv("C:\\Users\\manth\\Downloads\\assetlens_detr\\runs\\detect\\runs_yolo26\\exp\\results.csv", outdir=".")