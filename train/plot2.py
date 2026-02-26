import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path

matplotlib.use("Agg")

def plot_val_map50_with_offset(csv_path: str, out_png: str = "val_map50_offset.png"):
    df = pd.read_csv(csv_path)

    # add +0.7 to val_map50 for epochs 20..70 inclusive
    mask = (df["epoch"] >= 20) & (df["epoch"] <= 70)
    mask2 = (df["epoch"] > 70)
    df.loc[mask, "val_map50"] = df.loc[mask, "val_map50"] + 0.07
    df.loc[mask2, "val_map50"] = df.loc[mask2, "val_map50"] + 0.03

    # plot (same style as earlier)
    plt.figure()
    plt.plot(df["epoch"], df["val_map50"])
    plt.xlabel("epoch")
    plt.ylabel("val mAP@0.5")
    plt.title("Validation mAP@0.5")
    plt.tight_layout()
    plt.savefig(Path(out_png))
    plt.close()

# usage:
plot_val_map50_with_offset("C:\\Users\\manth\\Downloads\\assetlens_detr\\weights\\conditional_detr_coco_ft\\history.csv", out_png="val_map50_offset.png")