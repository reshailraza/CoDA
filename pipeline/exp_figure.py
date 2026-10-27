"""Experimental phantom: ground truth and every method applied to CAFNet (one colour scale per row = ground-truth range)."""
import os, numpy as np, pandas as pd
from scipy.io import loadmat
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__))
run = os.path.join(HERE, "runs", "EXP", "results")
d = loadmat(os.path.join(os.path.dirname(HERE), "mask1.22", "landseed_processed__new_data.mat"), squeeze_me=True)
mask = loadmat(os.path.join(HERE, "runs", "R_1.22", "samples_500.mat"), squeeze_me=True)["mask"].astype(bool)
gt = np.stack([np.asarray(d["Grid_MUA"], float), np.asarray(d["Grid_MUSP"], float)])
cols = [("Ground truth", None), ("CAFNet", "CAFNet__noadaptation"), ("Fine-Tuned", "Fine-Tuned__AR1.22labeledmodel"), ("AdaBN", "AdaBN__poolK=64,jitter0.05"),
        ("Tent", "Tent__poolK=64,jitter0.05"), ("CoDA", "CoDA__poolK=64,jitter0.05"), ("AdaptNet", "AdaptNet__per-sample(native)")]
fig, axes = plt.subplots(2, len(cols), figsize=(2.3 * len(cols), 5.2))
for r, (k, name) in enumerate(((0, "μa"), (1, "μs′"))):
    lo, hi = gt[k][mask].min(), gt[k][mask].max()
    for c, (t, f) in enumerate(cols):
        img = gt[k] if f is None else np.load(os.path.join(run, "pred_%s.npy" % f))[k]
        ax = axes[r, c]
        im = ax.imshow(np.where(mask, img, np.nan), vmin=lo, vmax=hi, cmap="inferno")
        ax.set_xticks([]); ax.set_yticks([])
        if r == 0: ax.set_title(t, fontsize=10)
        if c == 0: ax.set_ylabel(name, fontsize=11)
    fig.colorbar(im, ax=axes[r, :], fraction=0.012, pad=0.01)
fig.suptitle("Experimental phantom (mask1.22/landseed_processed__new_data.mat): every method applied directly to CAFNet; colour limits = ground-truth range", fontsize=9)
fig.savefig(os.path.join(HERE, "figures", "experimental_reconstructions.png"), dpi=170, bbox_inches="tight")
print("saved")
