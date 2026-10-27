"""Qualitative comparison: ONE held-out sample (same index in every dataset), every method, mu_a and mu_s', one figure per aspect ratio.
No numbers are written on the images; the sample properties and the per-method numbers go to tables (CSV files read by the document builder)."""
import os, numpy as np, pandas as pd
from scipy.io import loadmat
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ARS = ["1.06", "1.10", "1.16", "1.22"]
METHODS = ["CAFNet", "FineTuned", "Tent", "AdaBN", "AdaptNet", "CoDA"]
LABEL = {"CAFNet": "CAFNet", "FineTuned": "Fine-Tuned", "Tent": "Tent", "AdaBN": "AdaBN", "AdaptNet": "AdaptNet", "CoDA": "CoDA"}

# ---- sample choice (rule, not results): two well separated inclusions, both contrasts >= 2, frequency closest to the median of such samples
H22 = pd.read_csv(os.path.join(HERE, "sample_properties_heldout25_AR_1.22.csv"))
two = H22[H22.n_inclusions == 2].copy()
cx = lambda r, th, roc: roc * np.cos(np.deg2rad(th))
cy = lambda r, th, roc: roc * np.sin(np.deg2rad(th))
two["sep"] = np.hypot(cx(0, two.theta1_deg, two.roc1_mm) - cx(0, two.theta2_deg, two.roc2_mm), cy(0, two.theta1_deg, two.roc1_mm) - cy(0, two.theta2_deg, two.roc2_mm))
cand = two[(two.sep > two.r1_mm + two.r2_mm + 4) & (two[["ca1", "ca2", "csp1", "csp2"]].min(axis=1) >= 2.0)]
med = cand.frequency_MHz.median()
POS = int(cand.iloc[(cand.frequency_MHz - med).abs().argsort().iloc[0]].heldout_position)
print("candidates:", cand.heldout_position.tolist(), "chosen position:", POS)

rows = []
for ar in ARS:
    run = os.path.join(HERE, "runs", "R_" + ar, "results")
    gt = np.load(os.path.join(run, "gt_held.npy"))
    mask = loadmat(os.path.join(HERE, "runs", "R_" + ar, "samples_500.mat"), squeeze_me=True)["mask"].astype(bool)
    preds = {m: np.load(os.path.join(run, "pred_%s.npy" % m)) for m in METHODS}
    wb = {m: pd.read_excel(os.path.join(run, "%s_%s.xlsx" % (m, ar)), header=[0, 1], index_col=0) for m in METHODS}
    extra = {m: pd.read_csv(os.path.join(run, "extra_%s_%s.csv" % (m, ar))) for m in METHODS}
    fig, axes = plt.subplots(2, 1 + len(METHODS), figsize=(2.1 * (1 + len(METHODS)), 4.9))
    for r, (k, pc, name) in enumerate(((0, "mu_a", "μa"), (1, "mu_sp", "μs′"))):
        g = gt[POS, k]
        lo, hi = g[mask].min(), g[mask].max()
        for c in range(1 + len(METHODS)):
            ax = axes[r, c]
            img = g if c == 0 else preds[METHODS[c - 1]][POS, k]
            im = ax.imshow(np.where(mask, img, np.nan), vmin=lo, vmax=hi, cmap="inferno")
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title("Ground truth" if c == 0 else LABEL[METHODS[c - 1]], fontsize=10)
            if c == 0:
                ax.set_ylabel(name, fontsize=11)
        fig.colorbar(im, ax=axes[r, :], fraction=0.012, pad=0.01)
    fig.suptitle("Aspect ratio %s, held-out sample %d (colour limits = ground-truth range of each row)" % (ar, POS), fontsize=10)
    os.makedirs(os.path.join(HERE, "figures"), exist_ok=True)
    fig.savefig(os.path.join(HERE, "figures", "qual_AR_%s.png" % ar), dpi=160, bbox_inches="tight")
    plt.close(fig)
    for m in METHODS:
        rec = dict(AR=ar, heldout_position=POS, method=m)
        for k, pc, pn in ((0, "mu_a", "a"), (1, "mu_sp", "s")):
            rec["ssim_" + pn] = float(wb[m][("ssim", pc)].iloc[POS])
            rec["csd_" + pn] = float(wb[m][("R_csd", pc)].iloc[POS])
            rec["max_est_" + pn] = float(preds[m][POS, k][mask].max())
            rec["rmse_" + pn] = float(extra[m]["rmse_" + pn].iloc[POS]) * (1e3 if pn == "a" else 1.0)
        rec["gt_max_a"] = float(gt[POS, 0][mask].max()); rec["gt_max_s"] = float(gt[POS, 1][mask].max())
        rows.append(rec)
pd.DataFrame(rows).to_csv(os.path.join(HERE, "qualitative_sample_numbers.csv"), index=False)
pd.DataFrame([dict(heldout_position=POS)]).to_csv(os.path.join(HERE, "qualitative_samples.csv"), index=False)
print("done")
