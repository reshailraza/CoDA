"""RMSE, NRMSE, CNR, localization error (plus diagnostics used to explain CSD vs SSIM) from saved predictions.

Definitions (per sample, per property, inside the dataset's elliptical mask; grid 64x64, major axis = 64 px = d mm):
  RMSE      sqrt(mean((est - gt)^2)) over the mask, in mm^-1 (mu_a values are ~1e-3, so reported x1e-3)
  NRMSE     RMSE / (max(gt) - min(gt)) in percent
  ROI       gt pixels more than 5% above the background value (background = min of gt inside the mask); blobs < 5 px dropped
  BG        mask pixels with gt equal to the background, at least 2 px away from the ROI
  CNR       (mean_est[ROI] - mean_est[BG]) / sqrt(var_est[ROI] + var_est[BG])   (signed: negative = inverted contrast)
  CRC       (mean_est[ROI] - mean_est[BG]) / (mean_gt[ROI] - mean_gt[BG])        (contrast recovery coefficient, 1 = exact)
  Loc. err. for each ground-truth inclusion (connected ROI blob), distance from its centroid to the nearest centroid of the
            connected components of est > background_est + 0.5 (max_est - background_est), background_est = median(est in mask);
            mean over the inclusions of the sample; pixels and mm (mm = px * d / 64)
  rmse_bg / rmse_roi / bg_bias  error inside BG / ROI and mean(est[BG]) - gt background (diagnostics)
"""
import os
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from scipy.io import loadmat

HERE = os.path.dirname(os.path.abspath(__file__))
PROPS = ["a", "s"]


def components(binary, min_area):
    lab, n = ndi.label(binary)
    cents = []
    for i in range(1, n + 1):
        m = lab == i
        if m.sum() >= min_area:
            cents.append(np.array(ndi.center_of_mass(m)))
    return cents


def one(g, e, mask, mm_per_px=np.nan):
    r = {}
    gm, em = g[mask], e[mask]
    rmse = float(np.sqrt(np.mean((em - gm) ** 2)))
    r["rmse"] = rmse
    r["nrmse"] = 100.0 * rmse / (gm.max() - gm.min())
    bg_val = gm.min()
    roi = mask & (g > bg_val * 1.05)
    lab, n = ndi.label(roi)
    keep = [i for i in range(1, n + 1) if (lab == i).sum() >= 5]
    roi = np.isin(lab, keep) if keep else np.zeros_like(roi)
    gt_cents = [np.array(ndi.center_of_mass(lab == i)) for i in keep]
    bg = mask & (g <= bg_val * 1.001) & ~ndi.binary_dilation(roi, iterations=2)
    for k in ("cnr", "crc", "loc_px", "loc_mm", "rmse_roi", "rmse_bg", "bg_bias", "roi_mean_ratio", "n_incl"):
        r[k] = np.nan
    r["n_incl"] = len(gt_cents)
    if bg.sum() > 10:
        r["rmse_bg"] = float(np.sqrt(np.mean((e[bg] - g[bg]) ** 2)))
        r["bg_bias"] = float(e[bg].mean() - bg_val)
    if roi.sum() > 0 and bg.sum() > 10:
        mr, mb = e[roi].mean(), e[bg].mean()
        r["cnr"] = float((mr - mb) / np.sqrt(e[roi].var() + e[bg].var() + 1e-30))
        r["crc"] = float((mr - mb) / (g[roi].mean() - g[bg].mean()))
        r["rmse_roi"] = float(np.sqrt(np.mean((e[roi] - g[roi]) ** 2)))
        r["roi_mean_ratio"] = float(e[roi].mean() / g[roi].mean())
        bg_e = np.median(em)
        thr = bg_e + 0.5 * (em.max() - bg_e)
        est_c = components((e > thr) & mask, 3)
        if est_c:
            dist = [min(np.linalg.norm(c - ec) for ec in est_c) for c in gt_cents]
            r["loc_px"] = float(np.mean(dist))
            r["loc_mm"] = r["loc_px"] * mm_per_px
    return r


def per_sample(gt, pred, mask, d_mm):
    rows = []
    for i in range(len(gt)):
        row = {"sample": i}
        for k, p in enumerate(PROPS):
            res = one(gt[i, k], pred[i, k], mask, d_mm[i] / 64.0 if d_mm is not None else np.nan)
            for kk, v in res.items():
                row["%s_%s" % (kk, p)] = v
        rows.append(row)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    import sys
    ars = sys.argv[1:] or ["1.06", "1.10", "1.16", "1.22"]
    methods = ["CAFNet", "FineTuned", "Tent", "AdaBN", "AdaptNet", "CoDA"]
    for ar in ars:
        run = os.path.join(HERE, "runs", "R_" + ar)
        gt = np.load(os.path.join(run, "results", "gt_held.npy"))
        held = np.load(os.path.join(run, "results", "split.npz"))["held"]
        mask = loadmat(os.path.join(run, "samples_500.mat"), squeeze_me=True)["mask"].astype(bool)
        d_all = np.asarray(loadmat(os.path.join(run, "param_samples_500.mat"), squeeze_me=True)["d"], float)
        for m in methods:
            f = os.path.join(run, "results", "pred_%s.npy" % m)
            if not os.path.exists(f):
                continue
            df = per_sample(gt, np.load(f), mask, d_all[held])
            df.to_csv(os.path.join(run, "results", "extra_%s_%s.csv" % (m, ar)), index=False)
            print(ar, m, "done", df[["rmse_a", "cnr_a", "loc_px_a"]].mean().round(4).to_dict())
