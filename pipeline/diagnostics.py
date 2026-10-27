"""Diagnostics for 'CSD falls while SSIM rises': SSIM ceiling of an inclusion-free map, relative-contrast ratio, CRC, output-BN gamma."""
import os, sys, numpy as np, pandas as pd
from scipy import ndimage as ndi
from scipy.io import loadmat
from skimage.metrics import structural_similarity as ssim
HERE = os.path.dirname(os.path.abspath(__file__))
ars = sys.argv[1:] or ["1.06", "1.10", "1.16", "1.22"]
methods = ["CAFNet", "FineTuned", "Tent", "AdaBN", "AdaptNet", "CoDA"]
bn = pd.read_csv(os.path.join(HERE, "bn_output_params.csv"), dtype={"AR": str}) if os.path.exists(os.path.join(HERE, "bn_output_params.csv")) else None
rows, ceil_rows = [], []
for ar in ars:
    run = os.path.join(HERE, "runs", "R_" + ar)
    gt = np.load(os.path.join(run, "results", "gt_held.npy"))
    mask = loadmat(os.path.join(run, "samples_500.mat"), squeeze_me=True)["mask"].astype(bool)
    reg = []
    for i in range(len(gt)):
        per = {}
        for k in (0, 1):
            g = gt[i, k]
            bg = g[mask].min()
            roi = mask & (g > bg * 1.05)
            bgm = mask & (g <= bg * 1.001) & ~ndi.binary_dilation(roi, iterations=2)
            per[k] = (bg, roi, bgm)
        reg.append(per)
    # ceiling: exact background, no inclusion
    for k, p in ((0, "a"), (1, "s")):
        c, frac = [], []
        for i in range(len(gt)):
            g = gt[i, k]; bg, roi, bgm = reg[i][k]
            if roi.sum() == 0:
                continue
            e = g.copy(); e[roi] = bg
            c.append(ssim(g, e, data_range=max(g.max(), e.max()) - min(g.min(), e.min()), gaussian_weights=True))
            frac.append(roi.sum() / mask.sum())
        ceil_rows.append(dict(AR=ar, prop=p, ssim_ceiling=np.mean(c), incl_fraction=np.mean(frac)))
    for m in methods:
        f = os.path.join(run, "results", "pred_%s.npy" % m)
        wbp = os.path.join(run, "results", "%s_%s.xlsx" % (m, ar))
        if not (os.path.exists(f) and os.path.exists(wbp)):
            continue
        pr = np.load(f)
        w = pd.read_excel(wbp, header=[0, 1], index_col=0)
        for k, p, pc in ((0, "a", "mu_a"), (1, "s", "mu_sp")):
            ratios, crcs, peaks = [], [], []
            for i in range(len(gt)):
                bg, roi, bgm = reg[i][k]
                if roi.sum() == 0 or bgm.sum() < 10:
                    continue
                g, e = gt[i, k], pr[i, k]
                relg = (np.percentile(g[roi], 90) - np.percentile(g[bgm], 10)) / np.percentile(g[bgm], 10)
                q10 = np.percentile(e[bgm], 10)
                rele = (np.percentile(e[roi], 90) - q10) / q10 if q10 != 0 else np.nan
                ratios.append(rele / relg)
                crcs.append((e[roi].mean() - e[bgm].mean()) / (g[roi].mean() - g[bgm].mean()))
            csd = pd.to_numeric(w[("R_csd", pc)], errors="coerce")
            ratios = np.asarray(ratios, float)
            r = dict(AR=ar, method=m, prop=p, ssim=pd.to_numeric(w[("ssim", pc)], errors="coerce").mean(),
                     csd_mean=csd.mean(), csd_median=csd.median(), rel_contrast_ratio_median=np.nanmedian(ratios),
                     frac_overshoot=np.nanmean(ratios > 1.0), crc_mean=np.mean(crcs))
            if bn is not None:
                b = bn[(bn.AR == ar) & (bn.method == m)]
                if len(b):
                    r["gamma"] = float(b.iloc[0]["gamma_" + p]); r["beta"] = float(b.iloc[0]["beta_" + p])
            rows.append(r)
pd.DataFrame(rows).to_csv(os.path.join(HERE, "diagnostics.csv"), index=False)
pd.DataFrame(ceil_rows).to_csv(os.path.join(HERE, "ssim_ceiling.csv"), index=False)
print(pd.DataFrame(ceil_rows).round(3).to_string(index=False))
print(pd.DataFrame(rows).query("prop=='a'").round(3).to_string(index=False))
