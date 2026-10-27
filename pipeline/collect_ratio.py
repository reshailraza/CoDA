"""Merge per-sample CSD/SSIM workbooks with RMSE/CNR/localization CSVs for the ratio_* datasets; summaries, paired tests, boxplots."""
import os, warnings
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
ARS = ["1.06", "1.10", "1.16", "1.22"]
METHODS = ["CAFNet", "FineTuned", "Tent", "AdaBN", "AdaptNet", "CoDA"]
LABEL = {"CAFNet": "CAFNet", "FineTuned": "Fine-Tuned", "Tent": "Tent", "AdaBN": "AdaBN", "AdaptNet": "AdaptNet", "CoDA": "CoDA"}
COLOR = {"CAFNet": "#9e9e9e", "FineTuned": "#1f77b4", "Tent": "#ff7f0e", "AdaBN": "#2ca02c", "AdaptNet": "#8c564b", "CoDA": "#d62728"}
# (column suffix in merged frame, title, scale factor for mu_a RMSE)
PROP = {"a": "μa", "s": "μs′"}
METRICS = [("ssim", "SSIM"), ("csd", "CSD"), ("rmse", "RMSE"), ("nrmse", "NRMSE (%)"), ("cnr", "CNR"), ("loc_mm", "Loc. error (mm)")]
SUMMARY_METRICS = METRICS + [("crc", "CRC"), ("rmse_bg", "BG RMSE"), ("rmse_roi", "ROI RMSE"), ("bg_bias", "BG bias"), ("loc_px", "Loc. error (px)")]

frames = []
for ar in ARS:
    for m in METHODS:
        wb = os.path.join(HERE, "runs", "R_" + ar, "results", "%s_%s.xlsx" % (m, ar))
        ex = os.path.join(HERE, "runs", "R_" + ar, "results", "extra_%s_%s.csv" % (m, ar))
        if not (os.path.exists(wb) and os.path.exists(ex)):
            continue
        w = pd.read_excel(wb, header=[0, 1], index_col=0)
        e = pd.read_csv(ex)
        d = pd.DataFrame({"AR": ar, "method": m, "sample": range(len(w))})
        for p, pc in (("a", "mu_a"), ("s", "mu_sp")):
            d["ssim_" + p] = pd.to_numeric(w[("ssim", pc)], errors="coerce").to_numpy()
            d["csd_" + p] = pd.to_numeric(w[("R_csd", pc)], errors="coerce").to_numpy()
            d["rcont_" + p] = pd.to_numeric(w[("R_cont", pc)], errors="coerce").to_numpy()
            d["rsize_" + p] = pd.to_numeric(w[("R_size", pc)], errors="coerce").to_numpy()
        for c in e.columns:
            if c != "sample":
                d[c] = e[c].to_numpy()
        for p in ("a", "s"):
            d["rmse_" + p] = d["rmse_" + p] * (1e3 if p == "a" else 1.0)   # mu_a in 1e-3 mm^-1
        frames.append(d)
allm = pd.concat(frames, ignore_index=True)
allm.to_csv(os.path.join(HERE, "ratio_per_sample.csv"), index=False)

# ---- summary: mean, sd, ci, median, iqr, n
rows = []
for (ar, m), g in allm.groupby(["AR", "method"]):
    for key, title in SUMMARY_METRICS:
        for p in ("a", "s"):
            col = "%s_%s" % (key, p)
            v = g[col].to_numpy(float)
            v = v[np.isfinite(v)]
            n = len(v)
            sd = v.std(ddof=1) if n > 1 else np.nan
            rows.append(dict(AR=ar, method=m, metric=key, prop=p, n=n, mean=v.mean() if n else np.nan, sd=sd,
                             ci_hw=stats.t.ppf(.975, n - 1) * sd / np.sqrt(n) if n > 1 else np.nan,
                             median=np.median(v) if n else np.nan, q25=np.percentile(v, 25) if n else np.nan, q75=np.percentile(v, 75) if n else np.nan))
summ = pd.DataFrame(rows)
summ.to_csv(os.path.join(HERE, "ratio_summary.csv"), index=False)

# ---- paired tests: reference method vs each other method
def paired(ref):
    pr = []
    for ar in ARS:
        for m in [x for x in METHODS if x != ref]:
            for key, title in METRICS:
                for p in ("a", "s"):
                    col = "%s_%s" % (key, p)
                    ra = allm[(allm.AR == ar) & (allm.method == ref)].sort_values("sample")
                    rb = allm[(allm.AR == ar) & (allm.method == m)].sort_values("sample")
                    if ra.empty or rb.empty:
                        continue
                    a = ra[col].to_numpy(float); b = rb[col].to_numpy(float)
                    ok = np.isfinite(a) & np.isfinite(b)
                    if ok.sum() < 5:
                        continue
                    dd = (a - b)[ok]
                    t, pv = stats.ttest_rel(a[ok], b[ok])
                    se = dd.std(ddof=1) / np.sqrt(len(dd))
                    pr.append(dict(AR=ar, vs=m, metric=key, prop=p, n=len(dd), mean_diff=dd.mean(),
                                   ci_lo=dd.mean() - stats.t.ppf(.975, len(dd) - 1) * se, ci_hi=dd.mean() + stats.t.ppf(.975, len(dd) - 1) * se, p=pv))
    pr = pd.DataFrame(pr)
    pr["p_holm"] = np.nan
    for (m, key), g in pr.groupby(["vs", "metric"]):
        pv = g.p.to_numpy()
        order = np.argsort(pv)
        adj = np.empty_like(pv)
        run = 0.0
        for rank, i in enumerate(order):
            run = max(run, min(1.0, (len(pv) - rank) * pv[i]))
            adj[i] = run
        pr.loc[g.index, "p_holm"] = adj
    return pr

pr = paired("CoDA")
pr.to_csv(os.path.join(HERE, "ratio_paired_tests.csv"), index=False)

# ---- boxplots: one figure per AR, rows = property, columns = metric
os.makedirs(os.path.join(HERE, "figures"), exist_ok=True)
CLIP = {"csd": (-2.5, 1.1)}
for ar in ARS:
    g = allm[allm.AR == ar]
    fig, axes = plt.subplots(2, len(METRICS), figsize=(3.0 * len(METRICS), 7.0))
    for r, p in enumerate(("a", "s")):
        for c, (key, title) in enumerate(METRICS):
            ax = axes[r, c]
            data = [g[g.method == m]["%s_%s" % (key, p)].dropna().to_numpy(float) for m in METHODS]
            bp = ax.boxplot(data, patch_artist=True, widths=0.65, showfliers=True, flierprops=dict(marker="o", markersize=2.5, alpha=0.6),
                            medianprops=dict(color="black", linewidth=1.3))
            for patch, m in zip(bp["boxes"], METHODS):
                patch.set_facecolor(COLOR[m]); patch.set_alpha(0.85)
            ax.set_xticks(range(1, len(METHODS) + 1))
            ax.set_xticklabels([LABEL[m] for m in METHODS], rotation=60, ha="right", fontsize=8)
            unit = {"rmse": (" (×10⁻³ mm⁻¹)" if p == "a" else " (mm⁻¹)")}.get(key, "")
            ax.set_title("%s %s%s" % (title, PROP[p], unit), fontsize=9)
            ax.grid(alpha=0.3)
            if key in CLIP:
                lo, hi = CLIP[key]
                ax.set_ylim(lo, hi)
                off = [int((d < lo).sum()) for d in data]
                for i, k in enumerate(off):
                    if k:
                        ax.annotate("↓%d" % k, (i + 1, lo), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=7, color="#333")
    fig.suptitle("Aspect ratio %s (N = 25 held-out samples per method)" % ar, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(os.path.join(HERE, "figures", "boxplots_AR_%s.png" % ar), dpi=170)
    plt.close(fig)
print("rows", len(allm), "summary", len(summ), "paired", len(pr))
print(summ[(summ.metric.isin(["ssim", "csd", "rmse", "cnr", "loc_mm"])) & (summ.prop == "a")].pivot_table(index=["AR", "method"], columns="metric", values="mean").round(3).reindex(METHODS, level=1).to_string())
