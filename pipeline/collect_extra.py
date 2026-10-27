"""Aggregate extra_analyses.py outputs into Summary tables and the sweep figure."""
import os, numpy as np, pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ARS = ["1.06", "1.10", "1.16", "1.22"]
OUT = os.path.join(HERE, "extra_summary"); os.makedirs(OUT, exist_ok=True)
R = lambda ar: os.path.join(HERE, "runs", "R_" + ar, "results")

# ---- A: statistics-weight sweep
A = pd.concat([pd.read_csv(os.path.join(R(ar), "sweepA_%s.csv" % ar)) for ar in ARS])
A.to_csv(os.path.join(OUT, "sweepA_statistics_weight.csv"), index=False)
print(A.groupby("n")[["w", "ssim_a", "csd_a", "rmse_a", "cnr", "loc_mm"]].mean().round(3).to_string())
fig, axes = plt.subplots(1, 4, figsize=(15, 3.3))
for ax, (c, t) in zip(axes, [("ssim_a", "SSIM $\\mu_a$"), ("rmse_a", "RMSE $\\mu_a$ ($10^{-3}$ mm$^{-1}$)"), ("cnr", "CNR"), ("loc_mm", "Localization error (mm)")]):
    for ar in ARS:
        g = A[A.AR == ar].sort_values("w")
        ax.plot(g.w, g[c], "o-", ms=4, label="AR " + ar)
    for w, lab in ((0.39, "Tent"), (0.63, "CoDA"), (1.0, "AdaBN")):
        ax.axvline(w, color="gray", ls=":", lw=1)
        ax.text(w, ax.get_ylim()[1], lab, ha="center", va="bottom", fontsize=7, color="gray")
    ax.set_xlabel("statistics weight $w=1-0.99^n$"); ax.set_title(t, fontsize=10, pad=14); ax.grid(alpha=.3)
axes[0].legend(fontsize=7)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "sweep_statistics_weight.png"), dpi=170, bbox_inches="tight")
plt.close(fig)

# ---- C: Tent / CoDA sensitivity
Cc = pd.concat([pd.read_csv(os.path.join(R(ar), "sweepC_%s.csv" % ar)) for ar in ARS])
Cc.to_csv(os.path.join(OUT, "sweepC_budget_lr.csv"), index=False)
print(Cc.groupby(["method", "lr", "n_iter"])[["ssim_a", "csd_a", "rmse_a", "cnr", "loc_mm"]].mean().round(3).to_string())

# ---- B: all-500 scoring
M = ["CAFNet", "AdaBN", "Tent", "CoDA", "CoDAemaOnly"]
rows, pairs = [], []
METR = ["ssim_a", "ssim_s", "csd_a", "csd_s", "rmse_a", "rmse_s", "cnr", "loc_mm"]
for ar in ARS:
    d = {m: pd.read_csv(os.path.join(R(ar), "all500_%s.csv" % m)) for m in M}
    for m in M:
        for c in METR:
            v = d[m][c].to_numpy(float)
            rows.append(dict(AR=ar, method=m, metric=c, n=int(np.isfinite(v).sum()), mean=np.nanmean(v), sd=np.nanstd(v, ddof=1), median=np.nanmedian(v)))
    for a, b in (("AdaBN", "CAFNet"), ("CoDA", "CAFNet"), ("Tent", "CAFNet"), ("CoDA", "AdaBN"), ("CoDA", "Tent"), ("AdaBN", "Tent"), ("CoDA", "CoDAemaOnly")):
        for c in METR:
            x, y = d[a][c].to_numpy(float), d[b][c].to_numpy(float)
            ok = np.isfinite(x) & np.isfinite(y); df = (x - y)[ok]
            se = df.std(ddof=1) / np.sqrt(len(df)); t = stats.t.ppf(0.975, len(df) - 1)
            pairs.append(dict(AR=ar, a=a, b=b, metric=c, n=len(df), mean_diff=df.mean(), ci_lo=df.mean() - t * se, ci_hi=df.mean() + t * se,
                              p=stats.ttest_rel(x[ok], y[ok]).pvalue, p_wilcoxon=stats.wilcoxon(df).pvalue if np.any(df != 0) else 1.0))
S = pd.DataFrame(rows); P = pd.DataFrame(pairs)
P["p_holm"] = np.nan
for (a, b, c), g in P.groupby(["a", "b", "metric"]):
    pv = g.p.to_numpy(); o = np.argsort(pv); adj = np.empty(len(pv)); r = 0.0
    for i, j in enumerate(o):
        r = max(r, min(1.0, (len(pv) - i) * pv[j])); adj[j] = r
    P.loc[g.index, "p_holm"] = adj
S.to_csv(os.path.join(OUT, "all500_summary.csv"), index=False); P.to_csv(os.path.join(OUT, "all500_paired.csv"), index=False)
print(S[S.metric.isin(["ssim_a", "rmse_a", "cnr", "csd_a", "loc_mm"])].pivot_table(index=["metric", "method"], columns="AR", values="mean").round(3).to_string())
print(P[P.metric.isin(["ssim_a", "rmse_a", "cnr", "csd_a", "loc_mm"])].groupby(["a", "b", "metric"]).agg(lo=("mean_diff", "min"), hi=("mean_diff", "max"), pmax=("p_holm", "max")).round(3).to_string())
