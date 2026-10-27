"""CoDA vs CoDA without the consistency step (EMA statistics only, same 50 iterations): means per aspect ratio and paired differences."""
import os, numpy as np, pandas as pd
from scipy import stats
from scipy.io import loadmat
import extra_metrics as EM

HERE = os.path.dirname(os.path.abspath(__file__))
ARS = ["1.06", "1.10", "1.16", "1.22"]
rows, diffs = [], []
for ar in ARS:
    run = os.path.join(HERE, "runs", "R_" + ar)
    gt = np.load(os.path.join(run, "results", "gt_held.npy"))
    held = np.load(os.path.join(run, "results", "split.npz"))["held"]
    mask = loadmat(os.path.join(run, "samples_500.mat"), squeeze_me=True)["mask"].astype(bool)
    d = np.asarray(loadmat(os.path.join(run, "param_samples_500.mat"), squeeze_me=True)["d"], float)[held]
    per = {}
    for name in ("CoDA", "CoDAemaOnly", "AdaBN"):
        w = pd.read_excel(os.path.join(run, "results", "%s_%s.xlsx" % (name, ar)), header=[0, 1], index_col=0)
        e = EM.per_sample(gt, np.load(os.path.join(run, "results", "pred_%s.npy" % name)), mask, d)
        rec = {"ssim_a": pd.to_numeric(w[("ssim", "mu_a")], errors="coerce").to_numpy(float), "ssim_s": pd.to_numeric(w[("ssim", "mu_sp")], errors="coerce").to_numpy(float),
               "csd_a": pd.to_numeric(w[("R_csd", "mu_a")], errors="coerce").to_numpy(float), "csd_s": pd.to_numeric(w[("R_csd", "mu_sp")], errors="coerce").to_numpy(float),
               "rmse_a": e["rmse_a"].to_numpy(float) * 1e3, "rmse_s": e["rmse_s"].to_numpy(float), "cnr_a": e["cnr_a"].to_numpy(float), "loc_mm_a": e["loc_mm_a"].to_numpy(float)}
        per[name] = rec
        for k, v in rec.items():
            rows.append(dict(AR=ar, method=name, metric=k, mean=np.nanmean(v), sd=np.nanstd(v, ddof=1)))
    for k in per["CoDA"]:
        a, b = per["CoDA"][k], per["CoDAemaOnly"][k]
        ok = np.isfinite(a) & np.isfinite(b)
        dd = (a - b)[ok]
        t, p = stats.ttest_rel(a[ok], b[ok])
        diffs.append(dict(AR=ar, metric=k, n=int(ok.sum()), mean_diff=dd.mean(), p=p))
S = pd.DataFrame(rows)
D = pd.DataFrame(diffs)
D["p_holm"] = np.nan
for k, g in D.groupby("metric"):
    pv = g.p.to_numpy(); order = np.argsort(pv); adj = np.empty_like(pv); run_ = 0.0
    for rank, i in enumerate(order):
        run_ = max(run_, min(1.0, (len(pv) - rank) * pv[i])); adj[i] = run_
    D.loc[g.index, "p_holm"] = adj
S.to_csv(os.path.join(HERE, "ema_only_summary.csv"), index=False)
D.to_csv(os.path.join(HERE, "ema_only_paired.csv"), index=False)
print(S.groupby(["method", "metric"])["mean"].mean().unstack().round(3).to_string())
print(D.groupby("metric").agg(lo=("mean_diff", "min"), hi=("mean_diff", "max"), pmin=("p_holm", "min"), pmax=("p_holm", "max")).round(3).to_string())
