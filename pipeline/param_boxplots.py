"""Boxplots of the simulation parameters (d, freq, MUa, MUsp, r, theta, roc, ca, csp) for each aspect ratio.
Usage: python param_boxplots.py collect <AR>   (run inside runs/R_<AR>; reads the helper's dictionaries for key 'training_20')
       python param_boxplots.py plot
"""
import os, sys, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
ORIG = os.environ.get("CODA_ROOT", os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")))
ARS = ["1.06", "1.10", "1.16", "1.22"]
if sys.argv[1] == "collect":
    ar = sys.argv[2]
    run = os.path.join(HERE, "runs", "R_" + ar)
    os.chdir(run); sys.path.insert(0, ORIG)
    import helper_20_1 as H
    key = "training_20"
    d_ = H.d[key]; f_ = H.freq[key]; Mua_ = H.MUa[key]; Musp_ = H.MUsp[key]
    r = H.r_data[key]; theta = H.theta_data[key]; roc = H.roc_data[key]; ca = H.ca_data[key]; csp = H.csp_data[key]
    held = np.load(os.path.join(run, "results", "split.npz"))["held"]
    np.savez(os.path.join(HERE, "params_%s.npz" % ar), d=np.ravel(d_), freq=np.ravel(f_), MUa=np.ravel(Mua_), MUsp=np.ravel(Musp_),
             r=r, theta=theta, roc=roc, ca=ca, csp=csp, held=held)
    print("collected", ar, d_.shape, r.shape)
else:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    P = {ar: np.load(os.path.join(HERE, "params_%s.npz" % ar)) for ar in ARS}
    # (key, title, scale, is_inclusion_level)
    SPEC = [("d", "Phantom diameter d (mm)", 1.0, False), ("freq", "Modulation frequency (MHz)", 1e-6, False),
            ("MUa", "Background \u03bca (mm\u207b\u00b9)", 1.0, False), ("MUsp", "Background \u03bcs\u2032 (mm\u207b\u00b9)", 1.0, False),
            ("r", "Inclusion radius r (mm)", 1.0, True), ("theta", "Inclusion angle \u03b8 (deg)", 1.0, True),
            ("roc", "Inclusion distance from centre, roc (mm)", 1.0, True), ("ca", "\u03bca contrast ca", 1.0, True), ("csp", "\u03bcs\u2032 contrast csp", 1.0, True)]

    def values(p, key, scale, incl, subset=None):
        x = np.asarray(p[key], float)
        if subset is not None:
            x = x[subset]
        if incl:
            m = np.asarray(p["r"], float)
            if subset is not None:
                m = m[subset]
            x = x[m > 0]                       # drop padding slots (samples with fewer than two inclusions)
        return np.ravel(x) * scale

    os.makedirs(os.path.join(HERE, "figures"), exist_ok=True)
    # one figure per aspect ratio: all 500 samples vs the 25 held-out samples
    for ar in ARS:
        p = P[ar]
        fig, axes = plt.subplots(3, 3, figsize=(11, 9))
        for ax, (key, title, sc, inc) in zip(axes.ravel(), SPEC):
            data = [values(p, key, sc, inc), values(p, key, sc, inc, subset=p["held"])]
            bp = ax.boxplot(data, patch_artist=True, widths=0.6, medianprops=dict(color="black"), flierprops=dict(marker="o", markersize=3, alpha=0.6))
            for patch, c in zip(bp["boxes"], ("#1f77b4", "#d62728")):
                patch.set_facecolor(c); patch.set_alpha(0.8)
            ax.set_xticks([1, 2]); ax.set_xticklabels(["all 500 samples" + chr(10) + "(n=%d)" % len(data[0]), "held-out 25" + chr(10) + "(n=%d)" % len(data[1])], fontsize=8)
            ax.set_title(title, fontsize=9); ax.grid(alpha=0.3)
        fig.suptitle("Simulation parameters, aspect ratio %s (inclusion-level parameters pool both inclusions of a sample)" % ar, fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        fig.savefig(os.path.join(HERE, "figures", "params_AR_%s.png" % ar), dpi=160)
        plt.close(fig)
    # overview: the four aspect ratios side by side, all 500 samples
    fig, axes = plt.subplots(3, 3, figsize=(12, 9.5))
    for ax, (key, title, sc, inc) in zip(axes.ravel(), SPEC):
        data = [values(P[ar], key, sc, inc) for ar in ARS]
        bp = ax.boxplot(data, patch_artist=True, widths=0.6, medianprops=dict(color="black"), flierprops=dict(marker="o", markersize=3, alpha=0.6))
        for patch, c in zip(bp["boxes"], ("#4c72b0", "#55a868", "#c44e52", "#8172b2")):
            patch.set_facecolor(c); patch.set_alpha(0.85)
        ax.set_xticks(range(1, 5)); ax.set_xticklabels(["AR " + a for a in ARS], fontsize=8)
        ax.set_title(title, fontsize=9); ax.grid(alpha=0.3)
    fig.suptitle("Simulation parameters of the four datasets (all 500 samples each)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(os.path.join(HERE, "figures", "params_overview_all_AR.png"), dpi=160)
    plt.close(fig)
    # histograms of the same parameters: one figure per aspect ratio (all 500 samples, held-out 25 outlined) and an overview overlay
    for ar in ARS:
        p = P[ar]
        fig, axes = plt.subplots(3, 3, figsize=(11, 9))
        for ax, (key, title, sc, inc) in zip(axes.ravel(), SPEC):
            allv = values(p, key, sc, inc)
            hv = values(p, key, sc, inc, subset=p["held"])
            bins = np.histogram_bin_edges(allv, bins=20)
            ax.hist(allv, bins=bins, color="#1f77b4", alpha=0.75, label="all 500 samples (n=%d)" % len(allv), density=True)
            ax.hist(hv, bins=bins, histtype="step", color="#d62728", linewidth=1.8, label="held-out 25 (n=%d)" % len(hv), density=True)
            ax.set_title(title, fontsize=9); ax.set_ylabel("density", fontsize=8); ax.grid(alpha=0.3)
            if key == "d":
                ax.legend(fontsize=7)
        fig.suptitle("Histograms of the simulation parameters, aspect ratio %s" % ar, fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        fig.savefig(os.path.join(HERE, "figures", "params_hist_AR_%s.png" % ar), dpi=160)
        plt.close(fig)
    fig, axes = plt.subplots(3, 3, figsize=(12, 9.5))
    cols = ("#4c72b0", "#55a868", "#c44e52", "#8172b2")
    for ax, (key, title, sc, inc) in zip(axes.ravel(), SPEC):
        vals = [values(P[ar], key, sc, inc) for ar in ARS]
        bins = np.histogram_bin_edges(np.concatenate(vals), bins=20)
        for ar, v, c in zip(ARS, vals, cols):
            ax.hist(v, bins=bins, histtype="step", color=c, linewidth=1.6, density=True, label="AR " + ar)
        ax.set_title(title, fontsize=9); ax.grid(alpha=0.3)
        if key == "d":
            ax.legend(fontsize=7)
    fig.suptitle("Histograms of the simulation parameters, the four aspect ratios overlaid (all 500 samples each)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(os.path.join(HERE, "figures", "params_hist_overview_all_AR.png"), dpi=160)
    plt.close(fig)
    # numeric summary
    import pandas as pd
    rows = []
    for ar in ARS:
        for key, title, sc, inc in SPEC:
            v = values(P[ar], key, sc, inc)
            rows.append(dict(AR=ar, parameter=title, n=len(v), min=v.min(), q25=np.percentile(v, 25), median=np.median(v), q75=np.percentile(v, 75), max=v.max(), mean=v.mean(), sd=v.std(ddof=1)))
    pd.DataFrame(rows).to_csv(os.path.join(HERE, "param_summary.csv"), index=False)
    print("plots done")
