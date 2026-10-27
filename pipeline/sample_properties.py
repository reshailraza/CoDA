"""Properties of every simulated sample per aspect ratio (from the helper's parameter arrays) and of the 25 held-out samples."""
import os, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
ARS = ["1.06", "1.10", "1.16", "1.22"]
rows_all = {}
for ar in ARS:
    P = np.load(os.path.join(HERE, "params_%s.npz" % ar))
    n = len(P["d"])
    rec = []
    for i in range(n):
        inc = [k for k in range(P["r"].shape[1]) if P["r"][i, k] > 0]
        row = dict(AR=ar, original_index=i, held_out=bool(i in set(P["held"].tolist())), n_inclusions=len(inc), diameter_mm=float(P["d"][i]), frequency_MHz=float(P["freq"][i]) * 1e-6,
                   mu_a_background=float(P["MUa"][i]), mu_sp_background=float(P["MUsp"][i]))
        for j in range(3):
            if j < len(inc):
                k = inc[j]
                row.update({"r%d_mm" % (j + 1): float(P["r"][i, k]), "roc%d_mm" % (j + 1): float(P["roc"][i, k]), "theta%d_deg" % (j + 1): float(P["theta"][i, k]),
                            "ca%d" % (j + 1): float(P["ca"][i, k]), "csp%d" % (j + 1): float(P["csp"][i, k])})
            else:
                row.update({"r%d_mm" % (j + 1): np.nan, "roc%d_mm" % (j + 1): np.nan, "theta%d_deg" % (j + 1): np.nan, "ca%d" % (j + 1): np.nan, "csp%d" % (j + 1): np.nan})
        rec.append(row)
    df = pd.DataFrame(rec)
    df = df.dropna(axis=1, how="all")
    df.to_csv(os.path.join(HERE, "sample_properties_all500_AR_%s.csv" % ar), index=False)
    held = df[df.held_out].copy()
    held.insert(1, "heldout_position", [list(P["held"]).index(i) for i in held.original_index])
    held = held.sort_values("heldout_position")
    held.to_csv(os.path.join(HERE, "sample_properties_heldout25_AR_%s.csv" % ar), index=False)
    rows_all[ar] = held
print(rows_all["1.22"][["heldout_position", "original_index", "n_inclusions", "diameter_mm", "r1_mm", "r2_mm", "ca1", "ca2", "csp1", "csp2"]].round(2).to_string(index=False))
print({ar: int(v.n_inclusions.sum()) for ar, v in rows_all.items()})
