"""Output-BN parameters (mu_a branch = BN 82, mu_s' branch = BN 83) of every saved model, per AR (read straight from the .h5 files)."""
import os, sys, h5py, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
ars = sys.argv[1:] or ["1.06", "1.10", "1.16", "1.22"]
files = {"CAFNet": "CAFNet.h5", "FineTuned": "FineTuned_model.h5", "Tent": "Tent_model.h5", "AdaBN": "AdaBN_model.h5", "CoDA": "CoDA_model.h5"}
rows = []
for ar in ars:
    for m, f in files.items():
        p = os.path.join(os.path.dirname(HERE), f) if m == "CAFNet" else os.path.join(HERE, "runs", "R_" + ar, f)   # source model lives in the project root
        if not os.path.exists(p):
            continue
        vals = {}
        with h5py.File(p, "r") as h:
            def visit(name, obj):
                if isinstance(obj, h5py.Dataset) and "batch_normalization_8" in name:
                    vals[name] = obj[()]
            h.visititems(visit)
        def g(layer, field):
            for k, v in vals.items():
                if ("/%s/" % layer) in k and k.endswith(field + ":0"):
                    return float(np.ravel(v)[0])
            return np.nan
        rows.append(dict(AR=ar, method=m,
                         gamma_a=g("batch_normalization_82", "gamma"), beta_a=g("batch_normalization_82", "beta"),
                         gamma_s=g("batch_normalization_83", "gamma"), beta_s=g("batch_normalization_83", "beta"),
                         mean_out=g("batch_normalization_82", "moving_mean"), var_out=g("batch_normalization_82", "moving_variance")))
df = pd.DataFrame(rows)
df.to_csv(os.path.join(HERE, "bn_output_params.csv"), index=False)
print(df.round(6).to_string(index=False))
