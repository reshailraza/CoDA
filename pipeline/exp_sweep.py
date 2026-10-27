"""(setup copied from exp_from_cafnet.py) Statistics-weight sweep on the experimental phantom.
Experimental phantom, every method applied directly to the source model CAFNet (no stacking, original CoDA).
Data: mask1.22/landseed_processed__new_data.mat only (one phantom: calibrated measurement PHI_CLB_meas and ground-truth grids).
(The 12-sample landseed file in the run folder is only needed so that helper_20_1.py can be imported; nothing is read from it.)

A single measurement cannot give batch statistics (a dense-layer BN over a batch of one has zero variance), so the label-free methods
adapt on a pool of K jittered copies of the measurement (Gaussian jitter, std POOL_SIGMA); the consistency noise of CoDA is the usual 0.05.
  CAFNet      : source model, no adaptation
  Fine-Tuned  : BN-only fine-tuned on the labeled ratio_1.22 simulation pool (not adapted on the experiment; labels are not available there)
  AdaBN       : statistics replaced by those of the pool (blend 1.0)
  Tent        : entropy of spatial softmax, Adam 1e-5, 50 iterations (optimizer steps), batch 8 (TenT.py logic)
  CoDA        : original CoDA (EMA statistics + consistency loss on all BN gamma/beta incl. the output layers), Adam 1e-5, 50 iterations (optimizer steps), batch 8, sigma 0.05
  AdaptNet    : per-sample test-time adaptation of the measurement (its native mode)
"""
import os, sys, time, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
ORIG = r"D:\Machine_Learning\SPIE Conference"; HERE = os.path.dirname(os.path.abspath(__file__))
EXPFILE = os.path.join(ORIG, "mask1.22", "landseed_processed__new_data.mat")
K = int(os.environ.get("POOL_K", "64"))
os.chdir(os.path.join(HERE, "runs", "EXP")); sys.path.insert(0, ORIG); sys.path.insert(0, HERE)
os.makedirs("results", exist_ok=True)
import matplotlib; matplotlib.use("Agg")
import builtins
from keras import backend as _b
builtins.backend = _b
T0 = time.time()
def log(*a): print("[%7.1fs]" % (time.time() - T0), *a, flush=True)
import AdABN as A
import TenT as TT
import helper_20_1 as H
import tensorflow as tf
from scipy.io import loadmat
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import get_custom_objects
import mask_utils, extra_metrics as EM
get_custom_objects().update(H.custom_objects)
_lam = sys.modules[tf.keras.layers.Lambda.__module__]
for k_, v_ in H.custom_objects.items():
    setattr(_lam, k_, v_)
DS_MASK = loadmat("samples_500.mat", squeeze_me=True)["mask"].astype(bool)      # AR 1.22 ellipse
mask_utils.set_mask(H, DS_MASK)

# ---------------- data: this file only
d = loadmat(EXPFILE, squeeze_me=True)
def prep(phi):
    lg = np.log(np.asarray(phi).astype(complex).reshape(-1))
    ph = H.unwrap_sample(np.imag(lg).copy(), 16)
    return np.concatenate([lg.real, ph]).reshape(1, 2, 16, 15).astype(np.float32)
Y = prep(d["PHI_CLB_meas"])
GT = np.stack([np.asarray(d["Grid_MUA"], float), np.asarray(d["Grid_MUSP"], float)])           # (2,64,64)
# helper-compatible entries for this single phantom (inclusions sorted by radius, padded with zeros)
r_, roc_, th_, ca_, cs_ = [np.atleast_1d(np.asarray(d[k], float)) for k in ("r", "roc", "theta", "ca", "csp")]
order = np.argsort(-r_)
def pad(x):
    out = np.zeros((1, H.MAX_INC)); out[0, :len(x)] = x[order]; return out
KEY = "calibrated"
H.d[KEY] = np.array([[float(d["d"])]]); H.r_data[KEY] = pad(r_); H.roc_data[KEY] = pad(roc_); H.theta_data[KEY] = pad(th_)
H.ca_data[KEY] = pad(ca_); H.csp_data[KEY] = pad(cs_); H.MU[KEY] = GT[None]
log("measurement:", d["freq"] / 1e6, "MHz, d =", d["d"], "mm, inclusions r", r_.tolist(), "ca", ca_.tolist(), "csp", cs_.tolist())

COLS = [("ssim", "mu_a"), ("ssim", "mu_sp"), ("R_csd", "mu_a"), ("R_csd", "mu_sp")]
NAMES = ["ssim_a", "ssim_s", "csd_a", "csd_s"]
def evaluate(pred):
    MU = np.concatenate([np.asarray(pred[0]), np.asarray(pred[1])], axis=0).reshape(1, 2, 64, 64).astype(np.float32)
    df = H.ground_truth_vs_estimated_direct(KEY, MU, s_range=[0])[0]
    r = {n: float(df[c].iloc[0]) for n, c in zip(NAMES, COLS)}
    for k, p in ((0, "a"), (1, "s")):
        m = EM.one(GT[k], MU[0, k], DS_MASK)
        r["rmse_" + p] = m["rmse"] * (1e3 if p == "a" else 1.0); r["nrmse_" + p] = m["nrmse"]; r["cnr_" + p] = m["cnr"]; r["crc_" + p] = m["crc"]; r["loc_px_" + p] = m["loc_px"]
    return r, MU[0]
def load(p): return load_model(p, compile=False, custom_objects=H.custom_objects)
def predict(m): return [p[0] for p in m.predict(Y, batch_size=1, verbose=0)]


rows = []

import CODA_Github as C            # simulated AR 1.22 pool (475 unlabeled measurements; labels not used)
SIM_POOL = tf.cast(C.x_train_20_2d[:475], tf.float32).numpy()
LAB_OUT = []


def stats_only(pool, n, batch=8):
    m = load("CAFNet.h5")
    for l in m.layers:
        l.trainable = isinstance(l, BatchNormalization)
    batches = list(tf.data.Dataset.from_tensor_slices(pool).batch(batch))
    for it in range(n):
        _ = m(batches[it % len(batches)], training=True)
    return m


pools = {}
rng = np.random.RandomState(19)
pools["64 jittered copies (sigma 0.05)"] = (Y + 0.05 * rng.randn(64, 2, 16, 15)).astype(np.float32)
pools["simulated AR1.22 pool (475)"] = SIM_POOL
for pname, pool in pools.items():
    for n in (0, 10, 25, 50, 100, 200, 400, 1200):
        pred = predict(stats_only(pool, n))
        r, mu = evaluate(pred)
        r.update(pool=pname, n=n, w=1 - 0.99 ** n)
        rows.append(r)
        np.save("results/pred_sweep_%s_n%d.npy" % (pname.split(" ")[0], n), mu)
        log("%-34s n=%4d w=%.2f ssim_a %.3f ssim_s %.3f csd_a %7.3f rmse_a %.2f cnr %.2f loc_px %.1f" % (pname, n, r["w"], r["ssim_a"], r["ssim_s"], r["csd_a"], r["rmse_a"], r["cnr_a"], r["loc_px_a"]))
pd.DataFrame(rows).to_csv("results/exp_sweep.csv", index=False)
log("DONE")
