"""Additional analyses requested after review (per aspect ratio):
 A. statistics-weight sweep: AdaBN-style recalibration (no gradient) with n training-mode passes over the pool, w = 1 - 0.99^n
 B. transductive scoring of the label-free methods on all 500 samples (the adaptation uses measurements only, so scoring the pool is legitimate test-time adaptation)
 C. sensitivity of Tent and CoDA to iterations and learning rate (budget fairness)
Usage: python extra_analyses.py <AR> [A] [B] [C]
"""
import os, sys, time, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
ORIG = os.environ.get("CODA_ROOT", os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))); HERE = os.path.dirname(os.path.abspath(__file__))
AR = sys.argv[1]
PARTS = set(sys.argv[2:]) or {"A", "B", "C"}
os.chdir(os.path.join(HERE, "runs", "R_" + AR)); sys.path.insert(0, ORIG); sys.path.insert(0, HERE)
import matplotlib; matplotlib.use("Agg")
import builtins
from keras import backend as _b
builtins.backend = _b
import CODA_Github as C
import helper_20_1 as H
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import get_custom_objects
from scipy.io import loadmat
import mask_utils, extra_metrics as EM
get_custom_objects().update(H.custom_objects)
_lam = sys.modules[tf.keras.layers.Lambda.__module__]
for k, v in H.custom_objects.items():
    setattr(_lam, k, v)
MASK = loadmat("samples_500.mat", squeeze_me=True)["mask"].astype(bool)
mask_utils.set_mask(H, MASK)
D_MM = np.asarray(loadmat("param_samples_500.mat", squeeze_me=True)["d"], float)
HELD = C.selected_indices_20_2d[C.test_indices_20_2d]
ALL = np.arange(500)
X_ALL = np.asarray(H.PHI_meas_16x15["training_20"], np.float32)
GT = np.asarray(H.MU["training_20"])
POOL = tf.cast(C.x_train_20_2d[:475], tf.float32)
BATCHES = list(tf.data.Dataset.from_tensor_slices(POOL).batch(8))
T0 = time.time()


def log(*a):
    print("[%7.1fs]" % (time.time() - T0), *a, flush=True)


def predict(m, x):
    p = m.predict(x, batch_size=8, verbose=0)
    return np.concatenate(list(p)[:2], axis=1)


def score(pred, idx):
    """pred: (len(idx),2,64,64); returns per-sample frame with ssim, csd, rmse, cnr, loc (both properties)."""
    MU = np.zeros((500, 2, 64, 64), np.float32)
    MU[idx] = pred
    df = H.ground_truth_vs_estimated_direct("training_20", MU, s_range=idx)[0]
    ex = EM.per_sample(GT[idx], pred, MASK, D_MM[idx])
    out = pd.DataFrame({"ssim_a": pd.to_numeric(df[("ssim", "mu_a")], errors="coerce").to_numpy(float), "ssim_s": pd.to_numeric(df[("ssim", "mu_sp")], errors="coerce").to_numpy(float),
                        "csd_a": pd.to_numeric(df[("R_csd", "mu_a")], errors="coerce").to_numpy(float), "csd_s": pd.to_numeric(df[("R_csd", "mu_sp")], errors="coerce").to_numpy(float),
                        "rmse_a": ex["rmse_a"].to_numpy(float) * 1e3, "rmse_s": ex["rmse_s"].to_numpy(float), "cnr": ex["cnr_a"].to_numpy(float), "loc_mm": ex["loc_mm_a"].to_numpy(float)})
    return out


def summarize(df):
    return {c: float(np.nanmean(df[c])) for c in df.columns}


def fresh(train_bn=True):
    m = load_model("CAFNet.h5", compile=False, custom_objects=H.custom_objects)
    for l in m.layers:
        l.trainable = isinstance(l, BatchNormalization) and train_bn
    return m



# ---- CoDA without its gradient step (EMA only, same 50 iterations, same noisy second pass), scored on all 500 samples
m = fresh()
tf.random.set_seed(19)
for it in range(C.N_ITERATIONS):
    x = BATCHES[it % len(BATCHES)]
    noise = tf.random.normal(tf.shape(x), stddev=C.NOISE_STD)
    _ = m(x, training=True)
    _ = m(x + noise, training=True)
s = score(predict(m, X_ALL), ALL)
s.insert(0, "sample", ALL); s.insert(1, "in_pool", ~np.isin(ALL, HELD))
s.to_csv("results/all500_CoDAemaOnly.csv", index=False)
log("EMA-only all500", {k: round(v, 3) for k, v in summarize(s).items()})
