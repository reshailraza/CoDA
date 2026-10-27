"""CoDA without its consistency step: the same 50 iterations (same batches, same noisy second forward pass, so the same EMA updates), but no gradient step on gamma/beta.
Isolates what the consistency-driven affine update adds to the EMA statistics update. Usage: python coda_ema_only.py <AR>"""
import os, sys, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
ORIG = os.environ.get("CODA_ROOT", os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))); HERE = os.path.dirname(os.path.abspath(__file__))
AR = sys.argv[1]
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
from tensorflow.keras.utils import get_custom_objects
from scipy.io import loadmat
import mask_utils
get_custom_objects().update(H.custom_objects)
_lam = sys.modules[tf.keras.layers.Lambda.__module__]
for k, v in H.custom_objects.items():
    setattr(_lam, k, v)
mask_utils.set_mask(H, loadmat("samples_500.mat", squeeze_me=True)["mask"].astype(bool))
HELD = C.selected_indices_20_2d[C.test_indices_20_2d]
m = load_model("CAFNet.h5", compile=False, custom_objects=H.custom_objects)
for l in m.layers:
    l.trainable = isinstance(l, BatchNormalization)      # BN in training mode (EMA updates), as in CoDA
batches = list(tf.data.Dataset.from_tensor_slices(tf.cast(C.x_train_20_2d[:475], tf.float32)).batch(8))
tf.random.set_seed(19)
for it in range(C.N_ITERATIONS):
    x = batches[it % len(batches)]
    noise = tf.random.normal(tf.shape(x), stddev=C.NOISE_STD)
    _ = m(x, training=True)
    _ = m(x + noise, training=True)                       # same two forward passes as the CoDA step; no optimizer step
pred = m.predict(C.x_test_20_2d, batch_size=8, verbose=0)
MU = np.zeros((500, 2, 64, 64), np.float32)
MU[HELD] = np.concatenate(list(pred)[:2], axis=1)
df = H.ground_truth_vs_estimated_direct("training_20", MU, s_range=HELD)[0]
df.to_excel("results/CoDAemaOnly_%s.xlsx" % AR)
np.save("results/pred_CoDAemaOnly.npy", MU[HELD])
print("RESULT", AR, "ssim_a %.3f csd_a %.3f csd_s %.3f" % (df[("ssim", "mu_a")].mean(), df[("R_csd", "mu_a")].mean(), df[("R_csd", "mu_sp")].mean()), flush=True)
