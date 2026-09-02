import tensorflow as tf
from tensorflow.keras import layers, models
from helper_20_1 import *
np.random.seed(19)
from sklearn.model_selection import train_test_split
import numpy as np
from keras import backend
from keras.layers import BatchNormalization, Dropout, Flatten, Lambda
from keras.layers.advanced_activations import ELU, LeakyReLU
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
from keras.regularizers import l2
from sklearn.decomposition import PCA
import pandas as pd
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from sklearn.model_selection import KFold
import os
backend.set_image_data_format('channels_first')
print(backend.image_data_format())
smooth = 1.
dropout_rate = 0.5
act = "relu"
###############################################################################

def fn_mask(x):
    from scipy.io import loadmat
    mask = loadmat('mask.mat')['mask'].astype(np.float)
    return multiply_constant(x, mask)


##########################################################################################################
from keras import backend
backend.set_image_data_format('channels_first')
print(backend.image_data_format())

lr = 0.001
beta_1 = 0.5


key_3_2d = "training_20"

x0_20_2d = PHI_meas_16x15[key_3_2d]
y0_20_2d = MU[key_3_2d]
total_num_20_2d = x0_20_2d.shape[0]

# Step 1: Select 10,000 random samples from the original dataset (20,000 samples)
total_selected_samples = 500

# Generate a list of indices and shuffle them
indices_20_2d = np.arange(total_num_20_2d)
np.random.shuffle(indices_20_2d)

# Select 10,000 random indices
selected_indices_20_2d = indices_20_2d[:total_selected_samples]

# Get the corresponding data for the 10,000 chosen indices
x_selected_20_2d = x0_20_2d[selected_indices_20_2d]
y_selected_20_2d = y0_20_2d[selected_indices_20_2d]

# Step 2: Split the 10,000 samples into training (80%), validation (15%), and testing (5%)
train_prop = 0.95
val_prop = 0.0
test_prop = 0.05

# Compute the number of samples for each set
train_num_20_2d = int(train_prop * total_selected_samples)  # 80% of 10k
val_num_20_2d = int(val_prop * total_selected_samples)  # 15% of 10k
test_num_20_2d = total_selected_samples - train_num_20_2d - val_num_20_2d  # 5% of 10k

# Generate a list of indices for the 10k selected samples and shuffle them
indices_selected_20_2d = np.arange(total_selected_samples)
np.random.shuffle(indices_selected_20_2d)

# Split into training, validation, and testing sets
train_indices_20_2d = indices_selected_20_2d[:train_num_20_2d]
val_indices_20_2d = indices_selected_20_2d[train_num_20_2d:train_num_20_2d + val_num_20_2d]
test_indices_20_2d = indices_selected_20_2d[train_num_20_2d + val_num_20_2d:]
print(test_indices_20_2d)
# Get the corresponding data for training, validation, and testing sets
x_train_20_2d = x_selected_20_2d[train_indices_20_2d]
y_train_20_2d = y_selected_20_2d[train_indices_20_2d]




x_val_20_2d = x_selected_20_2d[val_indices_20_2d]
y_val_20_2d = y_selected_20_2d[val_indices_20_2d]

x_test_20_2d = x_selected_20_2d[test_indices_20_2d]
y_test_20_2d = y_selected_20_2d[test_indices_20_2d]

# Prepare the target arrays (assuming the target is 2D, with two outputs)
y_train_20_2d = [y_train_20_2d[:, [0]], y_train_20_2d[:, [1]]]
y_val_20_2d = [y_val_20_2d[:, [0]], y_val_20_2d[:, [1]]]
y_test_20_2d = [y_test_20_2d[:, [0]], y_test_20_2d[:, [1]]]


PHI_meas_in_20_2d = PHI_meas_16x15[key_3_2d]
input_shape_20_2d = PHI_meas_in_20_2d.shape[1:]
inputs_20_2d = Input(input_shape_20_2d)
# input_ = Reshape((1,) + input_shape)(inputs[0])
inputs_20_2d.shape

import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model

# ============================================================
# BN STATISTICS UTILITIES
# ============================================================

def compute_empirical_bn_statistics(model, dataset, num_passes=20):
    """
    Proper AdaBN statistics estimation with multiple full passes.
    Fixes weak/1-iteration BN updates.
    """

    bn_layers = [l for l in model.layers if isinstance(l, tf.keras.layers.BatchNormalization)]

    # Backup original stats
    backup_mean = [l.moving_mean.numpy().copy() for l in bn_layers]
    backup_var = [l.moving_variance.numpy().copy() for l in bn_layers]

    # IMPORTANT: do NOT aggressively reset to destroy pretrained stability
    # (soft reset instead of hard reset)
    for l in bn_layers:
        l.moving_mean.assign(l.moving_mean * 0.0)
        l.moving_variance.assign(l.moving_variance * 1.0)

    # 🔥 MULTIPLE PASSES (KEY FIX)
    for _ in range(num_passes):
        for batch in dataset:
            _ = model(batch, training=True)

    # Extract stats
    target_stats = [
        (l.moving_mean.numpy().copy(), l.moving_variance.numpy().copy())
        for l in bn_layers
    ]

    # Restore original stats
    for l, m, v in zip(bn_layers, backup_mean, backup_var):
        l.moving_mean.assign(m)
        l.moving_variance.assign(v)

    return target_stats

def compute_alignment(stats_A, stats_B):
    total = 0.0
    for (m_a, v_a), (m_b, v_b) in zip(stats_A, stats_B):
        total += np.linalg.norm(m_a - m_b) ** 2
        total += np.linalg.norm(v_a - v_b) ** 2
    return total


def compute_layer_wise_alignment(stats_A, stats_B):
    return [
        np.linalg.norm(m_a - m_b) ** 2 + np.linalg.norm(v_a - v_b) ** 2
        for (m_a, v_a), (m_b, v_b) in zip(stats_A, stats_B)
    ]


# ============================================================
# ADA-BN APPLICATION
# ============================================================

def apply_adabn(model, target_stats, momentum_blend=0.7):
    """
    Stable AdaBN injection.
    Instead of hard overwrite, blend stats to avoid collapse.
    """

    bn_layers = [l for l in model.layers if isinstance(l, tf.keras.layers.BatchNormalization)]

    for layer, (mu_t, var_t) in zip(bn_layers, target_stats):

        # current stats
        mu_s = layer.moving_mean.numpy()
        var_s = layer.moving_variance.numpy()

        # 🔥 BLENDED UPDATE (IMPORTANT FIX FOR YOUR CASE)
        new_mu = (1 - momentum_blend) * mu_s + momentum_blend * mu_t
        new_var = (1 - momentum_blend) * var_s + momentum_blend * var_t

        layer.moving_mean.assign(new_mu)
        layer.moving_variance.assign(new_var)

# ============================================================
# MAIN ADA-BN PIPELINE
# ============================================================

def run_adabn_full_analysis():

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------
    model = load_model("CAFNet.h5", compile=False)

    ADAPTED_MODEL_PATH = "CAFNet_AdaBN_1.06.h5"

    # Freeze everything (no training)
    for layer in model.layers:
        layer.trainable = False

    bn_layers = [l for l in model.layers if isinstance(l, tf.keras.layers.BatchNormalization)]

    print("\n" + "="*70)
    print(f"ADA-BN MODE | BN layers: {len(bn_layers)}")
    print("="*70)

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------
    x_adapt = tf.cast(x_test_20_2d[:475], tf.float32)
    dataset = tf.data.Dataset.from_tensor_slices(x_adapt).batch(8)

    # --------------------------------------------------------
    # Save source BN stats
    # --------------------------------------------------------
    source_stats = [
        (l.moving_mean.numpy().copy(), l.moving_variance.numpy().copy())
        for l in bn_layers
    ]

    # --------------------------------------------------------
    # Compute target stats (AdaBN step)
    # --------------------------------------------------------
    print("\n[ADABN] Computing target BN statistics...")
    target_stats = compute_empirical_bn_statistics(model, dataset)

    # --------------------------------------------------------
    # Alignment BEFORE adaptation
    # --------------------------------------------------------
    A_initial = compute_alignment(source_stats, target_stats)
    A_initial_layerwise = compute_layer_wise_alignment(source_stats, target_stats)

    print(f"\nInitial BN Misalignment: {A_initial:.6f}")

    # --------------------------------------------------------
    # Apply AdaBN
    # --------------------------------------------------------
    print("\n[ADABN] Injecting target statistics...")
    apply_adabn(model, target_stats)

    # --------------------------------------------------------
    # Alignment AFTER adaptation
    # --------------------------------------------------------
    adapted_stats = [
        (l.moving_mean.numpy().copy(), l.moving_variance.numpy().copy())
        for l in bn_layers
    ]

    A_final = compute_alignment(adapted_stats, target_stats)
    A_final_layerwise = compute_layer_wise_alignment(adapted_stats, target_stats)

    print(f"Final BN Misalignment: {A_final:.6f}")

    improvement = (A_initial - A_final) / (A_initial + 1e-8) * 100
    print(f"AdaBN Alignment Improvement: {improvement:.2f}%")

    # --------------------------------------------------------
    # Per-layer report
    # --------------------------------------------------------
    print("\n--- Layer-wise Alignment ---")
    for i, (a0, a1) in enumerate(zip(A_initial_layerwise, A_final_layerwise)):
        print(f"Layer {i+1:2d}: {a0:.4f} → {a1:.4f}")

    # --------------------------------------------------------
    # Predictions
    # --------------------------------------------------------
    print("\n[ADABN] Running inference...")

    predictions = model.predict(
        x_test_20_2d,
        batch_size=8,
        verbose=1
    )

    # --------------------------------------------------------
    # Save model
    # --------------------------------------------------------
    model.save(ADAPTED_MODEL_PATH)

    print(f"\nSaved AdaBN model → {ADAPTED_MODEL_PATH}")
    print("\n=== ADA-BN COMPLETE ===")

    return predictions


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    preds = run_adabn_full_analysis()