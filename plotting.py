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
from keras.models import Model
import pywt
import cv2
lr = 0.001
beta_1 = 0.5
key_3_2d = "training_20"


PHI_meas_={}

# PHI_meas_['training_16']=PHI_meas['training_16']
# PHI_meas_['training_16']=PHI_meas['training_16']
PHI_meas_['training_20']=PHI_meas['training_20']
PHI_meas_['calibrated_16']=PHI_meas['calibrated_16']
PHI_meas_['calibrated']=PHI_meas['calibrated']
PHI_meas_['uncalibrated']=PHI_meas['uncalibrated']
PHI_meas_['simulation']=PHI_meas['simulation']
PHI_meas_16x15_={}

# PHI_meas_16x15_['training_16']=PHI_meas_16x15['training_16']
PHI_meas_16x15_['training_20']=PHI_meas_16x15['training_20']
PHI_meas_16x15_['calibrated_16']=PHI_meas_16x15['calibrated_16']
PHI_meas_16x15_['calibrated']=PHI_meas_16x15['calibrated']
PHI_meas_16x15_['uncalibrated']=PHI_meas_16x15['uncalibrated']
PHI_meas_16x15_['simulation']=PHI_meas_16x15['simulation']




MU_deep_1D = {}  #1d net output
MU_deep_2D = {}   #2d net output
MU_deep_De = {}   #dense net output


import keras
loaded_1 = keras.models.load_model("adapted_CAFNet_CoDA_0.05_50.h5")
model_20_de = loaded_1

loaded_2 = keras.models.load_model("adaptnet_model.h5")
model_20_1d = loaded_2


loaded_3 = keras.models.load_model("adapted_CAFNet_BN_1.22.h5")
model_20_2d = loaded_3


def obtain_results1(param_model=model_20_1d, param_input=input_for_model, weights=None, sel=slice(2), sel_bg=slice(4, 6),
                   print_summary=False, exclude_keys=[]):
    global MU_deep_1D, deep_output, input_for_model
    set_model(param_model)
    if weights is not None:
        set_weights(weights)
    if param_input is not input_for_model:
        if isinstance(param_input, list):
            input_for_model = {}
            for k in param_input[0].keys():
                if k in exclude_keys:
                    continue
                input_for_model[k] = [inp[k] for inp in param_input]
        else:
            input_for_model = param_input.copy()
            for k in exclude_keys:
                input_for_model.pop(k)
    deep_output = {}
    for k, v in input_for_model.items():
        deep_output[k] = model_20_1d.predict(v)
    if print_summary:
        model_20_1d.summary()
    if isinstance(deep_output[list(deep_output.keys())[0]], list):
        for k, v in deep_output.items():
            if isinstance(sel, int):
                MU_deep_1D[k] = v[sel]
            elif isinstance(sel, slice):
                MU_deep_1D[k] = np.concatenate(v[sel], axis=1)
            else:
                MU_deep_1D[k] = np.concatenate([v[i] for i in sel], axis=1)
            if isinstance(sel_bg, int):
                MU0_deep[k] = v[sel_bg]
            elif isinstance(sel_bg, slice):
                MU0_deep[k] = np.concatenate(v[sel_bg], axis=1)
            else:
                MU0_deep[k] = np.concatenate([v[i] for i in sel_bg], axis=1)
    else:
        MU_deep_1D = deep_output
    for k in deep_output.keys():
        if k in train_names:
            datasets[k].set_MU_deep(MU_deep_1D[k])
    return MU_deep_1D



def obtain_results2(param_model=model_20_2d, param_input=input_for_model, weights=None, sel=slice(2), sel_bg=slice(4, 6),
                   print_summary=False, exclude_keys=[]):
    global MU_deep_2D, deep_output, input_for_model
    set_model(param_model)
    if weights is not None:
        set_weights(weights)
    if param_input is not input_for_model:
        if isinstance(param_input, list):
            input_for_model = {}
            for k in param_input[0].keys():
                if k in exclude_keys:
                    continue
                input_for_model[k] = [inp[k] for inp in param_input]
        else:
            input_for_model = param_input.copy()
            for k in exclude_keys:
                input_for_model.pop(k)
    deep_output = {}
    for k, v in input_for_model.items():
        deep_output[k] = model_20_2d.predict(v)
    if print_summary:
        model_20_2d.summary()
    if isinstance(deep_output[list(deep_output.keys())[0]], list):
        for k, v in deep_output.items():
            if isinstance(sel, int):
                MU_deep_2D[k] = v[sel]
            elif isinstance(sel, slice):
                MU_deep_2D[k] = np.concatenate(v[sel], axis=1)
            else:
                MU_deep_2D[k] = np.concatenate([v[i] for i in sel], axis=1)
            if isinstance(sel_bg, int):
                MU0_deep[k] = v[sel_bg]
            elif isinstance(sel_bg, slice):
                MU0_deep[k] = np.concatenate(v[sel_bg], axis=1)
            else:
                MU0_deep[k] = np.concatenate([v[i] for i in sel_bg], axis=1)
    else:
        MU_deep_2D = deep_output
    for k in deep_output.keys():
        if k in train_names:
            datasets[k].set_MU_deep(MU_deep_2D[k])
    return MU_deep_2D



def obtain_results3(param_model=model_20_de, param_input=input_for_model, weights=None, sel=slice(2), sel_bg=slice(4, 6),
                   print_summary=False, exclude_keys=[]):
    global MU_deep_De, deep_output, input_for_model
    set_model(param_model)
    if weights is not None:
        set_weights(weights)
    if param_input is not input_for_model:
        if isinstance(param_input, list):
            input_for_model = {}
            for k in param_input[0].keys():
                if k in exclude_keys:
                    continue
                input_for_model[k] = [inp[k] for inp in param_input]
        else:
            input_for_model = param_input.copy()
            for k in exclude_keys:
                input_for_model.pop(k)
    deep_output = {}
    for k, v in input_for_model.items():
        deep_output[k] = model_20_de.predict(v)
    if print_summary:
        model_20_de.summary()
    if isinstance(deep_output[list(deep_output.keys())[0]], list):
        for k, v in deep_output.items():
            if isinstance(sel, int):
                MU_deep_De[k] = v[sel]
            elif isinstance(sel, slice):
                MU_deep_De[k] = np.concatenate(v[sel], axis=1)
            else:
                MU_deep_De[k] = np.concatenate([v[i] for i in sel], axis=1)
            if isinstance(sel_bg, int):
                MU0_deep[k] = v[sel_bg]
            elif isinstance(sel_bg, slice):
                MU0_deep[k] = np.concatenate(v[sel_bg], axis=1)
            else:
                MU0_deep[k] = np.concatenate([v[i] for i in sel_bg], axis=1)
    else:
        MU_deep_De = deep_output
    for k in deep_output.keys():
        if k in train_names:
            datasets[k].set_MU_deep(MU_deep_De[k])
    return MU_deep_De



MU_deep_1D = obtain_results1(model_20_1d,param_input=[PHI_meas_16x15_],sel_bg=1)
MU_deep_2D = obtain_results2(model_20_2d,param_input=[PHI_meas_16x15_],sel_bg=1)
MU_deep_De = obtain_results3(model_20_de,param_input=[PHI_meas_16x15_],sel_bg=1)






def plot_result_ablation2(s_num, dataset_name='training', color_range='normal', vmin=None, vmax=None,
                roc=None, roc_index=None, contrast=False, title=None):
    if not isinstance(vmin, list) or len(vmin) != 2:
        vmin = [None, None]
    if not isinstance(vmax, list) or len(vmax) != 2:
        vmax = [None, None]
    if title is None:
        title = ''
    title = str(title)
    scale = d[dataset_name][s_num] / 2
    plt.rcParams.update({'font.size': 10})
    fig, axes = plt.subplots(nrows=2, ncols=4, sharex=True, sharey=True, figsize=(20, 10),
                             gridspec_kw={'hspace': 0.1, 'wspace': 0.1})
    plt.rcParams.update({'font.size': 18})
    fig.suptitle(title, fontsize=10)
    plt.rcParams.update({'font.size': 10})
    axes = axes.flatten()
    for ax in axes:
        ax.set_aspect('equal')

    image_dense = MU_deep_De[dataset_name][s_num]
    truth_image = MU[dataset_name][s_num]
    image_2d    = MU_deep_2D[dataset_name][s_num]
    try:
        image_1d = MU_deep_1D[dataset_name][s_num]
    except:
        temp = mask * np.nan
        image_1d = np.concatenate([[temp], [temp]])

    if contrast:
        MU0_      = np.reshape(MU0[dataset_name][s_num],      MU0[dataset_name][s_num].shape      + (1, 1))
        MU0_deep_ = np.reshape(MU0_deep[dataset_name][s_num], MU0_deep[dataset_name][s_num].shape + (1, 1))
        truth_image /= MU0_
        image_1d    /= MU0_deep_
        image_2d    /= MU0_deep_
        image_dense /= MU0_deep_

    gt_vmin = [0, 0]
    if color_range == 'normal':
        gt_vmax = [np.max(mask_image(truth_image[i], -np.inf)) * 2 for i in range(2)]
    elif color_range == 'auto':
        gt_vmin = [np.min(mask_image(truth_image[i],  np.inf)) for i in range(2)]
        gt_vmax = [np.max(mask_image(truth_image[i], -np.inf)) for i in range(2)]
    else:
        gt_vmin = [
            vmin[i] if vmin[i] is not None else np.min(mask_image(truth_image[i],  np.inf))
            for i in range(2)
        ]
        gt_vmax = [
            vmax[i] if vmax[i] is not None else np.max(mask_image(truth_image[i], -np.inf))
            for i in range(2)
        ]

    images = [
        axes[0].pcolormesh(X2 * scale, Y2 * scale, mask_image(truth_image[0], np.nan), cmap='inferno'),
        axes[1].pcolormesh(X2 * scale, Y2 * scale, mask_image(image_1d[0],    np.nan), cmap='inferno'),
        axes[2].pcolormesh(X2 * scale, Y2 * scale, mask_image(image_2d[0],    np.nan), cmap='inferno'),
        axes[3].pcolormesh(X2 * scale, Y2 * scale, mask_image(image_dense[0], np.nan), cmap='inferno'),
    ]
    axes[0].set_title('Ground Truth', fontsize=15)
    axes[1].set_title('CAFNet',        fontsize=15)
    axes[2].set_title('Fine-Tuned',    fontsize=15)
    axes[3].set_title('CoDA',          fontsize=15)
    axes[0].set_ylabel(r'$\mu_a$', fontsize=20)

    images += [
        axes[4].pcolormesh(X2 * scale, Y2 * scale, mask_image(truth_image[1], np.nan), cmap='jet'),
        axes[5].pcolormesh(X2 * scale, Y2 * scale, mask_image(image_1d[1],    np.nan), cmap='jet'),
        axes[6].pcolormesh(X2 * scale, Y2 * scale, mask_image(image_2d[1],    np.nan), cmap='jet'),
        axes[7].pcolormesh(X2 * scale, Y2 * scale, mask_image(image_dense[1], np.nan), cmap='jet'),
    ]
    axes[4].set_ylabel(r"$\mu^\prime_s$", labelpad=0.9, fontsize=20)

    for im in images[:4]:
        im.set_clim(vmin=gt_vmin[0], vmax=gt_vmax[0])
    for im in images[4:]:
        im.set_clim(vmin=gt_vmin[1], vmax=gt_vmax[1])

    for im, ax in zip(images[0:4], axes[0:4]):
        fig.colorbar(im, ax=ax, orientation='horizontal', fraction=.1)

    for im, ax in zip(images[4:8], axes[4:8]):
        fig.colorbar(im, ax=ax, orientation='horizontal', fraction=.1)

    if roc_index is not None:
        roc = roc_data[dataset_name][s_num, roc_index]
    if roc is not None:
        for ax in axes[:4]:
            ax.plot(roc * np.cos(p_theta), roc * np.sin(p_theta), c='#00ff00')
        for ax in axes[4:]:
            ax.plot(roc * np.cos(p_theta), roc * np.sin(p_theta), c='#000000')

    for ax in axes:
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    plt.show()



for i in range(0,12):
    plot_result_ablation2(i,'calibrated',None)
