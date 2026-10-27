"""Per-AR evaluation mask, replicating helper_20_1.py lines 553-579 (which hard-code one AR by editing line 571)."""
import numpy as np


def ellipse_mask(ar, n=64):
    cx = cy = n // 2
    x, y = np.meshgrid(np.arange(n), np.arange(n))
    dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    rx = n // 2
    ry = round(rx / float(ar))
    return (dist <= rx) & ((y - cy) ** 2 / ry ** 2 + (x - cx) ** 2 / rx ** 2 <= 1)


def set_mask(H, mask):
    """Overwrite the module-level globals that ground_truth_vs_estimated() reads."""
    mask = np.asarray(mask).astype(bool)   # the dataset's own mask (the helper's formula is offset by half a pixel)
    H.mask = mask
    with np.errstate(all="ignore"):
        H.empty_image = mask / mask * ~mask
    H.X_kmeans = (H.X[mask].reshape(-1, 1) + 1) * H.c1_kmeans
    H.Y_kmeans = (H.Y[mask].reshape(-1, 1) + 1) * H.c1_kmeans
    H.boundary_kmeans = (mask & H.boundary_mask)[mask]
    return mask
