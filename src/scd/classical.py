import numpy as np
from skimage.filters import threshold_otsu
import rs_toolkit.indices as indices

def compute_rgb_diff(t1, t2):
    """
    Computes magnitude of RGB difference.
    t1, t2: [C, H, W] where first 3 channels are RGB
    Returns: [H, W] difference magnitude
    """
    # Euclidean distance across RGB channels
    diff = t2[:3] - t1[:3]
    return np.sqrt(np.sum(diff**2, axis=0))

def compute_ndvi_diff(t1, t2):
    """
    Computes absolute NDVI difference.
    t1, t2: [C, H, W] where B04 is index 2 (Red) and B08 is index 3 (NIR)
    Returns: [H, W] NDVI absolute difference
    """
    ndvi1 = indices.ndvi(t1[3], t1[2])
    ndvi2 = indices.ndvi(t2[3], t2[2])
    return np.abs(ndvi2 - ndvi1)

def compute_cva(t1, t2):
    """
    Computes Change Vector Analysis (magnitude).
    t1, t2: [C, H, W] (uses all channels)
    Returns: [H, W] CVA magnitude
    """
    diff = t2 - t1
    return np.sqrt(np.sum(diff**2, axis=0))

def fit_threshold(diff_maps, labels):
    """
    Fits a threshold on the difference maps to maximize validation IoU.
    diff_maps: list of [H, W] arrays or a stacked array
    labels: list of [H, W] boolean/int arrays
    """
    diff_flat = np.concatenate([d.flatten() for d in diff_maps])
    labels_flat = np.concatenate([l.flatten() for l in labels]).astype(bool)
    
    # Check 100 thresholds between min and max difference
    min_val, max_val = np.nanmin(diff_flat), np.nanmax(diff_flat)
    thresholds = np.linspace(min_val, max_val, 100)
    
    best_iou = -1.0
    best_thresh = min_val
    
    for th in thresholds:
        pred = diff_flat > th
        tp = np.sum(pred & labels_flat)
        fp = np.sum(pred & ~labels_flat)
        fn = np.sum(~pred & labels_flat)
        iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0
        
        if iou > best_iou:
            best_iou = iou
            best_thresh = th
            
    return best_thresh

def predict_change(diff_map, threshold):
    """
    Predicts change mask using a given threshold.
    """
    return (diff_map > threshold).astype(np.uint8)
