import json
import numpy as np
from pathlib import Path

from scd.data import load_scene, get_splits
from scd.classical import compute_rgb_diff, compute_ndvi_diff, compute_cva, fit_threshold, predict_change
from scd.evaluate import compute_metrics, print_metrics

ROOT = Path(__file__).resolve().parents[1]

def load_norm_stats():
    with open(ROOT / "results" / "norm_stats.json", "r") as f:
        stats = json.load(f)
    means = np.array([stats[f"band_{i}"]["mean"] for i in range(4)])[:, None, None]
    stds = np.array([stats[f"band_{i}"]["std"] for i in range(4)])[:, None, None]
    return means, stds

def normalize(img, means, stds):
    return (img - means) / np.maximum(stds, 1e-8)

def run_baselines():
    splits = get_splits()
    means, stds = load_norm_stats()
    
    print("Loading validation scenes to fit Otsu thresholds...")
    val_rgb_diffs, val_ndvi_diffs, val_cva_diffs, val_labels = [], [], [], []
    for scene in splits["val"]:
        t1, t2, lbl = load_scene(scene, "val")
        
        t1_norm = normalize(t1, means, stds)
        t2_norm = normalize(t2, means, stds)
        
        val_rgb_diffs.append(compute_rgb_diff(t1_norm, t2_norm))
        val_ndvi_diffs.append(compute_ndvi_diff(t1, t2)) # NDVI uses raw reflectance!
        val_cva_diffs.append(compute_cva(t1_norm, t2_norm))
        val_labels.append(lbl)
        
    print("Fitting thresholds...")
    rgb_thresh = fit_threshold(val_rgb_diffs, val_labels)
    ndvi_thresh = fit_threshold(val_ndvi_diffs, val_labels)
    cva_thresh = fit_threshold(val_cva_diffs, val_labels)
    
    print(f"Thresholds -> RGB: {rgb_thresh:.4f}, NDVI: {ndvi_thresh:.4f}, CVA: {cva_thresh:.4f}")
    
    print("\nEvaluating on test scenes...")
    def create_metrics_dict():
        return {"iou": [], "f1": [], "precision": [], "recall": [], "tp": 0, "fp": 0, "fn": 0, "tn": 0}
        
    metrics_e0 = create_metrics_dict()
    metrics_rgb = create_metrics_dict()
    metrics_ndvi = create_metrics_dict()
    metrics_cva = create_metrics_dict()
    
    def update_metrics(m_dict, m_new):
        for k in ["iou", "f1", "precision", "recall"]:
            m_dict[k].append(m_new[k])
        for k in ["tp", "fp", "fn", "tn"]:
            m_dict[k] += m_new[k]
            
    for scene in splits["test"]:
        t1, t2, lbl = load_scene(scene, "test")
        t1_norm = normalize(t1, means, stds)
        t2_norm = normalize(t2, means, stds)
        
        # E0: Predict no change (all zeros)
        e0_pred = np.zeros_like(lbl)
        update_metrics(metrics_e0, compute_metrics(lbl, e0_pred))
        
        # E1: RGB Diff
        rgb_pred = predict_change(compute_rgb_diff(t1_norm, t2_norm), rgb_thresh)
        update_metrics(metrics_rgb, compute_metrics(lbl, rgb_pred))
        
        # E1: NDVI Diff
        ndvi_pred = predict_change(compute_ndvi_diff(t1, t2), ndvi_thresh)
        update_metrics(metrics_ndvi, compute_metrics(lbl, ndvi_pred))
        
        # E1: CVA
        cva_pred = predict_change(compute_cva(t1_norm, t2_norm), cva_thresh)
        update_metrics(metrics_cva, compute_metrics(lbl, cva_pred))

    def summarize(name, m_dict):
        m_avg = {k: np.mean(v) for k, v in m_dict.items() if k in ["iou", "f1", "precision", "recall"]}
        print_metrics(name, m_avg)
        
    summarize("E0: Predict No Change", metrics_e0)
    summarize("E1: RGB Difference", metrics_rgb)
    summarize("E1: NDVI Difference", metrics_ndvi)
    summarize("E1: CVA", metrics_cva)
    
    import csv
    rows = {"E0_NoChange": metrics_e0, "E1_RGBDiff": metrics_rgb, "E1_NDVIDiff": metrics_ndvi, "E1_CVA": metrics_cva}
    with open(ROOT / "results/metrics/e1_classical.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["experiment", "iou", "f1", "precision", "recall", "aggregation"])
        for n, m in rows.items():
            # Mean per scene
            w.writerow([n] + [f"{np.mean(m[k]):.4f}" for k in ("iou", "f1", "precision", "recall")] + ["mean_per_scene"])
            # Pooled
            tp, fp, fn = m["tp"], m["fp"], m["fn"]
            iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            w.writerow([n] + [f"{v:.4f}" for v in (iou, f1, precision, recall)] + ["pooled"])

if __name__ == "__main__":
    run_baselines()
