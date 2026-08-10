import os
import csv
import json
import yaml
import torch
import numpy as np
import argparse
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm

from scd.data import OSCDPatches, get_splits
from scd.models import EarlyFusionUNet
from scd.losses import BCEDiceLoss
from scd.evaluate import compute_metrics, print_metrics

ROOT = Path(__file__).resolve().parents[2]

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_norm_stats():
    with open(ROOT / "results" / "norm_stats.json", "r") as f:
        stats = json.load(f)
    means = np.array([stats[f"band_{i}"]["mean"] for i in range(4)])[:, None, None]
    stds = np.array([stats[f"band_{i}"]["std"] for i in range(4)])[:, None, None]
    return means, stds

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for batch in tqdm(loader, desc="Train", leave=False):
        t1 = batch["t1"].to(device)
        t2 = batch["t2"].to(device)
        lbl = batch["label"].to(device)
        
        optimizer.zero_grad()
        out = model(t1, t2)
        loss = criterion(out, lbl)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
    return total_loss / len(loader)

def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    metrics_accum = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Eval", leave=False):
            t1 = batch["t1"].to(device)
            t2 = batch["t2"].to(device)
            lbl = batch["label"].to(device)
            
            out = model(t1, t2)
            loss = criterion(out, lbl)
            total_loss += loss.item()
            
            # metrics
            preds = (torch.sigmoid(out) > 0.5).long()
            lbl_b = lbl.long()
            if preds.dim() == 4:
                preds = preds.squeeze(1)
            
            metrics_accum["tp"] += (preds & lbl_b).sum().item()
            metrics_accum["fp"] += (preds & ~lbl_b).sum().item()
            metrics_accum["fn"] += (~preds & lbl_b).sum().item()
            metrics_accum["tn"] += (~preds & ~lbl_b).sum().item()
            
    avg_loss = total_loss / len(loader)
    tp, fp, fn = metrics_accum["tp"], metrics_accum["fp"], metrics_accum["fn"]
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
    
    return avg_loss, iou, metrics_accum

def run(config_path, seed_override=None):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
        
    exp_name = cfg["experiment_name"]
    seed = seed_override if seed_override is not None else cfg["seed"]
    set_seed(seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running {exp_name} on {device} (seed={seed})")
    
    splits = get_splits()
    means, stds = load_norm_stats()
    
    train_ds = OSCDPatches(
        splits["train"], "train", 
        size=cfg["data"]["size"], overlap=cfg["data"]["overlap"],
        means=means, stds=stds, oversample=cfg["data"]["oversample"]
    )
    val_ds = OSCDPatches(
        splits["val"], "val",
        size=cfg["data"]["size"], overlap=0, # no overlap needed for val patches, or 64?
        means=means, stds=stds, oversample=False
    )
    test_ds = OSCDPatches(
        splits["test"], "test",
        size=cfg["data"]["size"], overlap=0,
        means=means, stds=stds, oversample=False
    )
    
    train_loader = DataLoader(train_ds, batch_size=cfg["data"]["batch_size"], shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=cfg["data"]["batch_size"], shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=cfg["data"]["batch_size"], shuffle=False, num_workers=0)
    
    model = EarlyFusionUNet(
        in_channels=cfg["model"]["in_channels"],
        features=cfg["model"]["features"]
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["train"]["lr"])
    criterion = BCEDiceLoss(bce_weight=cfg["train"]["bce_weight"])
    
    best_iou = -1.0
    patience_counter = 0
    epochs = cfg["train"]["epochs"]
    patience = cfg["train"]["patience"]
    
    chkpt_dir = ROOT / "results" / "checkpoints"
    chkpt_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = chkpt_dir / f"{exp_name}_s{seed}.pt"
    
    for ep in range(epochs):
        t_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        v_loss, v_iou, _ = eval_epoch(model, val_loader, criterion, device)
        
        print(f"Ep {ep:02d}: Train Loss={t_loss:.4f} | Val Loss={v_loss:.4f} | Val IoU={v_iou:.4f}")
        
        if v_iou > best_iou:
            best_iou = v_iou
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"  --> Saved new best model (IoU: {best_iou:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {ep}")
                break
                
    # Test Evaluation
    print("Loading best model for test evaluation...")
    model.load_state_dict(torch.load(best_model_path))
    _, t_iou, t_mets = eval_epoch(model, test_loader, criterion, device)
    
    tp, fp, fn = t_mets["tp"], t_mets["fp"], t_mets["fn"]
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    print("\nTest Results (Pooled):")
    print(f"IoU:       {t_iou:.4f}")
    print(f"F1:        {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    
    import subprocess
    try:
        git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").strip()
    except Exception:
        git_commit = "unknown"
        
    out_csv = ROOT / "results" / "metrics" / f"{exp_name}_results.csv"
    file_exists = out_csv.exists()
    
    with open(out_csv, "a", newline="") as f:
        w = csv.writer(f)
        if not file_exists:
            w.writerow(["experiment", "seed", "git_commit", "iou", "f1", "precision", "recall", "aggregation"])
        w.writerow([exp_name, seed, git_commit, f"{t_iou:.4f}", f"{f1:.4f}", f"{precision:.4f}", f"{recall:.4f}", "pooled"])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--seed", type=int, default=None, help="Override seed from config")
    args = parser.parse_args()
    run(args.config, args.seed)
