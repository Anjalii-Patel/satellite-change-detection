import json
from pathlib import Path
import numpy as np
import rasterio

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "raw"
IMAGES_DIR = DATA_DIR / "Onera Satellite Change Detection dataset - Images"
TRAIN_LABELS_DIR = DATA_DIR / "Onera Satellite Change Detection dataset - Train Labels"
TEST_LABELS_DIR = DATA_DIR / "Onera Satellite Change Detection dataset - Test Labels"

def get_splits():
    splits_path = ROOT / "results" / "splits.json"
    with open(splits_path, "r") as f:
        return json.load(f)

def _read_band(path):
    with rasterio.open(path) as src:
        # read(1) returns HxW
        return src.read(1).astype(np.float32)

def load_scene(scene_id, split="train", bands=("B02", "B03", "B04", "B08")):
    """
    Loads t1, t2, and label (if available) for a given scene.
    Uses the _rect folders for aligned 10m bands.
    
    Returns:
        t1: np.ndarray [C, H, W]
        t2: np.ndarray [C, H, W]
        label: np.ndarray [H, W] or None
    """
    scene_dir = IMAGES_DIR / scene_id
    
    t1_bands = []
    t2_bands = []
    
    for b in bands:
        t1_bands.append(_read_band(scene_dir / "imgs_1_rect" / f"{b}.tif"))
        t2_bands.append(_read_band(scene_dir / "imgs_2_rect" / f"{b}.tif"))
        
    t1 = np.stack(t1_bands, axis=0)
    t2 = np.stack(t2_bands, axis=0)
    
    label = None
    if split in ["train", "val"]:
        lbl_path = TRAIN_LABELS_DIR / scene_id / "cm" / f"{scene_id}-cm.tif"
    elif split == "test":
        lbl_path = TEST_LABELS_DIR / scene_id / "cm" / f"{scene_id}-cm.tif"
    else:
        lbl_path = None
        
    if lbl_path and lbl_path.exists():
        with rasterio.open(lbl_path) as src:
            raw_label = src.read(1)
            label = np.zeros_like(raw_label, dtype=np.uint8)
            label[raw_label == 2] = 1
            
    return t1, t2, label

import torch
from torch.utils.data import Dataset
from rs_toolkit.tiling import extract_patches

class OSCDPatches(Dataset):
    """
    Extracts 128x128 patches from a list of scenes.
    Normalizes t1 and t2 using pre-computed statistics.
    If oversample=True, it will oversample patches containing change pixels.
    """
    def __init__(self, scenes, split="train", size=128, overlap=64, means=None, stds=None, oversample=False):
        self.patches_t1 = []
        self.patches_t2 = []
        self.patches_lbl = []
        
        for scene in scenes:
            t1, t2, lbl = load_scene(scene, split=split)
            
            if means is not None and stds is not None:
                t1 = (t1 - means) / np.maximum(stds, 1e-8)
                t2 = (t2 - means) / np.maximum(stds, 1e-8)
                
            p_t1, _ = extract_patches(t1, size=size, overlap=overlap)
            p_t2, _ = extract_patches(t2, size=size, overlap=overlap)
            
            if lbl is not None:
                # Add channel dim to label for extract_patches
                lbl_c = lbl[None, ...]
                p_lbl, _ = extract_patches(lbl_c, size=size, overlap=overlap)
                
                # Oversampling logic
                if oversample and split == "train":
                    has_change = np.sum(p_lbl, axis=(1, 2, 3)) > 0
                    change_idx = np.where(has_change)[0]
                    # duplicate change patches
                    if len(change_idx) > 0:
                        p_t1 = np.concatenate([p_t1, p_t1[change_idx]], axis=0)
                        p_t2 = np.concatenate([p_t2, p_t2[change_idx]], axis=0)
                        p_lbl = np.concatenate([p_lbl, p_lbl[change_idx]], axis=0)
                        
                self.patches_lbl.append(p_lbl)
                
            self.patches_t1.append(p_t1)
            self.patches_t2.append(p_t2)
            
        self.patches_t1 = np.concatenate(self.patches_t1, axis=0)
        self.patches_t2 = np.concatenate(self.patches_t2, axis=0)
        if len(self.patches_lbl) > 0:
            self.patches_lbl = np.concatenate(self.patches_lbl, axis=0)
            
    def __len__(self):
        return len(self.patches_t1)
        
    def __getitem__(self, idx):
        t1 = torch.from_numpy(self.patches_t1[idx]).float()
        t2 = torch.from_numpy(self.patches_t2[idx]).float()
        
        sample = {"t1": t1, "t2": t2}
        if len(self.patches_lbl) > 0:
            lbl = torch.from_numpy(self.patches_lbl[idx][0]).float() # [H, W]
            sample["label"] = lbl
            
        return sample
