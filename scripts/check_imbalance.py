import os
import json
import rasterio
from pathlib import Path
import numpy as np

DATA_DIR = Path('data/raw')
IMAGES_DIR = DATA_DIR / 'Onera Satellite Change Detection dataset - Images'
TRAIN_LABELS_DIR = DATA_DIR / 'Onera Satellite Change Detection dataset - Train Labels'
TEST_LABELS_DIR = DATA_DIR / 'Onera Satellite Change Detection dataset - Test Labels'

def check_alignment_and_imbalance():
    train_scenes = sorted([d.name for d in TRAIN_LABELS_DIR.iterdir() if d.is_dir()])
    test_scenes = sorted([d.name for d in TEST_LABELS_DIR.iterdir() if d.is_dir()])
    
    # Select 3 validation scenes arbitrarily, maybe with different characteristics
    val_scenes = ['bordeaux', 'montpellier', 'rennes'] # montpellier is in test? Let's check splits.json.
    # Ah, montpellier is in test! We must only use train scenes for validation.
    val_scenes = ['bordeaux', 'mumbai', 'paris']
    train_only_scenes = [s for s in train_scenes if s not in val_scenes]
    
    Path('results/splits.json').parent.mkdir(parents=True, exist_ok=True)
    with open('results/splits.json', 'w') as f:
        json.dump({'train': train_only_scenes, 'val': val_scenes, 'test': test_scenes}, f, indent=2)
        
    print(f"Validation scenes: {val_scenes}")
    
    # Calculate imbalance and shape checking
    imbalance_data = {}
    total_change = 0
    total_pixels = 0
    
    for scene_id in train_scenes:
        label_path = TRAIN_LABELS_DIR / scene_id / 'cm' / f'{scene_id}-cm.tif'
        img_path = IMAGES_DIR / scene_id / 'imgs_1_rect' / 'B02.tif'
        
        with rasterio.open(img_path) as src_img:
            img_shape = src_img.shape
            
        with rasterio.open(label_path) as src_lbl:
            lbl_shape = src_lbl.shape
            lbl_data = src_lbl.read(1)
            
        if img_shape != lbl_shape:
            print(f"Shape mismatch in {scene_id}: img={img_shape}, lbl={lbl_shape}")
            
        change_pixels = np.sum(lbl_data == 1)
        valid_pixels = np.prod(lbl_shape)
        
        # In OSCD, 1 is change, 0 is no change. Wait, some borders might be 2? 
        # The README says 0 is no change, 1 is change.
        fraction = change_pixels / valid_pixels * 100
        
        imbalance_data[scene_id] = {
            'change_pixels': int(change_pixels),
            'total_pixels': int(valid_pixels),
            'fraction': fraction
        }
        total_change += change_pixels
        total_pixels += valid_pixels
        
    print(f"Overall change fraction in training set: {total_change / total_pixels * 100:.2f}%")
    
    # Write to a JSON for reference or we just print it to append to data/README.md
    with open('results/imbalance.json', 'w') as f:
        json.dump(imbalance_data, f, indent=2)
        
    print("Alignment checked and imbalance computed.")

if __name__ == '__main__':
    check_alignment_and_imbalance()
