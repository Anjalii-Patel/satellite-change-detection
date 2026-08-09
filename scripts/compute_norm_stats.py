import json
import numpy as np
from pathlib import Path
from scd.data import load_scene, get_splits

def compute_stats():
    print("Computing normalization statistics on the training set...")
    splits = get_splits()
    train_scenes = splits["train"]
    
    # We will accumulate all pixels for t1 and t2 separately, or together?
    # Usually satellite images from the same sensor share statistics across time,
    # so we can combine t1 and t2 to get robust band-wise statistics.
    all_pixels = []
    
    for scene_id in train_scenes:
        t1, t2, _ = load_scene(scene_id, split="train")
        # t1, t2 shape: (C, H, W) -> reshape to (C, H*W)
        C = t1.shape[0]
        t1_flat = t1.reshape(C, -1)
        t2_flat = t2.reshape(C, -1)
        all_pixels.append(t1_flat)
        all_pixels.append(t2_flat)
        
    all_pixels = np.concatenate(all_pixels, axis=1) # (C, Total_Pixels)
    
    stats = {}
    for c in range(all_pixels.shape[0]):
        band_data = all_pixels[c, :]
        valid_data = band_data[np.isfinite(band_data)]
        
        mean = float(np.mean(valid_data))
        std = float(np.std(valid_data))
        p2 = float(np.percentile(valid_data, 2))
        p98 = float(np.percentile(valid_data, 98))
        
        stats[f"band_{c}"] = {
            "mean": mean,
            "std": std,
            "p2": p2,
            "p98": p98
        }
        
    # Save to results/norm_stats.json
    out_path = Path(__file__).resolve().parents[1] / "results" / "norm_stats.json"
    with open(out_path, "w") as f:
        json.dump(stats, f, indent=2)
        
    print(f"Saved normalization statistics to {out_path}")
    print(json.dumps(stats, indent=2))

if __name__ == '__main__':
    compute_stats()
