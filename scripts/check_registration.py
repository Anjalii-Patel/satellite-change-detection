import numpy as np
import rasterio
from skimage.registration import phase_cross_correlation
from skimage.filters import sobel
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "raw"
IMAGES_DIR = DATA_DIR / "Onera Satellite Change Detection dataset - Images"

def check_registration():
    print("Checking t1/t2 phase cross-correlation for each scene...")
    
    # Iterate through all scenes
    for scene_dir in sorted(IMAGES_DIR.iterdir()):
        if not scene_dir.is_dir(): continue
        scene_id = scene_dir.name
        
        t1_path = scene_dir / "imgs_1_rect" / "B02.tif"
        t2_path = scene_dir / "imgs_2_rect" / "B02.tif"
        
        if not t1_path.exists() or not t2_path.exists():
            continue
            
        with rasterio.open(t1_path) as src1, rasterio.open(t2_path) as src2:
            img1 = src1.read(1).astype(np.float32)
            img2 = src2.read(1).astype(np.float32)
        # compute shift using phase cross correlation on gradient maps
        edge1 = sobel(img1)
        edge2 = sobel(img2)
        shift, error, diffphase = phase_cross_correlation(edge1, edge2, upsample_factor=10)
        
        # log the shift
        print(f"Scene: {scene_id:<12} | Shift (y, x): [{shift[0]:.2f}, {shift[1]:.2f}] | Error: {error:.4f}")

if __name__ == '__main__':
    check_registration()
