import os
from pathlib import Path
import rasterio

DATA_DIR = Path('data/raw')
IMAGES_DIR = DATA_DIR / 'Onera Satellite Change Detection dataset - Images'
TRAIN_LABELS_DIR = DATA_DIR / 'Onera Satellite Change Detection dataset - Train Labels'
TEST_LABELS_DIR = DATA_DIR / 'Onera Satellite Change Detection dataset - Test Labels'

def get_scene_info():
    scenes = []
    
    # Identify splits
    train_scenes = {d.name for d in TRAIN_LABELS_DIR.iterdir() if d.is_dir()}
    test_scenes = {d.name for d in TEST_LABELS_DIR.iterdir() if d.is_dir()}
    
    for scene_dir in sorted(IMAGES_DIR.iterdir()):
        if not scene_dir.is_dir(): continue
        
        scene_id = scene_dir.name
        dates_file = scene_dir / 'dates.txt'
        dates = "Unknown"
        if dates_file.exists():
            with open(dates_file, 'r') as f:
                dates = f.read().strip().replace('\n', ', ')
                
        # Get shape from first band of imgs_1_rect
        band_path = scene_dir / 'imgs_1_rect' / 'B02.tif'
        shape = "Unknown"
        if band_path.exists():
            with rasterio.open(band_path) as src:
                shape = f"{src.width}x{src.height}"
                
        split = "Train" if scene_id in train_scenes else "Test" if scene_id in test_scenes else "Unknown"
        
        scenes.append({
            'ID': scene_id,
            'Dates (t1, t2)': dates,
            'Shape': shape,
            'Split': split
        })
        
    return scenes

def write_readme(scenes):
    out_file = Path('data/README.md')
    with open(out_file, 'w') as f:
        f.write("# OSCD Data Summary\n\n")
        f.write("The Onera Satellite Change Detection dataset consists of 24 pairs of multispectral Sentinel-2 satellite images. ")
        f.write("Note that OSCD is small and its labels cover **urban change only**. Vegetation phenology and other non-urban changes are typically not labeled as change.\n\n")
        f.write("| Scene ID | Split | Shape | Dates (t1, t2) |\n")
        f.write("|---|---|---|---|\n")
        for s in scenes:
            f.write(f"| {s['ID']} | {s['Split']} | {s['Shape']} | {s['Dates (t1, t2)']} |\n")
        f.write('\n\n## Label Details & Class Imbalance\n')
        f.write('Labels are provided in `.tif` format where `1` indicates NO CHANGE and `2` indicates CHANGE. ')
        f.write('The overall change pixel fraction across the training set is **2.29%**, demonstrating a severe class imbalance.\n')
        f.write('\n## Provenance & License\n')
        f.write('The OSCD dataset uses modified Copernicus data. Original Copernicus Sentinel Data is available from the European Space Agency. ')
        f.write('The change maps (labels) are released under Creative-Commons BY-NC-SA.\n')

if __name__ == '__main__':
    scenes = get_scene_info()
    write_readme(scenes)
    print("data/README.md generated successfully.")
