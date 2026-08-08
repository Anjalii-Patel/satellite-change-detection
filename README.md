# Bi-Temporal Sentinel-2 Change Detection: Classical Differencing vs Siamese Learning

## Overview

This project explores the effectiveness of learned Siamese representations for detecting land-cover change between two Sentinel-2 observations, comparing them with classical image-differencing baselines. 

**Important Context:** Image difference is *not* semantic change. Vegetation phenology (seasonal variation) and illumination differences cause spectral differences but no actual land-cover change. Building construction or deforestation, however, causes both spectral and spatial differences and constitutes real semantic change. Distinguishing between these two types of differences is the central problem this project demonstrates.

## Dataset

This project uses the **OSCD (Onera Satellite Change Detection)** dataset, consisting of multispectral Sentinel-2 pairs with manual urban-change masks. 

**Limitations:** 
- The OSCD dataset is relatively small (24 scene pairs).
- The dataset's labels exclusively cover *urban* change, meaning other types of semantic change (e.g., deforestation) are not explicitly labeled and might be considered false positives if detected, or ignored.
- **Data Level:** The provided images are Level-1C (L1C) Top-of-Atmosphere (TOA) reflectances, not Bottom-of-Atmosphere (BOA) surface reflectances. This lack of atmospheric correction exacerbates differences caused by illumination and atmospheric conditions, feeding directly into the illumination failure category.

## Methodology

The project compares several approaches to change detection:

1. **Classical Baselines:**
   - RGB difference magnitude, thresholded via Otsu's method.
   - NDVI difference.
   - Change Vector Analysis (CVA).
2. **Early-Fusion U-Net:** A standard U-Net architecture where both temporal observations (`t1` and `t2`) are channel-concatenated at the input.
3. **Siamese U-Net:** A U-Net with a shared-weight Siamese encoder for `t1` and `t2`, fusing the features via absolute difference and concatenation before the decoder.

### Evaluation Protocol
- **Imbalance-Aware Evaluation:** Change pixels are rare. The primary metric is **change-class Intersection over Union (IoU)**, not pixel accuracy.
- **Data Splits:** The official training scenes are used to fit models and preprocessing statistics. A subset of training scenes is held out as a validation set for threshold tuning (e.g., Otsu). Official test scenes are evaluated strictly once.
- **Reporting:** Metrics are reported as Mean ± Standard Deviation across multiple training seeds.

## Setup & Reproducibility

### Installation
Ensure Python >= 3.9 is installed. Install the package and its dependencies:

```bash
pip install -e ".[dev]"
```

### Data Preparation
Download the OSCD dataset by running the provided script. The script verifies checksums and extracts the data.
```bash
python scripts/download_oscd.py
```
*Note: Raw data is never committed to the repository.*

## Acknowledgements & Citations
- **OSCD Dataset:** Daudt, R. C., Le Saux, B., & Boulch, A. "Urban Change Detection for Multispectral Earth Observation Using Convolutional Neural Networks." IGARSS 2018.
- **rs-toolkit:** Remote Sensing Toolkit by Anjalii-Patel.
