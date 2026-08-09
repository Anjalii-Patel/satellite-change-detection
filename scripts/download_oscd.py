import os
import urllib.request
import hashlib
import zipfile
from pathlib import Path

URLS = {
    'Onera Satellite Change Detection dataset - Images.zip': 'https://hf.co/datasets/hkristen/oscd/resolve/4958d786c1389ede1511d91a6ecf1a75c4074933/Onera%20Satellite%20Change%20Detection%20dataset%20-%20Images.zip',
    'Onera Satellite Change Detection dataset - Train Labels.zip': 'https://hf.co/datasets/hkristen/oscd/resolve/4958d786c1389ede1511d91a6ecf1a75c4074933/Onera%20Satellite%20Change%20Detection%20dataset%20-%20Train%20Labels.zip',
    'Onera Satellite Change Detection dataset - Test Labels.zip': 'https://hf.co/datasets/hkristen/oscd/resolve/4958d786c1389ede1511d91a6ecf1a75c4074933/Onera%20Satellite%20Change%20Detection%20dataset%20-%20Test%20Labels.zip',
}

SHA256S = {
    'Onera Satellite Change Detection dataset - Images.zip': '940b87887511058a933e67cd6d0e43e2eb825a55d8e79a50983dee7f23003656',
    'Onera Satellite Change Detection dataset - Train Labels.zip': '89fb54cd12ad0dbea6c447528139dec305b865294215434bf6dd170fb8fd3ca5',
    'Onera Satellite Change Detection dataset - Test Labels.zip': '2e195eaa1b788b99fa93ea8073e3780bc0b763000b0c49dbf70548acf1e5d67d',
}

DATA_DIR = Path('data/raw')

def compute_sha256(filepath):
    sha256 = hashlib.sha256()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha256.update(chunk)
    return sha256.hexdigest()

def download_and_extract():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    for filename, url in URLS.items():
        filepath = DATA_DIR / filename
        if not filepath.exists():
            print(f"Downloading {filename}...")
            urllib.request.urlretrieve(url, filepath)
        else:
            print(f"{filename} already exists.")
            
        print(f"Verifying checksum for {filename}...")
        checksum = compute_sha256(filepath)
        if checksum != SHA256S[filename]:
            raise ValueError(f"Checksum mismatch for {filename}! Expected {SHA256S[filename]}, got {checksum}")
        print("Checksum OK.")
        
        print(f"Extracting {filename}...")
        with zipfile.ZipFile(filepath, 'r') as zip_ref:
            zip_ref.extractall(DATA_DIR)
        print(f"Extracted {filename}.")

if __name__ == '__main__':
    download_and_extract()
