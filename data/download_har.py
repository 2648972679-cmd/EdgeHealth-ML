"""
Download and extract the UCI HAR Dataset (Human Activity Recognition).
https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones

The dataset contains smartphone accelerometer + gyroscope data
from 30 subjects performing 6 activities:
  - walking, walking_upstairs, walking_downstairs, sitting, standing, laying
"""
import os
import sys
import zipfile
from urllib.request import urlretrieve

# Multiple mirror URLs — tries in order
HAR_URLS = [
    "https://archive.ics.uci.edu/static/public/240/"
    "human+activity+recognition+using+smartphones.zip",
    "https://github.com/guillaume-chevalier/HAR-stacked-residual-bidir-LSTMs/"
    "raw/master/data/UCI%20HAR%20Dataset.zip",
]

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
EXTRACT_DIR = os.path.join(DATA_DIR, "UCI_HAR_Dataset")


def download_and_extract():
    """Download and extract the UCI HAR Dataset."""
    zip_path = os.path.join(DATA_DIR, "har.zip")

    if os.path.exists(EXTRACT_DIR):
        print(f"[OK] Dataset already exists at {EXTRACT_DIR}")
        return EXTRACT_DIR

    # Try each mirror URL
    downloaded = False
    for url in HAR_URLS:
        try:
            print(f"Trying: {url[:80]}...")
            urlretrieve(url, zip_path)
            # Verify the downloaded file is a valid zip
            with zipfile.ZipFile(zip_path, "r") as test:
                pass
            downloaded = True
            print(f"[OK] Downloaded to {zip_path}")
            break
        except Exception as e:
            print(f"  Failed: {e}")
            if os.path.exists(zip_path):
                os.remove(zip_path)
            continue

    if not downloaded:
        print("[ERROR] All download mirrors failed. "
              "Please download the UCI HAR Dataset manually and place it at:")
        print(f"  {EXTRACT_DIR}")
        print("Expected structure: UCI_HAR_Dataset/train/Inertial Signals/...")
        sys.exit(1)

    with zipfile.ZipFile(zip_path, "r") as zf:
        # The zip contains a top-level 'UCI HAR Dataset' folder (or 'UCI HAR Dataset')
        zf.extractall(DATA_DIR)
    print(f"[OK] Extracted to {EXTRACT_DIR}")

    os.remove(zip_path)
    print(f"[OK] Removed {zip_path}")
    return EXTRACT_DIR


if __name__ == "__main__":
    download_and_extract()
