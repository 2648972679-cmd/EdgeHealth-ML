"""
Multi-modal data loader for UCI HAR Dataset.

Vision modality: spectrograms computed from raw IMU sensor signals (STFT).
  - Shape: [batch, 3, 224, 224] — 3-channel spectrogram image (acc-x, acc-y, acc-z)

IMU modality: raw time-series sensor data.
  - Shape: [batch, 6, 128] — 6 channels (3 acc + 3 gyro), 128 time steps

Labels: 6 activity classes
"""
import os
import numpy as np
import scipy.signal

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms


LABELS = [
    "walking",
    "walking_upstairs",
    "walking_downstairs",
    "sitting",
    "standing",
    "laying",
]

SIGNAL_NAMES = [
    "body_acc_x", "body_acc_y", "body_acc_z",
    "body_gyro_x", "body_gyro_y", "body_gyro_z",
]


def _load_txt(path):
    """Load a whitespace-delimited text file as numpy array."""
    with open(path, "r") as f:
        return np.array(
            [[float(x) for x in line.strip().split()] for line in f]
        )


# ---------------------------------------------------------------------------
# Spectrogram generation (IMU → 2D time-frequency "image")
# ---------------------------------------------------------------------------
def _signal_to_spectrogram(signal_1d, nperseg=64, noverlap=48):
    """Convert a 1D IMU signal to a 2D log-magnitude spectrogram."""
    f, t, Zxx = scipy.signal.stft(
        signal_1d, nperseg=nperseg, noverlap=noverlap, window="hann"
    )
    mag = np.abs(Zxx)
    log_mag = np.log1p(mag)
    return log_mag  # [freq_bins, time_frames]


def _build_spectrogram_image(signals_3ch, target_size=224):
    """
    Build a 3-channel spectrogram 'image' from 3 IMU channels.
    signals_3ch: (3, T) — e.g. body_acc_x, body_acc_y, body_acc_z
    Returns: (3, target_size, target_size) float32 ndarray
    """
    channels = []
    for c in range(3):
        spec = _signal_to_spectrogram(signals_3ch[c])
        # Resize to target_size × target_size via simple binning
        spec = _resize_2d(spec, target_size, target_size)
        channels.append(spec)
    return np.stack(channels, axis=0).astype(np.float32)


def _resize_2d(arr, h, w):
    """Resize a 2D array to (h, w) using mean-pool binning."""
    import math
    H, W = arr.shape
    # Simple crop or pad + interpolate via repeat / mean
    # Use a Fourier-based or simple slice-based approach
    # For simplicity, use scipy's zoom or basic block-mean
    from scipy.ndimage import zoom
    return zoom(arr, (h / H, w / W), order=1)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class HARDataLoader(Dataset):
    """
    Multi-modal HAR dataset.

    Each sample provides:
      - spectrogram: (3, 224, 224) float32  [vision modality]
      - imu_signal:  (6, 128)    float32  [sensor modality]
      - label:        int
    """

    def __init__(
        self,
        data_dir: str,
        split: str = "train",   # "train" or "test"
        seq_len: int = 128,
        spec_size: int = 224,
    ):
        self.data_dir = data_dir
        self.split = split
        self.seq_len = seq_len
        self.spec_size = spec_size

        subdir = "train" if split == "train" else "test"
        raw_dir = os.path.join(data_dir, subdir, "Inertial Signals")

        # Load all 9 raw signal files
        self.body_acc_x = _load_txt(os.path.join(raw_dir, "body_acc_x_" + subdir + ".txt"))
        self.body_acc_y = _load_txt(os.path.join(raw_dir, "body_acc_y_" + subdir + ".txt"))
        self.body_acc_z = _load_txt(os.path.join(raw_dir, "body_acc_z_" + subdir + ".txt"))
        self.body_gyro_x = _load_txt(os.path.join(raw_dir, "body_gyro_x_" + subdir + ".txt"))
        self.body_gyro_y = _load_txt(os.path.join(raw_dir, "body_gyro_y_" + subdir + ".txt"))
        self.body_gyro_z = _load_txt(os.path.join(raw_dir, "body_gyro_z_" + subdir + ".txt"))

        # Load labels
        label_path = os.path.join(data_dir, subdir, "y_" + subdir + ".txt")
        self.labels = _load_txt(label_path).squeeze().astype(np.int64) - 1  # 1-indexed → 0-indexed

        self.num_samples = self.labels.shape[0]
        self.signal_len = self.body_acc_x.shape[1]  # 128

        # Simple normalize transform for spectrogram images
        self.spec_norm = transforms.Normalize(
            mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
        )

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # --- Vision modality: spectrogram from acc channels ---
        acc_signals = np.stack([
            self.body_acc_x[idx],
            self.body_acc_y[idx],
            self.body_acc_z[idx],
        ], axis=0)  # (3, 128)

        spec = _build_spectrogram_image(acc_signals, self.spec_size)
        spec = torch.from_numpy(spec)
        spec = self.spec_norm(spec)

        # --- IMU modality: raw time-series ---
        imu_signal = np.stack([
            self.body_acc_x[idx],
            self.body_acc_y[idx],
            self.body_acc_z[idx],
            self.body_gyro_x[idx],
            self.body_gyro_y[idx],
            self.body_gyro_z[idx],
        ], axis=0)  # (6, 128)
        imu_signal = torch.from_numpy(imu_signal).float()

        label = int(self.labels[idx])
        return spec, imu_signal, label


def create_dataloaders(data_dir, batch_size=32, seq_len=128, spec_size=224):
    """Create train and test DataLoaders."""
    train_ds = HARDataLoader(data_dir, split="train", seq_len=seq_len, spec_size=spec_size)
    test_ds = HARDataLoader(data_dir, split="test", seq_len=seq_len, spec_size=spec_size)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, test_loader, len(LABELS)
