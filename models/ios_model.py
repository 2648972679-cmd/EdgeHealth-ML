"""
iOS-ready model: accepts RAW accelerometer data and runs STFT internally.

Input:  (B, 3, 128)  — raw 3-axis accelerometer, 128 time steps @ 50Hz
Output: (B, 6)       — 6-class activity logits

The STFT spectrogram generation is fully inside the ONNX graph,
so the iOS app only needs to collect raw sensor data — no preprocessing.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as mobilenet


class STFTPreprocessor(nn.Module):
    """On-device STFT spectrogram generator — replaces scipy.signal.stft.

    Input:  (B, 3, 128)   raw accelerometer signals
    Output: (B, 3, 224, 224)  log-magnitude spectrogram images
    """

    def __init__(self, n_fft=64, hop_length=16, win_length=64, target_size=224):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.target_size = target_size
        # Register window as a buffer (saved with the model)
        self.register_buffer("window", torch.hann_window(win_length))

    def forward(self, x):
        """
        Args:
            x: (B, 3, 128) raw 3-channel accelerometer signal
        Returns:
            spec_img: (B, 3, 224, 224) spectrogram images, normalized
        """
        B, C, T = x.shape  # batch, 3 channels, 128 time steps

        # Reshape to (B*C, T) for batched STFT
        x_flat = x.reshape(B * C, T)

        # STFT: uses torch.stft (ONNX-exportable in PyTorch >= 2.0)
        # We use return_complex=False to get (real, imag) on last dim
        # Then compute magnitude = sqrt(real^2 + imag^2)
        stft_result = torch.stft(
            x_flat,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=True,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=True,  # Complex tensor
        )  # (B*C, freq_bins, time_frames)

        # Magnitude spectrum: |STFT|
        mag = torch.abs(stft_result)  # (B*C, freq_bins, time_frames)

        # Log compression: log(1 + x) to compress dynamic range
        log_mag = torch.log1p(mag)

        # Reshape back to (B, C, freq_bins, time_frames)
        _, freq_bins, time_frames = log_mag.shape
        log_mag = log_mag.reshape(B, C, freq_bins, time_frames)

        # Resize to (B, C, 224, 224) via bilinear interpolation
        spec_img = F.interpolate(
            log_mag,
            size=(self.target_size, self.target_size),
            mode="bilinear",
            align_corners=False,
        )  # (B, C, 224, 224)

        # Normalize to [-1, 1] range (matching training Normalize(0.5, 0.5))
        spec_img = (spec_img - 0.5) / 0.5

        return spec_img


class IMUEncoder(nn.Module):
    """1D-CNN encoder for raw IMU time-series data (unchanged from original)."""

    def __init__(self, in_channels=6, hidden=64, out_dim=128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, hidden, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden, hidden * 2, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(hidden * 2),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden * 2, out_dim, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        return x.squeeze(-1)


class VisionEncoder(nn.Module):
    """MobileNetV3-Small encoder (unchanged from original)."""

    def __init__(self, out_dim=576):
        super().__init__()
        backbone = mobilenet.mobilenet_v3_small(weights="DEFAULT")
        self.features = backbone.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.out_dim = out_dim

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        return x.view(x.size(0), -1)


class iOSReadyHARModel(nn.Module):
    """End-to-end HAR model for iOS deployment.

    Input:  raw 6-channel IMU data (B, 6, 128)
            - Channels 0-2: accelerometer (body_acc_x, y, z)
            - Channels 3-5: gyroscope (body_gyro_x, y, z)

    Processing:
        acc[0:3] → STFT → spectrogram → VisionEncoder → 576-dim
        imu[0:6] → IMUEncoder → 128-dim
        concat → MLP → 6-class logits
    """

    def __init__(self, num_classes=6, vision_dim=576, imu_dim=128, fusion_dim=256):
        super().__init__()
        self.stft = STFTPreprocessor(n_fft=64, hop_length=16, win_length=64)
        self.vision_encoder = VisionEncoder(out_dim=vision_dim)
        self.imu_encoder = IMUEncoder(in_channels=6, out_dim=imu_dim)

        self.fusion = nn.Sequential(
            nn.Linear(vision_dim + imu_dim, fusion_dim),
            nn.BatchNorm1d(fusion_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(fusion_dim, num_classes),
        )

    def forward(self, imu_signal):
        """
        Args:
            imu_signal: (B, 6, 128) — 6-channel raw sensor data
        Returns:
            logits: (B, num_classes)
        """
        # Split: first 3 channels = accelerometer → STFT → Vision
        acc = imu_signal[:, :3, :]       # (B, 3, 128)
        spec = self.stft(acc)             # (B, 3, 224, 224)
        v_feat = self.vision_encoder(spec)  # (B, 576)

        # All 6 channels → IMU encoder
        i_feat = self.imu_encoder(imu_signal)  # (B, 128)

        # Fusion
        fused = torch.cat([v_feat, i_feat], dim=1)  # (B, 704)
        return self.fusion(fused)


def count_parameters(model: nn.Module) -> dict:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}


if __name__ == "__main__":
    model = iOSReadyHARModel()
    print(model)
    stats = count_parameters(model)
    print(f"Parameters: {stats['total']:,} total, {stats['trainable']:,} trainable")

    # Test forward pass with random input
    x = torch.randn(4, 6, 128)  # batch=4, 6 channels, 128 time steps
    out = model(x)
    print(f"Input: {x.shape} → Output: {out.shape}")
    print(f"Output (first sample): {out[0].detach()}")
