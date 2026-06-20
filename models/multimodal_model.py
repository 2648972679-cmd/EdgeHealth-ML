"""
Multi-modal Human Activity Recognition model.

Architecture:
  ┌─────────────────────────┐    ┌─────────────────────────┐
  │  Vision Branch          │    │  IMU Branch             │
  │  MobileNetV3-Small      │    │  1D-CNN (3-layer)       │
  │  Input: (3, 224, 224)   │    │  Input: (6, 128)        │
  │  Output: 576-dim        │    │  Output: 128-dim        │
  └──────────┬──────────────┘    └──────────┬──────────────┘
             │                              │
             └──────────┬───────────────────┘
                        ▼
              ┌─────────────────┐
              │  Fusion Layer   │  concat → Linear(576+128, 256)
              └────────┬────────┘
                       ▼
              ┌─────────────────┐
              │  Classification │  Linear(256, 6)
              └─────────────────┘
"""
import torch
import torch.nn as nn
import torchvision.models as mobilenet


class IMUEncoder(nn.Module):
    """1D-CNN encoder for raw IMU time-series data.

    Input:  (B, 6, 128)  — 6 channels (3 acc + 3 gyro), 128 time steps
    Output: (B, 128)     — compact feature vector
    """

    def __init__(self, in_channels=6, hidden=64, out_dim=128):
        super().__init__()
        self.conv = nn.Sequential(
            # Block 1: (B, 6, 128) → (B, 64, 64)
            nn.Conv1d(in_channels, hidden, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            # Block 2: (B, 64, 64) → (B, 128, 32)
            nn.Conv1d(hidden, hidden * 2, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(hidden * 2),
            nn.ReLU(inplace=True),
            # Block 3: (B, 128, 32) → (B, 128, 16)
            nn.Conv1d(hidden * 2, out_dim, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)  # → (B, 128, 1)

    def forward(self, x):
        x = self.conv(x)       # (B, 128, 16)
        x = self.pool(x)       # (B, 128, 1)
        return x.squeeze(-1)   # (B, 128)


class VisionEncoder(nn.Module):
    """MobileNetV3-Small encoder for spectrogram images.

    Input:  (B, 3, 224, 224)
    Output: (B, 576)
    """

    def __init__(self, out_dim=576):
        super().__init__()
        backbone = mobilenet.mobilenet_v3_small(weights="DEFAULT")
        # Remove the final classifier; keep the feature extractor
        self.features = backbone.features  # → (B, 576, 7, 7)
        self.pool = nn.AdaptiveAvgPool2d(1)  # → (B, 576, 1, 1)
        self.out_dim = out_dim

    def forward(self, x):
        x = self.features(x)   # (B, 576, 7, 7)
        x = self.pool(x)       # (B, 576, 1, 1)
        return x.view(x.size(0), -1)  # (B, 576)


class MultimodalHARModel(nn.Module):
    """Multi-modal fusion model for Human Activity Recognition.

    Fuses vision (spectrogram) and IMU (raw sensor) branches
    with a 2-layer MLP fusion head.
    """

    def __init__(self, num_classes=6, vision_dim=576, imu_dim=128, fusion_dim=256):
        super().__init__()
        self.vision_encoder = VisionEncoder(out_dim=vision_dim)
        self.imu_encoder = IMUEncoder(in_channels=6, out_dim=imu_dim)

        self.fusion = nn.Sequential(
            nn.Linear(vision_dim + imu_dim, fusion_dim),
            nn.BatchNorm1d(fusion_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(fusion_dim, num_classes),
        )

    def forward(self, spectrogram, imu_signal):
        """
        Args:
            spectrogram: (B, 3, 224, 224)  float32
            imu_signal:  (B, 6, 128)       float32
        Returns:
            logits: (B, num_classes)
        """
        v_feat = self.vision_encoder(spectrogram)  # (B, 576)
        i_feat = self.imu_encoder(imu_signal)       # (B, 128)
        fused = torch.cat([v_feat, i_feat], dim=1)  # (B, 704)
        return self.fusion(fused)


# ---------------------------------------------------------------------------
# Utility: count parameters
# ---------------------------------------------------------------------------
def count_parameters(model: nn.Module) -> dict:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}


if __name__ == "__main__":
    # Quick smoke test
    model = MultimodalHARModel()
    print(model)
    stats = count_parameters(model)
    print(f"Parameters: {stats['total']:,} total, {stats['trainable']:,} trainable")

    # Test forward pass
    b = 4
    spec = torch.randn(b, 3, 224, 224)
    imu = torch.randn(b, 6, 128)
    out = model(spec, imu)
    print(f"Input: spec {spec.shape}, imu {imu.shape} → Output {out.shape}")
