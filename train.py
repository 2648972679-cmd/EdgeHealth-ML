"""
Training script for the EdgeHealth-ML multimodal HAR model.

Usage:
    python train.py [--epochs 30] [--batch_size 32] [--lr 0.001]
"""
import os
import sys
import argparse
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.download_har import download_and_extract
from utils.data_loader import create_dataloaders, LABELS
from models.multimodal_model import MultimodalHARModel, count_parameters

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
LOG_DIR = os.path.join(PROJECT_ROOT, "runs")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Train EdgeHealth-ML model")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cpu", "cuda"])
    return parser.parse_args()


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for spec, imu, labels in loader:
        spec, imu, labels = spec.to(device), imu.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(spec, imu)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * spec.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == labels).sum().item()
        total += spec.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    for spec, imu, labels in loader:
        spec, imu, labels = spec.to(device), imu.to(device), labels.to(device)

        logits = model(spec, imu)
        loss = criterion(logits, labels)

        total_loss += loss.item() * spec.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == labels).sum().item()
        total += spec.size(0)

        all_preds.extend(pred.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    acc = correct / total
    return total_loss / total, acc, all_preds, all_labels


def main():
    args = parse_args()

    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    # Data
    print("\n[1/4] Preparing data...")
    data_dir = download_and_extract()
    train_loader, test_loader, num_classes = create_dataloaders(
        data_dir, batch_size=args.batch_size
    )
    print(f"  Train samples: {len(train_loader.dataset)}")
    print(f"  Test  samples: {len(test_loader.dataset)}")
    print(f"  Classes: {num_classes} — {LABELS}")

    # Model
    print("\n[2/4] Building model...")
    model = MultimodalHARModel(num_classes=num_classes)
    model.to(device)
    stats = count_parameters(model)
    print(f"  Parameters: {stats['total']:,} total, {stats['trainable']:,} trainable")

    # Optimizer & loss
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )
    criterion = nn.CrossEntropyLoss()

    # Logger
    writer = SummaryWriter(log_dir=os.path.join(LOG_DIR, f"run_{int(time.time())}"))

    # Training loop
    print(f"\n[3/4] Training ({args.epochs} epochs)...")
    best_acc = 0.0
    best_path = os.path.join(CHECKPOINT_DIR, "best_model.pth")

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )
        test_loss, test_acc, _, _ = evaluate(
            model, test_loader, criterion, device
        )
        scheduler.step()

        writer.add_scalars("Loss", {"train": train_loss, "test": test_loss}, epoch)
        writer.add_scalars("Accuracy", {"train": train_acc, "test": test_acc}, epoch)
        writer.add_scalar("LR", optimizer.param_groups[0]["lr"], epoch)

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), best_path)
            improved = " *"
        else:
            improved = ""

        print(
            f"  Epoch {epoch:3d} | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.3f} | "
            f"Test Loss: {test_loss:.4f} Acc: {test_acc:.3f}{improved}"
        )

    writer.close()

    # Final evaluation
    print(f"\n[4/4] Final evaluation — best model (acc={best_acc:.4f})")

    model.load_state_dict(torch.load(best_path, map_location=device))
    test_loss, test_acc, preds, labels = evaluate(
        model, test_loader, criterion, device
    )

    # Per-class accuracy
    from sklearn.metrics import classification_report, confusion_matrix
    print("\n" + "=" * 60)
    print("Classification Report (Test Set)")
    print("=" * 60)
    print(classification_report(labels, preds, target_names=LABELS, digits=4))

    print("Confusion Matrix:")
    cm = confusion_matrix(labels, preds)
    print(cm)

    # Save final model
    final_path = os.path.join(CHECKPOINT_DIR, "final_model.pth")
    torch.save(model.state_dict(), final_path)
    print(f"\nModels saved:")
    print(f"  Best:     {best_path}")
    print(f"  Final:    {final_path}")
    print(f"  Test Acc: {best_acc:.4f}")

    return best_acc


if __name__ == "__main__":
    main()
