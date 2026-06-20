"""
Quantize the trained model (INT8 dynamic quantization) and export to ONNX.

Outputs:
  - checkpoints/model_int8.pth       Quantized PyTorch model
  - checkpoints/model_fp32.onnx      ONNX model (FP32)
  - checkpoints/model_int8.onnx      ONNX model (quantized) [if supported]
"""
import os
import sys
import argparse
import time

import numpy as np
import torch
import torch.nn as nn
import torch.quantization as quant

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models.multimodal_model import MultimodalHARModel, count_parameters

CHECKPOINT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")


def parse_args():
    parser = argparse.ArgumentParser(description="Quantize & export model")
    parser.add_argument("--model", type=str, default="checkpoints/best_model.pth",
                        help="Path to trained model weights")
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args()


def export_onnx(model, filepath, spec_shape=(1, 3, 224, 224), imu_shape=(1, 6, 128)):
    """Export model to ONNX format."""
    model.eval()
    model.cpu()

    dummy_spec = torch.randn(*spec_shape)
    dummy_imu = torch.randn(*imu_shape)

    torch.onnx.export(
        model,
        (dummy_spec, dummy_imu),
        filepath,
        input_names=["spectrogram", "imu_signal"],
        output_names=["logits"],
        dynamic_axes={
            "spectrogram": {0: "batch"},
            "imu_signal": {0: "batch"},
            "logits": {0: "batch"},
        },
        opset_version=18,  # Use 18+ for PyTorch 2.12 compatibility
    )
    print(f"  ONNX exported -> {filepath} ({os.path.getsize(filepath):,} bytes)")


def quantize_model(model, train_loader, device="cpu"):
    """Apply dynamic INT8 quantization to Linear and Conv1d layers."""
    model.eval()
    model.to("cpu")

    # Dynamic quantization: only Linear/Conv1d weights -> INT8
    model_q = torch.quantization.quantize_dynamic(
        model,
        {nn.Linear, nn.Conv1d},
        dtype=torch.qint8,
    )
    return model_q


def compare_models(model_fp32, model_int8, test_loader, device="cpu"):
    """Accuracy & latency comparison between FP32 and INT8 models."""
    from sklearn.metrics import accuracy_score

    @torch.no_grad()
    def run_inference(model, loader):
        all_preds, all_labels = [], []
        times = []
        model.eval()
        model.to(device)

        for spec, imu, labels in loader:
            spec, imu = spec.to(device), imu.to(device)

            t0 = time.perf_counter()
            logits = model(spec, imu)
            times.append(time.perf_counter() - t0)

            pred = logits.argmax(dim=1).cpu()
            all_preds.extend(pred.tolist())
            all_labels.extend(labels.tolist())

        acc = accuracy_score(all_labels, all_preds)
        avg_time = np.mean(times) * 1000  # ms per batch
        return acc, avg_time, len(all_labels)

    print("\n" + "=" * 60)
    print("FP32 vs INT8 Comparison")
    print("=" * 60)

    acc_fp32, lat_fp32, n = run_inference(model_fp32, test_loader)
    acc_int8, lat_int8, _ = run_inference(model_int8, test_loader)

    # Model size
    fp32_path = os.path.join(CHECKPOINT_DIR, "_tmp_fp32.pth")
    int8_path = os.path.join(CHECKPOINT_DIR, "_tmp_int8.pth")
    torch.save(model_fp32.state_dict(), fp32_path)
    torch.save(model_int8.state_dict(), int8_path)
    size_fp32 = os.path.getsize(fp32_path) / 1024  # KB
    size_int8 = os.path.getsize(int8_path) / 1024  # KB
    os.remove(fp32_path)
    os.remove(int8_path)

    # Parameters
    fp32_params = count_parameters(model_fp32)["total"]
    int8_params = count_parameters(model_int8)["total"]

    print(f"{'Metric':<30} {'FP32':>12} {'INT8':>12} {'Change':>12}")
    print("-" * 66)
    print(f"{'Accuracy':<30} {acc_fp32:>11.4f} {acc_int8:>11.4f} {'':>12}")
    print(f"{'Latency (ms/batch)':<30} {lat_fp32:>11.2f} {lat_int8:>11.2f} {lat_fp32/lat_int8:>11.2f}x")
    print(f"{'Model size (KB)':<30} {size_fp32:>11.1f} {size_int8:>11.1f} {size_fp32/size_int8:>11.2f}x")
    print(f"{'Parameters':<30} {fp32_params:>11,} {int8_params:>11,} {'':>12}")

    return {
        "acc_fp32": acc_fp32, "acc_int8": acc_int8,
        "lat_fp32": lat_fp32, "lat_int8": lat_int8,
        "size_fp32": size_fp32, "size_int8": size_int8,
    }


def main():
    args = parse_args()

    print("[1/4] Loading data...")
    from data.download_har import download_and_extract
    from utils.data_loader import create_dataloaders
    data_dir = download_and_extract()
    _, test_loader, num_classes = create_dataloaders(data_dir, batch_size=32)

    print("\n[2/4] Loading FP32 model...")
    model_fp32 = MultimodalHARModel(num_classes=num_classes)
    model_path = args.model
    if os.path.exists(model_path):
        model_fp32.load_state_dict(torch.load(model_path, map_location="cpu"))
        print(f"  Loaded weights from {model_path}")
    else:
        print(f"  [WARN] {model_path} not found -- using untrained model for demo")

    model_fp32.eval()

    # Export FP32 ONNX
    print("\n[3/4] Exporting ONNX...")
    onnx_fp32 = os.path.join(CHECKPOINT_DIR, "model_fp32.onnx")
    export_onnx(model_fp32, onnx_fp32)

    # Dynamic quantization
    print("\n[4/4] Applying dynamic quantization...")
    model_int8 = quantize_model(model_fp32, test_loader)

    # Export INT8 ONNX (may not be supported with newer PyTorch)
    onnx_int8 = os.path.join(CHECKPOINT_DIR, "model_int8.onnx")
    try:
        export_onnx(model_int8, onnx_int8)
    except Exception as e:
        print(f"  [SKIP] INT8 ONNX export not supported with PyTorch 2.12+")
        print(f"  INT8 PyTorch model is saved for inference.")

    # Save quantized model
    torch.save(model_int8.state_dict(), os.path.join(CHECKPOINT_DIR, "model_int8.pth"))
    print(f"  Saved -> {os.path.join(CHECKPOINT_DIR, 'model_int8.pth')}")

    # Compare
    bench = compare_models(model_fp32, model_int8, test_loader)
    print("\nDone!")
    return bench


if __name__ == "__main__":
    main()
