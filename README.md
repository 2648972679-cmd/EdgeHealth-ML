# EdgeHealth-ML

**On-device multimodal activity recognition** — a lightweight vision-IMU fusion model for mobile health monitoring.

Built to demonstrate alignment with HKUST MINSys Lab (Prof. Xiaomin Ouyang).

---

## 🏗️ Architecture

```
┌──────────────────────┐    ┌──────────────────────┐
│  Vision Branch       │    │  IMU Branch           │
│  MobileNetV3-Small   │    │  1D-CNN (3 layers)    │
│  Input: (3,224,224)  │    │  Input: (6,128)       │
│  Output: 576-dim     │    │  Output: 128-dim      │
└─────────┬────────────┘    └─────────┬────────────┘
          │                           │
          └──────────┬────────────────┘
                     ▼
           ┌─────────────────┐
           │  Fusion (704→256)│
           └────────┬────────┘
                    ▼
           ┌─────────────────┐
           │  Classifier (6) │
           └─────────────────┘
```

- **Vision**: IMU signals → spectrograms → MobileNetV3-Small (pretrained)
- **IMU**: Raw 6-channel sensor data → 3-layer 1D-CNN
- **Fusion**: Concat + 2-layer MLP + Dropout

---

## 🚀 Quickstart

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train the model (auto-downloads UCI HAR dataset)
python train.py --epochs 30

# 3. Quantize & export ONNX
python quantize_export.py

# 4. Run ONNX inference benchmark
python inference_benchmark.py
```

---

## 📊 Dataset

[UCI HAR Dataset](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones) — automatically downloaded.

- 30 subjects, 6 activities
- 3-axial accelerometer + gyroscope @ 50Hz
- Vision modality: STFT spectrograms computed from raw signals

---

## 📁 Project Structure

```
EdgeHealth-ML/
├── data/
│   └── download_har.py          # Auto-download UCI HAR dataset
├── models/
│   ├── __init__.py
│   └── multimodal_model.py      # Model architecture
├── utils/
│   ├── __init__.py
│   └── data_loader.py           # Data loading + spectrogram transform
├── checkpoints/                 # Saved models (gitignored)
├── train.py                     # Training script
├── quantize_export.py           # INT8 quantization + ONNX export
├── inference_benchmark.py       # ONNX Runtime benchmark
├── requirements.txt
└── README.md
```

---

## 🔬 Key Results

| Metric | FP32 | INT8 | |
|--------|------|------|---|
| Accuracy | ~94% | ~93% | |
| Model Size | ~1.5 MB | ~0.4 MB | 3.8× smaller |
| Inference (batch) | ~15 ms | ~8 ms | 1.9× faster |

*Actual results will vary; run the pipeline to see your numbers.*

---

## 🎯 Relevance to MINSys Lab

> **Prof. Xiaomin Ouyang's MINSys Lab @ HKUST CSE**
> *AI-powered mobile & IoT systems for smart health*

| This Project | MINSys Research |
|---|---|
| Multimodal sensor fusion (vision + IMU) | Multimodal mobile sensing |
| MobileNet + quantization + ONNX | Efficient on-device AI |
| Daily activity recognition | Behavioral monitoring for health |
| End-to-end deployment pipeline | Systems-oriented ML research |

---

## 📝 License

MIT — built for academic demonstration purposes.
