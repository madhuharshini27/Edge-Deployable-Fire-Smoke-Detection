<p align="center">
  <img src="https://img.shields.io/badge/YOLOv8n-Ultralytics-blue?style=for-the-badge&logo=yolo" alt="YOLOv8">
  <img src="https://img.shields.io/badge/ONNX-Runtime-orange?style=for-the-badge&logo=onnx" alt="ONNX">
  <img src="https://img.shields.io/badge/Raspberry%20Pi-4-red?style=for-the-badge&logo=raspberrypi" alt="Raspberry Pi">
  <img src="https://img.shields.io/badge/OpenCV-RTSP-green?style=for-the-badge&logo=opencv" alt="OpenCV">
  <img src="https://img.shields.io/badge/Python-3.11-yellow?style=for-the-badge&logo=python" alt="Python">
</p>

<h1 align="center">🔥 Edge-Based Real-Time Fire & Smoke Detection</h1>

<p align="center">
  <strong>YOLOv8 · ONNX Runtime · Raspberry Pi 4 · RTSP IP Cameras</strong>
</p>

<p align="center">
  A production-grade, privacy-preserving edge AI system for real-time fire and smoke detection<br/>
  using existing CCTV infrastructure — <strong>zero cloud dependency</strong>.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/mAP%4050-86.1%25-brightgreen?style=flat-square" alt="mAP">
  <img src="https://img.shields.io/badge/Precision-84.3%25-blue?style=flat-square" alt="Precision">
  <img src="https://img.shields.io/badge/Recall-78.4%25-blue?style=flat-square" alt="Recall">
  <img src="https://img.shields.io/badge/Inference-<200ms%20(RPi4)-orange?style=flat-square" alt="Latency">
  <img src="https://img.shields.io/badge/Model%20Size-11.8MB%20(ONNX)-purple?style=flat-square" alt="Size">
</p>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [System Architecture](#-system-architecture)
- [Performance](#-performance)
- [Dataset](#-dataset)
- [Model Details](#-model-details)
- [Installation](#-installation)
- [Usage](#-usage)
- [Edge Deployment (Raspberry Pi)](#-edge-deployment-raspberry-pi)
- [RTSP Camera Setup](#-rtsp-camera-setup)
- [Alert System](#-alert-system)
- [Project Structure](#-project-structure)
- [Results](#-results)
- [Future Roadmap](#-future-roadmap)
- [Contributing](#-contributing)
- [License](#-license)
- [Acknowledgements](#-acknowledgements)

---

## 🎯 Overview

Traditional fire detection relies on particle sensors with limited range, high false-alarm rates, and zero visual evidence. Cloud-based AI alternatives introduce latency, privacy risks, and recurring costs.

**This project solves both problems** by deploying a lightweight YOLOv8n object detection model on a Raspberry Pi 4 that processes live RTSP camera feeds — fully offline, fully on-premise.

### Why This Project?

| Problem | Our Solution |
|---------|-------------|
| Smoke detectors fail in open/large spaces | Camera-based detection covers 100–1000m² per camera |
| No visual evidence of fire location | Bounding boxes + confidence scores on every detection |
| Cloud AI = latency + privacy risk + cost | 100% offline edge processing, <200ms latency, $0 recurring |
| False alarms from steam/dust/sunlight | 5-frame temporal verification suppresses transient noise |
| Requires internet 24/7 | Works without internet; only needs network for alerts |

---

## ✨ Key Features

- 🔥 **Real-time fire & smoke detection** at <200ms end-to-end latency on Raspberry Pi 4
- 🔒 **Privacy-preserving** — video never leaves the device, zero cloud upload
- 📡 **RTSP IP camera support** — works with any standard CCTV/IP camera
- ⏱️ **Temporal verification** — 5-frame consistency check eliminates false alarms
- 📲 **Multi-channel alerts** — Telegram, Email (SMTP), and SMS (Twilio)
- 🌐 **Fully offline** — runs without internet; network only needed for sending alerts
- ⚡ **Lightweight model** — 3M parameters, 8.1 GFLOPs, 11.8MB ONNX
- 🔄 **Auto-reconnect** — handles camera disconnections with exponential backoff
- 📊 **Explainability** — Grad-CAM heatmaps for visual justification of detections

---

## 🏗️ System Architecture

```
┌──────────────────┐     ┌─────────────────────────────────────────────────┐     ┌──────────────────┐
│   IP Camera      │     │          Raspberry Pi 4 (Edge Device)           │     │   Alert System   │
│  (RTSP Stream)   │────▶│                                                 │────▶│                  │
│                  │     │  ┌─────────┐  ┌──────────┐  ┌───────────────┐  │     │  📱 Telegram     │
│  rtsp://cam:554  │     │  │ Frame   │  │ YOLOv8n  │  │  Temporal     │  │     │  📧 Email        │
│                  │     │  │ Capture │─▶│ ONNX     │─▶│  Verification │  │     │  💬 SMS          │
│                  │     │  │ OpenCV  │  │ Inference │  │  (5 frames)   │  │     │                  │
└──────────────────┘     │  └─────────┘  └──────────┘  └───────────────┘  │     └──────────────────┘
                         │                                                 │
                         │  Model: best.onnx (11.8 MB)                    │
                         │  Runtime: ONNX Runtime (ARM64)                 │
                         └─────────────────────────────────────────────────┘
```

```mermaid
flowchart LR
    CAM[IP Camera\nRTSP] --> CAPTURE[Frame\nCapture]
    CAPTURE --> SAMPLE[Frame\nSampling]
    SAMPLE --> PREPROCESS[Resize\nNormalize]
    PREPROCESS --> INFER[YOLOv8n\nONNX Inference]
    INFER --> TEMPORAL[Temporal\nVerification]
    TEMPORAL --> |Confirmed| ALERT[Alert\nGeneration]
    TEMPORAL --> |Not confirmed| SAMPLE
```

---

## 📊 Performance

### Detection Metrics

| Metric | All Classes | Fire 🔥 | Smoke 💨 |
|--------|:-----------:|:-------:|:--------:|
| **mAP@50** | **86.1%** | 88.2% | 84.0% |
| **mAP@50-95** | **55.3%** | 58.7% | 51.9% |
| **Precision** | **84.3%** | 87.3% | 81.4% |
| **Recall** | **78.4%** | 80.3% | 76.5% |

### Inference Speed

| Platform | Preprocess | Inference | Postprocess | **Total** |
|----------|:----------:|:---------:|:-----------:|:---------:|
| Tesla T4 (GPU) | 0.2ms | 1.8ms | 0.7ms | **~4ms** |
| Raspberry Pi 4 (ONNX) | ~15ms | ~150ms | ~20ms | **~185ms** |

### Model Specifications

| Spec | Value |
|------|-------|
| Architecture | YOLOv8n (Nano) |
| Parameters | 3,006,038 |
| GFLOPs | 8.1 |
| Model Size (PyTorch) | 6.0 MB |
| Model Size (ONNX) | 11.8 MB |
| Classes | 2 (Fire, Smoke) |
| Input Size | 640×640 |

---

## 📦 Dataset

A curated multi-source dataset combining three complementary sources:

| Dataset | Images | Format | Description |
|---------|:------:|--------|-------------|
| [D-Fire](https://github.com/gaiasd/DFireDataset) | 21,527 | YOLO | Large-scale fire/smoke with diverse backgrounds |
| Nishanth FireSmoke | 2,611 | YOLO | Varied indoor/outdoor fire scenarios |
| DFS (Datacluster Labs) | 100 | VOC → YOLO | High-quality professional fire/smoke imagery |
| **Total** | **24,238** | YOLO | After merge and deduplication |
| **Augmented** | **24,963** | YOLO | After class-aware augmentation |

### Data Pipeline

```
Raw Datasets ─┬─▶ Format Conversion (VOC → YOLO)
               ├─▶ Label Validation & Correction
               ├─▶ Corrupt Image Detection
               ├─▶ 80/20 Train/Val Split (seed=42)
               └─▶ Class-Aware Augmentation (2× smoke)
```

### Augmentation Strategy

| Transform | Probability | Purpose |
|-----------|:-----------:|---------|
| RandomBrightnessContrast | 0.6 | Lighting variation |
| HorizontalFlip | 0.5 | Orientation diversity |
| ShiftScaleRotate | 0.5 | Perspective changes |
| GaussNoise | 0.2 | Camera noise simulation |
| Blur | 0.2 | Low-quality camera handling |

> **Class-aware:** Smoke images receive 2× augmentation to address under-representation.

---

## 🧠 Model Details

### Architecture

YOLOv8n uses a **CSPDarknet backbone** with C2f modules, an **FPN+PAN neck** for multi-scale feature fusion, and an **anchor-free decoupled detection head** — critical for detecting irregularly shaped smoke plumes.

```
Backbone (CSPDarknet)         Neck (FPN + PAN)          Head (Anchor-Free)
┌─────────────────┐      ┌──────────────────┐      ┌──────────────────┐
│ Conv → C2f → ×4 │ ───▶ │ Upsample + C2f   │ ───▶ │ P3: Small objects │
│ SPPF at P5      │      │ Downsample + C2f  │      │ P4: Medium       │
└─────────────────┘      └──────────────────┘      │ P5: Large        │
  3M parameters            Multi-scale fusion       │ → 2 classes      │
                                                    └──────────────────┘
```

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Pretrained weights | YOLOv8n (COCO) |
| Epochs | 75 |
| Batch size | 16 |
| Image size (train) | 768×768 |
| Image size (inference) | 640×640 |
| Optimizer | SGD + Cosine LR |
| Learning rate | 0.01 → 0.0001 |
| Early stopping | Patience = 25 |
| AMP | Enabled |
| GPU | 2× Tesla T4 (Kaggle) |
| Training time | 7.3 hours |

---

## 🚀 Installation

### Prerequisites

- Python 3.9+
- pip

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/edge-fire-smoke-detection.git
cd edge-fire-smoke-detection

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

### Requirements

```txt
ultralytics>=8.0.0
onnxruntime>=1.15.0
opencv-python>=4.8.0
albumentations>=1.3.0
numpy>=1.24.0
Pillow>=9.0.0
requests>=2.28.0      # Telegram alerts
```

---

## 💻 Usage

### Training

```python
from ultralytics import YOLO

# Load pretrained YOLOv8n
model = YOLO('yolov8n.pt')

# Train on fire-smoke dataset
results = model.train(
    data='data.yaml',
    epochs=75,
    imgsz=768,
    batch=16,
    seed=42,
    cos_lr=True,
    patience=25
)

# Validate
metrics = model.val(imgsz=640)
print(f"mAP@50: {metrics.box.map50:.3f}")

# Export to ONNX
model.export(format='onnx', imgsz=768, simplify=True)
```

### Inference (PyTorch)

```python
from ultralytics import YOLO

model = YOLO('best.pt')
results = model.predict(
    source='path/to/image_or_video',
    conf=0.25,
    imgsz=640,
    save=True
)
```

### Inference (ONNX — Edge Deployment)

```python
import onnxruntime as ort
import cv2
import numpy as np

# Load ONNX model
session = ort.InferenceSession('best.onnx')

# Capture frame from RTSP
cap = cv2.VideoCapture('rtsp://username:password@camera_ip:554/stream')
ret, frame = cap.read()

# Preprocess
img = cv2.resize(frame, (640, 640))
img = img.astype(np.float32) / 255.0
img = np.transpose(img, (2, 0, 1))  # HWC → CHW
img = np.expand_dims(img, axis=0)    # Add batch dim

# Run inference
outputs = session.run(None, {'images': img})

# Process detections
# outputs[0] shape: (1, 6, 8400)
# Format: [x, y, w, h, fire_conf, smoke_conf]
```

---

## 🍓 Edge Deployment (Raspberry Pi)

### Hardware Requirements

| Component | Specification |
|-----------|--------------|
| Board | Raspberry Pi 4 Model B |
| RAM | 4GB (minimum) |
| Storage | 16GB+ microSD |
| OS | Raspberry Pi OS 64-bit |
| Cooling | Passive heatsink (recommended) |

### Setup on Raspberry Pi

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install dependencies
sudo apt install python3-pip python3-opencv -y

# Install ONNX Runtime for ARM64
pip3 install onnxruntime

# Copy model file
scp best.onnx pi@raspberrypi:~/fire-detection/

# Run detection
python3 detect_edge.py --model best.onnx --source rtsp://camera_ip:554/stream
```

### Performance on Raspberry Pi

| Metric | Value |
|--------|-------|
| End-to-end latency | <200ms |
| Sustainable FPS | 3–5 FPS |
| CPU utilization | 40–60% |
| RAM usage | ~500MB |
| Power consumption | ~4W |

> **Tip:** Set `OMP_NUM_THREADS=4` to optimize ONNX Runtime for the Pi's quad-core CPU.

---

## 📡 RTSP Camera Setup

### Supported Cameras

Any IP camera with RTSP support works. Tested with:
- Hikvision DS series
- Dahua IPC series
- TP-Link Tapo C200/C310
- Generic ONVIF-compatible cameras

### Connection

```python
import cv2

# Standard RTSP URL format
rtsp_url = "rtsp://admin:password@192.168.1.100:554/stream1"

# Open stream with optimized buffer
cap = cv2.VideoCapture(rtsp_url)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize latency

while True:
    ret, frame = cap.read()
    if not ret:
        # Auto-reconnect logic
        cap.release()
        cap = cv2.VideoCapture(rtsp_url)
        continue
    
    # Process frame...
```

---

## 🔔 Alert System

The system supports multi-channel alerting with automatic fallback:

```
Detection Confirmed → Telegram Bot API (primary)
                    → Email via SMTP (secondary)
                    → SMS via Twilio (tertiary)
```

### Features

- **Deduplication** — Minimum 60-second interval between alerts
- **Evidence attachment** — Detection frame with bounding boxes sent with every alert
- **Fallback routing** — If primary channel fails, automatically tries the next
- **Configurable** — All channels and thresholds via YAML config

### Telegram Setup

```python
import requests

TELEGRAM_BOT_TOKEN = "your_bot_token"
TELEGRAM_CHAT_ID = "your_chat_id"

def send_alert(image_path, message):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
    with open(image_path, 'rb') as photo:
        requests.post(url, data={
            'chat_id': TELEGRAM_CHAT_ID,
            'caption': message
        }, files={'photo': photo})
```

---

## 📁 Project Structure

```
edge-fire-smoke-detection/
├── 📓 edge_fire_smoke.ipynb      # Complete training pipeline notebook
├── 🏋️ best.pt                    # Trained PyTorch weights (6.0 MB)
├── ⚡ best.onnx                   # ONNX optimized model (11.8 MB)
├── 📊 data.yaml                   # Dataset configuration
├── 📄 Project_Report.md           # Full professional project report
├── 📄 Executive_Summary.md        # 2-page executive summary
├── 📐 Architecture_Diagrams.md    # System architecture diagrams
├── 📋 requirements.txt            # Python dependencies
├── 📜 README.md                   # This file
└── 📂 runs/                       # Training outputs
    └── detect/
        └── train/
            ├── weights/
            │   ├── best.pt
            │   └── last.pt
            ├── results.csv
            └── confusion_matrix.png
```

---

## 📈 Results

### Training Convergence

| Phase | Epochs | mAP@50 |
|-------|:------:|:------:|
| Warmup | 1–3 | 0.29 → 0.35 |
| Rapid Learning | 4–15 | 0.44 → 0.68 |
| Fine-Tuning | 16–65 | 0.69 → 0.75 |
| Post-Mosaic | 66–75 | 0.75 (stable) |

### Final Validation (640×640)

| Class | Precision | Recall | mAP@50 | mAP@50-95 |
|-------|:---------:|:------:|:------:|:---------:|
| **All** | **84.3%** | **78.4%** | **86.1%** | **55.3%** |
| Fire 🔥 | 87.3% | 80.3% | 88.2% | 58.7% |
| Smoke 💨 | 81.4% | 76.5% | 84.0% | 51.9% |

### Temporal Verification Impact

| Scenario | Without Temporal Check | With 5-Frame Check |
|----------|:---------------------:|:------------------:|
| Sunlight glare | ❌ False alarm | ✅ Suppressed |
| Camera reflection | ❌ False alarm | ✅ Suppressed |
| Passing shadow | ❌ False alarm | ✅ Suppressed |
| Actual fire | ✅ Detected (~200ms) | ✅ Detected (~1.7s) |
| Actual smoke | ✅ Detected (~200ms) | ✅ Detected (~1.7s) |

---

## 🗺️ Future Roadmap

- [ ] 🚀 **Edge TPU support** — Google Coral USB for 10× speedup
- [ ] 📉 **INT8 quantization** — 50% model size reduction
- [ ] 🎥 **Multi-camera support** — Round-robin processing of 4–8 cameras
- [ ] 🌡️ **Thermal camera fusion** — IR + visual dual-modality
- [ ] 📱 **Mobile dashboard** — Real-time monitoring web app
- [ ] 🧠 **Continual learning** — Auto-retrain with deployment feedback
- [ ] ⚡ **TensorRT deployment** — Jetson Nano/Orin support
- [ ] 🔄 **Adaptive frame sampling** — Dynamic FPS based on scene activity

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgements

- [Ultralytics](https://github.com/ultralytics/ultralytics) — YOLOv8 framework
- [D-Fire Dataset](https://github.com/gaiasd/DFireDataset) — Primary dataset source
- [ONNX Runtime](https://onnxruntime.ai/) — Edge inference engine
- [Albumentations](https://albumentations.ai/) — Data augmentation library
- VR Siddhartha Engineering College, Vijayawada — Infrastructure and academic support

---

<p align="center">
  <strong>Built with ❤️ for fire safety</strong><br/>
  <sub>If this project helps you, consider giving it a ⭐</sub>
</p>
