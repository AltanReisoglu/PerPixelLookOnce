# 🧠 PerPixelLookOnce

**PerPixelLookOnce** is a prompt-based object detection system that identifies and highlights only the objects specified in a natural language query. The model architecture is based on a **YOLO-style detection backbone**, enhanced with **Transformer-based self-attention mechanisms** for improved contextual understanding and accuracy.

## 🚀 Key Features

- 🗣️ **Prompt-guided detection**: Detects objects only if they match the user's prompt (e.g., "bicycle", "person").
- ⚡ **YOLO-inspired fast inference**: Uses a single-shot detection strategy for high-speed processing.
- 🧠 **Transformer-enhanced**: Integrates self-attention modules to better capture spatial and semantic relationships.
- 🖼️ **Per-pixel decision making**: Each pixel is processed once, reducing redundancy while preserving detail.
- 🎯 **Selective bounding box generation**: Only draws boxes around prompt-relevant objects.

## 📁 Project Structure

```bash
PerPixelLookOnce/
├── models/
│   └── model.py           # YOLOv3 based + Self-Attention +Axial Attention architecture
├── data/
│   └── sample.jpg         # Sample image for testing
├── detect.py              # Inference script
├── utils.py               # Utility functions (e.g., drawing boxes)
├── config.yaml            # Model config
├── requirements.txt       # Required libraries
└── README.md              # Project documentation

## Usage
python detect.py --image ./data/sample.jpg --prompt "car"
