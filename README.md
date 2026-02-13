# Trash Classification with MobileNetV3

A deep learning project for classifying trash bin images using transfer learning with MobileNetV3-Large architecture. This project implements a computer vision model to distinguish between different types of trash bins based on their color and visual characteristics.

## 📋 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Training](#training)
- [Evaluation Metrics](#evaluation-metrics)
- [Results](#results)
- [Acknowledgments](#acknowledgments)

## 🎯 Overview

This project uses **MobileNetV3-Large** with transfer learning to classify trash bin images into 4 categories. The model leverages ImageNet-pretrained weights and fine-tunes a custom 3-layer classification head to achieve high accuracy on trash bin classification.

**Key Features:**
- Transfer learning from ImageNet weights
- Custom aspect-ratio-preserving image preprocessing
- Comprehensive data augmentation pipeline
- Differential learning rates for backbone and classifier
- Early stopping and learning rate scheduling
- Detailed evaluation metrics including confusion matrix, F1 scores, and inference time

## 📊 Dataset

**Source:** CVPR 2024 Trash Classification Dataset

**Classes (4):**
- Black
- Blue
- Green
- TTR (Trash/Recycling/Recycle)

**Dataset Split:**
- Training Set
- Validation Set
- Test Set

**Image Properties:**
- Variable dimensions (analyzed via EDA)
- RGB color images
- Aspect ratios preserved during preprocessing

## 🏗️ Model Architecture

### Base Model
- **Backbone:** MobileNetV3-Large (pretrained on ImageNet)
- **Input Size:** 224×224×3
- **Preprocessing:** ResizeAndPad (maintains aspect ratio)

### Custom Classifier Head (3 Layers)

```
Input (960) 
   ↓
FC1: Linear(960 → 512) + BatchNorm + ReLU + Dropout(0.3)
   ↓
FC2: Linear(512 → 256) + BatchNorm + ReLU + Dropout(0.2)
   ↓
FC3: Linear(256 → 4) [Output Layer]
   ↓
Softmax (for inference)
```

**Total Classifier Parameters:** ~625,924

## 🚀 Installation

### Prerequisites
- Python 3.12 or higher
- CUDA-capable GPU (optional, but recommended)
- [uv](https://github.com/astral-sh/uv) package manager

### Setup

1. **Clone the repository:**
```bash
git clone https://github.com/NotAlpha45/trash-classification-project.git
cd trash-classification-project
```

2. **Install dependencies using uv:**

If you are starting with the `pyptoject.toml` file, then 

```bash
uv sync
```

If you are starting a new development bundle with your own configuration files:

```bash
# Install PyTorch with CUDA 13.0 support
uv add torch torchvision --index https://download.pytorch.org/whl/cu130 --index-strategy unsafe-best-match

# Install other dependencies
uv add scikit-learn matplotlib seaborn pillow
```

3. **Verify installation:**
```bash
uv run python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

## 📁 Project Structure

```
trash-classification-project/
├── notebooks/
│   └── image_model_experiment_mobilenetv3.ipynb  # Main training notebook
├── dataset/
│   ├── CVPR_2024_dataset_Train/
│   ├── CVPR_2024_dataset_Val/
│   └── CVPR_2024_dataset_Test/
├── models/                                        # Saved model checkpoints
│   ├── mobilenetv3_best.pth
│   ├── mobilenetv3_epoch_*.pth
│   ├── training_history.png
│   └── test_metrics.txt
├── AGENT_INSTRUCTIONS.md                          # Development guidelines
├── pyproject.toml                                 # Project dependencies
└── README.md                                      # This file
```

## 💻 Usage

### Training the Model

Open and run the Jupyter notebook:
```bash
jupyter notebook notebooks/image_model_experiment_mobilenetv3.ipynb
```

**Training Pipeline:**
1. **Data Loading & EDA** (Sections 1-3)
2. **Data Preprocessing** (Section 4)
3. **Model Setup** (Section 5)
4. **Training** (Sections 6-7)
5. **Visualization** (Section 8)
6. **Inference** (Section 9)
7. **Evaluation** (Section 10)

### Loading a Pre-trained Model

```python
from pathlib import Path
from load_pretrained_model import load_pretrained_model

# Load best model
model = load_pretrained_model(
    checkpoint_path=Path("models/mobilenetv3_best.pth"),
    num_classes=4,
    device=device
)
```

### Making Predictions

```python
# Single image prediction
result = predict_single_image(model, image_tensor)
print(f"Predicted class: {result['predicted_class_name']}")
print(f"Probabilities: {result['probabilities']}")
```

## 🎓 Training

### Hyperparameters

| Parameter | Value |
|-----------|-------|
| Epochs | 50 (with early stopping) |
| Batch Size | 32 |
| Learning Rate (Classifier) | 0.001 |
| Learning Rate (Backbone) | 0.0001 |
| Optimizer | AdamW |
| Weight Decay | 1e-4 |
| Scheduler | ReduceLROnPlateau |
| Early Stopping Patience | 15 epochs |

### Data Augmentation (Training Only)

- Random Resized Crop (scale: 0.8-1.0)
- Random Horizontal Flip (p=0.5)
- Random Rotation (±15°)
- Color Jitter (brightness, contrast, saturation, hue)
- Random Affine Translation (±10%)

### Loss Function
- **CrossEntropyLoss** (handles multi-class classification)

## 📈 Evaluation Metrics

The model is evaluated using the following metrics:

1. **Accuracy** - Overall classification accuracy
2. **Macro-averaged F1 Score** - Balanced performance across all classes
3. **Per-class Precision** - Precision for each trash bin class
4. **Inference Time** - Average time per sample (milliseconds)
5. **Confusion Matrix** - Visual heatmap of predictions vs. ground truth

### Evaluation Function

```python
metrics = evaluate_model(
    model=model,
    dataloader=test_loader,
    class_names=class_names,
    device=device,
    plot_confusion_matrix=True
)
```

## 🏆 Results

*Results will be populated after training completes*

**Test Set Performance:**
- Accuracy: TBD
- Macro F1 Score: TBD
- Inference Time: TBD ms/sample

**Per-class Precision:**
- Black: TBD
- Blue: TBD
- Green: TBD
- TTR: TBD

## 🔍 Key Features

### Custom ResizeAndPad Transform
Maintains aspect ratio while resizing images to prevent distortion:
```python
class ResizeAndPad(nn.Module):
    """Resize to target size maintaining aspect ratio, then pad to square."""
```

### Prediction Output Options
Get both raw logits and probability distributions:
```python
logits, probs = get_predictions(model, images, return_type='both')
```

### Checkpoint System
- Saves best model based on validation accuracy
- Periodic checkpoints every 5 epochs
- Includes optimizer state for training resumption

## 🛠️ Technologies Used

- **PyTorch** (2.10.0+) - Deep learning framework
- **torchvision** (0.25.0+) - Computer vision utilities
- **scikit-learn** - Evaluation metrics
- **matplotlib & seaborn** - Visualization
- **PIL/Pillow** - Image processing
- **NumPy** - Numerical operations

## 📝 Acknowledgments

- **MobileNetV3** architecture from [Searching for MobileNetV3](https://arxiv.org/abs/1905.02244)
- **CVPR 2024** dataset for trash classification
- **University of Calgary** - ENSF617 Introduction to Machine Learning (Winter 2026)

## 📄 License

This project is for educational purposes as part of ENSF617 coursework.

## 👤 Author

**NotAlpha45**
- GitHub: [@NotAlpha45](https://github.com/NotAlpha45)

---

**Note:** This project demonstrates transfer learning techniques for image classification using state-of-the-art mobile architectures suitable for deployment on resource-constrained devices.