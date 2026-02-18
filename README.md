# Multimodal Trash Classification: Image + Text Fusion

A comprehensive deep learning project implementing **three complementary approaches** to trash bin classification: computer vision (MobileNetV3), natural language processing (DistilBERT), and multimodal fusion. This project demonstrates how combining visual and textual information can improve classification accuracy beyond single-modality approaches.

## 📋 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Model Architectures](#model-architectures)
  - [Image Model (MobileNetV3)](#image-model-mobilenetv3)
  - [Text Model (DistilBERT)](#text-model-distilbert)
  - [Multimodal Fusion](#multimodal-fusion)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Training](#training)
- [Evaluation Metrics](#evaluation-metrics)
- [Results](#results)
- [Acknowledgments](#acknowledgments)

## 🎯 Overview

This project implements a **multimodal learning system** that classifies trash bin images into 4 categories using three different approaches:

1. **Image Classification (MobileNetV3-Large)** - Computer vision approach using transfer learning from ImageNet
2. **Text Classification (DistilBERT)** - NLP approach using image filenames as textual features
3. **Multimodal Fusion** - Weighted logit-level fusion combining both image and text predictions

**Key Features:**
- 🖼️ **Image Model:** Transfer learning with custom 3-layer classifier, aspect-ratio-preserving preprocessing
- 📝 **Text Model:** DistilBERT fine-tuned on image filenames with optimized training configuration
- 🔥 **Multimodal Fusion:** Weighted logit combination with alpha sensitivity analysis
- 📊 Comprehensive evaluation metrics across all three approaches
- 🎯 Alpha parameter tuning to find optimal image-text fusion weights
- ⚡ Early stopping, learning rate scheduling, and gradient clipping
- 📈 Detailed visualizations: confusion matrices, performance comparisons, sample predictions

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

**Text Properties:**
- Image filenames used as textual features
- Preprocessed and tokenized for transformer input
- CSV format: `text` (filename), `label` (class)

## 🏗️ Model Architectures

### Image Model (MobileNetV3)

**Base Model:**
- **Backbone:** MobileNetV3-Large (pretrained on ImageNet)
- **Input Size:** 224×224×3
- **Preprocessing:** ResizeAndPad (maintains aspect ratio)

**Custom Classifier Head (3 Layers):**

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

### Text Model (DistilBERT)

**Base Model:**
- **Architecture:** DistilBERT-base-uncased
- **Input:** Image filenames (tokenized, max length 64)
- **Fine-tuning:** Full model fine-tuning with classification head

**Architecture:**

```
Input (Filename Text)
   ↓
Tokenizer (distilbert-base-uncased)
   ↓
DistilBERT Transformer (6 layers, 768 hidden)
   ↓
Classification Head: Linear(768 → 4)
   ↓
Softmax (for inference)
```

**Total Parameters:** ~67M (66M pretrained + 1M fine-tuned)

### Multimodal Fusion

**Fusion Strategy:** Weighted Logit-Level Fusion

```
Image Model → Image Logits (W_image)
                                           ↓
Text Model  → Text Logits (W_text)   →   W_fused = α·W_text + (1-α)·W_image
                                           ↓
                                      Softmax → Final Predictions
```

**Parameters:**
- **α (alpha):** Fusion weight controlling text vs. image contribution
  - α = 0.0: Image only
  - α = 0.5: Equal weight
  - α = 1.0: Text only
- **Optimization:** Alpha sensitivity analysis to find optimal fusion weight

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

# Install transformers for text model
uv add transformers --index-strategy unsafe-best-match

# Install other dependencies
uv add scikit-learn matplotlib seaborn pillow pandas
```

3. **Verify installation:**
```bash
uv run python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

## 📁 Project Structure

```
trash-classification-project/
├── notebooks/
│   ├── image_model_experiment_mobilenetv3.ipynb  # Image classification (MobileNetV3)
│   ├── distilbert_text_model.ipynb               # Text classification (DistilBERT)
│   └── multimodal_fusion.ipynb                   # Multimodal fusion (Image + Text)
├── dataset/
│   ├── CVPR_2024_dataset_Train/                  # Training images
│   ├── CVPR_2024_dataset_Val/                    # Validation images
│   └── CVPR_2024_dataset_Test/                   # Test images
├── dataset_text/                                  # Text dataset (CSV files)
│   ├── train.csv
│   ├── val.csv
│   └── test.csv
├── models/                                        # Image model checkpoints
│   ├── mobilenetv3_best.pth
│   ├── mobilenetv3_epoch_*.pth
│   ├── training_history.png
│   └── test_metrics.txt
├── text_models/                                   # Text model checkpoints
│   ├── distilbert_best.pth
│   ├── distilbert_epoch_*.pth
│   └── multimodal_*.png                          # Fusion visualizations
├── AGENT_INSTRUCTIONS.md                          # Development guidelines
├── pyproject.toml                                 # Project dependencies
└── README.md                                      # This file
```

## 💻 Usage

### Training the Models

#### 1. Image Model (MobileNetV3)
```bash
jupyter notebook notebooks/image_model_experiment_mobilenetv3.ipynb
```

**Pipeline:**
1. Data Loading & EDA (Sections 1-3)
2. Data Preprocessing (Section 4)
3. Model Setup (Section 5)
4. Training (Sections 6-7)
5. Visualization (Section 8)
6. Inference (Section 9)
7. Evaluation (Section 10)

#### 2. Text Model (DistilBERT)
```bash
jupyter notebook notebooks/distilbert_text_model.ipynb
```

**Pipeline:**
1. Import Libraries & Setup
2. Load Text Dataset (CSV files)
3. Create Dataset & DataLoader
4. Load Pretrained DistilBERT
5. Training Configuration & Functions
6. Training Loop (15 epochs with early stopping)
7. Evaluation & Metrics
8. Inference Functions

#### 3. Multimodal Fusion
```bash
jupyter notebook notebooks/multimodal_fusion.ipynb
```

**Pipeline:**
1. Load Both Trained Models
2. Create Multimodal Dataset
3. Implement Fusion Functions
4. Run Test Set Evaluation
5. Performance Comparison (Image vs Text vs Fused)
6. Alpha Sensitivity Analysis
7. Generate Visualizations

### Loading Pre-trained Models

```python
from pathlib import Path
import torch

# Load image model
image_checkpoint = torch.load("models/mobilenetv3_best.pth")
image_model.load_state_dict(image_checkpoint["model_state_dict"])

# Load text model
text_checkpoint = torch.load("text_models/distilbert_best.pth")
text_model.load_state_dict(text_checkpoint["model_state_dict"])
```

### Making Predictions

#### Single Image Prediction
```python
# Image model only
result = predict_single_image(image_model, image_tensor)
print(f"Predicted class: {result['predicted_class_name']}")

# Text model only
result = predict_single_text(text_model, filename, tokenizer)
print(f"Predicted class: {result['class_name']}")

# Multimodal fusion
result = predict_multimodal(
    image_path=image_path,
    image_model=image_model,
    text_model=text_model,
    tokenizer=tokenizer,
    transform=transform,
    device=device,
    class_names=class_names,
    alpha=0.5  # Fusion weight
)
print(f"Image prediction: {result['image_prediction']}")
print(f"Text prediction: {result['text_prediction']}")
print(f"Fused prediction: {result['fused_prediction']}")
```

## 🎓 Training

### Image Model Hyperparameters

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

**Data Augmentation (Training Only):**
- Random Resized Crop (scale: 0.8-1.0)
- Random Horizontal Flip (p=0.5)
- Random Rotation (±15°)
- Color Jitter (brightness, contrast, saturation, hue)
- Random Affine Translation (±10%)

### Text Model Hyperparameters

| Parameter | Value |
|-----------|-------|
| Epochs | 15 (with early stopping) |
| Batch Size | 64 |
| Learning Rate | 3e-5 |
| Optimizer | AdamW |
| Weight Decay | 0.01 |
| Warmup Ratio | 0.1 (10% of total steps) |
| Gradient Clipping | Max norm 1.0 |
| Early Stopping Patience | 4 epochs |
| Max Sequence Length | 64 tokens |

**Key Optimizations:**
- Warmup learning rate schedule for stable training
- Gradient clipping to prevent exploding gradients
- Higher batch size (64) for faster convergence on simple text features
- Lower epoch count (15-25 sufficient for filename classification)

### Multimodal Fusion Configuration

| Parameter | Value |
|-----------|-------|
| Fusion Method | Weighted logit combination |
| Alpha (α) | 0.5 (default, tunable) |
| Alpha Range Tested | [0.0, 0.1, 0.2, ..., 1.0] |
| Evaluation Metric | Accuracy & F1 (weighted) |

**Fusion Formula:**
```
W_fused = α · W_text + (1 - α) · W_image
Predictions = softmax(W_fused)
```

### Loss Functions
- **Image Model:** CrossEntropyLoss (with optional class weights)
- **Text Model:** CrossEntropyLoss
- **Fusion:** Inference only (no training)

## 📈 Evaluation Metrics

The models are evaluated using comprehensive metrics across all three approaches:

### Core Metrics

1. **Accuracy** - Overall classification accuracy
2. **Macro F1 Score** - Balanced performance across all classes (handles class imbalance)
3. **Weighted F1 Score** - F1 weighted by class support
4. **Per-class Precision & Recall** - Individual class performance
5. **Confusion Matrix** - Visual heatmap of predictions vs. ground truth
6. **Inference Time** - Average time per sample (milliseconds)

### Evaluation Functions

#### Image Model
```python
metrics = evaluate_model(
    model=image_model,
    dataloader=test_loader,
    class_names=class_names,
    device=device,
    plot_confusion_matrix=True
)
```

#### Text Model
```python
results = evaluate_model(
    model=text_model,
    test_dataloader=test_loader,
    device=device,
    class_names=class_names
)
```

#### Multimodal Fusion
```python
results = evaluate_multimodal(
    image_model=image_model,
    text_model=text_model,
    dataloader=test_loader,
    device=device,
    alpha=0.5  # Fusion weight
)
```

### Performance Comparison

The multimodal fusion notebook generates comprehensive comparisons:
- Side-by-side confusion matrices (Image | Text | Fused)
- Performance bar charts (Accuracy & F1 scores)
- Per-class classification reports
- Alpha sensitivity analysis (finding optimal fusion weight)
- Sample prediction visualizations with all three models

## 🏆 Results

*Results will be populated after training all three models*

### Test Set Performance

| Model | Accuracy | F1 (Macro) | F1 (Weighted) | Inference Time |
|-------|----------|------------|---------------|----------------|
| **Image Only** (MobileNetV3) | TBD | TBD | TBD | TBD ms/sample |
| **Text Only** (DistilBERT) | TBD | TBD | TBD | TBD ms/sample |
| **Multimodal Fusion** (α=0.5) | TBD | TBD | TBD | TBD ms/sample |

### Per-Class Performance

**Image Model:**
- Black: TBD
- Blue: TBD
- Green: TBD
- TTR: TBD

**Text Model:**
- Black: TBD
- Blue: TBD
- Green: TBD
- TTR: TBD

**Fused Model:**
- Black: TBD
- Blue: TBD
- Green: TBD
- TTR: TBD

### Key Findings

- **Best Single Model:** TBD (Image or Text)
- **Fusion Improvement:** TBD% over best single model
- **Optimal Alpha:** TBD (from sensitivity analysis)
- **Trade-offs:** 
  - Image model: Strong visual feature learning but slower inference
  - Text model: Fast inference but limited to filename patterns
  - Fusion: Best accuracy at cost of running both models

### Visualizations Generated

All notebooks generate detailed visualizations saved to respective model directories:

**Image Model:** (`models/`)
- Training history plots (loss, accuracy, learning rate curves)
- Confusion matrix heatmap
- Sample predictions with probability distributions

**Text Model:** (`text_models/`)
- Training history plots
- Confusion matrix heatmap
- Sample predictions from test set

**Multimodal Fusion:** (`text_models/`)
- `multimodal_confusion_matrices.png` - Side-by-side comparison
- `multimodal_performance_comparison.png` - Bar chart comparison
- `alpha_sensitivity_analysis.png` - Fusion weight optimization
- `sample_predictions.png` - Visual comparison of all three models

## 🔍 Key Features

### 1. Image Model Innovations

**Custom ResizeAndPad Transform:**
Maintains aspect ratio while resizing images to prevent distortion:
```python
class ResizeAndPad(nn.Module):
    """Resize to target size maintaining aspect ratio, then pad to square."""
```

**Prediction Output Options:**
Get both raw logits and probability distributions:
```python
logits, probs = get_predictions(model, images, return_type='both')
```

**Checkpoint System:**
- Saves best model based on validation accuracy
- Periodic checkpoints every 5 epochs
- Includes optimizer state for training resumption

### 2. Text Model Innovations

**Optimized Training Configuration:**
- Warmup learning rate schedule for transformer stability
- Gradient clipping to prevent exploding gradients
- Fast convergence (15 epochs vs 50 for images)

**Filename-Based Classification:**
Leverages patterns in image filenames to extract class information without visual processing

**Inference Functions:**
```python
# Get predictions with multiple return types
get_predictions(model, texts, tokenizer, return_type='logits')  # Raw logits
get_predictions(model, texts, tokenizer, return_type='probs')   # Probabilities
get_predictions(model, texts, tokenizer, return_type='both')    # Both
```

### 3. Multimodal Fusion Innovations

**Logit-Level Fusion:**
Combines raw model outputs before softmax for better calibration than probability fusion

**Alpha Sensitivity Analysis:**
Automatically tests multiple fusion weights (α ∈ [0.0, 1.0]) to find optimal combination

**Comprehensive Evaluation:**
Side-by-side comparison of all three approaches with unified metrics

**Visualization Suite:**
Automatic generation of comparison plots, confusion matrices, and sample predictions

## 🛠️ Technologies Used

### Deep Learning Frameworks
- **PyTorch** (2.10.0+) - Deep learning framework
- **torchvision** (0.25.0+) - Computer vision utilities
- **Transformers** (4.x) - HuggingFace library for NLP models

### Models
- **MobileNetV3-Large** - Efficient CNN architecture for image classification
- **DistilBERT** - Lightweight transformer for text classification

### Data & Evaluation
- **scikit-learn** - Evaluation metrics and preprocessing
- **pandas** - Text dataset management (CSV files)
- **NumPy** - Numerical operations

### Visualization
- **matplotlib** - Plotting and visualizations
- **seaborn** - Statistical visualizations and heatmaps
- **PIL/Pillow** - Image processing

## 📝 Acknowledgments

- **MobileNetV3** architecture from [Searching for MobileNetV3](https://arxiv.org/abs/1905.02244)
- **DistilBERT** from [DistilBERT, a distilled version of BERT](https://arxiv.org/abs/1910.01108)
- **HuggingFace Transformers** library for pretrained NLP models
- **CVPR 2024** dataset for trash classification
- **University of Calgary** - ENSF617 Introduction to Machine Learning (Winter 2026)

## 🎯 Learning Outcomes

This project demonstrates:

1. **Transfer Learning:** Leveraging pretrained models (ImageNet, BERT) for downstream tasks
2. **Multimodal Learning:** Combining different data modalities (vision + language)
3. **Model Fusion:** Weighted logit combination for ensemble predictions
4. **Hyperparameter Optimization:** Finding optimal training configurations for different architectures
5. **Evaluation & Analysis:** Comprehensive metrics, visualizations, and performance comparison
6. **Production Considerations:** Checkpoint management, inference optimization, model deployment readiness

## 📄 License

This project is for educational purposes as part of ENSF617 coursework.

## 👤 Author

**NotAlpha45**
- GitHub: [@NotAlpha45](https://github.com/NotAlpha45)

---

**Note:** This project demonstrates advanced multimodal learning techniques combining computer vision and natural language processing. The fusion approach showcases how different modalities can complement each other, with the lightweight MobileNetV3 architecture making it suitable for deployment on resource-constrained devices while DistilBERT provides fast text-based inference.