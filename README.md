# 🌿 PlantDetect — AI Plant Disease Detection

An end-to-end deep learning application that detects diseases in plant leaves using a fine-tuned ResNet18 model. Upload a photo of a plant leaf and get instant disease diagnosis with confidence scores.

### 🔗 [Live Demo](https://plantdetect-ai-plant-disease-detection-1.onrender.com)

![Python](https://img.shields.io/badge/Python-3.11-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-ResNet18-red)
![Flask](https://img.shields.io/badge/Flask-Backend-green


## 🎯 Features

- **15 Disease Classes** — Detects diseases across Tomato, Potato, and Pepper plants
- **Real-time Predictions** — Upload a leaf image and get results in seconds
- **Top-3 Predictions** — Shows confidence scores for the most likely diagnoses
- **Leaf Validation** — Rejects non-leaf images (photos of people, objects, etc.) using color heuristics and confidence thresholds
- **Modern Web UI** — Responsive, dark-themed interface with smooth animations
- **Early Stopping & LR Scheduling** — Training with smart convergence strategies
- **Mixed Precision Training** — Faster GPU training with automatic mixed precision

## 🏗️ Project Structure

```
plant-disease-detection/
├── train.py           # Model training script (ResNet18 fine-tuning)
├── predict.py         # Inference module with leaf validation
├── app.py             # Flask API server
├── index.html         # Frontend UI
├── styles.css         # Styling
├── requirements.txt   # Python dependencies
└── .gitignore
```

## 📋 Disease Classes

| Plant   | Conditions Detected                                                        |
|---------|----------------------------------------------------------------------------|
| Tomato  | Bacterial Spot, Early Blight, Late Blight, Leaf Mold, Septoria Leaf Spot, Spider Mites, Target Spot, Yellow Leaf Curl Virus, Mosaic Virus, Healthy |
| Potato  | Early Blight, Late Blight, Healthy                                         |
| Pepper  | Bacterial Spot, Healthy                                                    |

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- [PlantVillage Dataset](https://www.kaggle.com/datasets/emmarex/plantdisease) placed in `PlantVillage/train/`

### Installation

```bash
# Clone the repository
git clone https://github.com/Akhil866/plant-disease-detection.git
cd plant-disease-detection

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### Training the Model

```bash
python train.py
```

This will:
1. Automatically split data into train/validation (80/20)
2. Fine-tune a pre-trained ResNet18 with dropout regularization
3. Apply early stopping and learning rate scheduling
4. Save the trained model to `PlantVillage_model.pth`
5. Generate training curves in `training_curves.png`

### Running the Web App

```bash
python app.py
```

Open [http://localhost:5000](http://localhost:5000) in your browser and upload a leaf image.

## 🛠️ Tech Stack

- **Model**: ResNet18 (pre-trained on ImageNet, fine-tuned on PlantVillage)
- **Framework**: PyTorch
- **Backend**: Flask
- **Frontend**: HTML, CSS, JavaScript
- **Training**: Mixed precision, early stopping, ReduceLROnPlateau scheduler

## 📊 Model Details

- **Architecture**: ResNet18 with custom FC head (Dropout 0.5 → Linear)
- **Input Size**: 224 × 224 RGB
- **Data Augmentation**: Random horizontal flip, rotation (±10°), color jitter
- **Optimizer**: Adam (lr=0.001)
- **Regularization**: Dropout (0.5) + Early stopping (patience=2)


