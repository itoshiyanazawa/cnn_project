# CIFAR-10 Image Classification using CNNs + Grad-CAM

This project implements a Convolutional Neural Network (CNN) to classify images from the CIFAR-10 dataset. It includes preprocessing, model training, evaluation, visualization with Grad-CAM, and performance optimization using grayscale and RGB images.

## 📂 Project Structure

- `main_backup.ipynb`: Main training and evaluation logic
- `README.md`: Project overview and insights (you’re here!)

---

## 🧠 Main Objectives

- Build an effective CNN for CIFAR-10 image classification.
- Understand model decisions using **Grad-CAM** visualizations.
- Compare grayscale and RGB input performance.
- Apply performance metrics and analyze misclassifications.

---

## 🔧 Methods & Architecture

### 🔹 Preprocessing

- Normalize images to [0, 1].
- Convert to grayscale for baseline.
- Stratified split: 80% train, 20% validation.
- Optional: Retain RGB channels for richer feature learning.

### 🔹 CNN Architecture (Grayscale)

- Stacked `Conv2D + ReLU + MaxPooling`
- `GlobalAveragePooling2D` for dimensionality reduction
- Dropout to prevent overfitting
- Final Dense layer with `softmax` activation

### 🔹 CNN Architecture (RGB)

- Deepened model with filter progression: 32 → 64 → 128 → 256
- Multiple convolutions per pooling stage
- Same padding + dropout regularization
- Significantly increased capacity to extract texture and color-based features

---

## 📊 Results

| Model Type       | Accuracy | Precision | Recall | F1 Score |
|------------------|----------|-----------|--------|----------|
| Grayscale CNN    | 79.6%    | 80.2%     | 79.6%  | 79.3%    |
| RGB CNN (deep)   | 94.77% | 94.89% | 94.77% | 94.80% |

> 📝 RGB model showed better performance but caused overfitting. We assume this confusion was caused due to similar images in the test samples.

---

## 📈 Visualization: Grad-CAM

Used **Grad-CAM** to visualize model attention on cat and dog images:

- Correct predictions were made.
- Heatmaps showed **diffused attention**, not always focused on key object parts.
- Suggests CNN may rely on global patterns rather than localized discriminative features.

---

## 🔍 Analysis & Findings

- CNN performs moderately well on CIFAR-10 color images.
- Struggles particularly between **cat vs. dog** classes.
- Using RGB inputs and deeper layers helps extract more meaningful features.
- Early stopping and dropout were crucial for avoiding overfitting.

---

## 📌 Key Learnings

- Grayscale inputs simplify models but risk losing class-specific info.
- Grad-CAM is an excellent tool for interpreting CNN decisions.
- Balanced architecture and hyperparameter tuning are key to high performance.
- Color channels significantly boost classification potential—especially for animals.

---

## 🚀 Next Steps

- Integrate advanced architectures (e.g., ResNet, EfficientNet).
- Implement batch normalization and learning rate schedules.
- Perform grid search for hyperparameter tuning.
- Expand the dataset via augmentation for improved generalization.

---

## 📚 Requirements

- Python 3.11
- TensorFlow 2.15
- OpenCV
- Matplotlib, Seaborn, scikit-learn

Install dependencies:

```bash
pip install tensorflow==2.15 keras opencv-python matplotlib seaborn scikit-learn

