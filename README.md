# Rice Leaf Disease Classification Framework Using CNN and Transfer Learning

## Project Overview

This project develops a deep learning-based framework for automatic rice leaf disease classification using Convolutional Neural Networks (CNNs) and Transfer Learning techniques.

The objective is to classify rice leaf images into one of four categories:

* BrownSpot
* Healthy
* Hispa
* LeafBlast

The project was developed as part of the **DSC01 Machine Learning 120 A** module and follows the requirements specified in the CNN-Based Image Dataset Project.

The framework aims to support agricultural disease screening and decision-making by providing an accurate and explainable image classification system.

---

# Dataset Information

### Dataset Name

Rice Diseases Image Dataset

### Source

Kaggle

### Dataset Link

https://www.kaggle.com/datasets/minhhuy2810/rice-diseases-image-dataset

### Dataset Characteristics

| Property            | Value       |
| ------------------- | ----------- |
| Total Images        | 2,092       |
| Classes             | 4           |
| Image Format        | JPG         |
| Classification Type | Multi-Class |
| Input Size          | 224 × 224   |

### Classes

| Class     |
| --------- |
| BrownSpot |
| Healthy   |
| Hispa     |
| LeafBlast |

---

# Project Objectives

The primary objectives of this project are:

* Develop a custom CNN model for rice disease classification.
* Implement transfer learning models.
* Compare CNN and transfer learning performance.
* Evaluate model performance using multiple metrics.
* Generate explainability visualizations using Grad-CAM.
* Analyze classification errors and deployment limitations.

---

# Project Structure

```text
Rice-Leaf-Disease-Classification
│
├── dataset
│   ├── train
│   ├── validation
│   └── test
│
├── notebooks
│   └── rice_leaf_disease.ipynb
│
├── outputs
│   │
│   ├── figures
│   │   ├── dataset_distribution.png
│   │   ├── accuracy_curve.png
│   │   ├── loss_curve.png
│   │   ├── confusion_matrix.png
│   │   ├── roc_curve.png
│   │   ├── gradcam_correct.png
│   │   ├── gradcam_false_positive.png
│   │   └── gradcam_false_negative.png
│   │
│   ├── models
│   │   ├── custom_cnn.keras
│   │   ├── mobilenetv2.keras
│   │   ├── efficientnetb0.keras
│   │   ├── custom_cnn_summary.csv
│   │   ├── mobilenetv2_summary.csv
│   │   └── efficientnetb0_summary.csv
│   │
│   └── reports
│       ├── classification_report.txt
│       ├── model_comparison.csv
│       └── error_analysis.csv
│
├── README.md
└── requirements.txt
```

---

# Methodology

The project follows the following workflow:

## 1. Dataset Exploration

* Dataset loading
* Class distribution analysis
* Sample image visualization

## 2. Data Preprocessing

* Image resizing (224 × 224)
* Pixel normalization
* Dataset validation

## 3. Data Augmentation

* Rotation
* Horizontal Flip
* Zoom
* Width Shift
* Height Shift

## 4. Custom CNN Development

Architecture:

```text
Input Layer
    ↓
Conv2D (32)
    ↓
MaxPooling
    ↓
Conv2D (64)
    ↓
MaxPooling
    ↓
Conv2D (128)
    ↓
MaxPooling
    ↓
Flatten
    ↓
Dense (256)
    ↓
Dropout (0.5)
    ↓
Softmax Output (4 Classes)
```

## 5. Transfer Learning

The following pretrained models are implemented:

### MobileNetV2

Lightweight architecture optimized for mobile deployment.

### EfficientNetB0

Efficient architecture balancing accuracy and computational cost.

---

# Evaluation Metrics

The following metrics are used:

### Classification Metrics

* Accuracy
* Precision
* Recall
* F1-Score

### Visualization Metrics

* Confusion Matrix
* ROC Curve
* Accuracy Curve
* Loss Curve

### Explainability Metrics

* Grad-CAM

### Comparative Analysis

* Model Comparison Table
* Error Analysis

---

# Research Questions

### RQ1

How accurately can CNN-based models classify rice leaf diseases?

### RQ2

Which model provides the best balance between predictive performance and computational efficiency?

### RQ3

How do preprocessing and augmentation techniques affect performance?

### RQ4

Do Grad-CAM visualizations indicate meaningful attention regions?

### RQ5

What are the major failure patterns and deployment challenges?

---

# Installation

## Clone Repository

```bash
git clone https://github.com/yourusername/Rice-Leaf-Disease-Classification.git
```

```bash
cd Rice-Leaf-Disease-Classification
```

---

# Create Virtual Environment

Windows:

```bash
python -m venv venv
```

Activate:

```bash
venv\Scripts\activate
```

---

# Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Required Packages

```text
tensorflow
keras
numpy
pandas
matplotlib
seaborn
opencv-python
scikit-learn
jupyter
```

---

# Running the Project

## Launch Jupyter Notebook

```bash
jupyter notebook
```

Open:

```text
rice_leaf_disease.ipynb
```

Run all cells sequentially.

---

# Generated Outputs

The notebook automatically generates:

### Figures

* Dataset Distribution
* Accuracy Curve
* Loss Curve
* Confusion Matrix
* ROC Curve
* Grad-CAM Visualizations

### Reports

* Classification Report
* Error Analysis
* Model Comparison

### Models

* Custom CNN
* MobileNetV2
* EfficientNetB0

---

# Explainable AI

Grad-CAM visualizations are generated for:

### Correct Prediction

Demonstrates successful disease localization.

### False Positive

Shows why incorrect predictions occur.

### False Negative

Highlights cases where disease symptoms were missed.

---

# Future Work

Potential improvements include:

* Larger rice disease datasets
* Additional disease categories
* Hyperparameter optimization
* Real-time mobile deployment
* Cloud-based agricultural monitoring
* Advanced explainability techniques

---

# Author

**MANIKANTESWARA REDDY MUKKAMALLA**

DSC01 Machine Learning 120 A

University of Europe for Applied Sciences

---
