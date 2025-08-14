# Weapon Detection Pipeline

A hybrid weapon detection system that combines **YOLOv11** deep learning object detection with **classical computer vision techniques** (HOG, color histograms, edge features) and **XGBoost classification** to enhance detection accuracy.

## Overview

This pipeline is designed to detect weapons (knives, guns, and heavy weapons) in images using a two-stage approach:

1. **Object Detection with YOLOv11**  
   Detects potential weapon regions in an image.

2. **Feature-Based Classification with XGBoost**  
   Cropped detection results are analyzed using traditional features:
   - Histogram of Oriented Gradients (HOG)
   - Color histograms
   - Edge features  
   These features are fed into an XGBoost classifier for refined classification.

## Dataset

The project expects datasets in YOLO format. You can:
- Use your own dataset, or
- Download an example dataset from [Roboflow](https://roboflow.com/) or other public repositories.

Expected folder structure:
dataset/
├── train/
│ ├── images/
│ ├── labels/
├── valid/
│ ├── images/
│ ├── labels/


**Negative Images:** Place images without weapons in a folder named `negative_images/`. The notebook will generate empty label files for them.

## Installation

Clone this repository:
```bash
git clone https://github.com/mhamedkhalil/Weapon-Detection-Pipeline.git
cd Weapon-Detection-Pipeline
```

## Dependencies

pip install -r requirements.txt

Requirements include:

Python 3.8+

OpenCV

NumPy

Matplotlib

scikit-image

scikit-learn

xgboost

ultralytics (for YOLOv11)

## Usage

### 1. Prepare the Dataset
Run the preprocessing cells in **`SiftWithYolo.ipynb`** to:
- Adjust YOLO label files.
- Create empty labels for negative images.

### 2. Train YOLOv11
Use the YOLO training code (in the same notebook) to train on your dataset.

### 3. Extract Weapons and Features
Open **`our model.ipynb`** to:
- Crop detected weapons from images.
- Extract features (HOG, color histograms, edges).
- Train an XGBoost model on extracted features.

### 4. Run Inference
Use the trained YOLOv11 + XGBoost pipeline to run detection on new images.

## Results
The hybrid approach aims to reduce false positives compared to YOLO alone, leveraging classical CV features for extra verification.
