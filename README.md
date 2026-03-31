# Disease Detection (Multi-Fruit Leaf Disease Classification)

This repository contains a Kaggle notebook for building a leaf disease classification model using a multi-fruit leaf disease dataset.

## Project Overview

The notebook `disease-detection.ipynb` performs the following steps:

- Inspects the Kaggle dataset structure from `/kaggle/input/multi-fruit-leaf-disease-dataset`
- Creates training, validation, and test splits from the original dataset
- Builds a TensorFlow image classification pipeline using transfer learning with `EfficientNetB1`
- Applies data augmentation to improve generalization
- Handles class imbalance using class weighting
- Trains the model in two phases: frozen base model then fine-tuning
- Evaluates the model on a held-out test set
- Generates accuracy/loss plots, a confusion matrix, and a classification report
- Saves the trained model and class names mapping

## Files

- `disease-detection.ipynb` - main notebook containing dataset setup, model training, and evaluation
- `home.htm` - unrelated HTML file present in the workspace

## Dataset

The notebook expects the Kaggle dataset to be available at:

- `/kaggle/input/multi-fruit-leaf-disease-dataset/Dataset/Train`

The notebook copies images into a working dataset directory with the following split:

- `train` — 70%
- `val` — 15%
- `test` — 15%

## Model and Training

Key model details:

- Backbone: `EfficientNetB1` pretrained on ImageNet
- Input size: `240x240`
- Output: softmax over all dataset classes
- Loss: `sparse_categorical_crossentropy`
- Metrics: `accuracy`
- Optimizer: `Adam`
- Callbacks: `EarlyStopping` and `ReduceLROnPlateau`

The notebook computes class weights from the training labels to reduce bias toward majority classes.

## Evaluation

The notebook evaluates the trained model on the test dataset and produces:

- Test loss and accuracy
- Confusion matrix visualization
- Classification report with precision, recall, and F1-score

## Output

Saved output files:

- `/kaggle/working/leaf_disease_model.keras`
- `/kaggle/working/class_names.json`

## Requirements

The notebook is designed for a Kaggle/Colab-style Python environment with these packages installed:

- `tensorflow`
- `numpy`
- `scikit-learn`
- `matplotlib`
- `seaborn`

## How to Use

1. Open `disease-detection.ipynb` in a Jupyter or Kaggle environment.
2. Ensure the dataset is mounted at `/kaggle/input/multi-fruit-leaf-disease-dataset`.
3. Run the notebook cells sequentially.
4. Review the training metrics, plots, and evaluation outputs.

## Notes

- The notebook was written for use on Kaggle and relies on Kaggle-specific paths like `/kaggle/input` and `/kaggle/working`.
- If running locally, update dataset and output paths accordingly.
