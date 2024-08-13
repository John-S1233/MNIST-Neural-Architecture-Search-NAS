# MNIST Neural Architecture Search (NAS) with Keras Tuner and K-Fold Cross-Validation

This repository contains a TensorFlow-based implementation for building and tuning a Convolutional Neural Network (CNN) model using the MNIST dataset. The project leverages Keras Tuner for Neural Architecture Search (NAS) and utilizes K-Fold Cross-Validation to evaluate model performance.

## Key Features

- **Neural Architecture Search (NAS)**: Automatically searches for the optimal neural network architecture using Keras Tuner.
- **K-Fold Cross-Validation**: Implements 7-fold cross-validation to ensure robust evaluation of the model's performance.
- **GPU Utilization**: Leverages available GPU resources to accelerate model training.
- **Exponential Learning Rate Decay**: Uses a learning rate scheduler with exponential decay to optimize training efficiency.
- **Model Saving**: Automatically saves the best model configuration to the specified directory.

## Dataset
![image](https://github.com/user-attachments/assets/c93cefb2-0849-4495-acff-fed408f8357f)

*Image: MNIST Dataset.*
The model is trained and tested on the **MNIST dataset**, which is a large database of handwritten digits commonly used for training various image processing systems. The dataset is split into training and test sets:
- **Training Set**: 60,000 images
- **Test Set**: 10,000 images

## Installation

To use this project, follow these steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/John-S1233/mnist-nas-tuning.git
    cd mnist-nas-tuning
    ```

2. Install the required dependencies:
    ```bash
    pip install tensorflow keras keras-tuner matplotlib scikit-learn
    ```

## Usage

To train the model and perform NAS with hyperparameter tuning, run the script `train_model.py`:

```bash
python train_model.py
