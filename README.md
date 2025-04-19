# Face Real/Fake Prediction with Machine Learning and CNN

## Table of Contents

1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Model Overview](#model-overview)
4. [Dataset](#dataset)
5. [FastAPI API](#fastapi-api)
6. [Example Usage](#example-usage)
7. [License](#license)

## Introduction

This project uses a combination of traditional machine learning techniques (PCA and LDA) and deep learning (CNN) for face real/fake image classification. The goal is to detect whether a given image of a face is real or fake using a deep learning-based Convolutional Neural Network (CNN), combined with additional feature engineering such as PCA, LDA, and LBPH (Local Binary Patterns).

## Installation

To run this project, clone the repository and install the necessary dependencies. You can use the following commands:

```bash
git clone https://github.com/yourusername/face-real-fake-prediction.git
cd face-real-fake-prediction
pip install -r requirements.txt
```

## 📘 Project Requirements

This project depends on the following Python libraries:

- Python 3.x  
- TensorFlow 2.x  
- scikit-learn  
- OpenCV  
- FastAPI  
- Joblib  
- scikit-image  

These dependencies can be installed using the following `requirements.txt` file:

```txt
tensorflow>=2.0.0
scikit-learn
opencv-python
fastapi
joblib
scikit-image
```
## Model Overview

The model is based on the following steps:

### Image Preprocessing:

Load and preprocess the images, converting them to grayscale and resizing them to 64x64 pixels.

Feature Engineering:

PCA (Principal Component Analysis) is applied for dimensionality reduction.

LDA (Linear Discriminant Analysis) is used for enhancing class separation.

LBPH (Local Binary Patterns Histogram) is used to extract texture features.

### Feature Fusion:

The features from PCA, LDA, and LBPH are combined to form a feature vector.

### CNN Model:

 A simple Convolutional Neural Network (CNN) is used for final classification, achieving high accuracy with the fused feature set.

Prediction:

The trained model can be used to predict whether a given image is real or fake.

## Dataset

The dataset used for training and testing is the Real vs Fake Faces Dataset. It contains images labeled as "real" and "fake" faces.

The directory structure should look like:

train
    real
    fake

real contains real face images.

fake contains fake face images.

Ensure that the dataset is placed correctly in the train/real and train/fake directories.


## FastAPI API

This project includes a FastAPI application that allows you to upload an image and get a prediction of whether the face is real or fake.
Running the FastAPI Server

Run the following command to start the FastAPI server:
```bash
python -m uvicorn main:app --reload
```
Visit http://127.0.0.1:8000 in your browser to access the upload form.
#Upload an Image for Prediction

    Open the webpage.

    Click the "Choose File" button to upload an image.

    The model will process the image and provide a prediction: whether the image is real or fake.

