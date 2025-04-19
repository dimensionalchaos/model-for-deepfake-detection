Face Real/Fake Prediction with CNN and Feature Fusion

This repository contains a machine learning model built to predict whether a given face image is real or fake. The model uses a combination of PCA, LDA, and LBPH feature extraction techniques, followed by a CNN classifier. The code also includes a FastAPI application to provide a web interface for image uploads and predictions.
Table of Contents

    Project Overview

    Setup Instructions

    Model Training

    FastAPI Application

    Predicting with the Model

    Dependencies

    License

Project Overview

This project aims to classify face images as either real or fake using a deep learning model. The approach is a multi-step process that involves the following:

    Data Preprocessing: The images are loaded, resized, and split into real and fake classes.

    Feature Extraction: The images are processed with PCA, LDA, and LBPH techniques to extract relevant features.

    Model Architecture: A Convolutional Neural Network (CNN) is used to classify the fused features (LBPH + LDA) for face image classification.

    Web Interface: A FastAPI-based web application allows users to upload face images and get predictions from the trained model.

Setup Instructions

To set up this project on your local machine, follow these steps:

    Clone the Repository:

git clone https://github.com/yourusername/face-real-fake-prediction.git
cd face-real-fake-prediction

Install Dependencies: Create a virtual environment and activate it:

python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate

Install the necessary dependencies:

pip install -r requirements.txt

Model Files:

    The pre-trained models (pca_model.pkl, lda_model.pkl, and my_model.h5) should be placed in the model/ directory. You can either use pre-trained models or train your own using the training scripts.

Run FastAPI Application: Once the environment is set up and the models are in place, you can run the FastAPI app:

    uvicorn main:app --reload

    The FastAPI app will run on http://127.0.0.1:8000/. You can visit this URL in your browser to access the web interface for uploading images and getting predictions.

Model Training
Steps to Train the Model:

    Data Loading:

        The dataset contains two folders: real and fake, which hold face images. Images are loaded and resized into a standard size of (64, 64).

    Feature Extraction:

        PCA (Principal Component Analysis) is used for dimensionality reduction.

        LDA (Linear Discriminant Analysis) is applied to reduce dimensionality while maintaining class separability.

        LBPH (Local Binary Pattern Histogram) is used to extract texture features from the images.

    Feature Fusion:

        The LBPH and LDA features are concatenated together to form the input for the CNN model.

    CNN Model:

        A 1D CNN model is used to classify the fused features (LBPH + LDA).

        The model is trained using the binary_crossentropy loss function and accuracy as the evaluation metric.

    Training:

        The model is trained using EarlyStopping to prevent overfitting. The best weights are restored after training completes.

    Model Saving:

        The trained model, PCA, and LDA models are saved to disk for later use in the FastAPI app.

FastAPI Application

The FastAPI application is designed to provide a simple interface to upload images for real/fake face prediction.
How it works:

    Home Page: The main page (/) allows the user to upload a face image via a file input form.

    Prediction Endpoint:

        The predict endpoint receives the uploaded image, processes it using the same feature extraction pipeline (PCA, LDA, and LBPH), and feeds it to the CNN model for classification.

        The output prediction is returned as a label (real or fake) based on the model's confidence.

Example Usage:

    Navigate to the / endpoint in your browser (e.g., http://127.0.0.1:8000/).

    Upload a face image, and the app will display whether the image is real or fake.

Predicting with the Model

You can also use the trained model to predict whether a face image is real or fake by calling the predict_image function.

Example usage:

prediction = predict_image("path/to/your/image.jpg")
print(f"Prediction: {prediction}")

Prediction Output:

    If the predicted value is greater than 0.5, the image is classified as fake.

    Otherwise, it is classified as real.

Dependencies

The following dependencies are required to run this project:

    Python 3.x

    FastAPI

    TensorFlow

    scikit-learn

    OpenCV

    joblib

    numpy

    scikit-image

To install the dependencies:

pip install -r requirements.txt
