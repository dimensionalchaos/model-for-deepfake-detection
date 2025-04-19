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
