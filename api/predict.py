from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import numpy as np
import cv2
import io
from tensorflow.keras.models import load_model
from PIL import Image
import joblib  # for loading PCA/LDA if you saved them
import os
from sklearn.decomposition import PCA
app = FastAPI()

# Load model and any preprocessing objects
MODEL_PATH = os.path.join(os.path.dirname(__file__), "../model/model.h5")
model = load_model(MODEL_PATH)

# If you used PCA/LDA, load them too
# pca = joblib.load("model/pca.pkl")
# lda = joblib.load("model/lda.pkl")
x = []
x = np.array(x)
def preproccess_images(images):
    img_size = (64, 64)  # Resize images to a fixed size
    folder = "./imges"
    load_images_from_folder(folder,label="data")
    num_pca_components = 100  # Number of principal components to retain
    pca = PCA(n_components=num_pca_components)
    x = pca.fit_transform(x)
    num_lda_components = min(len(np.unique(y_train)) - 1, num_pca_components)  # LDA max components = classes - 1
    lda = LDA(n_components=num_lda_components)
    x = lda.fit_transform(x)
# Lists to store image data and labels



# Function to load images
def load_images_from_folder(folder, label):
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Convert to grayscale
        if img is not None:
            img = cv2.resize(img, img_size)  # Resize to standard size
            x.append(img.flatten())  # Flatten image into 1D array
            
            
def applyldaandpca():

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    features = extract_features(contents)
    pred = model.predict(features)[0][0]
    label = "fake" if pred > 0.5 else "real"
    return JSONResponse(content={"label": label, "confidence": float(pred)})
