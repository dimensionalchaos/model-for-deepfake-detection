from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import numpy as np
import cv2
import io
from tensorflow.keras.models import load_model
from PIL import Image
import joblib  # for loading PCA/LDA if you saved them
import os

app = FastAPI()

# Load model and any preprocessing objects
MODEL_PATH = os.path.join(os.path.dirname(__file__), "../model/model.h5")
model = load_model(MODEL_PATH)

# If you used PCA/LDA, load them too
# pca = joblib.load("model/pca.pkl")
# lda = joblib.load("model/lda.pkl")

def extract_features(image_bytes):
    # Convert bytes to PIL image
    img = Image.open(io.BytesIO(image_bytes)).convert('L')
    img = img.resize((64, 64))
    img_np = np.array(img).astype("uint8")

    # --- Your preprocessing goes here ---
    img_flat = img_np.flatten().reshape(1, -1)

    # Apply PCA + LDA if needed
    # img_pca = pca.transform(img_flat)
    # img_lda = lda.transform(img_pca)
    # features = img_lda

    # Or use LBPH if needed
    # features = extract_lbph_features(img_np)

    # If you fused LBPH + LDA
    # features = np.concatenate([lbph_feats, lda_feats], axis=1)

    # For now, let’s assume you used raw pixels
    return img_flat

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    features = extract_features(contents)
    pred = model.predict(features)[0][0]
    label = "fake" if pred > 0.5 else "real"
    return JSONResponse(content={"label": label, "confidence": float(pred)})
