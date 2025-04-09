from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import JSONResponse, HTMLResponse
import numpy as np
import cv2
from tensorflow.keras.models import load_model
import joblib
import os

app = FastAPI()

# Paths
MODEL_PATH = os.path.join(os.path.dirname(__file__), "../model/model.h5")
PCA_PATH = os.path.join(os.path.dirname(__file__), "../model/lda_and_pca_models/pca_model.pkl")
LDA_PATH = os.path.join(os.path.dirname(__file__), "../model/lda_and_pca_models/lda_model.pkl")
IMG_SIZE = (64, 64)

# Load models
model = load_model(MODEL_PATH)
pca = joblib.load(PCA_PATH)
lda = joblib.load(LDA_PATH)

# Preprocessing function
def preprocess_uploaded_image(contents: bytes):
    np_arr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Could not decode image")

    img = cv2.resize(img, IMG_SIZE)
    img_flattened = img.flatten().reshape(1, -1)

    img_pca = pca.transform(img_flattened)
    img_lda = lda.transform(img_pca)

    return img_lda

# HTML form
@app.get("/", response_class=HTMLResponse)
async def homepage():
    return """
    <html>
    <head>
        <title>Face Real/Fake Prediction</title>
    </head>
    <body>
        <h2>Upload a Face Image</h2>
        <form id="upload-form" enctype="multipart/form-data">
            <input type="file" name="file" id="file-input" accept="image/*" required />
            <button type="submit">Upload & Predict</button>
        </form>
        <p id="result"></p>

        <script>
            const form = document.getElementById("upload-form");
            const resultText = document.getElementById("result");

            form.addEventListener("submit", async (e) => {
                e.preventDefault();
                const formData = new FormData(form);
                resultText.textContent = "Processing...";

                try {
                    const response = await fetch("/predict", {
                        method: "POST",
                        body: formData
                    });
                    const data = await response.json();
                    resultText.textContent = `Prediction: ${data.label} (Confidence: ${data.confidence.toFixed(4)})`;
                } catch (err) {
                    resultText.textContent = "Error: " + err.message;
                }
            });
        </script>
    </body>
    </html>
    """

# Predict endpoint
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        features = preprocess_uploaded_image(contents)
        pred = model.predict(features)[0][0]
        label = "fake" if pred > 0.5 else "real"
        return JSONResponse(content={"label": label, "confidence": float(pred)})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
