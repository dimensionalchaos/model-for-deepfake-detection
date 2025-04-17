from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, HTMLResponse
import numpy as np
import cv2
from tensorflow.keras.models import load_model
import joblib
import os
from skimage.feature import local_binary_pattern
from tempfile import NamedTemporaryFile

app = FastAPI()

# Paths (update the paths if necessary)
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "../model/my_model.h5")
PCA_PATH = os.path.join(BASE_DIR, "../model/lda_and_pca_models/pca_model.pkl")
LDA_PATH = os.path.join(BASE_DIR, "../model/lda_and_pca_models/lda_model.pkl")
img_size = (64, 64)

# Load models only once at startup
model = load_model(MODEL_PATH)
pca = joblib.load(PCA_PATH)
lda = joblib.load(LDA_PATH)

def extract_lbph_features(img, P=8, R=1, grid_x=8, grid_y=8):
    """
    Extracts Local Binary Pattern Histogram features from a given image.
    """
    lbp = local_binary_pattern(img, P, R, method='uniform')
    n_bins = int(lbp.max() + 1)
    h, w = img.shape
    cell_h, cell_w = h // grid_y, w // grid_x
    features = []
    for i in range(grid_y):
        for j in range(grid_x):
            cell = lbp[i*cell_h:(i+1)*cell_h, j*cell_w:(j+1)*cell_w]
            hist, _ = np.histogram(cell.ravel(), bins=n_bins, range=(0, n_bins))
            hist = hist.astype("float")
            hist /= (hist.sum() + 1e-6)
            features.extend(hist)
    return np.array(features)

def predict_image(img_path: str) -> float:
    """
    Accepts an image path, processes it with the same steps as training, and returns the CNN's output probability.
    """
    # Load image in grayscale and resize it
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Image not found or unreadable.")
    img = cv2.resize(img, img_size)
    
    # Preprocess image for PCA: flatten and reshape to (1, -1)
    img_flat = img.flatten().reshape(1, -1)
    
    # Apply PCA then LDA transformation using the globally loaded models
    img_pca = pca.transform(img_flat)
    img_lda = lda.transform(img_pca)
    
    # Ensure img_lda is 2D (i.e. [1, n_features])
    if len(img_lda.shape) == 1:
        img_lda = img_lda.reshape(1, -1)
    
    # Extract LBPH features from the resized grayscale image
    lbph_features = extract_lbph_features(img)  # returns a 1D vector
    lbph_features = lbph_features.reshape(1, -1)  # shape (1, feature_length)
    
    # Fuse LBPH and LDA features by concatenating along the feature axis
    fused_features = np.concatenate((lbph_features, img_lda), axis=1)
    
    # Reshape for CNN input: add a channel dimension so shape becomes (1, num_features, 1)
    fused_features = fused_features[..., np.newaxis]
    
    # Predict using the globally loaded CNN model
    prediction = model.predict(fused_features)
    print("Prediction (probability):", prediction[0][0])
    return prediction[0][0]

# HTML form updated for clarity
@app.get("/", response_class=HTMLResponse)
async def homepage():
    return """
    <html>
    <head>
        <title>Face Real/Fake Prediction</title>
    </head>
    <body>
        <h2>Upload a Face Image for Prediction</h2>
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

                    if (!response.ok) {
                        throw new Error(data.error || "Server error");
                    }

                    resultText.textContent = "Prediction: " + data.label;
                } catch (err) {
                    console.error("Prediction error:", err);
                    resultText.textContent = "Error: " + err.message;
                }
            });
        </script>
    </body>
    </html>
    """

# Predict endpoint now uses the new predict_image function
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Read file contents from the upload
        contents = await file.read()
        # Write the uploaded bytes to a temporary file for processing
        with NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            tmp.write(contents)
            tmp_path = tmp.name

        # Call the updated predict_image function on the temporary file
        pred = predict_image(tmp_path)
        
        # Clean up the temporary file
        os.remove(tmp_path)
        
        # Use threshold 0.5 to decide between "fake" and "real"
        label = "fake" if pred > 0.5 else "real"
        return JSONResponse(content={"label": label})   
    except Exception as e:
        print("Error during prediction:", str(e))
        return JSONResponse(status_code=500, content={"error": str(e)})
