import joblib
import numpy as np
from PIL import Image

model = joblib.load('models/image_model.pkl')
scaler = joblib.load('models/image_scaler.pkl')

def predict_image(path):
    try:
        img = Image.open(path).convert('L').resize((64, 64))
        arr = np.array(img).flatten().reshape(1, -1)
        arr_scaled = scaler.transform(arr)
        pred = model.predict(arr_scaled)
        return "AI-Generated" if pred[0] == 1 else "Human"
    except Exception as e:
        return f"Image Prediction Error: {e}"
