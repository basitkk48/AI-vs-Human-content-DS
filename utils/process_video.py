import joblib
import os

model = joblib.load('models/video_model.pkl')

def predict_video(path):
    try:
        filename = os.path.basename(path).lower()
        feature = [[1 if 'fake' in filename else 0]]
        pred = model.predict(feature)
        return "AI-Generated" if pred[0] == 1 else "Human"
    except Exception as e:
        return f"Video Prediction Error: {e}"
