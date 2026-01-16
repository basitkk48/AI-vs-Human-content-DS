# train_video_model.py
import os
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression

VIDEO_PATH = 'models/video_dataset'
MODEL_PATH = 'models/video_model.pkl'

def extract_features_from_filename(filename):
    return [1 if 'fake' in filename.lower() else 0]

def load_videos():
    X, y = [], []
    for filename in os.listdir(VIDEO_PATH + "/fake"):
        if filename.endswith(".mp4"):
            X.append([1])
            y.append(1)
    for filename in os.listdir(VIDEO_PATH + "/real"):
        if filename.endswith(".mp4"):
            X.append([0])
            y.append(0)
    return np.array(X), np.array(y)

if __name__ == "__main__":
    print("🎥 Training dummy video model...")
    X, y = load_videos()
    model = LogisticRegression()
    model.fit(X, y)
    joblib.dump(model, MODEL_PATH)
    print(f"✅ Video model saved to {MODEL_PATH}")
