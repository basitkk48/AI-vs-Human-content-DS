import os
import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

# Define dataset paths and image size
FAKE_PATH = 'models/image_dataset/fake'
REAL_PATH = 'models/image_dataset/real'
IMG_SIZE = (64, 64)

# Function to load and preprocess images
def load_images(folder, label):
    X, y = [], []
    files = os.listdir(folder)
    print(f"🔍 Loading {len(files)} images from {folder}...")

    for i, filename in enumerate(files):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            path = os.path.join(folder, filename)
            try:
                img = Image.open(path).convert('L').resize(IMG_SIZE)
                arr = np.array(img).flatten()
                X.append(arr)
                y.append(label)
                if i % 100 == 0:
                    print(f"✅ Processed {i} files...")
            except Exception as e:
                print(f"⚠️ Skipped {filename} due to error: {e}")
    return X, y

# Main training pipeline
if __name__ == "__main__":
    print("📂 Starting image model training...")

    # Load both fake and real images
    X_fake, y_fake = load_images(FAKE_PATH, 1)
    X_real, y_real = load_images(REAL_PATH, 0)

    X = np.array(X_fake + X_real)
    y = np.array(y_fake + y_real)

    print(f"📊 Total samples: {len(X)}")

    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Train a logistic regression model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # Save the trained model and scaler
    joblib.dump(model, 'models/image_model.pkl')
    joblib.dump(scaler, 'models/image_scaler.pkl')

    print("✅ Model trained and saved to 'models/image_model.pkl'")
