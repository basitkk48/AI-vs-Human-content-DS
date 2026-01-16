import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib
import os

CSV_PATH = "models/AI_Human.csv"

if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(f"❌ File not found: {CSV_PATH}")

df = pd.read_csv(CSV_PATH)

if 'text' not in df.columns or 'label' not in df.columns:
    raise ValueError("❌ CSV must have 'text' and 'label' columns.")

X = df['text']
y = df['label']

vectorizer = TfidfVectorizer()
X_vec = vectorizer.fit_transform(X)

model = LogisticRegression(max_iter=1000)
model.fit(X_vec, y)

joblib.dump(model, 'models/text_model.pkl')
joblib.dump(vectorizer, 'models/text_vectorizer.pkl')

print("✅ Text model and vectorizer saved to 'models/'")
