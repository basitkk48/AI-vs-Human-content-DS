import joblib

model = joblib.load('models/text_model.pkl')
vectorizer = joblib.load('models/text_vectorizer.pkl')

def predict_text(path):
    try:
        with open(path, 'r', encoding='utf-8') as file:
            text = file.read()
        X = vectorizer.transform([text])
        pred = model.predict(X)
        return "AI-Generated" if pred[0] == 1 else "Human"
    except Exception as e:
        return f"Text Prediction Error: {e}"
