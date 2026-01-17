# 🛡️ AI vs Human Content Detection System

A Flask-based machine learning project to detect whether uploaded content — images, videos, or text — is human-generated or created by AI.

---

## 🚀 Features

- 🔍 Detect AI-generated vs Human-generated:
  - 🖼 Images
  - 🎥 Videos
  - 📝 Text
- 🧠 Uses Logistic Regression and TF-IDF for classification
- 📦 Built with Python, Flask, OpenCV, Scikit-learn
- 🧪 Trained on real-world AI/human datasets

---

## 🧱 Project Structure

├── app.py # Flask application
├── templates/index.html # Upload form
├── models/ # Trained model files (.pkl)
│ ├── image_model.pkl
│ ├── video_model.pkl
│ ├── text_model.pkl
│ ├── text_vectorizer.pkl
│ └── AI_Human.csv
├── utils/ # Prediction scripts
│ ├── process_image.py
│ ├── process_video.py
│ └── process_text.py
├── uploads/ # Temp file uploads
├── requirements.txt
└── README.md


---

## 📂 Datasets Used

| Type   | Source |
|--------|--------|
| Image  | [ThisPersonDoesNotExist, CelebA, Kaggle Subsets] |
| Video  | Synthetic + Real video sample classification |
| Text   | Manually labeled AI-Human sentences (see `AI_Human.csv`) |

---

## ⚙️ Run Locally

```bash
# Step 1: Create virtual environment
python -m venv venv
venv\Scripts\activate

# Step 2: Install dependencies
pip install -r requirements.txt

# Step 3: Train models (once)
python train_image_model_from_your_dataset.py
python train_video_model.py
python train_text_model.py

# Step 4: Run the app
python app.py
Then open: http://127.0.0.1:5000

📜 Policy
A full information assurance policy is included:
📄 Content_Authenticity_Policy.pdf

📌 Author
Sufyan-Ayaz,Ahmed-Farzan-&-Abdul-Basit
🎓 Built for ethical AI detection, academic submission, or portfolio use.

🌐 License
MIT License. For educational & ethical purposes only.


