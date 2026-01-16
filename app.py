from flask import Flask, render_template, request
from utils.process_image import predict_image
from utils.process_video import predict_video
from utils.process_text import predict_text
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return render_template("result.html", file_type="N/A", result="No file uploaded", status="error")

    file = request.files['file']
    if file.filename == '':
        return render_template("result.html", file_type="N/A", result="No selected file", status="error")

    file_type = request.form.get('file_type', '').lower().strip()

    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    try:
        if file_type == 'image':
            result = predict_image(filepath)
        elif file_type == 'video':
            result = predict_video(filepath)
        elif file_type == 'text':
            result = predict_text(filepath)
        else:
            return render_template("result.html", file_type="N/A", result="Invalid type selected.", status="error")

        status = "success"

    except Exception as e:
        result = f"Error while processing: {e}"
        status = "error"

    return render_template(
        "result.html",
        file_type=file_type.capitalize(),
        result=result,
        status=status,
        filename=filename
    )

if __name__ == '__main__':
    app.run(debug=True)
