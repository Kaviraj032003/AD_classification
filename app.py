from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)

# Create the uploads folder if not exists
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load the trained model
model = load_model("models/alzheimer_model.h5")

# Class labels (ensure these match your training classes)
class_labels = ['Mild Dementia', 'Moderate Dementia', 'No Dementia', 'Very Mild Dementia']

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Handle file upload
        uploaded_file = request.files['file']
        if uploaded_file:
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], uploaded_file.filename)
            uploaded_file.save(file_path)

            # Preprocess the image
            img = image.load_img(file_path, target_size=(128, 128), color_mode='rgb')
            img_array = image.img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            # Predict the class
            predictions = model.predict(img_array)
            predicted_class = np.argmax(predictions, axis=1)[0]
            result = class_labels[predicted_class]

            return render_template("index.html", prediction=result)

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
