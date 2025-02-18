from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
import os
from werkzeug.utils import secure_filename
from PIL import Image

app = Flask(__name__)
model = load_model("models/alzheimers_model.h5")

UPLOAD_FOLDER = 'static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return "No file part"
        file = request.files['file']
        if file.filename == '':
            return "No selected file"
        if file:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            
            image = Image.open(file_path).convert('L').resize((128, 128))
            image = img_to_array(image) / 255.0
            image = np.expand_dims(image, axis=0)
            
            prediction = model.predict(image)
            result = np.argmax(prediction, axis=1)
            labels = ['Mild Demented', 'Moderate Demented', 'Non Demented', 'Very Mild Demented']
            label = labels[result[0]]

            return render_template('result.html', label=label, image_url=file_path)
    return render_template('index.html')

if __name__ == '__main__':
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
