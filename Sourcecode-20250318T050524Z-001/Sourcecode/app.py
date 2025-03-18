from flask import Flask, render_template, request, url_for
import os
from PIL import Image
import numpy as np
from keras.models import load_model
from io import BytesIO
from PIL import Image, ImageChops, ImageEnhance

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static'

def convert_to_ela_image(image_path, quality):
    # Open the image
    image = Image.open(image_path).convert('RGB')
    
    # Save the image to a BytesIO object
    with BytesIO() as image_bytes:
        image.save(image_bytes, 'JPEG', quality=quality)
        image_bytes.seek(0)
        
        # Open the image from BytesIO
        temp_image = Image.open(image_bytes)
        
        # Compute ELA
        ela_image = ImageChops.difference(image, temp_image)
        
        # Get extrema
        extrema = ela_image.getextrema()
        max_diff = max([ex[1] for ex in extrema])
        
        # Calculate scale
        if max_diff == 0:
            max_diff = 1
        scale = 255.0 / max_diff
        
        # Enhance brightness
        ela_image = ImageEnhance.Brightness(ela_image).enhance(scale)
        
        return ela_image
    
image_size = (128, 128)


def prepare_image(image_path):
    return np.array(convert_to_ela_image(image_path, 90).resize(image_size)).flatten() / 255.0

model = load_model('model_casia_run1.h5')

def classify_image(image_path, model):
    class_names = ['fake', 'real']
    image = prepare_image(image_path)
    image = image.reshape(-1, 128, 128, 3)
    y_pred = model.predict(image)
    y_pred_class = np.argmax(y_pred, axis=1)[0]
    confidence = np.amax(y_pred) * 100
    image_class = class_names[y_pred_class]
    return image_class, confidence

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    file_path = None

    if request.method == 'POST':
        uploaded_file = request.files['file']
        if uploaded_file:
            filename = uploaded_file.filename
            full_image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            uploaded_file.save(full_image_path)

            # Classify the image and get prediction
            image_class, confidence = classify_image(full_image_path, model)

            # Set values for rendering in the template
            prediction = f"{image_class} with confidence {confidence:.2f}%"
            file_path = url_for('static', filename=filename)

    return render_template('index.html', prediction=prediction, file_path=file_path)

if __name__ == "__main__":
    app.run(debug=True)
