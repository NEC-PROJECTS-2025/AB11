import os
os.environ['TF_USE_LEGACY_KERAS'] = '1'
from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras.preprocessing import image
import numpy as np

app = Flask(__name__)

# Load the pre-trained EfficientNet model from TensorFlow Hub
model_url = "https://tfhub.dev/google/efficientnet/b0/classification/1"

# Wrap the hub layer in a Lambda layer
hub_layer = tf.keras.layers.Lambda(lambda x: hub.KerasLayer(model_url)(x), input_shape=(224, 224, 3))

# Build the functional model
inputs = tf.keras.Input(shape=(224, 224, 3))
x = hub_layer(inputs)
outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)  # Binary classification
model = tf.keras.Model(inputs, outputs)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
print("Model loaded successfully")

# Define function to preprocess the image
def preprocess_image(file_path):
    img = image.load_img(file_path, target_size=(224, 224))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    img = tf.keras.applications.efficientnet.preprocess_input(img)  # Preprocess for EfficientNet
    return img

# Define route for the main page
@app.route('/')
def index():
    return render_template('base.html')

# Define route to handle image upload and prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    try:
        # Ensure the 'uploads' directory exists
        if not os.path.exists('uploads'):
            os.makedirs('uploads')

        # Save the uploaded file to the 'uploads' directory
        file_path = os.path.join('uploads', 'temp_img.jpg')
        file.save(file_path)

        # Check if the file was saved successfully
        if not os.path.exists(file_path):
            return jsonify({'error': 'File not saved successfully'})

        # Preprocess the image
        processed_img = preprocess_image(file_path)
        print(f"Preprocessed image shape: {processed_img.shape}")

        # Ensure processed image is a tensor before prediction
        processed_img_tensor = tf.convert_to_tensor(processed_img)
        print(f"Processed image tensor: {processed_img_tensor}")

        # Make prediction
        result = model.predict(processed_img_tensor)
        print(f"Raw prediction result: {result}")

        # Interpret the prediction result
        prediction = 'Real' if result[0] < 0.5 else 'Fake'
        print(f"Prediction: {prediction}")

        # Remove the temporary file
        os.remove(file_path)

        return jsonify({'prediction': prediction})

    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)