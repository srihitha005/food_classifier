from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import sys

# Force UTF-8 encoding for console output
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

# Initialize Flask app
app = Flask(__name__)
CORS(app, resources={r"/predict": {"origins": "http://localhost:8080"}})

# Load the TFLite model
try:
    interpreter = tf.lite.Interpreter(model_path='models/rotten_classifier_model.tflite')
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print("TFLite model loaded successfully.", flush=True)
except Exception as e:
    print(f"Error loading TFLite model: {e}", flush=True)
    raise

# Define the prediction function using TFLite
def predict_rotten(img):
    # Preprocess the image
    img = img.resize((224, 224))
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Set input tensor and run inference
    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()

    # Get prediction
    output = interpreter.get_tensor(output_details[0]['index'])
    prediction = output[0][0]
    result = "rotten" if prediction > 0.5 else "fresh"
    confidence = prediction if result == "rotten" else 1 - prediction
    return result, confidence

# Define the API endpoint for image prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    try:
        img = Image.open(io.BytesIO(file.read())).convert('RGB')
        result, confidence = predict_rotten(img)
        return jsonify({'result': result, 'confidence': float(confidence)})
    except Exception as e:
        print(f"Error during prediction: {e}", flush=True)
        return jsonify({'error': str(e)}), 500

# Run the Flask application
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
