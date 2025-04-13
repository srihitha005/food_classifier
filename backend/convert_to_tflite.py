import tensorflow as tf

# Load the Keras model
model = tf.keras.models.load_model('models/rotten_classifier_model.h5')

# Convert the model to TFLite format (optional: add quantization for smaller size)
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]  # Enable quantization
tflite_model = converter.convert()

# Save the TFLite model
with open('models/rotten_classifier_model.tflite', 'wb') as f:
    f.write(tflite_model)

print("Model converted to TFLite and saved as models/rotten_classifier_model.tflite")