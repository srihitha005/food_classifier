import tensorflow as tf

try:
    # Load the Keras model
    model = tf.keras.models.load_model('models/rotten_classifier_model.h5')

    # Convert to TFLite with compatibility settings
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]  # Enable quantization
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,  # Include all TFLite built-in ops
        tf.lite.OpsSet.SELECT_TF_OPS      # Include select TensorFlow ops for compatibility
    ]
    converter.allow_custom_ops = False  # Disable custom ops to ensure standard support

    # Convert the model
    tflite_model = converter.convert()

    # Save the TFLite model
    with open('models/rotten_classifier_model.tflite', 'wb') as f:
        f.write(tflite_model)

    print("Model converted to TFLite and saved as models/rotten_classifier_model.tflite")

except Exception as e:
    print(f"Error during conversion: {e}")