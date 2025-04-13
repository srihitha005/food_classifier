import tensorflow as tf

try:
    # Load the Keras model
    model = tf.keras.models.load_model('models/rotten_classifier_model.h5')

    # Convert to TFLite with strict compatibility
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    # Disable optimizations to avoid newer op versions
    converter.optimizations = []  # Remove quantization for now
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,  # Use only TFLite built-in ops
        # Remove SELECT_TF_OPS to avoid TensorFlow-specific ops
    ]
    converter.allow_custom_ops = False  # Ensure no custom ops
    converter._experimental_lower_tensor_list_ops = False  # Avoid experimental ops

    # Convert the model
    tflite_model = converter.convert()

    # Save the TFLite model
    with open('models/rotten_classifier_model.tflite', 'wb') as f:
        f.write(tflite_model)

    print("Model converted to TFLite and saved as models/rotten_classifier_model.tflite")

except Exception as e:
    print(f"Error during conversion: {e}")