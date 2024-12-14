import tensorflow as tf

# Load the existing T5 model saved in .h5 format
model = tf.keras.models.load_model('tf_model.h5')

# Save the model in TensorFlow SavedModel format
model.save('t5-small-savedmodel', save_format='tf')

print("Model saved as TensorFlow SavedModel format.")
