import streamlit as st
import tensorflow as tf
import numpy as np
import cv2  # Assuming you've installed OpenCV

# Load the saved model
model = tf.keras.models.load_model('ASL_Model_2epochs.h5')

img_file_buffer = st.camera_input("Take a picture")

if img_file_buffer is not None:
    # To read image file buffer as a 3D uint8 tensor with TensorFlow:
    bytes_data = img_file_buffer.getvalue()
    img_tensor = tf.io.decode_image(bytes_data, channels=3)

    # Resize the image to the required dimensions (28x28 in this case)
    img_tensor = tf.image.resize(img_tensor, [28, 28])

    # Convert the image to grayscale
    img_tensor = tf.image.rgb_to_grayscale(img_tensor)

    # Normalize the image if your training data was normalized
    img_tensor = img_tensor / 255.0  # Assuming your images are scaled to [0,1]

    # Expand dimensions to match the input shape of the model
    img_tensor = np.expand_dims(img_tensor, axis=0)  # Adding batch dimension

    # Make prediction
    prediction = model.predict(img_tensor)

    # Process prediction (assuming softmax activation in the output layer)
    predicted_class_idx = np.argmax(prediction[0])
    confidence = np.max(prediction[0])

    # Assuming classes is the list of class names in the same order as during training
    classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'del', 'nothing', 'space']
    predicted_class_name = classes[predicted_class_idx]

    st.write(f'Predicted: {predicted_class_name} ({confidence:.2f})')
