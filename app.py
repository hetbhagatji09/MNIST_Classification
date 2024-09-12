import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import io

# Load the trained model
model = tf.keras.models.load_model('mnist_cnn_model.h5')

# Define a function to preprocess the image
def preprocess_image(image):
    # Convert image to grayscale
    image = image.convert('L')
    # Resize the image to 28x28
    image = image.resize((28, 28))
    # Convert image to numpy array
    image_array = np.array(image)
    # Normalize the image
    image_array = image_array / 255.0
    # Expand dimensions to fit the model input
    image_array = np.expand_dims(image_array, axis=(0, -1))
    return image_array

# Define a function to predict the label
def predict_label(image_array):
    # Make prediction
    predictions = model.predict(image_array)
    # Get the class with the highest probability
    predicted_class = np.argmax(predictions, axis=1)
    return predicted_class[0]

# Streamlit app
st.title('MNIST Digit Classification')

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type="png")

if uploaded_file is not None:
    # Open and display the image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # Preprocess the image
    image_array = preprocess_image(image)
    
    # Predict the label
    predicted_label = predict_label(image_array)
    
    # Display the prediction
    st.write(f'Predicted Label: {predicted_label}')
