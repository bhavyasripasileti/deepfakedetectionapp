import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Load your CNN+LSTM model (do this once at startup)
model = tf.keras.models.load_model('deepfake_model.h5')

st.title("Deepfake Face Detection")
uploaded_file = st.file_uploader("Upload a face image/video", type=["jpg", "png", "mp4"])

if uploaded_file:
    # Preprocess and predict
    if uploaded_file.type == "video/mp4":
        st.warning("Video processing with LSTM...")  # Add your LSTM logic here
    else:
        image = Image.open(uploaded_file)
        img_array = np.array(image.resize((128, 128))) / 255.0
        prediction = model.predict(np.expand_dims(img_array, axis=0))
        st.write("Fake" if prediction[0][0] > 0.5 else "Real")