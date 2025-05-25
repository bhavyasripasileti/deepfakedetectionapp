import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

@st.cache_resource
def load_model():
    return tf.keras.models.load_model('deepfake_model.h5')

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

st.title("Deepfake Face Detection")
uploaded_file = st.file_uploader("Upload a face image/video", type=["jpg", "png", "mp4"])

if uploaded_file:
    if uploaded_file.type == "video/mp4":
        st.warning("Video processing with LSTM...") 
    else:
        image = Image.open(uploaded_file)
        img_array = np.array(image.resize((128, 128))) / 255.0
        prediction = model.predict(np.expand_dims(img_array, axis=0))
        st.write("Fake" if prediction[0][0] > 0.5 else "Real")

