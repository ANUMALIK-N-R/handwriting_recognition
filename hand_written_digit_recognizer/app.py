import streamlit as st
import joblib
import numpy as np
from PIL import Image, ImageOps

# Load model
model = joblib.load("model.pkl")
label_map = {0: 3, 1: 5}

st.title("Digit Classifier: 3 vs 5")
st.write("Upload an image of a digit (3 or 5) to classify.")

def preprocess_image(img):
    img = ImageOps.grayscale(img)
    img = img.resize((28, 28))
    img = ImageOps.invert(img)
    return np.array(img).reshape(1, -1)

# Only upload option
uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded image", use_column_width=False)
    input_data = preprocess_image(img)
    pred = model.predict(input_data)[0]
    st.success(f"Prediction: **{label_map[pred]}**")
