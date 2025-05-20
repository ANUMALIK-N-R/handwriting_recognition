import streamlit as st
import joblib
import numpy as np
from PIL import Image, ImageOps
from streamlit_drawable_canvas import st_canvas

# Load model
model = joblib.load("model.joblib")
label_map = {0: 3, 1: 5}

st.title("Digit Classifier: 3 vs 5")
st.write("Draw or upload a digit (3 or 5) to classify.")

# Input method selection
option = st.radio("Choose input method:", ["Draw Digit", "Upload Image"])

def preprocess_image(img):
    img = ImageOps.grayscale(img)
    img = img.resize((28, 28))
    img = ImageOps.invert(img)
    return np.array(img).reshape(1, -1)

if option == "Draw Digit":
    canvas_result = st_canvas(
        fill_color="white",
        stroke_width=10,
        stroke_color="black",
        background_color="white",
        height=280,
        width=280,
        drawing_mode="freedraw",
        key="canvas",
    )

    if canvas_result.image_data is not None:
        img = Image.fromarray((canvas_result.image_data[:, :, 0]).astype('uint8'))
        input_data = preprocess_image(img)
        pred = model.predict(input_data)[0]
        st.success(f"Prediction: **{label_map[pred]}**")

elif option == "Upload Image":
    uploaded_file = st.file_uploader("Upload an image of digit 3 or 5", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        st.image(img, caption="Uploaded image", use_column_width=False)
        input_data = preprocess_image(img)
        pred = model.predict(input_data)[0]
        st.success(f"Prediction: **{label_map[pred]}**")
