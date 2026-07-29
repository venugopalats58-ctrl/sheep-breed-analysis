import streamlit as st
import numpy as np
import cv2
from PIL import Image
from tensorflow.keras.models import load_model

st.set_page_config(
    page_title="Sheep Breed Analysis",
    page_icon="🐑",
    layout="wide"
)

st.title("🐑 Sheep Breed Analysis System")
st.write("Upload a sheep image to predict its breed and estimated weight.")

model = load_model("sheep_model.keras")

uploaded_file = st.file_uploader(
    "Upload Sheep Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

img = np.array(image)
img = cv2.resize(img, (224,224))
img = img / 255.0
img = np.expand_dims(img, axis=0)
prediction = model.predict(img)

predicted_class = np.argmax(prediction)

confidence = np.max(prediction) * 100
st.success(f"Predicted Class : {predicted_class}")

st.info(f"Confidence : {confidence:.2f}%")