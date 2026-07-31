
import streamlit as st
import numpy as np
import os
from PIL import Image
from tensorflow.keras.models import load_model

# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Sheep Breed Analysis",
    page_icon="🐑",
    layout="wide"
)

# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

.main {
    background-color: #f5f7fa;
}

.title {
    text-align: center;
    font-size: 42px;
    font-weight: bold;
    margin-bottom: 5px;
}

.subtitle {
    text-align: center;
    font-size: 18px;
    color: #666;
    margin-bottom: 30px;
}

.result-box {
    padding: 25px;
    border-radius: 15px;
    background-color: #ffffff;
    box-shadow: 0px 4px 15px rgba(0,0,0,0.10);
    margin-top: 20px;
}

.breed {
    font-size: 30px;
    font-weight: bold;
}

.confidence {
    font-size: 22px;
    font-weight: bold;
}

.info-box {
    padding: 18px;
    border-radius: 12px;
    background-color: #ffffff;
    box-shadow: 0px 3px 10px rgba(0,0,0,0.08);
    margin-top: 15px;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# TITLE
# =========================================================

st.markdown(
    '<div class="title">🐑 Sheep Breed Analysis</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'AI-Based Sheep Breed Classification System'
    '</div>',
    unsafe_allow_html=True
)

# =========================================================
# MODEL PATH
# =========================================================

MODEL_PATH = "new_sheep_model.keras"

# =========================================================
# BREED INFORMATION
# =========================================================

weight_data = {
    "Marino": "45 - 80 kg",
    "Poll Dorset": "70 - 120 kg",
    "Suffolk": "60 - 110 kg",
    "White Suffolk": "65 - 120 kg"
}

breed_description = {
    "Marino": "Marino sheep are famous for producing high-quality fine wool.",
    "Poll Dorset": "Poll Dorset sheep are excellent meat-producing sheep with fast growth.",
    "Suffolk": "Suffolk sheep are one of the world's most popular meat breeds.",
    "White Suffolk": "White Suffolk sheep are known for rapid growth, high fertility, and premium meat quality."
}

# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_sheep_model():

    if not os.path.exists(MODEL_PATH):

        st.error(
            "❌ Model file not found: "
            + MODEL_PATH
        )

        st.stop()

    return load_model(MODEL_PATH)


model = load_sheep_model()

# =========================================================
# BREED NAMES
# =========================================================

# IMPORTANT:
# These names must be in the SAME ORDER as your
# train_generator.class_indices.

class_names = [
    "Marino",
    "Poll Dorset",
    "Suffolk",
    "White Suffolk"
]

# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.header("🐑 About Project")

    st.write(
        """
        This application uses a Machine Learning
        image classification model to identify
        sheep breeds.
        """
    )

    st.divider()

    st.subheader("Features")

    st.write("✅ Sheep breed identification")
    st.write("✅ Prediction confidence")
    st.write("✅ Typical weight range")
    st.write("✅ Breed information")
    st.write("✅ Image upload")

# =========================================================
# IMAGE UPLOAD
# =========================================================

st.subheader("📷 Upload Sheep Image")

uploaded_file = st.file_uploader(
    "Choose a sheep image",
    type=["jpg", "jpeg", "png"]
)

# =========================================================
# PREDICTION
# =========================================================

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns(2)

    # -----------------------------------------------------
    # IMAGE
    # -----------------------------------------------------

    with col1:

        st.subheader("Uploaded Image")

        st.image(
            image,
            caption="Selected Sheep Image",
            use_container_width=True
        )

    # -----------------------------------------------------
    # PREDICTION
    # -----------------------------------------------------

    with col2:

        st.subheader("🔍 Analysis")

        # Resize image
        img = image.resize((224, 224))

        img_array = np.array(img, dtype=np.float32)
        img_array = img_array / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Convert to numpy
        img_array = np.array(img)

        # Normalize
        img_array = img_array / 255.0

        # Add batch dimension
        img_array = np.expand_dims(
            img_array,
            axis=0
        )

        # Model prediction
        prediction = model.predict(
            img_array,
            verbose=0
        )

        predicted_index = np.argmax(
            prediction[0]
        )

        confidence = (
            prediction[0][predicted_index] * 100
        )

        # Prevent index error
        if predicted_index < len(class_names):

            breed_name = class_names[
                predicted_index
            ]

        else:

            breed_name = "Unknown"

        # -------------------------------------------------
        # RESULT
        # -------------------------------------------------

        st.markdown(
            '<div class="result-box">',
            unsafe_allow_html=True
        )

        st.markdown(
            f'<div class="breed">'
            f'🐑 {breed_name}'
            f'</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            f'<div class="confidence">'
            f'Confidence: {confidence:.2f}%'
            f'</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            '</div>',
            unsafe_allow_html=True
        )

        # -------------------------------------------------
        # WEIGHT
        # -------------------------------------------------

        st.markdown(
            '<div class="info-box">',
            unsafe_allow_html=True
        )

        st.subheader("⚖️ Typical Weight")

        if breed_name in weight_data:

            st.success(
                weight_data[breed_name]
            )

        else:

            st.info(
                "Weight information is not available "
                "for this breed."
            )

        st.markdown(
            '</div>',
            unsafe_allow_html=True
        )

        # -------------------------------------------------
        # DESCRIPTION
        # -------------------------------------------------

        st.markdown(
            '<div class="info-box">',
            unsafe_allow_html=True
        )

        st.subheader("📋 Breed Information")

        if breed_name in breed_description:

            st.write(
                breed_description[breed_name]
            )

        else:

            st.write(
                "Information is not available."
            )

        st.markdown(
            '</div>',
            unsafe_allow_html=True
        )

        # -------------------------------------------------
        # PROBABILITY
        # -------------------------------------------------

        st.subheader("📊 Prediction Probability")

        for i, breed in enumerate(class_names):

            if i < len(prediction[0]):

                probability = (
                    prediction[0][i] * 100
                )

                st.write(
                    f"{breed}: "
                    f"{probability:.2f}%"
                )

                st.progress(
                    float(prediction[0][i])
                )

# =========================================================
# FOOTER
# =========================================================

st.divider()

st.markdown(
    """
    <div style="text-align:center; color:#777;">
    🐑 Sheep Breed Analysis | Machine Learning Project
    </div>
    """,
    unsafe_allow_html=True
)

