import numpy as np
from PIL import Image
import streamlit as st
import tensorflow as tf
from pathlib import Path
from tensorflow.keras.applications.resnet50 import preprocess_input



st.set_page_config(
    page_title="Oral Disease Classifier",
    page_icon="🦷",
    layout="centered"
)

img_size = (224, 224)

dir = Path(__file__).resolve().parent.parent
model_path = dir / "models" / "ResNet50.keras"

class_names = [
    "Calculus",
    "Caries",
    "Gingivitis",
    "Hypodontia",
    "Mouth Ulcer",
    "Tooth Discoloration"
]

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        model_path,
        custom_objects={
            "preprocess_input": preprocess_input
        },
        safe_mode=False
    )


model = load_model()

st.title("🦷 Oral Disease Image Classifier")

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    image = image.resize(img_size)

    image = np.array(image, dtype=np.float32)

    image = np.expand_dims(image, axis=0)

    with st.spinner("Classifying..."):
        prediction = model.predict(image, verbose=0)[0]

    predicted_class = np.argmax(prediction)
    confidence = prediction[predicted_class]

    st.success(
        f"Prediction: **{class_names[predicted_class]}**"
    )

    st.metric(
        label="Confidence",
        value=f"{confidence * 100:.2f}%"
    )

    st.subheader("Class Probabilities")

    probabilities = {
        class_names[i]: float(prediction[i])
        for i in range(len(class_names))
    }

    st.bar_chart(probabilities)

st.markdown("---")