import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications import VGG19
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten

# Title
st.title("🧠 Brain Tumor Detection from MRI Scans")

# Build and load the model
@st.cache_resource
def load_model():
    base_model = VGG19(include_top=False, input_shape=(224, 224, 3), weights='imagenet')
    x = Flatten()(base_model.output)
    x = Dense(128, activation='relu')(x)
    output = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=base_model.input, outputs=output)

    # Load your custom weights
    model.load_weights("./model_weights/vgg19_model_03.weights.h5")
    return model

model = load_model()

# Image preprocessing
def preprocess_image(image: Image.Image):
    image = image.resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(image)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalize
    return img_array

# File uploader
uploaded_file = st.file_uploader("Upload an MRI Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded MRI Scan", use_column_width=True)

    with st.spinner("Analyzing..."):
        processed_img = preprocess_image(image)
        prediction = model.predict(processed_img)[0][0]

    st.subheader("Prediction Result:")
    if prediction >= 0.5:
        st.error("⚠️ Tumorous Detected")
    else:
        st.success("✅ Non-Tumorous")
