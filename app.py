import os
import numpy as np
from PIL import Image
import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input
from pathlib import Path

# =====================
# 1. Load .env
# =====================

IMAGE_SIZE = int (224)
MODEL_PATH = 'my_resnet50_model.keras'

# =====================
# 2. Load Model (cache)
# =====================
@st.cache_resource
def load_my_model():
    st.write(f"📦 Đang tải model từ: `{MODEL_PATH}` ...")
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    return model

try:
    model = load_my_model()
    INPUT_SHAPE = model.input_shape
except Exception as e:
    st.error(f"❌ Lỗi khi load model: {e}")
    st.stop()

# =====================
# 3. Preprocess Image
# =====================
def preprocess_image(image: Image.Image):
    image = image.resize((IMAGE_SIZE, IMAGE_SIZE))
    image = image.convert("RGB")

    img_array = np.array(image)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array.astype(np.float32))

    return img_array

# =====================
# 4. Streamlit UI
# =====================
st.set_page_config(page_title="Human vs Non-Human Classifier", page_icon="🤖", layout="centered")

st.title("🤖 Human vs Non-Human Classifier")
st.write("Upload ảnh để model dự đoán **Human** hoặc **Non-Human**.")

st.info(f"📌 Model path: `{MODEL_PATH}`")
st.info(f"📌 Image size: `{IMAGE_SIZE}`")

uploaded_file = st.file_uploader("📤 Chọn một ảnh", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file)
        st.image(image, caption="Ảnh bạn upload", use_container_width=True)

        img_tensor = preprocess_image(image)

        # Predict
        if isinstance(INPUT_SHAPE, list) and len(INPUT_SHAPE) >= 2:
            prediction_raw = model.predict([img_tensor, img_tensor], verbose=0)
        else:
            prediction_raw = model.predict(img_tensor, verbose=0)

        prediction = float(prediction_raw[0][0])

        label = "Human" if prediction >= 0.5 else "Non-Human"
        confidence = prediction if prediction >= 0.5 else 1.0 - prediction

        st.success(f"✅ Prediction: **{label}**")
        st.write(f"🎯 Confidence: **{confidence:.4f}**")
        st.write(f"📌 Raw score: **{prediction:.4f}**")

    except Exception as e:
        st.error(f"❌ Lỗi dự đoán: {str(e)}")
