import streamlit as st
import torch

from model.inference import preprocess_image, get_embedding, predict_currency
from utils.constants import SAVED_MODELS_DIR, CLASSIFIER_MODEL_NAME
from utils.helpers import get_device, load_model


st.set_page_config(page_title="Currency Recognition App", layout="centered")

st.title("Currency Recognition App")
st.write("Upload a banknote image to predict its currency type and confidence score.")

uploaded_file = st.file_uploader("Choose a banknote image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Save uploaded file temporarily
    temp_path = "temp_uploaded_image.jpg"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Preprocess image and get embedding
    img = preprocess_image(temp_path)
    embedding_tensor = get_embedding(img)

    # Load classifier model
    device = get_device()
    classifier_model_path = SAVED_MODELS_DIR + CLASSIFIER_MODEL_NAME
    model = load_model(device)

    # Predict currency and confidence
    currency, confidence = predict_currency(embedding_tensor, model, device)

    st.success(f"Predicted Currency: **{currency}**")
    st.info(f"Confidence Score: **{confidence*100:.2f}%**")