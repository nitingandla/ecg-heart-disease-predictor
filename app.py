import streamlit as st
import numpy as np
import cv2
import json
from keras.models import load_model
import gdown
import os

MODEL_PATH = "ecg_best.keras"

if not os.path.exists(MODEL_PATH):
    url = "https://drive.google.com/uc?id=1n_gG5GKrASevebgZRVm5J-W1xTsg8uWQ"
    gdown.download(url, MODEL_PATH, quiet=False)

model = load_model(MODEL_PATH)

model = load_my_model()

with open("class_indices.json", "r") as f:
    class_indices = json.load(f)

classes = {v: k for k, v in class_indices.items()}

def preprocess_image(file_bytes):
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if img is None:
        raise ValueError("Image decode failed")

    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    return img


def draw_grid_and_extract(file_bytes):
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if img is None:
        raise ValueError("Image decode failed")

    img = cv2.resize(img, (600, 400))

    overlay = img.copy()

    h, w, _ = img.shape
    grid_h = h // 2
    grid_w = w // 4

    leads = []

    for i in range(2):
        for j in range(4):
            y1, y2 = i * grid_h, (i + 1) * grid_h
            x1, x2 = j * grid_w, (j + 1) * grid_w

            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)

            lead = img[y1:y2, x1:x2]

           
            if lead is not None and lead.size > 0:
                leads.append(lead)
            else:
                leads.append(np.zeros((50, 50, 3), dtype=np.uint8))

    return overlay, leads


st.title("ECG AI Diagnostic System")

file = st.file_uploader("Upload ECG Image", type=["jpg", "png", "jpeg"])

if file:

    file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)

  
    img_display = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if img_display is None:
        st.error("❌ Invalid image file")
        st.stop()

    st.image(cv2.cvtColor(img_display, cv2.COLOR_BGR2RGB), width=300)

  
    try:
        grid_img, leads = draw_grid_and_extract(file_bytes)

        st.subheader(" Lead Grid on ECG")
        st.image(cv2.cvtColor(grid_img, cv2.COLOR_BGR2RGB))

        st.subheader(" Extracted Leads")

        cols = st.columns(4)
        for i, lead in enumerate(leads):
            with cols[i % 4]:
                st.image(cv2.cvtColor(lead, cv2.COLOR_BGR2RGB), caption=f"Lead {i+1}")

    except Exception as e:
        st.error(f"Image processing error: {str(e)}")

    
    if st.button("Analyze ECG"):

        try:
            img = preprocess_image(file_bytes)
            preds = model.predict(img, verbose=0)[0]

            abnormal = preds[0]
            history_mi = preds[1]
            mi = preds[2]
            normal = preds[3]

            st.subheader("Final Diagnosis")

            if mi >= 0.095:
                st.error("🚨 Myocardial Infarction Detected")
            elif history_mi > 0.17:
                st.warning("⚠ History of MI Detected")
            elif abnormal >= 0.40:
                st.warning("⚠ Abnormal Heartbeat Detected")
            else:
                st.success("✅ Normal ECG")

        except Exception as e:
            st.error(f"Prediction error: {str(e)}")

        