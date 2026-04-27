import streamlit as st
import numpy as np
import cv2
import json
import onnxruntime as ort
import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


@st.cache_resource
def load_model():
    return ort.InferenceSession(
        "ecg_best.onnx",
        providers=["CPUExecutionProvider"]
    )


@st.cache_resource
def load_classes():
    with open("class_indices.json", "r") as f:
        class_indices = json.load(f)
    return {v: k for k, v in class_indices.items()}


session = load_model()
classes = load_classes()


def preprocess_image(file_bytes):
    img = cv2.imdecode(np.frombuffer(file_bytes, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Image decode failed")
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    img = np.expand_dims(img, axis=0).astype(np.float32)
    return img


def draw_grid_and_extract(file_bytes):
    img = cv2.imdecode(np.frombuffer(file_bytes, np.uint8), cv2.IMREAD_COLOR)
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
            leads.append(
                lead if lead.size > 0 else np.zeros((50, 50, 3), dtype=np.uint8)
            )

    return overlay, leads


# ================= UI =================
st.title("ECG AI Diagnostic System")

file = st.file_uploader("Upload ECG Image", type=["jpg", "png", "jpeg"])

if file:
    file_bytes = file.read()
    file_bytes = np.frombuffer(file_bytes, np.uint8)

    img_display = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if img_display is None:
        st.error("❌ Invalid image file")
        st.stop()

    st.image(cv2.cvtColor(img_display, cv2.COLOR_BGR2RGB), width=300)

    try:
        grid_img, leads = draw_grid_and_extract(file_bytes)

        st.subheader("Lead Grid on ECG")
        st.image(cv2.cvtColor(grid_img, cv2.COLOR_BGR2RGB))

        st.subheader("Extracted Leads")
        cols = st.columns(4)

        for i, lead in enumerate(leads):
            with cols[i % 4]:
                st.image(
                    cv2.cvtColor(lead, cv2.COLOR_BGR2RGB),
                    caption=f"Lead {i+1}"
                )

    except Exception as e:
        st.error(f"Image processing error: {str(e)}")

    if st.button("Analyze ECG"):
        try:
            img = preprocess_image(file_bytes)

            input_name = session.get_inputs()[0].name
            preds = session.run(None, {input_name: img})[0][0]

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

        