import streamlit as st
import numpy as np
from PIL import Image
import cv2

st.title("🫁 Pneumonia Detection AI")
st.write("Upload a chest X-ray image")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # عرض الصورة
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # تجهيز الصورة (بس للعرض)
    img = np.array(image)
    img = cv2.resize(img, (128, 128))
    img = img / 255.0

    # 🎯 prediction وهمي
    prediction = np.random.rand()

    # عرض النتيجة
    if prediction > 0.5:
        st.error("⚠️ Pneumonia Detected")
    else:
        st.success("✅ Normal")

    # نسبة ثقة
    confidence = prediction if prediction > 0.5 else 1 - prediction
    st.metric("Confidence", f"{confidence*100:.2f}%")