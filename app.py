# ==========================================
# 🩺 Breast Ultrasound AI Diagnostic App (Auto Model Loader - Updated)
# ==========================================

import os
import gdown
import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import altair as alt
from PIL import Image
from io import BytesIO
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.efficientnet import preprocess_input

# ==============================
# 🔹 Model configuration
# ==============================
SEG_MODEL_ID = "1CYBZRssHYWNErdU0SbcYdhzGIwHIL2ra"  # ✅ Model phân đoạn
CLF_MODEL_ID = "1fXPICuTkETep2oPiA56l0uMai2GusEJH"  # ✅ Model phân loại

SEG_MODEL_PATH = "seg_model.keras"
CLF_MODEL_PATH = "clf_model.h5"

# ==============================
# 🔹 Auto download models
# ==============================
def download_model(model_id, output_path, model_name):
    """Tự động tải model từ Google Drive nếu chưa tồn tại"""
    if not os.path.exists(output_path):
        st.info(f"📥 Đang tải {model_name} ...")
        gdown.download(f"https://drive.google.com/uc?id={model_id}", output_path, quiet=False)
        st.success(f"✅ {model_name} đã được tải xong!")

download_model(SEG_MODEL_ID, SEG_MODEL_PATH, "model phân đoạn")
download_model(CLF_MODEL_ID, CLF_MODEL_PATH, "model phân loại")

# ==============================
# 🔹 Load both models safely
# ==============================
@st.cache_resource
def load_models():
    from tensorflow import keras
    try:
        keras.config.enable_unsafe_deserialization()
    except Exception:
        pass

    classifier = tf.keras.models.load_model(CLF_MODEL_PATH, compile=False)
    segmentor = tf.keras.models.load_model(SEG_MODEL_PATH, compile=False)
    return classifier, segmentor

# ==============================
# 🔹 Image preprocessing
# ==============================
def classify_preprop(image_bytes):
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    image = image.resize((224, 224))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)
    return image

def segment_preprop(image_bytes):
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    image = image.resize((256, 256))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

def segment_postprop(image, mask):
    mask = np.argmax(mask, axis=-1)
    mask = np.expand_dims(mask, axis=-1)
    return np.squeeze(image) * (mask > 0)

# ==============================
# 🔹 Prediction pipeline
# ==============================
def predict_pipeline(file, classifier, segmentor):
    image_bytes = file.read()
    img_clf = classify_preprop(image_bytes)
    img_seg = segment_preprop(image_bytes)

    with tf.device("/CPU:0"):
        pred_class = classifier.predict(img_clf, verbose=0)
        pred_mask = segmentor.predict(img_seg, verbose=0)[0]

    seg_image = segment_postprop(img_seg, pred_mask)
    return pred_class, seg_image, image_bytes

# ==============================
# 🔹 Streamlit UI
# ==============================
st.set_page_config(page_title="Breast Ultrasound AI", layout="wide", page_icon="🩺")
st.sidebar.title("📘 Navigation")

app_mode = st.sidebar.selectbox(
    "Chọn trang",
    ["Ứng dụng chẩn đoán", "Thông tin chung", "Thống kê về dữ liệu huấn luyện"]
)

# -----------------------------
# Trang thông tin
# -----------------------------
if app_mode == "Thông tin chung":
    st.title("👨‍🎓 Giới thiệu về thành viên")
    st.markdown("<h4>Lê Vũ Anh Tin - 11TH</h4>", unsafe_allow_html=True)
    try:
        st.image("Tin.jpg", caption="Lê Vũ Anh Tin", width=250)
        st.image("school.jpg", caption="Trường THPT Chuyên Nguyễn Du", width=250)
    except:
        st.info("🖼️ Ảnh giới thiệu chưa được tải lên.")

# -----------------------------
# Trang thống kê dữ liệu
# -----------------------------
elif app_mode == "Thống kê về dữ liệu huấn luyện":
    st.title("📊 Thống kê tổng quan về tập dữ liệu")
    st.caption("""
    Tập dữ liệu **Breast Ultrasound Images (BUI)** được kết hợp từ hai nguồn:
    - BUSI (Arya Shah, Kaggle)
    - BUS-UCLM (Orvile, Kaggle)
    
    Tổng cộng **1578 ảnh siêu âm vú** có mặt nạ phân đoạn tương ứng.
    """)
    st.markdown(
        "[🔗 Link dataset gốc](https://drive.google.com/drive/folders/1eSAA5pMuEz1GgATBmvXbjjaihO1yBo1l?usp=drive_link)"
    )

# -----------------------------
# Trang ứng dụng chẩn đoán
# -----------------------------
elif app_mode == "Ứng dụng chẩn đoán":
    st.title("🩺 Ứng dụng chẩn đoán bệnh ung thư vú từ ảnh siêu âm")

    classifier, segmentor = load_models()
    file = st.file_uploader("📤 Tải ảnh siêu âm (JPG hoặc PNG)", type=["jpg", "png"])

    if file is None:
        st.info("👆 Vui lòng tải ảnh lên để bắt đầu chẩn đoán.")
    else:
        slot = st.empty()
        slot.text("⏳ Đang phân tích ảnh...")

        pred_class, seg_image, img_bytes = predict_pipeline(file, classifier, segmentor)
        input_image = Image.open(BytesIO(img_bytes))

        col1, col2 = st.columns(2)
        with col1:
            st.image(input_image, caption="Ảnh gốc", use_container_width=True)
        with col2:
            st.image(seg_image, caption="Kết quả phân đoạn", use_container_width=True)

        class_names = ["benign", "malignant", "normal"]
        result = class_names[np.argmax(pred_class)]
        if result == "benign":
            st.success("🟢 Kết luận: Khối u lành tính.")
        elif result == "malignant":
            st.error("🔴 Kết luận: Ung thư vú ác tính.")
        else:
            st.info("⚪ Kết luận: Không phát hiện khối u.")

        slot.success("✅ Hoàn tất chẩn đoán!")

        chart_df = pd.DataFrame({
            "Loại chẩn đoán": ["Lành tính", "Ác tính", "Bình thường"],
            "Xác suất (%)": [pred_class[0,0]*100, pred_class[0,1]*100, pred_class[0,2]*100]
        })
        chart = alt.Chart(chart_df).mark_bar().encode(
            x="Loại chẩn đoán",
            y="Xác suất (%)",
            color="Loại chẩn đoán"
        )
        st.altair_chart(chart, use_container_width=True)

        st.write(f"- **Khối u lành tính:** {pred_class[0,0]*100:.1f}%")
        st.write(f"- **Ung thư vú:** {pred_class[0,1]*100:.1f}%")
        st.write(f"- **Bình thường:** {pred_class[0,2]*100:.1f}%")
