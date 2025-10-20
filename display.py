# ==========================================
# 🩺 Breast Ultrasound AI Diagnostic App (with Language Toggle, no content loss)
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
SEG_MODEL_ID = "1axOg7N5ssJrMec97eV-JMPzID26ynzN1"
CLF_MODEL_ID = "1fXPICuTkETep2oPiA56l0uMai2GusEJH"

SEG_MODEL_PATH = "seg_model.keras"
CLF_MODEL_PATH = "clf_model.h5"

# ==============================
# 🔹 Custom Lambda functions
# ==============================
def spatial_mean(t): return tf.reduce_mean(t, axis=-1, keepdims=True)
def spatial_max(t): return tf.reduce_max(t, axis=-1, keepdims=True)
def spatial_output_shape(s): return (s[0], s[1], s[2], 1)

# ==============================
# 🔹 Auto download models
# ==============================
def download_model(model_id, output_path, model_name):
    if not os.path.exists(output_path):
        st.info(f"📥 Đang tải {model_name} (ID: {model_id})...")
        gdown.download(f"https://drive.google.com/uc?id={model_id}", output_path, quiet=False)
        st.success(f"✅ {model_name} đã được tải xong!")

download_model(SEG_MODEL_ID, SEG_MODEL_PATH, "model phân đoạn")
download_model(CLF_MODEL_ID, CLF_MODEL_PATH, "model phân loại")

# ==============================
# 🔹 Load both models
# ==============================
@st.cache_resource
def load_models():
    CUSTOM_OBJECTS = {"spatial_mean": spatial_mean, "spatial_max": spatial_max, "spatial_output_shape": spatial_output_shape}
    from tensorflow import keras
    try: keras.config.enable_unsafe_deserialization()
    except Exception: pass
    classifier = tf.keras.models.load_model(CLF_MODEL_PATH, compile=False)
    segmentor = tf.keras.models.load_model(SEG_MODEL_PATH, custom_objects=CUSTOM_OBJECTS, compile=False)
    return classifier, segmentor

# ==============================
# 🔹 Image preprocessing
# ==============================
def classify_preprop(image_bytes):
    image = Image.open(BytesIO(image_bytes)).convert("RGB").resize((224, 224))
    return preprocess_input(np.expand_dims(img_to_array(image), axis=0))

def segment_preprop(image_bytes):
    image = Image.open(BytesIO(image_bytes)).convert("RGB").resize((256, 256))
    return np.expand_dims(np.array(image) / 255.0, axis=0)

def segment_postprop(image, mask, alpha=0.5):
    original_img = np.squeeze(image[0])
    mask_indices = np.argmax(mask, axis=-1)
    color_map = np.zeros_like(original_img, dtype=np.float32)
    color_map[mask_indices == 1] = [0, 1, 0]
    color_map[mask_indices == 2] = [1, 0, 0]
    seg_image = original_img.copy()
    seg_image[mask_indices > 0] = (
        original_img[mask_indices > 0] * (1 - alpha) + color_map[mask_indices > 0] * alpha
    )
    return seg_image

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

# --- Session state for language ---
if "lang" not in st.session_state:
    st.session_state.lang = "vi"

# --- Language toggle (top-right corner) ---
top_cols = st.columns([8, 1])
with top_cols[1]:
    if st.session_state.lang == "vi":
        if st.button("🇬🇧 EN"):
            st.session_state.lang = "en"
            st.experimental_rerun()
    else:
        if st.button("🇻🇳 VN"):
            st.session_state.lang = "vi"
            st.experimental_rerun()

# --- Language dictionary ---
TEXT = {
    "vi": {
        "nav": "📘 Navigation",
        "pages": ["Ứng dụng chẩn đoán", "Thông tin chung", "Thống kê về dữ liệu huấn luyện"],
        "upload": "📤 Tải ảnh siêu âm (JPG hoặc PNG)",
        "wait": "👆 Vui lòng tải ảnh lên để bắt đầu chẩn đoán.",
        "analyzing": "⏳ Đang phân tích ảnh...",
        "app_title": "🩺 Ứng dụng chẩn đoán bệnh ung thư vú từ hình ảnh siêu âm",
    },
    "en": {
        "nav": "📘 Navigation",
        "pages": ["Diagnosis App", "About Members", "Dataset Overview"],
        "upload": "📤 Upload Ultrasound Image (JPG or PNG)",
        "wait": "👆 Please upload an image to start diagnosis.",
        "analyzing": "⏳ Analyzing image...",
        "app_title": "🩺 Breast Cancer Diagnosis from Ultrasound Images",
    }
}

lang = st.session_state.lang
st.sidebar.title(TEXT[lang]["nav"])
app_mode = st.sidebar.selectbox("Chọn trang" if lang == "vi" else "Select page", TEXT[lang]["pages"])

# ==============================
# 🔹 Pages
# ==============================
if app_mode == TEXT[lang]["pages"][1]:  # Thông tin chung
    st.title("👨‍🎓 Giới thiệu về thành viên")
    st.markdown("<h4>Lê Vũ Anh Tin - 11TH</h4>", unsafe_allow_html=True)
    try:
        st.image("Tin.jpg", width=500)
        st.markdown("<h4>Trường THPT Chuyên Nguyễn Du</h4>", unsafe_allow_html=True)
        st.image("school.jpg", width=500)
    except:
        st.info("🖼️ Ảnh giới thiệu chưa được tải lên.")

elif app_mode == TEXT[lang]["pages"][2]:  # Thống kê
    st.title("📊 Thống kê tổng quan về tập dữ liệu")
    st.caption("""
    Tập dữ liệu **Breast Ultrasound Images (BUI)** được kết hợp từ ba nguồn:
    - BUSI (Arya Shah, Kaggle): ~780 ảnh siêu âm vú với mặt nạ phân đoạn (benign, malignant, normal).
    - BUS-UCLM (Orvile, Kaggle): 683 ảnh siêu âm vú với mặt nạ phân đoạn (benign, malignant, normal).
    - Breast Lesions USG (Cancer Imaging Archive): 163 trường hợp với ảnh siêu âm vú (DICOM) và chú thích tổn thương.
    
    Tổng cộng **1578 ảnh siêu âm vú** có mặt nạ phân đoạn tương ứng.
    """)
    st.markdown("""
    ### 🔗 Nguồn dữ liệu và trích dẫn
    Dữ liệu được thu thập từ các nguồn công khai sau, với trích dẫn theo định dạng APA:
    
    | Nguồn | Số lượng | Mô tả | Link | Trích dẫn |
    |-------|----------|--------|------|-----------|
    | BUSI (Arya Shah, Kaggle) | ~780 ảnh | Ảnh siêu âm vú với mặt nạ phân đoạn (benign, malignant, normal) | [Link](https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset/data) | Shah, A. (2020). Breast Ultrasound Images Dataset [Dataset]. Kaggle. https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset/data |
    | BUS-UCLM (Orvile, Kaggle) | 683 ảnh | Ảnh siêu âm vú với mặt nạ phân đoạn (benign, malignant, normal) | [Link](https://www.kaggle.com/datasets/orvile/bus-uclm-breast-ultrasound-dataset) | Orvile. (2023). BUS-UCLM Breast Ultrasound Dataset [Dataset]. Kaggle. https://www.kaggle.com/datasets/orvile/bus-uclm-breast-ultrasound-dataset |
    | Breast Lesions USG (Cancer Imaging Archive) | 163 trường hợp | Ảnh siêu âm vú (DICOM) với chú thích tổn thương | [Link](https://www.cancerimagingarchive.net/collection/breast-lesions-usg/) | The Cancer Imaging Archive (TCIA). (2021). Breast Lesions USG [Dataset]. Cancer Imaging Archive. https://www.cancerimagingarchive.net/collection/breast-lesions-usg/ |
    
    **Tổng số ảnh:** 1578 ảnh siêu âm vú với mặt nạ phân đoạn.
    """)

else:  # Ứng dụng chẩn đoán
    st.title(TEXT[lang]["app_title"])
    classifier, segmentor = load_models()
    file = st.file_uploader(TEXT[lang]["upload"], type=["jpg", "png"])

    if file is None:
        st.info(TEXT[lang]["wait"])
    else:
        slot = st.empty()
        slot.text(TEXT[lang]["analyzing"])

        pred_class, seg_image, img_bytes = predict_pipeline(file, classifier, segmentor)
        input_image = Image.open(BytesIO(img_bytes))

        col1, col2 = st.columns(2)
        col1.image(input_image, caption="Ảnh gốc", use_container_width=True)
        col2.image(seg_image, caption="Kết quả phân đoạn (Đỏ: Ác tính, Xanh: Lành tính)", use_container_width=True)

        result_index = np.argmax(pred_class)
        result = ["benign", "malignant", "normal"][result_index]

        st.markdown("---")
        st.subheader("💡 Kết quả phân loại")

        if result == "benign":
            st.success("🟢 Kết luận: Khối u lành tính.")
        elif result == "malignant":
            st.error("🔴 Kết luận: Ung thư vú ác tính.")
        else:
            st.info("⚪ Kết luận: Không phát hiện khối u (Bình thường).")

        st.markdown("---")
        st.subheader("📈 Chi tiết xác suất")

        chart_df = pd.DataFrame({
            "Loại chẩn đoán": ["Lành tính", "Ác tính", "Bình thường"],
            "Xác suất (%)": [pred_class[0,0]*100, pred_class[0,1]*100, pred_class[0,2]*100]
        })
        chart = alt.Chart(chart_df).mark_bar().encode(
            x=alt.X("Loại chẩn đoán", sort=None),
            y=alt.Y("Xác suất (%)", scale=alt.Scale(domain=[0, 100])),
            color=alt.Color("Loại chẩn đoán", scale=alt.Scale(
                domain=["Lành tính", "Ác tính", "Bình thường"],
                range=["#10B981", "#EF4444", "#9CA3AF"]
            )),
            tooltip=["Loại chẩn đoán", alt.Tooltip("Xác suất (%)", format=".15f")]
        ).properties(title="Biểu đồ Xác suất Chẩn đoán")
        st.altair_chart(chart, use_container_width=True)
        slot.success("✅ Hoàn tất chẩn đoán!")
