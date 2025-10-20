# ==========================================
# 🩺 Breast Ultrasound AI Diagnostic App (with Language Toggle)
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
    color_map[mask_indices == 1] = [0, 1, 0]  # Green
    color_map[mask_indices == 2] = [1, 0, 0]  # Red
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
toggle_col = st.columns([8, 1])
with toggle_col[1]:
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
        "intro_title": "👨‍🎓 Giới thiệu về thành viên",
        "data_title": "📊 Thống kê tổng quan về tập dữ liệu",
        "app_title": "🩺 Ứng dụng chẩn đoán bệnh ung thư vú từ hình ảnh siêu âm",
        "upload": "📤 Tải ảnh siêu âm (JPG hoặc PNG)",
        "waiting": "👆 Vui lòng tải ảnh lên để bắt đầu chẩn đoán.",
        "analyzing": "⏳ Đang phân tích ảnh...",
        "benign": "🟢 Kết luận: Khối u lành tính.",
        "malignant": "🔴 Kết luận: Ung thư vú ác tính.",
        "normal": "⚪ Kết luận: Không phát hiện khối u (Bình thường).",
        "prob_title": "📈 Chi tiết xác suất",
        "chart_title": "Biểu đồ Xác suất Chẩn đoán",
    },
    "en": {
        "nav": "📘 Navigation",
        "pages": ["Diagnosis App", "About Members", "Dataset Overview"],
        "intro_title": "👨‍🎓 About Members",
        "data_title": "📊 Dataset Overview",
        "app_title": "🩺 Breast Cancer Diagnosis from Ultrasound Images",
        "upload": "📤 Upload Ultrasound Image (JPG or PNG)",
        "waiting": "👆 Please upload an image to start diagnosis.",
        "analyzing": "⏳ Analyzing image...",
        "benign": "🟢 Result: Benign tumor.",
        "malignant": "🔴 Result: Malignant breast cancer.",
        "normal": "⚪ Result: No tumor detected (Normal).",
        "prob_title": "📈 Probability Details",
        "chart_title": "Diagnosis Probability Chart",
    }
}

lang = st.session_state.lang
st.sidebar.title(TEXT[lang]["nav"])
app_mode = st.sidebar.selectbox("Chọn trang" if lang == "vi" else "Select page", TEXT[lang]["pages"])

# ==============================
# 🔹 Page Routing
# ==============================
if app_mode == TEXT[lang]["pages"][1]:
    st.title(TEXT[lang]["intro_title"])
    st.image("Tin.jpg", width=500)
    st.markdown("<h4>Trường THPT Chuyên Nguyễn Du</h4>", unsafe_allow_html=True)
    st.image("school.jpg", width=500)

elif app_mode == TEXT[lang]["pages"][2]:
    st.title(TEXT[lang]["data_title"])
    st.caption("Hiển thị thống kê và nguồn dữ liệu huấn luyện (bản tiếng Việt được giữ nguyên).")

else:
    st.title(TEXT[lang]["app_title"])
    classifier, segmentor = load_models()
    file = st.file_uploader(TEXT[lang]["upload"], type=["jpg", "png"])

    if file is None:
        st.info(TEXT[lang]["waiting"])
    else:
        slot = st.empty()
        slot.text(TEXT[lang]["analyzing"])
        pred_class, seg_image, img_bytes = predict_pipeline(file, classifier, segmentor)
        input_image = Image.open(BytesIO(img_bytes))
        col1, col2 = st.columns(2)
        col1.image(input_image, caption="Ảnh gốc" if lang == "vi" else "Original Image", use_container_width=True)
        col2.image(seg_image, caption="Kết quả phân đoạn (Đỏ: Ác tính, Xanh: Lành tính)" if lang == "vi" else "Segmentation Result (Red: Malignant, Green: Benign)", use_container_width=True)

        result_index = np.argmax(pred_class)
        result = ["benign", "malignant", "normal"][result_index]
        st.markdown("---")
        st.subheader("💡 Kết quả phân loại" if lang == "vi" else "💡 Classification Result")

        st.success(TEXT[lang]["benign"]) if result == "benign" else (
            st.error(TEXT[lang]["malignant"]) if result == "malignant" else st.info(TEXT[lang]["normal"])
        )

        st.markdown("---")
        st.subheader(TEXT[lang]["prob_title"])

        chart_df = pd.DataFrame({
            "Loại chẩn đoán" if lang == "vi" else "Diagnosis Type":
                ["Lành tính", "Ác tính", "Bình thường"] if lang == "vi" else ["Benign", "Malignant", "Normal"],
            "Xác suất (%)" if lang == "vi" else "Probability (%)":
                [pred_class[0,0]*100, pred_class[0,1]*100, pred_class[0,2]*100]
        })
        chart = alt.Chart(chart_df).mark_bar().encode(
            x=alt.X(chart_df.columns[0], sort=None),
            y=alt.Y(chart_df.columns[1], scale=alt.Scale(domain=[0, 100])),
            color=alt.Color(chart_df.columns[0], scale=alt.Scale(
                domain=["Lành tính", "Ác tính", "Bình thường"] if lang == "vi" else ["Benign", "Malignant", "Normal"],
                range=["#10B981", "#EF4444", "#9CA3AF"]
            )),
            tooltip=list(chart_df.columns)
        ).properties(title=TEXT[lang]["chart_title"])
        st.altair_chart(chart, use_container_width=True)
        slot.success("✅ Hoàn tất chẩn đoán!" if lang == "vi" else "✅ Diagnosis complete!")
