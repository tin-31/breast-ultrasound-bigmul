# ==========================================
# 🩺 Breast Ultrasound AI Diagnostic App (Song ngữ Việt - Anh)
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
# 🔹 Custom Lambda Functions
# ==============================
def spatial_mean(t):
    return tf.reduce_mean(t, axis=-1, keepdims=True)

def spatial_max(t):
    return tf.reduce_max(t, axis=-1, keepdims=True)

def spatial_output_shape(s):
    return (s[0], s[1], s[2], 1)

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
# 🔹 Load both models safely
# ==============================
@st.cache_resource
def load_models():
    CUSTOM_OBJECTS = {
        "spatial_mean": spatial_mean,
        "spatial_max": spatial_max,
        "spatial_output_shape": spatial_output_shape
    }
    from tensorflow import keras
    try:
        keras.config.enable_unsafe_deserialization()
    except Exception:
        pass
    classifier = tf.keras.models.load_model(CLF_MODEL_PATH, compile=False)
    segmentor = tf.keras.models.load_model(
        SEG_MODEL_PATH, custom_objects=CUSTOM_OBJECTS, compile=False
    )
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

def segment_postprop(image, mask, alpha=0.5):
    original_img = np.squeeze(image[0])
    mask_indices = np.argmax(mask, axis=-1)
    COLOR_BENIGN = np.array([0.0, 1.0, 0.0])
    COLOR_MALIGNANT = np.array([1.0, 0.0, 0.0])
    color_map = np.zeros_like(original_img, dtype=np.float32)
    color_map[mask_indices == 1] = COLOR_BENIGN
    color_map[mask_indices == 2] = COLOR_MALIGNANT
    segmented_image = original_img.copy()
    segment_locations = mask_indices > 0
    segmented_image[segment_locations] = (
        original_img[segment_locations] * (1 - alpha)
        + color_map[segment_locations] * alpha
    )
    return segmented_image

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
# 🔹 Streamlit UI (with language toggle)
# ==============================
st.set_page_config(page_title="Breast Ultrasound AI", layout="wide", page_icon="🩺")

# 🌐 Language toggle button
if "lang" not in st.session_state:
    st.session_state.lang = "vi"

lang_button_css = """
<style>
    div[data-testid="stToolbar"] { right: 120px !important; }
    #lang-toggle { position: fixed; top: 10px; right: 70px; z-index: 1000; }
    div[data-testid="stToolbarActions"] button {
        transform: scale(2.0) !important; /* Double GitHub button size */
    }
</style>
"""
st.markdown(lang_button_css, unsafe_allow_html=True)

lang_label = "🌏 English" if st.session_state.lang == "vi" else "🇻🇳 Tiếng Việt"
if st.button(lang_label, key="lang-btn"):
    st.session_state.lang = "en" if st.session_state.lang == "vi" else "vi"
    st.rerun()

# Sidebar
if st.session_state.lang == "vi":
    st.sidebar.title("📘 Điều hướng")
    app_mode = st.sidebar.selectbox("Chọn trang", ["Ứng dụng chẩn đoán", "Thông tin chung", "Thống kê về dữ liệu huấn luyện"])
else:
    st.sidebar.title("📘 Navigation")
    app_mode = st.sidebar.selectbox("Select page", ["Diagnostic App", "About", "Training Data Statistics"])

# =============== About Page ===============
if (st.session_state.lang == "vi" and app_mode == "Thông tin chung") or (st.session_state.lang == "en" and app_mode == "About"):
    if st.session_state.lang == "vi":
        st.title("👨‍🎓 Giới thiệu về thành viên")
        st.markdown("<h4>Lê Vũ Anh Tin - 11TH</h4>", unsafe_allow_html=True)
        try:
            st.image("Tin.jpg", width=500)
            st.markdown("<h4>Trường THPT Chuyên Nguyễn Du</h4>", unsafe_allow_html=True)
            st.image("school.jpg", width=500)
        except:
            st.info("🖼️ Ảnh giới thiệu chưa được tải lên.")
    else:
        st.title("👨‍🎓 Team Member Introduction")
        st.markdown("<h4>Lê Vũ Anh Tin - Grade 11TH</h4>", unsafe_allow_html=True)
        try:
            st.image("Tin.jpg", width=500)
            st.markdown("<h4>Nguyen Du High School for the Gifted</h4>", unsafe_allow_html=True)
            st.image("school.jpg", width=500)
        except:
            st.info("🖼️ Introduction images not uploaded yet.")

# =============== Dataset Statistics ===============
elif (st.session_state.lang == "vi" and app_mode == "Thống kê về dữ liệu huấn luyện") or (st.session_state.lang == "en" and app_mode == "Training Data Statistics"):
    if st.session_state.lang == "vi":
        st.title("📊 Thống kê tổng quan về tập dữ liệu")
        st.markdown("Tập dữ liệu **Breast Ultrasound Images (BUI)** được kết hợp từ ba nguồn chính...")
    else:
        st.title("📊 Overview of the Training Dataset")
        st.markdown("The **Breast Ultrasound Images (BUI)** dataset combines data from three main sources...")

# =============== Diagnostic App ===============
elif (st.session_state.lang == "vi" and app_mode == "Ứng dụng chẩn đoán") or (st.session_state.lang == "en" and app_mode == "Diagnostic App"):
    if st.session_state.lang == "vi":
        st.title("🩺 Ứng dụng chẩn đoán bệnh ung thư vú từ hình ảnh siêu âm")
        file_label = "📤 Tải ảnh siêu âm (JPG hoặc PNG)"
        info_text = "👆 Vui lòng tải ảnh lên để bắt đầu chẩn đoán."
        seg_caption = "Kết quả phân đoạn (Đỏ: Ác tính, Xanh: Lành tính)"
        result_labels = ["Lành tính", "Ác tính", "Bình thường"]
    else:
        st.title("🩺 Breast Cancer Diagnostic App from Ultrasound Images")
        file_label = "📤 Upload ultrasound image (JPG or PNG)"
        info_text = "👆 Please upload an image to begin diagnosis."
        seg_caption = "Segmentation Result (Red: Malignant, Green: Benign)"
        result_labels = ["Benign", "Malignant", "Normal"]

    classifier, segmentor = load_models()
    file = st.file_uploader(file_label, type=["jpg", "png"])

    if file is None:
        st.info(info_text)
    else:
        slot = st.empty()
        slot.text("⏳ Analyzing image..." if st.session_state.lang == "en" else "⏳ Đang phân tích ảnh...")

        pred_class, seg_image, img_bytes = predict_pipeline(file, classifier, segmentor)
        input_image = Image.open(BytesIO(img_bytes))

        col1, col2 = st.columns(2)
        with col1:
            st.image(input_image, caption="Original Image" if st.session_state.lang == "en" else "Ảnh gốc", use_container_width=True)
        with col2:
            st.image(seg_image, caption=seg_caption, use_container_width=True)

        class_names = ["benign", "malignant", "normal"]
        result_index = np.argmax(pred_class)
        result = class_names[result_index]

        st.markdown("---")
        st.subheader("💡 Diagnostic Result" if st.session_state.lang == "en" else "💡 Kết quả chẩn đoán")

        if result == "benign":
            st.success("🟢 Benign tumor detected." if st.session_state.lang == "en" else "🟢 Kết luận: Khối u lành tính.")
        elif result == "malignant":
            st.error("🔴 Malignant breast cancer detected." if st.session_state.lang == "en" else "🔴 Kết luận: Ung thư vú ác tính.")
        else:
            st.info("⚪ No tumor detected (Normal)." if st.session_state.lang == "en" else "⚪ Kết luận: Không phát hiện khối u (Bình thường).")

        st.markdown("---")
        st.subheader("📈 Probability Details" if st.session_state.lang == "en" else "📈 Chi tiết xác suất")

        chart_df = pd.DataFrame({
            "Diagnosis Type" if st.session_state.lang == "en" else "Loại chẩn đoán": result_labels,
            "Probability (%)": [pred_class[0,0]*100, pred_class[0,1]*100, pred_class[0,2]*100]
        })

        chart = alt.Chart(chart_df).mark_bar().encode(
            x=alt.X("Diagnosis Type" if st.session_state.lang == "en" else "Loại chẩn đoán", sort=None),
            y=alt.Y("Probability (%)", scale=alt.Scale(domain=[0, 100])),
            color=alt.Color("Diagnosis Type" if st.session_state.lang == "en" else "Loại chẩn đoán",
                            scale=alt.Scale(
                                domain=result_labels,
                                range=["#10B981", "#EF4444", "#9CA3AF"]
                            )),
            tooltip=[alt.Tooltip("Probability (%)", format=".2f")]
        ).properties(
            title="Diagnosis Probability Chart" if st.session_state.lang == "en" else "Biểu đồ Xác suất Chẩn đoán"
        )
        st.altair_chart(chart, use_container_width=True)

        slot.success("✅ Diagnosis completed!" if st.session_state.lang == "en" else "✅ Hoàn tất chẩn đoán!")
