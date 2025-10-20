# ==========================================
# 🩺 Breast Ultrasound AI Diagnostic App (Fixed Model Loader)
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
SEG_MODEL_ID = "1axOg7N5ssJrMec97eV-JMPzID26ynzN1"  # ✅ Model phân đoạn (FIXED)
CLF_MODEL_ID = "1fXPICuTkETep2oPiA56l0uMai2GusEJH"  # ✅ Model phân loại

SEG_MODEL_PATH = "seg_model.keras"
CLF_MODEL_PATH = "clf_model.h5"

# ==============================
# 🔹 Custom Lambda Functions (CBAM)
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
        SEG_MODEL_PATH,
        custom_objects=CUSTOM_OBJECTS,
        compile=False
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

    COLOR_BENIGN = np.array([0.0, 1.0, 0.0])     # Xanh lá
    COLOR_MALIGNANT = np.array([1.0, 0.0, 0.0])  # Đỏ

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
# 🔹 Language configuration
# ==============================
TEXTS = {
    "page_title": {"vi": "Breast Ultrasound AI", "en": "Breast Ultrasound AI"},
    "sidebar_title": {"vi": "📘 Navigation", "en": "📘 Navigation"},
    "selectbox_label": {"vi": "Chọn trang", "en": "Select Page"},
    "option_diagnostic": {"vi": "Ứng dụng chẩn đoán", "en": "Diagnostic App"},
    "option_info": {"vi": "Thông tin chung", "en": "General Info"},
    "option_stats": {"vi": "Thống kê về dữ liệu huấn luyện", "en": "Training Data Statistics"},
    "language_button": {"vi": "🌐 Chuyển sang Tiếng Anh", "en": "🌐 Switch to Vietnamese"},
    "info_title": {"vi": "👨‍🎓 Giới thiệu về thành viên", "en": "👨‍🎓 Member Introduction"},
    "info_name": {"vi": "Lê Vũ Anh Tin - 11TH", "en": "Le Vu Anh Tin - 11TH"},
    "info_school": {"vi": "Trường THPT Chuyên Nguyễn Du", "en": "Nguyen Du High School for the Gifted"},
    "info_image_alt": {"vi": "🖼️ Ảnh giới thiệu chưa được tải lên.", "en": "🖼️ Introduction image not uploaded."},
    "stats_title": {"vi": "📊 Thống kê tổng quan về tập dữ liệu", "en": "📊 Overview Statistics of the Dataset"},
    "stats_caption": {
        "vi": """
        Tập dữ liệu **Breast Ultrasound Images (BUI)** được kết hợp từ ba nguồn:
        - BUSI (Arya Shah, Kaggle)
        - BUS-UCLM (Orvile, Kaggle)
        - Breast Lesions USG (Cancer Imaging Archive)
        Tổng cộng **1578 ảnh siêu âm vú** có mặt nạ phân đoạn tương ứng.
        """,
        "en": """
        The **Breast Ultrasound Images (BUI)** dataset combines three sources:
        - BUSI (Arya Shah, Kaggle)
        - BUS-UCLM (Orvile, Kaggle)
        - Breast Lesions USG (Cancer Imaging Archive)
        Total **1578 ultrasound images** with segmentation masks.
        """
    },
    "stats_total": {"vi": "**Tổng số ảnh:** 1578 ảnh siêu âm vú.", "en": "**Total images:** 1578 ultrasound images."},
    "diagnostic_title": {"vi": "🩺 Ứng dụng chẩn đoán ung thư vú", "en": "🩺 Breast Cancer Diagnostic App"},
    "uploader_label": {"vi": "📤 Tải ảnh siêu âm (JPG/PNG)", "en": "📤 Upload Ultrasound Image (JPG/PNG)"},
    "uploader_info": {"vi": "👆 Vui lòng tải ảnh lên để bắt đầu chẩn đoán.", "en": "👆 Please upload an image to start diagnosis."},
    "processing_text": {"vi": "⏳ Đang phân tích ảnh...", "en": "⏳ Analyzing image..."},
    "original_caption": {"vi": "Ảnh gốc", "en": "Original Image"},
    "segmented_caption": {"vi": "Kết quả phân đoạn (Đỏ: Ác tính, Xanh: Lành tính)", "en": "Segmentation Result (Red: Malignant, Green: Benign)"},
    "result_title": {"vi": "💡 Kết quả phân loại", "en": "💡 Classification Result"},
    "benign_result": {"vi": "🟢 Kết luận: Khối u lành tính.", "en": "🟢 Conclusion: Benign tumor."},
    "malignant_result": {"vi": "🔴 Kết luận: Ung thư vú ác tính.", "en": "🔴 Conclusion: Malignant breast cancer."},
    "normal_result": {"vi": "⚪ Kết luận: Không phát hiện khối u.", "en": "⚪ Conclusion: No tumor detected."},
    "prob_title": {"vi": "📈 Chi tiết xác suất", "en": "📈 Probability Details"},
    "chart_title": {"vi": "Biểu đồ Xác suất Chẩn đoán", "en": "Diagnosis Probability Chart"},
    "prob_benign": {"vi": "Xác suất khối u lành tính", "en": "Probability of benign tumor"},
    "prob_malignant": {"vi": "Xác suất ung thư vú ác tính", "en": "Probability of malignant tumor"},
    "prob_normal": {"vi": "Xác suất bình thường", "en": "Probability of normal"},
    "success_message": {"vi": "✅ Hoàn tất chẩn đoán!", "en": "✅ Diagnosis completed!"}
}

# ==============================
# 🔹 Streamlit UI
# ==============================
if "language" not in st.session_state:
    st.session_state.language = "vi"

def get_text(key):
    return TEXTS[key][st.session_state.language]

st.set_page_config(page_title=get_text("page_title"), layout="wide", page_icon="🩺")

if st.sidebar.button(get_text("language_button")):
    st.session_state.language = "en" if st.session_state.language == "vi" else "vi"
    st.rerun()

st.sidebar.title(get_text("sidebar_title"))
app_mode = st.sidebar.selectbox(
    get_text("selectbox_label"),
    [get_text("option_diagnostic"), get_text("option_info"), get_text("option_stats")]
)

# -----------------------------
# Info page
# -----------------------------
if app_mode == get_text("option_info"):
    st.title(get_text("info_title"))
    st.markdown(f"<h4>{get_text('info_name')}</h4>", unsafe_allow_html=True)
    try:
        st.image("Tin.jpg", width=500)
        st.markdown(f"<h4>{get_text('info_school')}</h4>", unsafe_allow_html=True)
        st.image("school.jpg", width=500)
    except:
        st.info(get_text("info_image_alt"))

# -----------------------------
# Stats page
# -----------------------------
elif app_mode == get_text("option_stats"):
    st.title(get_text("stats_title"))
    st.caption(get_text("stats_caption"))
    st.markdown(get_text("stats_total"))

# -----------------------------
# Diagnostic app
# -----------------------------
elif app_mode == get_text("option_diagnostic"):
    st.title(get_text("diagnostic_title"))
    classifier, segmentor = load_models()
    file = st.file_uploader(get_text("uploader_label"), type=["jpg", "png"])

    if file is None:
        st.info(get_text("uploader_info"))
    else:
        slot = st.empty()
        slot.text(get_text("processing_text"))

        pred_class, seg_image, img_bytes = predict_pipeline(file, classifier, segmentor)
        input_image = Image.open(BytesIO(img_bytes))

        col1, col2 = st.columns(2)
        with col1:
            st.image(input_image, caption=get_text("original_caption"), use_container_width=True)
        with col2:
            st.image(seg_image, caption=get_text("segmented_caption"), use_container_width=True)

        class_names = ["benign", "malignant", "normal"]
        result_index = np.argmax(pred_class)
        result = class_names[result_index]

        st.markdown("---")
        st.subheader(get_text("result_title"))

        if result == "benign":
            st.success(get_text("benign_result"))
        elif result == "malignant":
            st.error(get_text("malignant_result"))
        else:
            st.info(get_text("normal_result"))

        st.markdown("---")
        st.subheader(get_text("prob_title"))

        format_spec = ".15f"
        chart_df = pd.DataFrame({
            "Loại chẩn đoán": ["Lành tính", "Ác tính", "Bình thường"] if st.session_state.language == "vi" else ["Benign", "Malignant", "Normal"],
            "Xác suất (%)": [pred_class[0, 0] * 100, pred_class[0, 1] * 100, pred_class[0, 2] * 100]
        })
        chart = alt.Chart(chart_df).mark_bar().encode(
            x=alt.X("Loại chẩn đoán", sort=None),
            y=alt.Y("Xác suất (%)", scale=alt.Scale(domain=[0, 100])),
            color=alt.Color("Loại chẩn đoán", scale=alt.Scale(
                domain=["Lành tính", "Ác tính", "Bình thường"] if st.session_state.language == "vi" else ["Benign", "Malignant", "Normal"],
                range=["#10B981", "#EF4444", "#9CA3AF"]
            )),
            tooltip=["Loại chẩn đoán", alt.Tooltip("Xác suất (%)", format=format_spec)]
        ).properties(title=get_text("chart_title"))
        st.altair_chart(chart, use_container_width=True)

        st.markdown(f"""
        - {get_text("prob_benign")}: **{pred_class[0,0]*100:{format_spec}}%**
        - {get_text("prob_malignant")}: **{pred_class[0,1]*100:{format_spec}}%**
        - {get_text("prob_normal")}: **{pred_class[0,2]*100:{format_spec}}%**
        """)

        st.success(get_text("success_message"))
