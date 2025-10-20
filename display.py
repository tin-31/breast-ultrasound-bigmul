# ==========================================
# 🩺 Breast Ultrasound AI Diagnostic App
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
# 🔹 Page config
# ==============================
st.set_page_config(page_title="Breast Ultrasound AI", layout="wide", page_icon="🩺")

# ==============================
# 🌐 Language Switch (Animated Style)
# ==============================
if "lang" not in st.session_state:
    st.session_state.lang = "vi"

# CSS cho nút chuyển kiểu iOS
st.markdown("""
<style>
.lang-switch {
    position: fixed;
    top: 10px;
    right: 130px;
    width: 78px;
    height: 34px;
    background: linear-gradient(145deg, #2d2d2d, #1a1a1a);
    border: 1px solid #444;
    border-radius: 50px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0 8px;
    box-shadow: inset 0 2px 5px rgba(0,0,0,0.4);
    cursor: pointer;
    transition: all 0.3s ease;
    z-index: 1000;
}
.lang-switch:hover {
    transform: scale(1.05);
    border-color: #777;
}
.flag {
    font-size: 16px;
    user-select: none;
}
.slider {
    position: absolute;
    top: 2px;
    left: 3px;
    width: 30px;
    height: 30px;
    background: linear-gradient(90deg, #3B82F6, #10B981);
    border-radius: 50%;
    transition: all 0.3s ease;
    box-shadow: 0 2px 6px rgba(0,0,0,0.4);
}
.lang-switch.en .slider {
    left: 44px;
}
</style>
""", unsafe_allow_html=True)

lang_class = "lang-switch en" if st.session_state.lang == "en" else "lang-switch"
lang_html = f"""
<div id="lang-toggle" class="{lang_class}">
    <span class="flag">🇻🇳</span>
    <div class="slider"></div>
    <span class="flag">🇺🇸</span>
</div>
"""
st.markdown(lang_html, unsafe_allow_html=True)

# JavaScript để xử lý click
st.components.v1.html("""
<script>
const langDiv = window.parent.document.querySelector('#lang-toggle');
if (langDiv) {
    langDiv.addEventListener('click', function() {
        window.parent.postMessage({type: 'langToggle'}, '*');
    });
}
</script>
""", height=0)

# Bắt sự kiện toggle
if "lang_toggle" not in st.session_state:
    st.session_state.lang_toggle = False

from streamlit.runtime.scriptrunner import add_script_run_ctx
import threading

def toggle_lang():
    st.session_state.lang = "en" if st.session_state.lang == "vi" else "vi"
    st.rerun()

# ==============================
# 🔹 Model Config
# ==============================
SEG_MODEL_ID = "1axOg7N5ssJrMec97eV-JMPzID26ynzN1"
CLF_MODEL_ID = "1fXPICuTkETep2oPiA56l0uMai2GusEJH"

SEG_MODEL_PATH = "seg_model.keras"
CLF_MODEL_PATH = "clf_model.h5"

# ==============================
# 🔹 Custom Lambda
# ==============================
def spatial_mean(t): return tf.reduce_mean(t, axis=-1, keepdims=True)
def spatial_max(t): return tf.reduce_max(t, axis=-1, keepdims=True)
def spatial_output_shape(s): return (s[0], s[1], s[2], 1)

# ==============================
# 🔹 Auto Download Models
# ==============================
def download_model(model_id, output_path, model_name):
    if not os.path.exists(output_path):
        st.info(f"📥 Đang tải {model_name} (ID: {model_id})...")
        gdown.download(f"https://drive.google.com/uc?id={model_id}", output_path, quiet=False)
        st.success(f"✅ {model_name} đã được tải xong!")

download_model(SEG_MODEL_ID, SEG_MODEL_PATH, "model phân đoạn")
download_model(CLF_MODEL_ID, CLF_MODEL_PATH, "model phân loại")

# ==============================
# 🔹 Load Models
# ==============================
@st.cache_resource
def load_models():
    CUSTOM_OBJECTS = {
        "spatial_mean": spatial_mean,
        "spatial_max": spatial_max,
        "spatial_output_shape": spatial_output_shape
    }
    from tensorflow import keras
    try: keras.config.enable_unsafe_deserialization()
    except Exception: pass
    classifier = tf.keras.models.load_model(CLF_MODEL_PATH, compile=False)
    segmentor = tf.keras.models.load_model(SEG_MODEL_PATH, custom_objects=CUSTOM_OBJECTS, compile=False)
    return classifier, segmentor

# ==============================
# 🔹 Pre/Post Process
# ==============================
def classify_preprop(image_bytes):
    image = Image.open(BytesIO(image_bytes)).convert("RGB").resize((224,224))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    return preprocess_input(image)

def segment_preprop(image_bytes):
    image = Image.open(BytesIO(image_bytes)).convert("RGB").resize((256,256))
    image = np.array(image) / 255.0
    return np.expand_dims(image, axis=0)

def segment_postprop(image, mask, alpha=0.5):
    original_img = np.squeeze(image[0])
    mask_indices = np.argmax(mask, axis=-1)
    COLOR_BENIGN = np.array([0.0, 1.0, 0.0])
    COLOR_MALIGNANT = np.array([1.0, 0.0, 0.0])
    color_map = np.zeros_like(original_img, dtype=np.float32)
    color_map[mask_indices == 1] = COLOR_BENIGN
    color_map[mask_indices == 2] = COLOR_MALIGNANT
    segmented_image = original_img.copy()
    seg_locations = mask_indices > 0
    segmented_image[seg_locations] = (
        original_img[seg_locations]*(1-alpha) + color_map[seg_locations]*alpha
    )
    return segmented_image

# ==============================
# 🔹 Prediction
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
# 🔹 UI
# ==============================
st.sidebar.title("📘 Navigation")
app_mode = st.sidebar.selectbox(
    "Chọn trang",
    ["Ứng dụng chẩn đoán", "Thông tin chung", "Thống kê về dữ liệu huấn luyện"]
)

if app_mode == "Thông tin chung":
    st.title("👨‍🎓 Giới thiệu về thành viên")
    st.markdown("<h4>Lê Vũ Anh Tin - 11TH</h4>", unsafe_allow_html=True)
    try:
        st.image("Tin.jpg", width=500)
        st.markdown("<h4>Trường THPT Chuyên Nguyễn Du</h4>", unsafe_allow_html=True)
        st.image("school.jpg", width=500)
    except:
        st.info("🖼️ Ảnh giới thiệu chưa được tải lên.")

elif app_mode == "Thống kê về dữ liệu huấn luyện":
    st.title("📊 Thống kê tổng quan về tập dữ liệu")
    st.caption("Tập dữ liệu Breast Ultrasound Images (BUI) ...")

elif app_mode == "Ứng dụng chẩn đoán":
    st.title("🩺 Ứng dụng chẩn đoán bệnh ung thư vú từ hình ảnh siêu âm")
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
            st.image(seg_image, caption="Kết quả phân đoạn (Đỏ: Ác tính, Xanh: Lành tính)", use_container_width=True)
        result_idx = np.argmax(pred_class)
        classes = ["benign","malignant","normal"]
        result = classes[result_idx]
        st.markdown("---")
        st.subheader("💡 Kết quả phân loại")
        if result=="benign":
            st.success("🟢 Kết luận: Khối u lành tính.")
        elif result=="malignant":
            st.error("🔴 Kết luận: Ung thư vú ác tính.")
        else:
            st.info("⚪ Kết luận: Không phát hiện khối u.")
        st.markdown("---")
        st.subheader("📈 Chi tiết xác suất")
        df = pd.DataFrame({
            "Loại chẩn đoán":["Lành tính","Ác tính","Bình thường"],
            "Xác suất (%)":[pred_class[0,0]*100,pred_class[0,1]*100,pred_class[0,2]*100]
        })
        chart = alt.Chart(df).mark_bar().encode(
            x="Loại chẩn đoán",
            y=alt.Y("Xác suất (%)", scale=alt.Scale(domain=[0,100])),
            color=alt.Color("Loại chẩn đoán", scale=alt.Scale(
                domain=["Lành tính","Ác tính","Bình thường"],
                range=["#10B981","#EF4444","#9CA3AF"]
            ))
        ).properties(title="Biểu đồ Xác suất Chẩn đoán")
        st.altair_chart(chart, use_container_width=True)
        slot.success("✅ Hoàn tất chẩn đoán!")
