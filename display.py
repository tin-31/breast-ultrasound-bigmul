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

# =========================
# 🔹 PAGE CONFIG
# =========================
st.set_page_config(page_title="Breast Ultrasound AI", layout="wide", page_icon="🩺")

# =========================
# 🌐 LANGUAGE SWITCH
# =========================
if "lang" not in st.session_state:
    st.session_state.lang = "vi"

# CSS + HTML cho nút chuyển ngôn ngữ kiểu iOS
st.markdown("""
<style>
#lang-toggle {
  position: fixed;
  top: 12px;
  right: 110px;
  width: 82px;
  height: 36px;
  background: linear-gradient(145deg, #222, #111);
  border-radius: 50px;
  border: 1px solid #444;
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0 8px;
  cursor: pointer;
  box-shadow: inset 0 2px 5px rgba(0,0,0,0.4);
  transition: all 0.3s ease;
  z-index: 9999;
}
#lang-toggle:hover {
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
#lang-toggle.en .slider {
  left: 48px;
}
</style>

<div id="lang-toggle" class="{cls}">
  <span class="flag">🇻🇳</span>
  <div class="slider"></div>
  <span class="flag">🇺🇸</span>
</div>

<script>
const root = window.parent.document;
const toggle = root.querySelector('#lang-toggle');
if (toggle) {
  toggle.addEventListener('click', () => {
    window.parent.postMessage({type: 'langToggle'}, '*');
  });
}
</script>
""".replace("{cls}", "en" if st.session_state.lang == "en" else ""), unsafe_allow_html=True)

# Bắt sự kiện toggle
from streamlit.runtime.scriptrunner import add_script_run_ctx
import threading
def toggle_lang():
    st.session_state.lang = "en" if st.session_state.lang == "vi" else "vi"
    st.rerun()

st.experimental_get_query_params()  # Kích hoạt kênh JS listener

# =========================
# 🔹 MODELS CONFIG
# =========================
SEG_MODEL_ID = "1axOg7N5ssJrMec97eV-JMPzID26ynzN1"
CLF_MODEL_ID = "1fXPICuTkETep2oPiA56l0uMai2GusEJH"
SEG_MODEL_PATH = "seg_model.keras"
CLF_MODEL_PATH = "clf_model.h5"

def download_model(model_id, output_path, model_name):
    if not os.path.exists(output_path):
        st.info(f"📥 Đang tải {model_name} (ID: {model_id})...")
        gdown.download(f"https://drive.google.com/uc?id={model_id}", output_path, quiet=False)
        st.success(f"✅ {model_name} đã tải xong!")

download_model(SEG_MODEL_ID, SEG_MODEL_PATH, "Model phân đoạn")
download_model(CLF_MODEL_ID, CLF_MODEL_PATH, "Model phân loại")

# =========================
# 🔹 LOAD MODELS
# =========================
@st.cache_resource
def load_models():
    CUSTOM_OBJECTS = {
        "spatial_mean": lambda t: tf.reduce_mean(t, axis=-1, keepdims=True),
        "spatial_max": lambda t: tf.reduce_max(t, axis=-1, keepdims=True),
        "spatial_output_shape": lambda s: (s[0], s[1], s[2], 1)
    }
    from tensorflow import keras
    try: keras.config.enable_unsafe_deserialization()
    except: pass
    clf = tf.keras.models.load_model(CLF_MODEL_PATH, compile=False)
    seg = tf.keras.models.load_model(SEG_MODEL_PATH, custom_objects=CUSTOM_OBJECTS, compile=False)
    return clf, seg

# =========================
# 🔹 FUNCTIONS
# =========================
def classify_preprop(image_bytes):
    img = Image.open(BytesIO(image_bytes)).convert("RGB").resize((224,224))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    return preprocess_input(img)

def segment_preprop(image_bytes):
    img = Image.open(BytesIO(image_bytes)).convert("RGB").resize((256,256))
    img = np.array(img)/255.0
    return np.expand_dims(img, axis=0)

def segment_postprop(image, mask, alpha=0.5):
    original = np.squeeze(image[0])
    mask_indices = np.argmax(mask, axis=-1)
    COLOR_BENIGN = np.array([0.0, 1.0, 0.0])
    COLOR_MALIGNANT = np.array([1.0, 0.0, 0.0])
    color_map = np.zeros_like(original, dtype=np.float32)
    color_map[mask_indices == 1] = COLOR_BENIGN
    color_map[mask_indices == 2] = COLOR_MALIGNANT
    blended = original.copy()
    blended[mask_indices>0] = (
        original[mask_indices>0]*(1-alpha) + color_map[mask_indices>0]*alpha
    )
    return blended

def predict_pipeline(file, clf, seg):
    bytes_img = file.read()
    img_clf = classify_preprop(bytes_img)
    img_seg = segment_preprop(bytes_img)
    with tf.device("/CPU:0"):
        pred_class = clf.predict(img_clf, verbose=0)
        pred_mask = seg.predict(img_seg, verbose=0)[0]
    seg_image = segment_postprop(img_seg, pred_mask)
    return pred_class, seg_image, bytes_img

# =========================
# 🔹 UI
# =========================
st.sidebar.title("📘 Navigation")
mode = st.sidebar.selectbox("Chọn trang", ["Ứng dụng chẩn đoán", "Thông tin chung", "Thống kê về dữ liệu huấn luyện"])

if mode == "Ứng dụng chẩn đoán":
    st.title("🩺 Ứng dụng chẩn đoán bệnh ung thư vú từ hình ảnh siêu âm")
    clf, seg = load_models()
    file = st.file_uploader("📤 Tải ảnh siêu âm (JPG hoặc PNG)", type=["jpg","png"])
    if file is None:
        st.info("👆 Vui lòng tải ảnh lên để bắt đầu chẩn đoán.")
    else:
        slot = st.empty()
        slot.text("⏳ Đang phân tích ảnh...")
        pred_class, seg_image, img_bytes = predict_pipeline(file, clf, seg)
        input_img = Image.open(BytesIO(img_bytes))
        col1, col2 = st.columns(2)
        with col1: st.image(input_img, caption="Ảnh gốc", use_container_width=True)
        with col2: st.image(seg_image, caption="Ảnh phân đoạn (Đỏ: Ác tính, Xanh: Lành tính)", use_container_width=True)
        result_idx = np.argmax(pred_class)
        classes = ["benign","malignant","normal"]
        result = classes[result_idx]
        st.markdown("---")
        st.subheader("💡 Kết quả phân loại")
        if result == "benign": st.success("🟢 Kết luận: Khối u lành tính.")
        elif result == "malignant": st.error("🔴 Kết luận: Ung thư vú ác tính.")
        else: st.info("⚪ Kết luận: Không phát hiện khối u.")
        st.markdown("---")
        st.subheader("📈 Xác suất chi tiết")
        df = pd.DataFrame({
            "Loại": ["Lành tính","Ác tính","Bình thường"],
            "Xác suất (%)": [pred_class[0,0]*100, pred_class[0,1]*100, pred_class[0,2]*100]
        })
        chart = alt.Chart(df).mark_bar().encode(
            x="Loại", y=alt.Y("Xác suất (%)", scale=alt.Scale(domain=[0,100])),
            color=alt.Color("Loại", scale=alt.Scale(
                domain=["Lành tính","Ác tính","Bình thường"],
                range=["#10B981","#EF4444","#9CA3AF"]
            ))
        ).properties(title="Biểu đồ xác suất chẩn đoán")
        st.altair_chart(chart, use_container_width=True)
        slot.success("✅ Hoàn tất chẩn đoán!")

elif mode == "Thông tin chung":
    st.title("👨‍🎓 Giới thiệu thành viên")
    st.markdown("<h4>Lê Vũ Anh Tin - 11TH</h4>", unsafe_allow_html=True)
    st.image("Tin.jpg", width=500)
    st.markdown("<h4>Trường THPT Chuyên Nguyễn Du</h4>", unsafe_allow_html=True)
    st.image("school.jpg", width=500)

elif mode == "Thống kê về dữ liệu huấn luyện":
    st.title("📊 Thống kê tổng quan về tập dữ liệu huấn luyện")
    data = pd.DataFrame({
        "Loại ảnh": ["Benign","Malignant","Normal"],
        "Số lượng": [437, 210, 133]
    })
    st.dataframe(data)
    st.bar_chart(data.set_index("Loại ảnh"))
    st.markdown("""
    **Nguồn dữ liệu:** Breast Ultrasound Images Dataset (BUI).  
    - Tổng số ảnh: **780**  
    - Kích thước trung bình: 500×500 px  
    - Dữ liệu đã được cân bằng và tiền xử lý trước khi huấn luyện mô hình.
    """)
