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
# SEG_MODEL_ID MỚI đã được cập nhật từ link bạn gửi
SEG_MODEL_ID = "1PI05-Z7K2TAN-v3Jh7ZPFqygKsQ4gCYV" # ✅ Model phân đoạn (FIXED)
CLF_MODEL_ID = "1fXPICuTkETep2oPiA56l0uMai2GusEJH" # ✅ Model phân loại

SEG_MODEL_PATH = "seg_model.keras"
CLF_MODEL_PATH = "clf_model.h5"

# ==============================
# 🔹 Custom Lambda Functions (Định nghĩa lại các hàm Lambda đã đặt tên trong CBAM)
# ==============================
def spatial_mean(t):
    """Channel Average Pooling cho Spatial Attention."""
    return tf.reduce_mean(t, axis=-1, keepdims=True)

def spatial_max(t):
    """Channel Max Pooling cho Spatial Attention."""
    return tf.reduce_max(t, axis=-1, keepdims=True)

def spatial_output_shape(s):
    """Output shape (batch, height, width, 1) cho Spatial Attention."""
    return (s[0], s[1], s[2], 1)

# ==============================
# 🔹 Auto download models
# ==============================
def download_model(model_id, output_path, model_name):
    """Tự động tải model từ Google Drive nếu chưa tồn tại"""
    if not os.path.exists(output_path):
        st.info(f"📥 Đang tải {model_name} (ID: {model_id})...")
        gdown.download(f"https://drive.google.com/uc?id={model_id}", output_path, quiet=False)
        st.success(f"✅ {model_name} đã được tải xong!")

download_model(SEG_MODEL_ID, SEG_MODEL_PATH, "model phân đoạn")
download_model(CLF_MODEL_ID, CLF_MODEL_PATH, "model phân loại")

# ==============================
# 🔹 Load both models safely (FIXED: Sử dụng custom_objects)
# ==============================
@st.cache_resource
def load_models():
    # Khai báo các đối tượng tùy chỉnh (Lambda functions)
    CUSTOM_OBJECTS = {
        "spatial_mean": spatial_mean,
        "spatial_max": spatial_max,
        "spatial_output_shape": spatial_output_shape
    }

    from tensorflow import keras
    try:
        # Cố gắng bật chế độ bỏ qua kiểm tra an toàn (nếu cần)
        keras.config.enable_unsafe_deserialization()
    except Exception:
        pass

    # Tải model phân loại
    classifier = tf.keras.models.load_model(CLF_MODEL_PATH, compile=False)
    
    # Tải model phân đoạn với custom_objects để giải quyết lỗi Lambda Layer
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

# CHỈNH SỬA: Hiển thị màu sắc cho các vùng phân đoạn: Đỏ (Ác tính), Vàng (Lành tính)
def segment_postprop(image, mask, alpha=0.5):
    """
    Tạo lớp phủ màu sắc lên ảnh gốc dựa trên kết quả phân đoạn.
    - Class 0 (Background/Normal): Giữ màu ảnh gốc
    - Class 1 (Benign/Lành tính): Màu Vàng (Yellow)
    - Class 2 (Malignant/Ác tính): Màu Đỏ (Red)
    """
    # Lấy ảnh gốc (256, 256, 3) từ batch đầu vào, range [0, 1]
    original_img = np.squeeze(image[0]) 

    # Lấy chỉ số lớp dự đoán (0, 1, 2)
    mask_indices = np.argmax(mask, axis=-1)

    # Định nghĩa màu sắc (RGB, 0-1)
    COLOR_BENIGN = np.array([1.0, 1.0, 0.0])    # Vàng
    COLOR_MALIGNANT = np.array([1.0, 0.0, 0.0]) # Đỏ

    # Tạo bản đồ màu
    color_map = np.zeros_like(original_img, dtype=np.float32)
    
    # Áp dụng màu cho Benign (1) và Malignant (2)
    color_map[mask_indices == 1] = COLOR_BENIGN
    color_map[mask_indices == 2] = COLOR_MALIGNANT
    
    # Tạo ảnh đã phân đoạn (bắt đầu bằng ảnh gốc)
    segmented_image = original_img.copy()
    
    # Lấy vị trí các pixel được phân loại là Benign hoặc Malignant
    segment_locations = mask_indices > 0
    
    # Trộn màu (Blending) chỉ tại các vị trí khối u
    # Blended = (Original * (1 - alpha)) + (Color * alpha)
    segmented_image[segment_locations] = (
        original_img[segment_locations] * (1 - alpha) + 
        color_map[segment_locations] * alpha
    )
    
    return segmented_image

# ==============================
# 🔹 Prediction pipeline
# ==============================
def predict_pipeline(file, classifier, segmentor):
    image_bytes = file.read()
    img_clf = classify_preprop(image_bytes)
    img_seg = segment_preprop(image_bytes)

    # Sử dụng CPU để dự đoán
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

    # Tải mô hình đã fix
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
            st.image(seg_image, caption="Kết quả phân đoạn (Đỏ: Ác tính, Vàng: Lành tính)", use_container_width=True)

        class_names = ["benign", "malignant", "normal"]
        # Phân loại là lành tính: index 0, ác tính: index 1, bình thường: index 2
        result_index = np.argmax(pred_class)
        result = class_names[result_index]
        
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

        # Đảm bảo thứ tự class_names khớp với thứ tự đầu ra của mô hình (benign, malignant, normal)
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
            tooltip=["Loại chẩn đoán", alt.Tooltip("Xác suất (%)", format=".1f")]
        ).properties(
            title="Biểu đồ Xác suất Chẩn đoán"
        )
        st.altair_chart(chart, use_container_width=True)

        st.write(f"- **Khối u lành tính:** **{pred_class[0,0]*100:.2f}%**")
        st.write(f"- **Ung thư vú (Ác tính):** **{pred_class[0,1]*100:.2f}%**")
        st.write(f"- **Bình thường:** **{pred_class[0,2]*100:.2f}%**")

        slot.success("✅ Hoàn tất chẩn đoán!")
