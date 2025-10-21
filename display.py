# ==========================================
# 🩺 ỨNG DỤNG TRÍ TUỆ NHÂN TẠO HỖ TRỢ PHÂN TÍCH ẢNH SIÊU ÂM VÚ
# ==========================================
# ⚠️ Phiên bản dành cho nghiên cứu học thuật - Không sử dụng cho mục đích y tế thực tế.
# ⚠️ Ứng dụng này chỉ mang tính minh họa kỹ thuật và học thuật.

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
# ⚙️ Cấu hình mô hình
# ==============================
SEG_MODEL_ID = "1axOg7N5ssJrMec97eV-JMPzID26ynzN1"
CLF_MODEL_ID = "1fXPICuTkETep2oPiA56l0uMai2GusEJH"
SEG_MODEL_PATH = "seg_model.keras"
CLF_MODEL_PATH = "clf_model.h5"

# ==============================
# 🔹 Hàm xử lý trung gian cho CBAM
# ==============================
def spatial_mean(t):
    return tf.reduce_mean(t, axis=-1, keepdims=True)

def spatial_max(t):
    return tf.reduce_max(t, axis=-1, keepdims=True)

def spatial_output_shape(s):
    return (s[0], s[1], s[2], 1)

# ==============================
# 🔹 Tự động tải mô hình
# ==============================
def download_model(model_id, output_path, model_name):
    if not os.path.exists(output_path):
        st.info(f"📥 Đang tải {model_name} (ID: {model_id})...")
        gdown.download(f"https://drive.google.com/uc?id={model_id}", output_path, quiet=False)
        st.success(f"✅ {model_name} đã được tải xong!")

download_model(SEG_MODEL_ID, SEG_MODEL_PATH, "mô hình phân đoạn")
download_model(CLF_MODEL_ID, CLF_MODEL_PATH, "mô hình phân loại")

# ==============================
# 🔹 Tải mô hình an toàn
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
    segmentor = tf.keras.models.load_model(SEG_MODEL_PATH, custom_objects=CUSTOM_OBJECTS, compile=False)
    return classifier, segmentor

# ==============================
# 🔹 Tiền xử lý ảnh
# ==============================
def classify_preprop(image_bytes):
    image = Image.open(BytesIO(image_bytes)).convert("RGB").resize((224, 224))
    image = preprocess_input(np.expand_dims(img_to_array(image), axis=0))
    return image

def segment_preprop(image_bytes):
    image = Image.open(BytesIO(image_bytes)).convert("RGB").resize((256, 256))
    image = np.expand_dims(np.array(image) / 255.0, axis=0)
    return image

# ==============================
# 🔹 Hậu xử lý ảnh phân đoạn
# ==============================
def segment_postprop(image, mask, alpha=0.5):
    goc = np.squeeze(image[0])
    chi_so = np.argmax(mask, axis=-1)

    MAU_LANH = np.array([0.0, 1.0, 0.0])    # Xanh lá
    MAU_AC = np.array([1.0, 0.0, 0.0])      # Đỏ

    mau = np.zeros_like(goc, dtype=np.float32)
    mau[chi_so == 1] = MAU_LANH
    mau[chi_so == 2] = MAU_AC

    kq = goc.copy()
    vi_tri = chi_so > 0
    kq[vi_tri] = goc[vi_tri] * (1 - alpha) + mau[vi_tri] * alpha
    return kq

# ==============================
# 🔹 Pipeline dự đoán
# ==============================
def du_doan(file, classifier, segmentor):
    image_bytes = file.read()
    img_clf = classify_preprop(image_bytes)
    img_seg = segment_preprop(image_bytes)

    with tf.device("/CPU:0"):
        pred_class = classifier.predict(img_clf, verbose=0)
        pred_mask = segmentor.predict(img_seg, verbose=0)[0]

    seg_image = segment_postprop(img_seg, pred_mask)
    return pred_class, seg_image, image_bytes

# ==============================
# 🔹 Giao diện Streamlit (Chỉ tiếng Việt)
# ==============================
st.set_page_config(page_title="AI Phân tích Siêu âm Vú", layout="wide", page_icon="🩺")
st.sidebar.title("📘 Danh mục")

chon_trang = st.sidebar.selectbox(
    "Chọn nội dung hiển thị",
    ["Giới thiệu", "Ứng dụng minh họa", "Nguồn dữ liệu & Bản quyền"]
)

# -----------------------------
# Trang Giới thiệu
# -----------------------------
if chon_trang == "Giới thiệu":
    st.title("👩‍🔬 ỨNG DỤNG AI TRONG PHÂN TÍCH SIÊU ÂM VÚ")
    st.markdown("""
    Dự án này được thực hiện với mục đích **nghiên cứu học thuật** trong lĩnh vực Trí tuệ nhân tạo và Y học hình ảnh.

    ⚠️ **Lưu ý quan trọng:**
    - Đây **không phải** là công cụ chẩn đoán y tế thật.
    - Ứng dụng chỉ dùng để **minh họa kỹ thuật xử lý ảnh và học sâu (Deep Learning)**.
    - Không nên sử dụng kết quả này để thay thế tư vấn hoặc chẩn đoán y tế từ bác sĩ.
    """)

# -----------------------------
# Trang minh họa chẩn đoán
# -----------------------------
elif chon_trang == "Ứng dụng minh họa":
    st.title("🩺 Minh họa mô hình AI trên ảnh siêu âm vú")

    classifier, segmentor = load_models()
    file = st.file_uploader("📤 Chọn ảnh siêu âm (JPG hoặc PNG)", type=["jpg", "png"])

    if file is None:
        st.info("👆 Hãy chọn một ảnh để mô hình tiến hành minh họa.")
    else:
        slot = st.empty()
        slot.text("⏳ Đang xử lý ảnh...")

        pred_class, seg_image, img_bytes = du_doan(file, classifier, segmentor)
        anh_goc = Image.open(BytesIO(img_bytes))

        cot1, cot2 = st.columns(2)
        with cot1:
            st.image(anh_goc, caption="Ảnh gốc", use_container_width=True)
        with cot2:
            st.image(seg_image, caption="Kết quả phân đoạn (Đỏ: Ác tính, Xanh: Lành tính)", use_container_width=True)

        ten_nhom = ["Lành tính", "Ác tính", "Bình thường"]
        idx = np.argmax(pred_class)
        ket_qua = ten_nhom[idx]

        st.markdown("---")
        st.subheader("💡 Kết quả minh họa")

        if ket_qua == "Lành tính":
            st.success("🟢 Mô hình dự đoán: Khối u lành tính (chỉ mang tính minh họa).")
        elif ket_qua == "Ác tính":
            st.error("🔴 Mô hình dự đoán: Khối u ác tính (chỉ mang tính minh họa).")
        else:
            st.info("⚪ Mô hình dự đoán: Không phát hiện bất thường (chỉ mang tính minh họa).")

        st.caption("Kết quả chỉ mang tính nghiên cứu học thuật, không có giá trị chẩn đoán y tế.")

# -----------------------------
# Trang nguồn dữ liệu & bản quyền
# -----------------------------
elif chon_trang == "Nguồn dữ liệu & Bản quyền":
    st.title("📊 Nguồn dữ liệu và bản quyền sử dụng")
    st.markdown("""
    Ứng dụng sử dụng dữ liệu từ ba nguồn công khai, tuân thủ giấy phép phi thương mại (CC BY-NC-SA 4.0):

    | Nguồn | Giấy phép | Liên kết |
    |-------|------------|----------|
    | **BUSI (Arya Shah, Kaggle)** | CC BY 4.0 | [Link](https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset) |
    | **BUS-UCLM (Orvile, Kaggle)** | CC BY-NC-SA 4.0 | [Link](https://www.kaggle.com/datasets/orvile/bus-uclm-breast-ultrasound-dataset) |
    | **Breast Lesions USG (TCIA)** | CC BY 3.0 | [Link](https://www.cancerimagingarchive.net/collection/breast-lesions-usg/) |

    ---
    **Giấy phép sử dụng:**  
    - Phi thương mại (Non-Commercial).  
    - Phải trích dẫn nguồn dữ liệu gốc.  
    - Không sử dụng cho mục đích y tế hoặc thương mại.

    ---
    **Trích dẫn APA:**  
    - Shah, A. (2020). *Breast Ultrasound Images Dataset* [Dataset]. Kaggle.  
    - Orvile. (2023). *BUS-UCLM Breast Ultrasound Dataset* [Dataset]. Kaggle.  
    - The Cancer Imaging Archive. (2021). *Breast Lesions USG* [Dataset].
    """)

# -----------------------------
# Chân trang (footer)
# -----------------------------
st.markdown("""
---
📘 **Tuyên bố miễn trừ trách nhiệm:**  
Ứng dụng này được phát triển phục vụ mục đích **nghiên cứu khoa học và giáo dục**.  
Không sử dụng cho **chẩn đoán, điều trị hoặc tư vấn y tế**.  
© 2025 – Dự án AI Siêu âm Vú. Tác giả: Lê Vũ Anh Tin – Trường THPT Chuyên Nguyễn Du.
""")
