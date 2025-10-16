import os
import gdown
import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import altair as alt
from PIL import Image
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from io import BytesIO

# ============================================================
# 🔹 1. CẤU HÌNH CÁC ID & TÊN FILE
# ============================================================

# 🧠 Model phân đoạn (.keras — phiên bản mới, không lỗi lambda)
SEG_MODEL_ID = "1YbX7lBQCjWXaSyCtwUXftjFHEaBjNnDa"

# 🧩 Model phân loại (.h5)
CLF_MODEL_ID = "1fXPICuTkETep2oPiA56l0uMai2GusEJH"

# Đường dẫn lưu file sau khi tải
SEG_MODEL_PATH = "best_model_cbam_attention_unet.keras"
CLF_MODEL_PATH = "Classifier_model.h5"

# ============================================================
# 🔹 2. TẢI FILE MODEL NẾU CHƯA CÓ
# ============================================================

if not os.path.exists(SEG_MODEL_PATH):
    st.info("📥 Đang tải model phân đoạn (.keras)...")
    gdown.download(f"https://drive.google.com/uc?id={SEG_MODEL_ID}", SEG_MODEL_PATH, quiet=False)
    st.success("✅ Model phân đoạn (.keras) đã tải xong!")

if not os.path.exists(CLF_MODEL_PATH):
    st.info("📥 Đang tải model phân loại (.h5)...")
    gdown.download(f"https://drive.google.com/uc?id={CLF_MODEL_ID}", CLF_MODEL_PATH, quiet=False)
    st.success("✅ Model phân loại đã tải xong!")

# ============================================================
# 🔹 3. LOAD MODELS
# ============================================================

@st.cache_resource(ttl=3600)
def load_models():
    try:
        # Load model phân loại
        clf = tf.keras.models.load_model(CLF_MODEL_PATH, compile=False)
        # Load model phân đoạn (.keras format — không cần custom_objects)
        seg = tf.keras.models.load_model(SEG_MODEL_PATH, compile=False)
        return clf, seg
    except Exception as e:
        st.error(f"❌ Lỗi khi load models: {e}")
        raise e

# ============================================================
# 🔹 4. XỬ LÝ ẢNH
# ============================================================

def classify_preprop(image_file):
    image = Image.open(BytesIO(image_file)).convert("RGB")
    image = image.resize((224, 224))
    arr = img_to_array(image)
    arr = np.expand_dims(arr, axis=0)
    arr = preprocess_input(arr)
    return arr

def segment_preprop(image_file):
    image = Image.open(BytesIO(image_file)).convert("RGB")
    image = image.resize((256, 256))
    arr = np.array(image, dtype=np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr

def segment_postprop(image, mask):
    image = np.squeeze(image)
    mask = np.argmax(mask, axis=-1)  # nếu mask là softmax đầu ra
    mask = np.expand_dims(mask, axis=-1)
    mask_rgb = np.repeat(mask, 3, axis=-1) * 255
    return mask_rgb.astype(np.uint8)

def preprocessing_uploader(file, classifier, segmentor):
    image_bytes = file.read()
    img_for_clf = classify_preprop(image_bytes)
    img_for_seg = segment_preprop(image_bytes)

    # Dự đoán
    clf_pred = classifier.predict(img_for_clf)
    seg_pred = segmentor.predict(img_for_seg)[0]
    seg_vis = segment_postprop(img_for_seg, seg_pred)

    return clf_pred, seg_vis

# ============================================================
# 🔹 5. GIAO DIỆN STREAMLIT
# ============================================================

st.sidebar.title("📘 Navigation")
app_mode = st.sidebar.selectbox(
    'Chọn trang',
    ['Ứng dụng chẩn đoán', 'Thông tin chung', 'Thống kê về dữ liệu huấn luyện']
)

if app_mode == 'Thông tin chung':
    st.title('👨‍🎓 Giới thiệu về thành viên')
    st.markdown('<h4>Lê Vũ Anh Tin - 11TH</h4>', unsafe_allow_html=True)
    try:
        st.image('Tin.jpg', caption='Lê Vũ Anh Tin')
        st.image('school.jpg', caption='Trường THPT Chuyên Nguyễn Du')
    except:
        st.warning("Không tìm thấy ảnh minh họa.")

elif app_mode == 'Thống kê về dữ liệu huấn luyện':
    st.title('📊 Thống kê tổng quan về tập dữ liệu')
    st.caption("""
    Tập dữ liệu **Breast Ultrasound Images (BUI)** được tổng hợp từ:
    - BUSI (Arya Shah, Kaggle)
    - BUS-UCLM (Orvile, Kaggle)
    Tổng cộng **1578 ảnh siêu âm vú** có mặt nạ phân đoạn tương ứng.
    """)
    st.caption('🔗 Nguồn dataset: [Google Drive](https://drive.google.com/drive/folders/1eSAA5pMuEz1GgATBmvXbjjaihO1yBo1l?usp=drive_link)')

elif app_mode == 'Ứng dụng chẩn đoán':
    st.title('🩺 Ứng dụng chẩn đoán bệnh ung thư vú từ ảnh siêu âm')

    classifier, segmentor = load_models()

    file = st.file_uploader("📤 Tải ảnh siêu âm vú (jpg/png)", type=["jpg", "png"])
    if not file:
        st.info('👆 Vui lòng tải ảnh siêu âm lên để bắt đầu chẩn đoán.')
    else:
        st.info("🔍 Đang phân tích ảnh...")
        clf_pred, seg_vis = preprocessing_uploader(file, classifier, segmentor)

        test_img = Image.open(file)
        st.image(test_img, caption="Ảnh đầu vào", width=350)
        st.image(seg_vis, caption="Ảnh phân đoạn khối u", width=350)

        class_names = ['Benign (Lành tính)', 'Malignant (Ác tính)', 'Normal (Bình thường)']
        result_idx = np.argmax(clf_pred)
        result_label = class_names[result_idx]

        if result_idx == 0:
            st.success('🟢 Kết quả: Khối u lành tính.')
        elif result_idx == 1:
            st.error('🔴 Kết quả: Ung thư vú (ác tính).')
        else:
            st.info('⚪ Kết quả: Không phát hiện khối u.')

        st.write("### 🔢 Xác suất dự đoán:")
        probs = [float(p * 100) for p in clf_pred[0]]
        df = pd.DataFrame({
            "Loại chẩn đoán": class_names,
            "Xác suất (%)": probs
        })
        chart = alt.Chart(df).mark_bar().encode(
            x="Loại chẩn đoán",
            y="Xác suất (%)",
            color="Loại chẩn đoán"
        )
        st.altair_chart(chart, use_container_width=True)

        for i, name in enumerate(class_names):
            st.write(f"- **{name}:** {probs[i]:.2f}%")

