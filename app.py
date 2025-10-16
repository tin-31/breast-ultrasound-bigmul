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

SEG_MODEL_ID = "1YbX7lBQCjWXaSyCtwUXftjFHEaBjNnDa"  # ID Google Drive của model phân đoạn .h5
CLF_MODEL_ID = "1fXPICuTkETep2oPiA56l0uMai2GusEJH"  # ID Google Drive của model phân loại .h5

SEG_MODEL_PATH = "best_model_cbam_attention_unet.h5"
CLF_MODEL_PATH = "Classifier_model.h5"

# ============================================================
# 🔹 2. TẢI FILE MODEL NẾU CHƯA CÓ
# ============================================================

if not os.path.exists(SEG_MODEL_PATH):
    st.info("📥 Đang tải model phân đoạn (.h5)...")
    gdown.download(f"https://drive.google.com/uc?id={SEG_MODEL_ID}", SEG_MODEL_PATH, quiet=False)
    st.success("✅ Model phân đoạn đã tải xong!")

if not os.path.exists(CLF_MODEL_PATH):
    st.info("📥 Đang tải model phân loại (.h5)...")
    gdown.download(f"https://drive.google.com/uc?id={CLF_MODEL_ID}", CLF_MODEL_PATH, quiet=False)
    st.success("✅ Model phân loại đã tải xong!")

# ============================================================
# 🔹 3. LOAD MODELS (CÓ CUSTOM LAYERS)
# ============================================================

from custom_layers import SEBlock, SpatialAttention, CBAM, AttentionGate

@st.cache_resource(ttl=3600)
def load_models():
    custom_objects = {
        "SEBlock": SEBlock,
        "SpatialAttention": SpatialAttention,
        "CBAM": CBAM,
        "AttentionGate": AttentionGate
    }

    # Load classifier
    classifier = tf.keras.models.load_model(CLF_MODEL_PATH, compile=False)

    # Load segmentation model
    segmentor = tf.keras.models.load_model(seg_model_path, compile=False)
    return classifier, segmentor

# ============================================================
# 🔹 4. XỬ LÝ ẢNH (PREPROCESS / POSTPROCESS)
# ============================================================

def classify_preprop(image_file):
    image = Image.open(BytesIO(image_file)).convert("RGB")
    image = image.resize((224, 224))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)
    return image

def segment_preprop(image_file):
    image = Image.open(BytesIO(image_file)).convert("RGB")
    image = image.resize((256, 256))
    arr = np.array(image) / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr

def segment_postprop(image, mask):
    image = np.squeeze(image)
    mask = np.squeeze(mask)
    if mask.ndim == 2:
        mask = np.expand_dims(mask, axis=2)
    return image * mask

def preprocessing_uploader(file, classifier, segmentor):
    image_file = file.read()
    image_to_classify = classify_preprop(image_file)
    image_to_segment = segment_preprop(image_file)

    classify_output = classifier.predict(image_to_classify)
    segment_output = segmentor.predict(image_to_segment)[0]
    segment_output = segment_postprop(image_to_segment, segment_output)
    return classify_output, segment_output

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
        tin_ava = Image.open('Tin.jpg')
        st.image(tin_ava, caption='Lê Vũ Anh Tin')
        school_ava = Image.open('school.jpg')
        st.image(school_ava, caption='Trường THPT Chuyên Nguyễn Du')
    except:
        pass

elif app_mode == 'Thống kê về dữ liệu huấn luyện':
    st.title('📊 Thống kê tổng quan về tập dữ liệu')
    st.caption("""
    Trong nghiên cứu này, tập dữ liệu **Breast Ultrasound Images (BUI)** được kết hợp từ hai nguồn công khai:
    - BUSI (Arya Shah, Kaggle)
    - BUS-UCLM (Orvile, Kaggle)
    Tổng cộng gồm **1578 ảnh siêu âm vú** với mặt nạ phân đoạn tương ứng.
    """)
    st.caption('Chi tiết dataset: https://drive.google.com/drive/folders/1eSAA5pMuEz1GgATBmvXbjjaihO1yBo1l?usp=drive_link')

elif app_mode == 'Ứng dụng chẩn đoán':
    st.title('🩺 Ứng dụng chẩn đoán bệnh ung thư vú từ ảnh siêu âm')

    classifier, segmentor = load_models()

    file = st.file_uploader("📤 Tải ảnh siêu âm vú (jpg/png)", type=["jpg", "png"])
    if file is None:
        st.info('👆 Vui lòng tải ảnh siêu âm lên để bắt đầu chẩn đoán.')
    else:
        slot = st.empty()
        slot.text('⏳ Đang phân tích ảnh...')

        classify_output, segment_output = preprocessing_uploader(file, classifier, segmentor)
        test_image = Image.open(file)
        st.image(test_image, caption="Ảnh đầu vào", width=400)

        class_names = ['benign', 'malignant', 'normal']
        result_name = class_names[np.argmax(classify_output)]

        st.image(segment_output, caption="Ảnh phân đoạn khối u", width=400)

        if result_name == 'benign':
            st.error('🟢 Chẩn đoán: Bệnh nhân có khối u lành tính.')
        elif result_name == 'malignant':
            st.warning('🔴 Chẩn đoán: Bệnh nhân mắc ung thư vú.')
        else:
            st.success('⚪ Chẩn đoán: Không phát hiện dấu hiệu khối u.')

        slot.success('✅ Hoàn tất chẩn đoán!')

        bar_frame = pd.DataFrame({
            'Loại chẩn đoán': ["Lành tính", "Ác tính", "Bình thường"],
            'Xác suất dự đoán (%)': [
                classify_output[0,0] * 100,
                classify_output[0,1] * 100,
                classify_output[0,2] * 100
            ]
        })
        bar_chart = alt.Chart(bar_frame).mark_bar().encode(
            x='Loại chẩn đoán',
            y='Xác suất dự đoán (%)',
            color='Loại chẩn đoán'
        )
        st.altair_chart(bar_chart, use_container_width=True)

        st.write(f"- **Khối u lành tính:** {classify_output[0,0]*100:.2f}%")
        st.write(f"- **Ung thư vú:** {classify_output[0,1]*100:.2f}%")
        st.write(f"- **Bình thường:** {classify_output[0,2]*100:.2f}%")
