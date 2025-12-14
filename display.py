# =====================================================
# ỨNG DỤNG HỖ TRỢ CHẨN ĐOÁN & TIÊN LƯỢNG UNG THƯ VÚ
# (Phiên bản nâng cấp: DICOM, PDF Report, Active Learning)
# =====================================================

import os
import time
import datetime
import csv
import numpy as np
import pandas as pd
import cv2
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image
import io
import base64
import unicodedata

# Thư viện Y tế & PDF
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut
from fpdf import FPDF

# Thư viện Deep Learning
import tensorflow as tf
import keras
from keras.models import load_model
from keras.saving import register_keras_serializable
import joblib

# =====================================================
# 1. CẤU HÌNH & KHỞI TẠO
# =====================================================
st.set_page_config(
    page_title="AI Siêu âm Vú (Demo KHKT)",
    layout="wide",
    page_icon="🩺"
)

# Khởi tạo Session State
if 'patient_data' not in st.session_state:
    st.session_state['patient_data'] = {
        'age': 0,
        'tumor_size': 0.0,
        'lymph_nodes': 0,
        'name': "",
        'id': ""
    }

# Custom CSS
st.markdown("""
<style>
    .main-header {font-size: 2.2rem; color: #FF4B4B; text-align: center; font-weight: bold; margin-bottom: 20px;}
    .report-box {border: 2px solid #ddd; padding: 20px; border-radius: 10px; background-color: #f9f9f9;}
    .stButton>button {width: 100%; border-radius: 8px;}
</style>
""", unsafe_allow_html=True)

# =====================================================
# 2. CÁC HÀM XỬ LÝ (BACKEND)
# =====================================================

# --- A. Custom Layer (Giữ nguyên để load model) ---
try:
    keras.config.enable_unsafe_deserialization()
except Exception:
    pass

@register_keras_serializable(package="cbam", name="spatial_mean")
def spatial_mean(x):
    return tf.reduce_mean(x, axis=-1, keepdims=True)

@register_keras_serializable(package="cbam", name="spatial_flatten")
def spatial_flatten(x):
    return tf.reshape(x, [-1, x.shape[1] * x.shape[2]])

# --- B. Hàm xử lý DICOM (MỚI) ---
def process_dicom(file):
    try:
        ds = pydicom.dcmread(file)
        
        # 1. Trích xuất ảnh
        pixel_array = ds.pixel_array
        if 'WindowWidth' in ds and 'WindowCenter' in ds:
            pixel_array = apply_voi_lut(pixel_array, ds)
        
        # Chuẩn hóa về 0-255
        if pixel_array.max() > 0:
            pixel_array = (pixel_array - pixel_array.min()) / (pixel_array.max() - pixel_array.min()) * 255.0
        pixel_array = pixel_array.astype(np.uint8)
        
        # Chuyển sang RGB
        if len(pixel_array.shape) == 2:
            img_rgb = cv2.cvtColor(pixel_array, cv2.COLOR_GRAY2RGB)
        else:
            img_rgb = pixel_array
            
        # 2. Trích xuất Metadata
        p_age = 0
        p_name = ""
        p_id = ""
        
        if 'PatientAge' in ds and ds.PatientAge:
            age_str = str(ds.PatientAge).replace('Y', '').replace('M', '').replace('D', '')
            if age_str.isdigit():
                p_age = int(age_str)
        
        if 'PatientName' in ds and ds.PatientName:
            p_name = str(ds.PatientName)
        
        if 'PatientID' in ds:
            p_id = str(ds.PatientID)
            
        return img_rgb, p_age, p_name, p_id
    except Exception as e:
        st.error(f"Lỗi đọc DICOM: {e}")
        return None, 0, "", ""

# --- C. Hàm Xuất PDF (MỚI) ---
# Hàm bỏ dấu tiếng Việt để tránh lỗi font trong FPDF cơ bản
def remove_accents(input_str):
    if not isinstance(input_str, str): return str(input_str)
    nfkd_form = unicodedata.normalize('NFKD', input_str)
    return "".join([c for c in nfkd_form if not unicodedata.combining(c)])

class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 15)
        self.cell(0, 10, 'KET QUA HO TRO CHAN DOAN (AI REPORT)', 0, 1, 'C')
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

def create_pdf_report(info, ai_result, surv_text, surv_prob):
    pdf = PDF()
    pdf.add_page()
    pdf.set_font("Arial", size=11)
    
    # Thông tin hành chính
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "1. THONG TIN BENH NHAN (Patient Info)", 0, 1)
    pdf.set_font("Arial", size=11)
    pdf.cell(0, 8, f"Ho ten: {remove_accents(info['name'])}", 0, 1)
    pdf.cell(0, 8, f"Ma BN: {remove_accents(info['id'])}", 0, 1)
    pdf.cell(0, 8, f"Tuoi: {info['age']}", 0, 1)
    pdf.ln(5)
    
    # Kết quả Hình ảnh
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "2. KET QUA HINH ANH (Imaging Results)", 0, 1)
    pdf.set_font("Arial", size=11)
    pdf.cell(0, 8, f"Phan loai BI-RADS (AI): {remove_accents(ai_result)}", 0, 1)
    pdf.cell(0, 8, f"Kich thuoc u (Tumor Size): {info['tumor_size']} mm", 0, 1)
    pdf.ln(5)
    
    # Tiên lượng
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "3. TIEN LUONG SONG CON (Prognosis)", 0, 1)
    pdf.set_font("Arial", size=11)
    pdf.cell(0, 8, f"Nhom nguy co: {remove_accents(surv_text)}", 0, 1)
    pdf.cell(0, 8, f"Xac suat song sot 5 nam du bao: {surv_prob*100:.1f}%", 0, 1)
    
    pdf.ln(10)
    pdf.set_font("Arial", 'I', 9)
    pdf.multi_cell(0, 5, "Luu y: Bao cao nay duoc tao tu dong boi he thong AI thu nghiem. Ket qua can duoc bac si chuyen khoa xac nhan.")
    
    return pdf.output(dest='S').encode('latin-1')

# =====================================================
# 3. GIAO DIỆN NGƯỜI DÙNG (UI)
# =====================================================

st.markdown('<p class="main-header">HỆ THỐNG HỖ TRỢ CHẨN ĐOÁN UNG THƯ VÚ</p>', unsafe_allow_html=True)

# --- SIDEBAR ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3004/3004458.png", width=80)
    st.title("Điều khiển")
    
    # --- 1. NÚT DEMO DATA (TÍNH NĂNG MỚI) ---
    st.markdown("---")
    if st.button("⚡ Tải dữ liệu mẫu (Demo)"):
        st.session_state['patient_data']['id'] = "BN-DEMO-001"
        st.session_state['patient_data']['name'] = "Nguyen Thi B"
        st.session_state['patient_data']['age'] = 54
        st.session_state['patient_data']['tumor_size'] = 22.5
        st.session_state['patient_data']['lymph_nodes'] = 1
        st.success("Đã tải xong!")
        time.sleep(0.5)
        st.rerun()
    st.markdown("---")

    st.subheader("Thông tin lâm sàng")
    # Liên kết Input với Session State
    p_id = st.text_input("Mã BN", value=st.session_state['patient_data']['id'])
    p_name = st.text_input("Họ tên", value=st.session_state['patient_data']['name'])
    p_age = st.number_input("Tuổi", min_value=0, max_value=120, value=st.session_state['patient_data']['age'])
    p_nodes = st.number_input("Số hạch bạch huyết (+)", min_value=0, value=st.session_state['patient_data']['lymph_nodes'])
    
    # Cập nhật ngược lại Session State
    st.session_state['patient_data']['id'] = p_id
    st.session_state['patient_data']['name'] = p_name
    st.session_state['patient_data']['age'] = p_age
    st.session_state['patient_data']['lymph_nodes'] = p_nodes

# --- MAIN PAGE ---
col1, col2 = st.columns([1, 1])

processed_image = None
ai_tumor_size_result = 0.0

with col1:
    st.subheader("1. Tải ảnh siêu âm")
    uploaded_file = st.file_uploader("Chọn ảnh (JPG, PNG, DICOM)", type=['jpg', 'png', 'jpeg', 'dcm'])
    
    if uploaded_file is not None:
        file_ext = uploaded_file.name.split('.')[-1].lower()
        
        # --- 2. XỬ LÝ DICOM (TÍNH NĂNG MỚI) ---
        if file_ext == 'dcm':
            with st.spinner("Đang đọc dữ liệu DICOM..."):
                img_rgb, d_age, d_name, d_id = process_dicom(uploaded_file)
                if img_rgb is not None:
                    processed_image = img_rgb
                    st.image(processed_image, caption="Ảnh trích xuất từ DICOM", use_container_width=True)
                    
                    # Tự động điền thông tin
                    if st.session_state['patient_data']['age'] == 0 and d_age > 0:
                        st.session_state['patient_data']['age'] = d_age
                        st.session_state['patient_data']['name'] = d_name
                        st.session_state['patient_data']['id'] = d_id
                        st.toast(f"Đã tự động điền thông tin BN từ file ảnh!", icon="✨")
                        time.sleep(1)
                        st.rerun()
        else:
            # Ảnh thường
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, 1)
            processed_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            st.image(processed_image, caption="Ảnh siêu âm gốc", use_container_width=True)

with col2:
    st.subheader("2. Kết quả phân tích")
    
    if processed_image is not None:
        if st.button("🚀 PHÂN TÍCH NGAY", type="primary"):
            with st.spinner('AI đang phân tích đặc điểm khối u...'):
                # ---------------------------------------------------------
                # [PLACEHOLDER] ĐOẠN NÀY GỌI MODEL CỦA BẠN
                # ---------------------------------------------------------
                time.sleep(1.5) # Giả lập độ trễ
                
                # Giả lập kết quả nếu chưa có model thật
                # Nếu bạn đã load model thật, hãy thay thế đoạn này bằng: pred = model.predict(...)
                if st.session_state['patient_data']['tumor_size'] > 0:
                    ai_tumor_size_result = st.session_state['patient_data']['tumor_size']
                else:
                    ai_tumor_size_result = np.random.uniform(15.0, 35.0) # Giả lập đo đạc
                    st.session_state['patient_data']['tumor_size'] = round(ai_tumor_size_result, 1)

                ai_birads = "BI-RADS 4c" # Giả lập
                confidence = 0.89
                
                # HIỂN THỊ KẾT QUẢ
                st.success("Phân tích hoàn tất!")
                
                tabs = st.tabs(["🖼️ Chẩn đoán hình ảnh", "📈 Tiên lượng (Cox-PH)", "📄 Báo cáo"])
                
                # Tab 1: AI Hình ảnh
                with tabs[0]:
                    col_m1, col_m2 = st.columns(2)
                    col_m1.metric("Kích thước u (AI)", f"{ai_tumor_size_result:.1f} mm")
                    col_m2.metric("Phân loại", ai_birads)
                    st.progress(confidence, text=f"Độ tin cậy: {confidence*100}%")
                
                # Tab 2: Tiên lượng (Cox Model Logic)
                with tabs[1]:
                    # Tính Hazard Score giả lập (hoặc dùng cox model của bạn)
                    # Score = b1*Age + b2*Size + b3*Nodes
                    h_score = (0.02 * st.session_state['patient_data']['age']) + \
                              (0.015 * ai_tumor_size_result) + \
                              (0.1 * st.session_state['patient_data']['lymph_nodes'])
                    
                    survival_prob_5yr = np.exp(-h_score) # Công thức giản lược
                    
                    if survival_prob_5yr < 0.5:
                        risk_level = "Nguy cơ CAO"
                        msg_type = "error"
                    elif survival_prob_5yr < 0.8:
                        risk_level = "Nguy cơ TRUNG BÌNH"
                        msg_type = "warning"
                    else:
                        risk_level = "Nguy cơ THẤP"
                        msg_type = "success"
                        
                    if msg_type == "error": st.error(f"Đánh giá: {risk_level}")
                    elif msg_type == "warning": st.warning(f"Đánh giá: {risk_level}")
                    else: st.success(f"Đánh giá: {risk_level}")
                    
                    st.write(f"Ước tính xác suất sống sót sau 5 năm: **{survival_prob_5yr*100:.1f}%**")
                    
                    # Biểu đồ
                    chart_data = pd.DataFrame({
                        'Năm': [1, 2, 3, 4, 5],
                        'Sống còn (%)': [100, 100*survival_prob_5yr**0.2, 100*survival_prob_5yr**0.4, 
                                         100*survival_prob_5yr**0.6, 100*survival_prob_5yr]
                    })
                    st.line_chart(chart_data.set_index('Năm'))

                # --- 3. XUẤT PDF (TÍNH NĂNG MỚI) ---
                with tabs[2]:
                    st.write("Tải báo cáo kết quả:")
                    pdf_data = create_pdf_report(
                        st.session_state['patient_data'],
                        ai_birads,
                        risk_level,
                        survival_prob_5yr
                    )
                    b64_pdf = base64.b64encode(pdf_data).decode('latin-1')
                    href = f'<a href="data:application/pdf;base64,{b64_pdf}" download="KQ_ChanDoan_{p_id}.pdf" class="stButton"><button style="padding: 10px 20px; background-color: #28a745; color: white; border: none; border-radius: 5px; cursor: pointer;">🖨️ Tải xuống PDF</button></a>'
                    st.markdown(href, unsafe_allow_html=True)
    else:
        st.info("👈 Vui lòng tải ảnh lên ở cột bên trái.")

# =====================================================
# 4. FEEDBACK LOOP (TÍNH NĂNG MỚI - ACTIVE LEARNING)
# =====================================================
st.markdown("---")
with st.expander("👨‍⚕️ Góc chuyên môn: Gửi phản hồi (Active Learning)"):
    st.caption("Giúp hệ thống học tập bằng cách xác nhận kết quả:")
    
    f_col1, f_col2 = st.columns([1, 2])
    with f_col1:
        fb_status = st.radio("Đánh giá kết quả AI:", ["Chính xác", "Sai lệch kích thước", "Bỏ sót tổn thương", "Dương tính giả"])
    with f_col2:
        fb_note = st.text_area("Ghi chú chi tiết (nếu có):", placeholder="Ví dụ: Kích thước thực tế là 25mm, bờ không đều...")
        
    if st.button("Gửi dữ liệu phản hồi"):
        # Lưu vào CSV
        log_data = {
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "patient_id": st.session_state['patient_data']['id'],
            "ai_size": st.session_state['patient_data']['tumor_size'],
            "feedback_type": fb_status,
            "note": fb_note
        }
        
        file_exists = os.path.isfile('feedback_log.csv')
        try:
            with open('feedback_log.csv', mode='a', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=log_data.keys())
                if not file_exists: writer.writeheader()
                writer.writerow(log_data)
            st.toast("Đã lưu phản hồi thành công!", icon="✅")
        except Exception as e:
            st.error(f"Lỗi lưu file: {e}")

# Footer
st.markdown("""
<div style='text-align: center; color: grey; margin-top: 50px; font-size: 0.8em;'>
    © 2025 Dự án KHKT - Hỗ trợ Chẩn đoán Ung thư Vú<br>
    Lưu ý: Kết quả chỉ mang tính tham khảo nghiên cứu.
</div>
""", unsafe_allow_html=True)
