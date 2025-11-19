import os
import gdown
import joblib
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image  # Thêm import PIL để xử lý ảnh

# Tải mô hình phân tích ảnh siêu âm đã huấn luyện (giữ nguyên chức năng có sẵn)
seg_model_path = "best_model_cbam_attention_unet.h5"
class_model_path = "breast_ultrasound_classifier.h5"
class_names_path = "class_names.npy"
if not os.path.exists(seg_model_path):
    gdown.download(id="your_seg_model_drive_id", output=seg_model_path, quiet=False)
if not os.path.exists(class_model_path):
    gdown.download(id="your_class_model_drive_id", output=class_model_path, quiet=False)
if not os.path.exists(class_names_path):
    gdown.download(id="your_class_names_drive_id", output=class_names_path, quiet=False)
seg_model = tf.keras.models.load_model(seg_model_path, compile=False)
class_model = tf.keras.models.load_model(class_model_path)
class_names = np.load(class_names_path)

# Tải mô hình Gradient Boosting phân tích thông tin lâm sàng (mới thêm)
MODEL_PATH = "clinical_epic_gb_model.pkl"
META_PATH = "clinical_epic_gb_metadata.pkl"
if not os.path.exists(MODEL_PATH):
    gdown.download(id="your_gb_model_drive_id", output=MODEL_PATH, quiet=False)
if not os.path.exists(META_PATH):
    gdown.download(id="your_gb_metadata_drive_id", output=META_PATH, quiet=False)
clinical_model = joblib.load(MODEL_PATH)
clinical_meta = joblib.load(META_PATH)

# Hàm mới: Dự đoán nguy cơ tử vong dựa trên thông tin lâm sàng
def predict_survival(age, tumor_size, lymph_nodes, hist_grade, tumor_stage, sex):
    """Dự đoán Overall Survival Status bằng mô hình lâm sàng Gradient Boosting."""
    # Chuẩn bị đặc trưng đầu vào (theo đúng pipeline tiền xử lý khi huấn luyện)
    num_features = np.array([[age, tumor_size, lymph_nodes]], dtype=float)
    scaler = None; ohe = None
    if isinstance(clinical_meta, dict):
        scaler = clinical_meta.get('scaler')
        ohe = clinical_meta.get('ohe')
    # Áp dụng StandardScaler cho đặc trưng số nếu có
    num_scaled = scaler.transform(num_features) if scaler else num_features
    # Áp dụng OneHotEncoder cho đặc trưng phân loại nếu có
    cat_features = [[hist_grade, tumor_stage, sex]]
    if ohe:
        cat_encoded = ohe.transform(cat_features)
        cat_encoded = cat_encoded.toarray() if hasattr(cat_encoded, "toarray") else cat_encoded
    else:
        # Nếu không có bộ mã, chuyển đổi thủ công (vd: Female/Male -> 0/1)
        sex_num = 1 if sex == "Male" else 0
        cat_encoded = np.array([[hist_grade, tumor_stage, sex_num]])
    # Ghép đặc trưng số và phân loại thành vector đầu vào
    X_input = np.hstack([num_scaled, cat_encoded])
    # Dự đoán xác suất tử vong (giả định mô hình nhị phân: 0=Living, 1=Deceased)
    prob = clinical_model.predict_proba(X_input)[0][1]
    return prob

# Sidebar - lựa chọn chế độ phân tích
st.sidebar.title("Chọn chế độ")
analysis_mode = st.sidebar.radio("", ["Phân tích ảnh siêu âm", "Phân tích thông tin lâm sàng"], index=0)

if analysis_mode == "Phân tích ảnh siêu âm":
    # *** Giao diện phân tích ảnh siêu âm (giữ nguyên) ***
    st.header("Phân tích ảnh siêu âm")
    uploaded_file = st.file_uploader("Tải ảnh siêu âm của bệnh nhân", type=["png", "jpg", "jpeg"])
    if uploaded_file is not None:
        # Hiển thị ảnh gốc
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Ảnh siêu âm gốc", use_column_width=True)
        img_array = np.array(image)
        # Tiền xử lý ảnh (resize về kích thước phù hợp và chuẩn hóa)
        img_resized = image.resize((256, 256))
        img_tensor = np.array(img_resized) / 255.0
        img_input = np.expand_dims(img_tensor, axis=0)
        # Dự đoán vùng tổn thương (phân đoạn)
        mask_pred = seg_model.predict(img_input)[0]
        mask_pred = np.argmax(mask_pred, axis=-1).astype(np.uint8)
        # Tô màu vùng tổn thương trên ảnh (xanh cho lành tính, đỏ cho ác tính)
        overlay = img_array.copy()
        overlay[mask_pred == 1] = 0.5 * overlay[mask_pred == 1] + 0.5 * np.array([0, 255, 0], dtype=np.uint8)
        overlay[mask_pred == 2] = 0.5 * overlay[mask_pred == 2] + 0.5 * np.array([255, 0, 0], dtype=np.uint8)
        overlay = overlay.astype(np.uint8)
        st.image(overlay, caption="Vùng tổn thương (xanh: lành tính, đỏ: ác tính)", use_column_width=True)
        # Dự đoán loại tổn thương (phân loại)
        preds = class_model.predict(img_input)
        class_idx = int(np.argmax(preds, axis=1)[0])
        class_name = class_names[class_idx]
        confidence = float(preds[0][class_idx]) * 100
        st.subheader(f"Kết quả phân loại: {class_name} ({confidence:.1f}%)")

elif analysis_mode == "Phân tích thông tin lâm sàng":
    # *** Giao diện phân tích thông tin lâm sàng (mới thêm) ***
    st.header("Phân tích thông tin lâm sàng")
    with st.form("clinical_form"):
        age = st.number_input("Tuổi chẩn đoán", min_value=0, max_value=120, value=50)
        tumor_size = st.number_input("Kích thước khối u (mm)", min_value=0.0, max_value=200.0, value=20.0)
        lymph_nodes = st.number_input("Số hạch dương tính", min_value=0, max_value=50, value=0)
        hist_grade = st.selectbox("Độ mô học (1-3)", [1, 2, 3], index=1)
        tumor_stage = st.selectbox("Giai đoạn (1-4)", [1, 2, 3, 4], index=0)
        sex_input = st.selectbox("Giới tính", ["Nữ", "Nam"], index=0)
        sex = "Female" if sex_input == "Nữ" else "Male"
        submitted = st.form_submit_button("Dự đoán")
    if submitted:
        # Thực hiện dự đoán với mô hình lâm sàng
        risk = predict_survival(age, tumor_size, lymph_nodes, hist_grade, tumor_stage, sex)
        risk_percent = risk * 100
        if risk >= 0.5:
            status_str = "Deceased (Tử vong)"
            explanation = "bệnh nhân có nguy cơ tử vong cao"
        else:
            status_str = "Living (Sống sót)"
            explanation = "bệnh nhân có khả năng sống sót cao"
        st.subheader("Kết quả dự đoán nguy cơ tử vong:")
        st.write(f"- Xác suất tử vong dự đoán: **{risk_percent:.1f}%**")
        st.write(f"- Phân loại: **{status_str}**")
        st.write(f"Giải thích: Theo mô hình, {explanation} trong thời gian theo dõi.")
