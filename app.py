import os
import gdown
import joblib
import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import cv2

# ==============================
# 🔹 Hàm xử lý trung gian cho CBAM (GIỐNG CODE CŨ)
# ==============================
def spatial_mean(t):
    return tf.reduce_mean(t, axis=-1, keepdims=True)

def spatial_max(t):
    return tf.reduce_max(t, axis=-1, keepdims=True)

def spatial_output_shape(s):
    return (s[0], s[1], s[2], 1)

CUSTOM_OBJECTS = {
    "spatial_mean": spatial_mean,
    "spatial_max": spatial_max,
    "spatial_output_shape": spatial_output_shape,
}

# Bật unsafe_deserialization giống code cũ (nếu Keras cho phép)
try:
    from tensorflow import keras
    keras.config.enable_unsafe_deserialization()
except Exception:
    pass

# ==============================
# 🔹 Đường dẫn model & dữ liệu
# ==============================
seg_model_path = "seg_model.keras"
class_model_path = "clf_model.h5"
class_names_path = "class_names.npy"
clinical_model_path = "clinical_epic_gb_model.pkl"
clinical_metadata_path = "clinical_epic_gb_metadata.pkl"
data_path = "Breast_Cancer_METABRIC_Epic_Hospital.csv"

# Google Drive IDs
seg_model_id = "1axOg7N5ssJrMec97eV-JMPzID26ynzN1"
class_model_id = "1fXPICuTkETep2oPiA56l0uMai2GusEJH"
clinical_model_id = "1z1wHVy9xyRXlRqxI8lYXMJhaJaUcKXnu"
clinical_metadata_id = "1WWlfeRqr99VL4nBQ-7eEptIxitKtXj6V"

# ==============================
# 🔹 Hàm tải file từ Google Drive
# ==============================
def ensure_download(file_id, output_path, description):
    """Download file from Google Drive by ID to the specified output path.
       Returns True if file exists or downloaded successfully, False if failed."""
    if os.path.exists(output_path):
        return True
    try:
        st.info(f"Đang tải {description} từ Google Drive...")
        # dùng dạng id= giống code mới
        gdown.download(id=file_id, output=output_path, quiet=False)
    except Exception as e:
        st.error(f"Không thể tải {description}. Kiểm tra kết nối và quyền truy cập. Lỗi: {e}")
        return False
    if not os.path.exists(output_path):
        st.error(f"Tải {description} thất bại, ứng dụng sẽ dừng.")
        return False
    return True

# ==============================
# 🔹 Tải các file mô hình
# ==============================
if not ensure_download(seg_model_id, seg_model_path, "mô hình phân đoạn ảnh"):
    st.stop()
if not ensure_download(class_model_id, class_model_path, "mô hình phân loại ảnh"):
    st.stop()
if not ensure_download(clinical_model_id, clinical_model_path, "mô hình lâm sàng"):
    st.stop()
if not ensure_download(clinical_metadata_id, clinical_metadata_path, "siêu dữ liệu mô hình lâm sàng"):
    st.stop()

# ==============================
# 🔹 Load mô hình ảnh (DÙNG custom_objects GIỐNG CODE CŨ)
# ==============================
@st.cache_resource
def load_image_models():
    try:
        seg_model = tf.keras.models.load_model(
            seg_model_path,
            custom_objects=CUSTOM_OBJECTS,
            compile=False,
        )
        class_model = tf.keras.models.load_model(
            class_model_path,
            compile=False,
        )
    except Exception as e:
        st.error(f"Lỗi khi tải các mô hình ảnh: {e}")
        st.stop()
    return seg_model, class_model

seg_model, class_model = load_image_models()

# ==============================
# 🔹 Load class names
# ==============================
try:
    class_names = np.load(class_names_path)
except Exception:
    class_names = np.array(["Bình thường", "Lành tính", "Ác tính"])

# ==============================
# 🔹 Load mô hình lâm sàng & metadata
# ==============================
try:
    clinical_model = joblib.load(clinical_model_path)
    clinical_metadata = joblib.load(clinical_metadata_path)
except Exception as e:
    st.error(f"Lỗi khi tải mô hình lâm sàng: {e}")
    st.stop()

# ==============================
# 🔹 Load CSV lâm sàng
# ==============================
df = None
if os.path.exists(data_path):
    try:
        df = pd.read_csv(data_path)
    except Exception as e:
        st.warning(f"Không thể đọc dữ liệu lâm sàng: {e}")

# ==============================
# 🔹 Giao diện chính
# ==============================
st.title("Hệ thống chẩn đoán ung thư vú thông minh")
st.write("""
Tải ảnh siêu âm vú để hệ thống xác định vị trí khối u và phân loại 
khối u đó là **lành tính**, **ác tính** hoặc **bình thường**.
""")

# ==============================
# 🔹 Phần xử lý ảnh
# ==============================
uploaded_file = st.file_uploader("Chọn ảnh siêu âm (định dạng JPG/PNG)", type=["jpg", "png"])
if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if image is None:
        st.error("Không thể đọc ảnh. Hãy thử lại với tệp hình ảnh hợp lệ.")
    else:
        orig_image = image.copy()
        orig_height, orig_width = orig_image.shape[0], orig_image.shape[1]

        # ---- Phân đoạn (U-Net xám) ----
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        seg_height, seg_width = seg_model.input_shape[1], seg_model.input_shape[2]
        resized_gray = cv2.resize(gray_image, (seg_width, seg_height))
        input_seg = np.expand_dims(resized_gray, axis=(0, -1))  # (1, H, W, 1)

        pred_mask = seg_model.predict(input_seg)[0]
        mask = (pred_mask.squeeze() >= 0.5).astype(np.uint8)
        mask_full_size = cv2.resize(mask, (orig_width, orig_height), interpolation=cv2.INTER_NEAREST)

        overlay_image = orig_image.copy()
        contours, _ = cv2.findContours(mask_full_size, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay_image, contours, -1, (0, 255, 255), 2)  # vàng

        # ---- Phân loại (ResNet / model RGB 224x224) ----
        class_height, class_width = class_model.input_shape[1], class_model.input_shape[2]
        resized_img = cv2.resize(orig_image, (class_width, class_height))
        img_rgb = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
        img_array = img_rgb.astype(np.float32)
        img_array = img_array / 127.5 - 1.0  # scale -1..1
        img_array = np.expand_dims(img_array, axis=0)

        pred_logits = class_model.predict(img_array)
        pred_probs = tf.nn.softmax(pred_logits[0]).numpy()
        class_idx = int(np.argmax(pred_probs))
        class_label = class_names[class_idx] if class_idx < len(class_names) else str(class_idx)
        confidence = float(np.max(pred_probs))

        col1, col2 = st.columns([1, 1])
        with col1:
            st.subheader("Kết quả từ ảnh siêu âm")
            st.write(f"**Chẩn đoán:** {class_label}")
            st.write(f"**Xác suất dự đoán:** {confidence*100:.2f}%")
        with col2:
            overlay_rgb = cv2.cvtColor(overlay_image, cv2.COLOR_BGR2RGB)
            st.image(overlay_rgb, caption="Ảnh siêu âm với vùng khối u được đánh dấu", use_column_width=True)

# ==============================
# 🔹 Phần mô hình lâm sàng METABRIC
# ==============================
if df is not None and 'clinical_model' in locals():
    st.markdown("---")
    st.header("Dự đoán tiên lượng lâm sàng (METABRIC)")
    st.write("Chọn một bệnh nhân từ bộ dữ liệu METABRIC để dự đoán khả năng sống sót:")

    patient_ids = df["Patient ID"].unique().tolist()
    selected_id = st.selectbox("Mã bệnh nhân:", patient_ids)

    if selected_id:
        patient = df[df["Patient ID"] == selected_id].iloc[0]

        # ---- Chuẩn bị đầu vào cho mô hình lâm sàng ----
        if isinstance(clinical_metadata, dict) and "features" in clinical_metadata:
            feature_cols = clinical_metadata["features"]
        else:
            cols = [c for c in df.columns if c not in [
                "Patient ID", "Overall Survival (Months)",
                "Overall Survival Status", "Relapse Free Status (Months)",
                "Relapse Free Status", "Patient's Vital Status"
            ]]
            feature_cols = cols

        X_input = patient[feature_cols].copy()

        if isinstance(clinical_metadata, dict):
            # encoders
            if "encoders" in clinical_metadata:
                for col, encoder in clinical_metadata["encoders"].items():
                    if col not in X_input.index:
                        continue
                    try:
                        X_input[col] = encoder.transform([X_input[col]])[0]
                    except Exception:
                        if isinstance(encoder, dict):
                            X_input[col] = encoder.get(X_input[col], X_input[col])

            # scaler
            if "scaler" in clinical_metadata:
                X_df = pd.DataFrame([X_input.values], columns=feature_cols)
                X_scaled = clinical_metadata["scaler"].transform(X_df)
            else:
                X_scaled = np.array([X_input.values])
        else:
            X_scaled = np.array([X_input.values])

        # ---- Dự đoán với mô hình lâm sàng ----
        y_pred = clinical_model.predict(X_scaled)
        pred_label = None
        pred_prob = None

        if hasattr(clinical_model, "predict_proba"):
            try:
                prob = clinical_model.predict_proba(X_scaled)
                pred_prob = float(np.max(prob))
            except Exception:
                pred_prob = None

        if isinstance(clinical_metadata, dict) and "target_encoder" in clinical_metadata:
            try:
                pred_label = clinical_metadata["target_encoder"].inverse_transform(y_pred)[0]
            except Exception:
                pred_label = str(y_pred[0])
        elif isinstance(clinical_metadata, dict) and "target_map" in clinical_metadata:
            inv_map = {v: k for k, v in clinical_metadata["target_map"].items()}
            pred_label = inv_map.get(int(y_pred[0]), str(y_pred[0]))
        else:
            pred_label = "Living" if int(y_pred[0]) == 0 else "Died of Disease"

        actual_label = patient["Patient's Vital Status"]

        st.subheader("Kết quả dự đoán cho bệnh nhân " + selected_id)
        result_text = f"**Dự đoán của mô hình:** {pred_label}"
        if pred_prob is not None:
            result_text += f" (xác suất {pred_prob*100:.1f}%)"
        st.write(result_text)
        st.write(f"**Tình trạng thực tế:** {actual_label}")
