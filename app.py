import os
import gdown
import joblib
import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import cv2

# Define file paths for models and data
seg_model_path = "seg_model.keras"
class_model_path = "clf_model.h5"
class_names_path = "class_names.npy"
clinical_model_path = "clinical_epic_gb_model.pkl"
clinical_metadata_path = "clinical_epic_gb_metadata.pkl"
data_path = "Breast_Cancer_METABRIC_Epic_Hospital.csv"

# Google Drive IDs for the model files (provided by user)
seg_model_id = "1axOg7N5ssJrMec97eV-JMPzID26ynzN1"
class_model_id = "1fXPICuTkETep2oPiA56l0uMai2GusEJH"
clinical_model_id = "1z1wHVy9xyRXlRqxI8lYXMJhaJaUcKXnu"
clinical_metadata_id = "1WWlfeRqr99VL4nBQ-7eEptIxitKtXj6V"

# Function to download a file from Google Drive if not already present
def ensure_download(file_id, output_path, description):
    """Download file from Google Drive by ID to the specified output path.
       Returns True if file exists or downloaded successfully, False if failed."""
    if os.path.exists(output_path):
        return True
    try:
        st.info(f"Đang tải {description} từ Google Drive...")
        gdown.download(id=file_id, output=output_path, quiet=False)
    except Exception as e:
        st.error(f"Không thể tải {description}. Kiểm tra kết nối và quyền truy cập. Lỗi: {e}")
        return False
    # Verify file was downloaded
    if not os.path.exists(output_path):
        st.error(f"Tải {description} thất bại, ứng dụng sẽ dừng.")
        return False
    return True

# Download required model files if not present
if not ensure_download(seg_model_id, seg_model_path, "mô hình phân đoạn ảnh"):
    st.stop()
if not ensure_download(class_model_id, class_model_path, "mô hình phân loại ảnh"):
    st.stop()
if not ensure_download(clinical_model_id, clinical_model_path, "mô hình lâm sàng"):
    st.stop()
if not ensure_download(clinical_metadata_id, clinical_metadata_path, "siêu dữ liệu mô hình lâm sàng"):
    st.stop()

# Load the models and data
try:
    seg_model = tf.keras.models.load_model(seg_model_path)
    class_model = tf.keras.models.load_model(class_model_path)
except Exception as e:
    st.error(f"Lỗi khi tải các mô hình ảnh: {e}")
    st.stop()

# Load class names for classification (benign/malignant/normal)
try:
    class_names = np.load(class_names_path)
except Exception as e:
    # Nếu không có file class_names.npy, có thể định nghĩa thủ công:
    class_names = np.array(["Bình thường", "Lành tính", "Ác tính"])

# Load clinical model and metadata
try:
    clinical_model = joblib.load(clinical_model_path)
    clinical_metadata = joblib.load(clinical_metadata_path)
except Exception as e:
    st.error(f"Lỗi khi tải mô hình lâm sàng: {e}")
    st.stop()

# Load clinical data CSV (if needed for UI)
df = None
if os.path.exists(data_path):
    try:
        df = pd.read_csv(data_path)
    except Exception as e:
        st.warning(f"Không thể đọc dữ liệu lâm sàng: {e}")

# Set up Streamlit interface
st.title("Hệ thống chẩn đoán ung thư vú thông minh")
st.write("""
Tải ảnh siêu âm vú để hệ thống xác định vị trí khối u và phân loại 
khối u đó là **lành tính**, **ác tính** hoặc **bình thường**.
""")

# Image upload and analysis
uploaded_file = st.file_uploader("Chọn ảnh siêu âm (định dạng JPG/PNG)", type=["jpg", "png"])
if uploaded_file is not None:
    # Đọc ảnh từ file tải lên
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if image is None:
        st.error("Không thể đọc ảnh. Hãy thử lại với tệp hình ảnh hợp lệ.")
    else:
        # Giữ ảnh gốc để hiển thị
        orig_image = image.copy()
        orig_height, orig_width = orig_image.shape[0], orig_image.shape[1]
        # Chuyển ảnh sang thang xám cho mô hình phân đoạn (nếu mô hình yêu cầu)
        # (Giả định mô hình phân đoạn U-Net sử dụng ảnh xám đầu vào)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Chuẩn bị ảnh cho mô hình phân đoạn (resize nếu cần)
        # Giả sử mô hình phân đoạn yêu cầu kích thước 256x256
        seg_height, seg_width = seg_model.input_shape[1], seg_model.input_shape[2]
        resized_gray = cv2.resize(gray_image, (seg_width, seg_height))
        # Chuẩn hóa định dạng đầu vào (thêm batch dimension và channel)
        input_seg = np.expand_dims(resized_gray, axis=(0, -1))  # shape (1, H, W, 1)
        # Dự đoán mask
        pred_mask = seg_model.predict(input_seg)[0]
        # Chuyển đổi mask thành nhị phân (ngưỡng 0.5)
        mask = (pred_mask.squeeze() >= 0.5).astype(np.uint8)
        # Resize mask về kích thước ảnh gốc
        mask_full_size = cv2.resize(mask, (orig_width, orig_height), interpolation=cv2.INTER_NEAREST)
        # Tô viền khối u trên ảnh gốc
        overlay_image = orig_image.copy()
        contours, _ = cv2.findContours(mask_full_size, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay_image, contours, -1, (0, 255, 255), 2)  # viền vàng cho khối u
        # Chuẩn bị ảnh cho mô hình phân loại (resize về 224x224 chẳng hạn)
        class_height, class_width = class_model.input_shape[1], class_model.input_shape[2]
        resized_img = cv2.resize(orig_image, (class_width, class_height))
        # Đảm bảo ảnh có 3 kênh màu (mô hình ResNet152V2 yêu cầu 3 kênh)
        img_rgb = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
        # Chuẩn hóa ảnh cho mô hình phân loại (scale -1..1)
        img_array = img_rgb.astype(np.float32)
        img_array = img_array / 127.5 - 1.0
        img_array = np.expand_dims(img_array, axis=0)
        # Dự đoán phân loại
        pred_logits = class_model.predict(img_array)
        # Tính xác suất dự đoán và lớp dự đoán
        pred_probs = tf.nn.softmax(pred_logits[0]).numpy()
        class_idx = int(np.argmax(pred_probs))
        class_label = class_names[class_idx] if class_idx < len(class_names) else str(class_idx)
        confidence = float(np.max(pred_probs))
        # Hiển thị kết quả
        col1, col2 = st.columns([1, 1])
        with col1:
            st.subheader("Kết quả từ ảnh siêu âm")
            st.write(f"**Chẩn đoán:** {class_label}")
            st.write(f"**Xác suất dự đoán:** {confidence*100:.2f}%")
        with col2:
            # Chuyển ảnh BGR sang RGB để hiển thị đúng màu
            overlay_rgb = cv2.cvtColor(overlay_image, cv2.COLOR_BGR2RGB)
            st.image(overlay_rgb, caption="Ảnh siêu âm với vùng khối u được đánh dấu", use_column_width=True)

# Clinical data analysis section (if data and model available)
if df is not None and 'clinical_model' in locals():
    st.markdown("---")
    st.header("Dự đoán tiên lượng lâm sàng (METABRIC)")
    st.write("Chọn một bệnh nhân từ bộ dữ liệu METABRIC để dự đoán khả năng sống sót:")
    patient_ids = df["Patient ID"].unique().tolist()
    selected_id = st.selectbox("Mã bệnh nhân:", patient_ids)
    if selected_id:
        patient = df[df["Patient ID"] == selected_id].iloc[0]
        # Chuẩn bị đầu vào cho mô hình lâm sàng
        # Lấy danh sách đặc trưng từ metadata nếu có, nếu không thì lấy tất cả trừ các cột ID/đích
        if isinstance(clinical_metadata, dict) and "features" in clinical_metadata:
            feature_cols = clinical_metadata["features"]
        else:
            # Suy ra đặc trưng: bỏ các cột không sử dụng
            cols = [c for c in df.columns if c not in ["Patient ID", "Overall Survival (Months)", 
                                                       "Overall Survival Status", "Relapse Free Status (Months)", 
                                                       "Relapse Free Status", "Patient's Vital Status"]]
            feature_cols = cols
        X_input = patient[feature_cols]
        # Nếu metadata có các encoder hoặc mapping, áp dụng chúng
        if isinstance(clinical_metadata, dict):
            # Áp dụng encoder cho từng cột nếu có
            if "encoders" in clinical_metadata:
                for col, encoder in clinical_metadata["encoders"].items():
                    try:
                        # Nếu encoder có phương thức transform (LabelEncoder, OneHotEncoder, etc.)
                        X_input[col] = encoder.transform([X_input[col]])[0]
                    except Exception:
                        # Nếu encoder được lưu dạng mapping dict
                        if isinstance(encoder, dict):
                            X_input[col] = encoder.get(X_input[col], X_input[col])
            # Áp dụng scaler nếu có
            if "scaler" in clinical_metadata:
                # Đảm bảo X_input ở dạng 2D cho scaler
                X_df = pd.DataFrame([X_input.values], columns=feature_cols)
                X_scaled = clinical_metadata["scaler"].transform(X_df)
            else:
                X_scaled = np.array([X_input.values])
        else:
            # Nếu metadata không phải dict (có thể là None), dùng dữ liệu thô
            X_scaled = np.array([X_input.values])
        # Dự đoán với mô hình lâm sàng
        y_pred = clinical_model.predict(X_scaled)
        # Nếu mô hình là classifier, lấy nhãn và xác suất
        pred_label = None
        pred_prob = None
        # Kiểm tra xem model có predict_proba (phân loại)
        if hasattr(clinical_model, "predict_proba"):
            try:
                prob = clinical_model.predict_proba(X_scaled)
                pred_prob = float(np.max(prob))
            except Exception:
                pred_prob = None
        # Xác định nhãn dự đoán từ y_pred
        # Nếu có target_encoder trong metadata để chuyển nhãn số -> chuỗi
        if isinstance(clinical_metadata, dict) and "target_encoder" in clinical_metadata:
            try:
                pred_label = clinical_metadata["target_encoder"].inverse_transform(y_pred)[0]
            except Exception:
                pred_label = str(y_pred[0])
        elif isinstance(clinical_metadata, dict) and "target_map" in clinical_metadata:
            inv_map = {v: k for k, v in clinical_metadata["target_map"].items()}
            pred_label = inv_map.get(int(y_pred[0]), str(y_pred[0]))
        else:
            # Nếu không có metadata, giả sử 0 = Sống, 1 = Tử vong
            pred_label = "Living" if int(y_pred[0]) == 0 else "Died of Disease"
        actual_label = patient["Patient's Vital Status"]
        # Hiển thị kết quả
        st.subheader("Kết quả dự đoán cho bệnh nhân " + selected_id)
        result_text = f"**Dự đoán của mô hình:** {pred_label}"
        if pred_prob is not None:
            result_text += f" (xác suất {pred_prob*100:.1f}%)"
        st.write(result_text)
        st.write(f"**Tình trạng thực tế:** {actual_label}")
