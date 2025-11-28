# ==========================================
# 🩺 ỨNG DỤNG TRÍ TUỆ NHÂN TẠO HỖ TRỢ PHÂN TÍCH ẢNH SIÊU ÂM VÚ
# ==========================================
# ⚠️ Phiên bản dành cho nghiên cứu học thuật - Không sử dụng cho mục đích y tế thực tế.
# ⚠️ Ứng dụng này chỉ mang tính minh họa kỹ thuật và học thuật.

import os
import json
import tempfile
from pathlib import Path

import gdown
import numpy as np
import pandas as pd
import cv2
import streamlit as st
import altair as alt

import tensorflow as tf
import keras
from keras.models import load_model
from keras.saving import register_keras_serializable
from keras.layers import Conv2D

import joblib

import nibabel as nib
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut

# =====================================================
# ⚙️ CẤU HÌNH CHUNG
# =====================================================

st.set_page_config(
    page_title="AI Phân tích Siêu âm Vú",
    layout="wide",
    page_icon="🩺"
)

# Cho phép load model cũ (Keras < 3)
try:
    keras.config.enable_unsafe_deserialization()
except Exception:
    pass

# ============================
# 0) CUSTOM OBJECTS CHO CBAM
# ============================
@register_keras_serializable(package="cbam", name="spatial_mean")
def spatial_mean(x):
    return tf.reduce_mean(x, axis=-1, keepdims=True)

@register_keras_serializable(package="cbam", name="spatial_max")
def spatial_max(x):
    return tf.reduce_max(x, axis=-1, keepdims=True)

@register_keras_serializable(package="cbam", name="spatial_output_shape")
def spatial_output_shape(input_shape):
    try:
        shape = tf.TensorShape(input_shape).as_list()
    except Exception:
        shape = list(input_shape)
    if len(shape) == 4:
        return (shape[0], shape[1], shape[2], 1)
    if len(shape) == 3:
        return (shape[0], shape[1], 1)
    return shape

CUSTOM_OBJECTS = {
    "spatial_mean": spatial_mean,
    "spatial_max": spatial_max,
    "spatial_output_shape": spatial_output_shape,
}

# ============================
# 1) TẢI MÔ HÌNH TỪ GOOGLE DRIVE
# ============================
MODEL_DIR = "models"

drive_files = {
    # Mô hình phân loại + phân đoạn ảnh siêu âm
    "Classifier_model_2.h5": "1fXPICuTkETep2oPiA56l0uMai2GusEJH",
    "best_model_cbam_attention_unet_fixed.keras": "1axOg7N5ssJrMec97eV-JMPzID26ynzN1",

    # Mô hình lâm sàng RandomForest + metadata
    "clinical_rf_model.joblib": "1zHBB05rVUK7H9eZ9y5N9stUZnhzYBafc",
    "clinical_rf_metadata.json": "1KHZWZXs8QV8jLNXBkAVsQa_DN3tHuXtx",
}

def download_models():
    os.makedirs(MODEL_DIR, exist_ok=True)
    for fname, fid in drive_files.items():
        path = os.path.join(MODEL_DIR, fname)
        if not os.path.exists(path):
            url = f"https://drive.google.com/uc?id={fid}"
            st.info(f"📥 Đang tải mô hình: `{fname}` ...")
            gdown.download(url, path, quiet=False)
            st.success(f"✅ Đã tải xong {fname}")

# ============================
# 2) LOAD CÁC MÔ HÌNH
# ============================
@st.cache_resource
def load_all_models():
    """Load mô hình phân đoạn, phân loại và mô hình lâm sàng."""
    seg_model = load_model(
        os.path.join(MODEL_DIR, "best_model_cbam_attention_unet_fixed.keras"),
        compile=False,
        custom_objects=CUSTOM_OBJECTS,
        safe_mode=False
    )

    class_model = load_model(
        os.path.join(MODEL_DIR, "Classifier_model_2.h5"),
        compile=False
    )

    clinical_model = None
    clinical_meta = None
    try:
        clinical_model = joblib.load(os.path.join(MODEL_DIR, "clinical_rf_model.joblib"))
        with open(os.path.join(MODEL_DIR, "clinical_rf_metadata.json"), "r") as f:
            clinical_meta = json.load(f)
    except Exception as e:
        st.error(f"❌ Không thể load mô hình lâm sàng: {e}")

    return seg_model, class_model, clinical_model, clinical_meta

# ============================
# 3) HÀM XỬ LÝ ẢNH CƠ BẢN
# ============================
def get_input_hwc(model):
    """Lấy kích thước (H, W, C) của input model Keras."""
    shape = model.input_shape
    if isinstance(shape, list):
        shape = shape[0]
    _, H, W, C = shape
    return int(H), int(W), int(C)

def prep(gray, target_shape):
    """Resize & chuẩn hóa ảnh xám theo kích thước model."""
    H, W, C = target_shape
    resized = cv2.resize(gray, (W, H))
    if C == 1:
        x = resized.astype(np.float32) / 255.0
        x = np.expand_dims(x, (0, -1))
    else:
        x = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB).astype(np.float32) / 255.0
        x = np.expand_dims(x, 0)
    return x, resized

COLOR_B = np.array([0, 255, 0], np.float32)   # Lành: xanh lá
COLOR_M = np.array([255, 0, 0], np.float32)   # Ác: đỏ
COLOR_G = (0, 255, 255)                       # Viền tổng: vàng

def overlay(gray, mask, alpha=0.6):
    """Vẽ lớp mask (1: lành, 2: ác) chồng lên ảnh xám."""
    base = np.stack([gray]*3, axis=-1).astype(np.float32)
    out = base.copy()

    ben = mask == 1
    mal = mask == 2

    if ben.any():
        out[ben] = (1 - alpha) * out[ben] + alpha * COLOR_B
    if mal.any():
        out[mal] = (1 - alpha) * out[mal] + alpha * COLOR_M

    general = ((ben | mal) * 255).astype(np.uint8)
    if general.any():
        ct, _ = cv2.findContours(general, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        out_uint8 = out.clip(0, 255).astype(np.uint8)
        cv2.drawContours(out_uint8, ct, -1, COLOR_G, 2)
        return out_uint8

    return out.clip(0, 255).astype(np.uint8)

# ============================
# 3.x) HÀM GRAD-CAM CHO MÔ HÌNH PHÂN LOẠI
# ============================
def get_last_conv_layer_name(model):
    """
    Tìm tên lớp Conv2D cuối cùng trong mô hình Keras.
    Dùng để làm Grad-CAM.
    """
    for layer in reversed(model.layers):
        if isinstance(layer, Conv2D):
            return layer.name
    raise ValueError("Không tìm thấy lớp Conv2D nào trong mô hình để làm Grad-CAM.")

def make_gradcam_heatmap(img_array,
                         model,
                         last_conv_layer_name,
                         class_index=None):
    """
    img_array: (1, H, W, C) – đầu vào đã preprocess (chính là x_clf).
    model: class_model (Classifier_model_2.h5)
    last_conv_layer_name: tên lớp conv cuối (tìm bằng get_last_conv_layer_name)
    class_index: chỉ số lớp cần Grad-CAM (0/1/2).
                 Nếu None → dùng lớp có xác suất cao nhất.
    """
    last_conv_layer = model.get_layer(last_conv_layer_name)
    grad_model = keras.Model(
        [model.inputs],
        [last_conv_layer.output, model.output],
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)

        # Nếu là list/tuple thì lấy phần tử đầu
        if isinstance(conv_outputs, (list, tuple)):
            conv_outputs = conv_outputs[0]
        if isinstance(predictions, (list, tuple)):
            predictions = predictions[0]

        if class_index is None:
            class_index = tf.argmax(predictions[0])

        class_channel = predictions[:, class_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]  # (H, W, C)
    heatmap = tf.reduce_mean(conv_outputs * pooled_grads, axis=-1)  # (H, W)

    heatmap = tf.nn.relu(heatmap)
    heatmap = heatmap / (tf.reduce_max(heatmap) + 1e-8)

    return heatmap.numpy()

def mask_heatmap_with_segmentation(heatmap, mask_resized):
    """
    Chỉ giữ Grad-CAM trên vùng có khối u (mask == 1 hoặc 2).

    heatmap: (Hc, Wc) – kích thước feature map (nhỏ)
    mask_resized: (H, W) – cùng kích thước với ảnh classifier (g_clf)
    => Hàm sẽ resize heatmap về (H, W) rồi mới mask.
    """
    # Đảm bảo heatmap là 2D
    heatmap = np.squeeze(heatmap)
    if heatmap.ndim == 3:
        # nếu lỡ là (H, W, 1) thì lấy kênh đầu
        heatmap = heatmap[..., 0]

    H, W = mask_resized.shape[:2]

    # Resize heatmap về đúng kích thước mask
    heatmap_resized = cv2.resize(heatmap, (W, H))
    heatmap_resized = heatmap_resized.astype(np.float32)

    # Tạo mask vùng tổn thương
    lesion = (mask_resized == 1) | (mask_resized == 2)

    masked = np.zeros_like(heatmap_resized, dtype=np.float32)
    masked[lesion] = heatmap_resized[lesion]

    if masked.max() > 0:
        masked /= masked.max()

    return masked


def apply_gradcam_on_gray(gray, heatmap, alpha=0.6, gamma=0.7, thresh=0.25):
    """
    gray: ảnh xám đã resize (g_clf)
    heatmap: (H, W) 0–1 (đã mask theo vùng u)
    alpha: độ đậm overlay
    gamma: <1 → tăng tương phản vùng nóng
    thresh: dưới ngưỡng coi như 0 để nền ít bị nhuộm màu
    """
    h, w = gray.shape[:2]
    heatmap_resized = cv2.resize(heatmap, (w, h))

    heatmap_resized = np.clip(heatmap_resized, 0.0, 1.0).astype(np.float32)

    # Tăng tương phản
    heatmap_gamma = np.power(heatmap_resized, gamma)

    # Cắt ngưỡng
    heatmap_gamma[heatmap_gamma < thresh] = 0.0

    # Làm mượt nhẹ
    heatmap_blur = cv2.GaussianBlur(heatmap_gamma, (5, 5), 0)

    heatmap_uint8 = np.uint8(255 * heatmap_blur)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

    base = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    cam = cv2.addWeighted(heatmap_color, alpha, base, 1 - alpha, 0)
    return cam

def overlay_mask_contour_on_color(color_img, mask, contour_color=(0, 255, 255), thickness=2):
    """
    color_img: ảnh màu (BGR) – ví dụ Grad-CAM đã tô màu.
    mask: (H, W), giá trị 0/1/2/... (1: lành, 2: ác).
    """
    h, w = mask.shape
    general = ((mask == 1) | (mask == 2)) * 255  # vùng có u
    general = general.astype(np.uint8)

    out = color_img.copy()
    if general.any():
        ct, _ = cv2.findContours(general, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, ct, -1, contour_color, thickness)

    return out

# ============================
# 3.y) HỖ TRỢ ĐỌC ẢNH 3D / DICOM
# ============================
def load_nifti_slice(file, slice_strategy="middle"):
    img = nib.load(file)
    vol = img.get_fdata()
    mid = vol.shape[2] // 2
    if slice_strategy == "middle":
        slice_img = vol[:, :, mid]
    elif slice_strategy == "max_std":
        idx = np.argmax([np.std(vol[:, :, i]) for i in range(vol.shape[2])])
        slice_img = vol[:, :, idx]
    return slice_img.astype(np.uint8)

def load_dicom_slice(file):
    ds = pydicom.dcmread(file)
    arr = apply_voi_lut(ds.pixel_array, ds)
    arr = arr.astype(np.float32)
    arr = (arr - arr.min()) / (arr.max() - arr.min()) * 255
    return arr.astype(np.uint8)

def load_3d_slice(upload):
    suffix = Path(upload.name).suffix.lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(upload.read())
        tmp_path = tmp.name
    try:
        if suffix in [".nii", ".gz"]:
            return load_nifti_slice(tmp_path), "3D"
        elif suffix == ".dcm":
            return load_dicom_slice(tmp_path), "DICOM"
        else:
            st.error("❌ Định dạng ảnh 3D chưa hỗ trợ đọc.")
            return None, None
    except Exception as e:
        st.error(f"❌ Không thể đọc ảnh: {e}")
        return None, None

# =====================================================
# 4) SIDEBAR & CHỌN TRANG
# =====================================================
st.sidebar.title("📘 Danh mục")
chon_trang = st.sidebar.selectbox(
    "Chọn nội dung hiển thị",
    ["Ứng dụng", "Giới thiệu", "Nguồn dữ liệu & Bản quyền"]
)

# =====================================================
# 5) TRANG 2: GIỚI THIỆU
# =====================================================
if chon_trang == "Giới thiệu":
    st.title("👩‍⚕️ ỨNG DỤNG AI HỖ TRỢ PHÂN TÍCH ẢNH SIÊU ÂM VÚ")

    st.markdown("""
### 🎯 Mục tiêu

Ứng dụng này được xây dựng với mục đích **nghiên cứu học thuật** trong lĩnh vực:

- Trí tuệ nhân tạo (AI)  
- Học sâu (Deep Learning)  
- Y học hình ảnh (Medical Imaging)  

Cụ thể, ứng dụng minh họa cách:
- Phân đoạn khối u trên **ảnh siêu âm tuyến vú** bằng mạng U-Net có cơ chế chú ý (CBAM).
- Phân loại khối u thành **lành tính / ác tính / bình thường**.
- Kết hợp thêm mô hình **dữ liệu lâm sàng** (RandomForest) để **hỗ trợ đánh giá nguy cơ**.
- Đưa ra **nhận định tổng hợp** từ cả hai mô hình (hình ảnh + lâm sàng).

---

### ⚠️ Lưu ý quan trọng

- Đây **không phải** là công cụ chẩn đoán y khoa thực tế.  
- Kết quả từ mô hình chỉ mang tính **minh họa kỹ thuật** và **hỗ trợ học thuật**.  
- **Tuyệt đối không** sử dụng kết quả từ ứng dụng này để:
  - Tự chẩn đoán bệnh.
  - Tự ý điều trị.
  - Thay thế ý kiến hay chỉ định của bác sĩ chuyên khoa.

---

### 🧪 Đối tượng sử dụng

- Học sinh, sinh viên, nhà nghiên cứu quan tâm đến AI trong y tế.  
- Người muốn tìm hiểu quy trình: **tiền xử lý ảnh → mô hình AI → diễn giải kết quả**.  

---

📌 **Tóm lại:**  
Ứng dụng này là một **mô hình nghiên cứu** (proof-of-concept) về AI trong chẩn đoán hình ảnh, không phải sản phẩm y tế lâm sàng.
""")

# =====================================================
# 6) TRANG 3: NGUỒN DỮ LIỆU & BẢN QUYỀN
# =====================================================
elif chon_trang == "Nguồn dữ liệu & Bản quyền":
    st.title("📊 Nguồn dữ liệu và bản quyền sử dụng")

    st.markdown("""
Ứng dụng sử dụng dữ liệu từ **các nguồn công khai** phục vụ mục đích **nghiên cứu phi thương mại**:

| Nguồn dữ liệu | Loại dữ liệu | Liên kết |
|---------------|-------------|---------|
| **BUSI – Breast Ultrasound Images Dataset** (Arya Shah, Kaggle) | Ảnh siêu âm tuyến vú | [Mở liên kết](https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset) |
| **BUS-UCLM Breast Ultrasound Dataset** (Orvile, Kaggle) | Ảnh siêu âm tuyến vú | [Mở liên kết](https://www.kaggle.com/datasets/orvile/bus-uclm-breast-ultrasound-dataset) |
| **Breast Lesions USG (TCIA)** | Ảnh siêu âm tổn thương vú | [Mở liên kết](https://www.cancerimagingarchive.net/collection/breast-lesions-usg/) |
| **Breast Cancer Clinical Data** (Mendeley Data) | Dữ liệu lâm sàng ung thư vú | [Mở liên kết](https://data.mendeley.com/datasets/dbz42w9x8h/2) |

---

### 📄 Giấy phép & phạm vi sử dụng

- Dữ liệu được sử dụng trong ứng dụng này **chỉ nhằm mục đích nghiên cứu khoa học và giáo dục**.
- Không sử dụng cho:
  - Mục đích thương mại.
  - Các hệ thống chẩn đoán y tế triển khai thực tế.
- Khi trích dẫn hoặc sử dụng lại dữ liệu, cần tuân thủ:
  - Điều khoản giấy phép ghi rõ trên từng trang dataset.
  - Trích dẫn tác giả/bộ sưu tập dữ liệu gốc.

---

### 📚 Gợi ý trích dẫn (APA, tham khảo)

- Shah, A. (2020). *Breast Ultrasound Images Dataset* [Dataset]. Kaggle.  
- Orvile. (2023). *BUS-UCLM Breast Ultrasound Dataset* [Dataset]. Kaggle.  
- The Cancer Imaging Archive. (2021). *Breast Lesions USG* [Dataset].  
- Mendeley Data (n.d.). *Breast Cancer Clinical Data* [Dataset]. Mendeley.  

---

🧾 **Tuyên bố bản quyền & miễn trừ trách nhiệm:**  
- Ứng dụng này không sở hữu bản quyền dữ liệu gốc, chỉ sử dụng lại theo đúng giấy phép của tác giả.  
- Tác giả ứng dụng **không chịu trách nhiệm** cho bất kỳ việc sử dụng sai mục đích nào từ phía người dùng.
""")

# =====================================================
# 7) TRANG 1: ỨNG DỤNG CHÍNH (ẢNH + LÂM SÀNG)
# =====================================================
elif chon_trang == "Ứng dụng":
    st.title("🩺 ỨNG DỤNG AI MINH HỌA PHÂN TÍCH SIÊU ÂM VÚ")
    st.markdown("""
Ứng dụng cho phép:
1. 📷 Tải lên **ảnh siêu âm tuyến vú** để mô hình:
   - Phân đoạn vùng nghi ngờ.
   - Phân loại: **Lành tính / Ác tính / Bình thường**.
2. 📊 Nhập **thông tin lâm sàng cơ bản** để mô hình RandomForest dự đoán **kết cục sống còn**.
3. 🧠 Xem **đánh giá tổng hợp** được kết hợp từ cả hai mô hình.

> ⚠️ Kết quả chỉ mang tính **minh họa học thuật**, không sử dụng cho chẩn đoán y khoa thực tế.
""")

    # Tải & load mô hình
    with st.spinner("🔧 Đang chuẩn bị mô hình..."):
        download_models()
        seg_model, class_model, clinical_model, clinical_meta = load_all_models()

    if clinical_model is None or clinical_meta is None:
        st.error("❌ Không thể tải đầy đủ mô hình lâm sàng. Vui lòng kiểm tra lại file mô hình.")
    
    # Biến lưu kết quả để dùng cho phần kết hợp
    image_pred_label_en = None
    image_pred_label_vi = None
    image_pred_probs = None
    clinical_pred_label = None
    clinical_prob_death = None

    labels_clf = ["benign", "malignant", "normal"]
    vi_map = {"benign": "U lành tính", "malignant": "U ác tính", "normal": "Bình thường"}

    # ---------------------------------------------
    # 7.1 PHÂN TÍCH ẢNH SIÊU ÂM (2D / 3D / DICOM + GRAD-CAM)
    # ---------------------------------------------
    upload = st.file_uploader(
        "📤 Chọn ảnh siêu âm (PNG/JPG hoặc NIfTI .nii/.gz hoặc DICOM .dcm)",
        ["png", "jpg", "jpeg", "nii", "nii.gz", "dcm"]
    )

    if upload:
        suffix = Path(upload.name).suffix.lower()
        if suffix in [".png", ".jpg", ".jpeg"]:
            arr = np.frombuffer(upload.read(), np.uint8)
            gray = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
            is_3d = False
        elif suffix in [".nii", ".gz", ".dcm"]:
            gray, dim = load_3d_slice(upload)
            is_3d = True
        else:
            st.error("❌ Định dạng ảnh không được hỗ trợ.")
            gray = None
            is_3d = False

        if gray is not None:
            st.info(f"📁 Hệ thống phát hiện ảnh {'3D' if is_3d else '2D'} – đang xử lý...")
            gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)

            # Chuẩn bị đầu vào cho segmentation và classification
            x_seg, g_seg = prep(gray, get_input_hwc(seg_model))
            x_clf, g_clf = prep(gray, get_input_hwc(class_model))

            # Dự đoán segmentation
            seg_pred = seg_model.predict(x_seg, verbose=0)[0]
            mask = np.argmax(seg_pred, -1).astype(np.uint8)
            overlay_img = overlay(g_seg, mask)

            # Dự đoán classification
            probs = class_model.predict(x_clf, verbose=0)[0]
            idx = int(np.argmax(probs))

            image_pred_label_en = labels_clf[idx]
            image_pred_label_vi = vi_map[image_pred_label_en]
            image_pred_probs = probs

            # ============================
            # 🔥 TÍNH GRAD-CAM ĐẸP HƠN
            # ============================
            gradcam_img = None
            gradcam_with_mask = None
            try:
                # 1) Tên lớp Conv2D cuối
                last_conv_name = get_last_conv_layer_name(class_model)

                # 2) Chọn lớp cần Grad-CAM (ác tính)
                class_idx_for_cam = labels_clf.index("malignant")
                # hoặc: class_idx_for_cam = idx

                # 3) Tính heatmap gốc
                heatmap_raw = make_gradcam_heatmap(
                    img_array=x_clf,
                    model=class_model,
                    last_conv_layer_name=last_conv_name,
                    class_index=class_idx_for_cam,
                )

                # 4) Resize mask về kích thước classifier
                mask_resized = cv2.resize(
                    mask,
                    (g_clf.shape[1], g_clf.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )

                # 5) chỉ giữ Grad-CAM trong vùng u
                heatmap_masked = mask_heatmap_with_segmentation(heatmap_raw, mask_resized)

                # 6) Overlay Grad-CAM (vùng nóng trong u)
                gradcam_img = apply_gradcam_on_gray(
                    g_clf,
                    heatmap_masked,
                    alpha=0.6,
                    gamma=0.7,
                    thresh=0.25,
                )

                # 7) Thêm contour khối u cho dễ nhìn
                gradcam_with_mask = overlay_mask_contour_on_color(
                    gradcam_img,
                    mask_resized,
                    contour_color=(0, 255, 255),
                    thickness=2,
                )

            except Exception as e:
                st.warning(f"⚠️ Không thể tạo Grad-CAM: {e}")

            # Hiển thị ảnh
            col1, col2, col3 = st.columns(3)
            with col1:
                st.image(g_clf, caption="Ảnh đầu vào (chuẩn hóa)", use_column_width=True)
            with col2:
                st.image(overlay_img, caption="Kết quả phân đoạn", use_column_width=True)
            with col3:
                if gradcam_with_mask is not None:
                    st.image(
                        gradcam_with_mask,
                        caption="Grad-CAM (lớp ác tính) + contour khối u",
                        use_column_width=True,
                    )
                elif gradcam_img is not None:
                    st.image(
                        gradcam_img,
                        caption="Grad-CAM (lớp ác tính)",
                        use_column_width=True,
                    )
                else:
                    st.info("Chưa tạo được Grad-CAM cho ảnh này.")

            # Thông tin phân loại
            st.success(f"🔍 Mô hình hình ảnh dự đoán: **{image_pred_label_vi}** ({probs[idx]*100:.1f}%)")

            df_img = pd.DataFrame({
                "Nhóm": ["Lành tính", "Ác tính", "Bình thường"],
                "Xác suất (%)": (probs * 100).round(2)
            })

            st.altair_chart(
                alt.Chart(df_img).mark_bar().encode(
                    x="Nhóm",
                    y="Xác suất (%)",
                    tooltip=["Nhóm", "Xác suất (%)"],
                ),
                use_container_width=True,
            )
    else:
        st.info("👆 Hãy tải lên một ảnh siêu âm để mô hình tiến hành minh họa.")

        # ---------------------------------------------
        # 7.2 MÔ HÌNH LÂM SÀNG (RANDOMFOREST)
        # ---------------------------------------------
        st.subheader("📊 Thông tin lâm sàng (minh họa)")

        if clinical_model is None or clinical_meta is None:
            st.warning("Không có mô hình lâm sàng khả dụng, bỏ qua phần này.")
        else:
            feature_names = clinical_model.feature_names_in_
            label_map = clinical_meta["label_map"]
            inv_label = {v: k for k, v in label_map.items()}

            with st.form("clinical_form"):
                col_a, col_b, col_c = st.columns(3)

                with col_a:
                    age = st.number_input("Tuổi tại chẩn đoán (Age at Diagnosis)", 0, 120, 50)
                    size = st.number_input("Kích thước khối u (Tumor Size, mm)", 0, 200, 20)
                    lymph = st.number_input("Số hạch dương tính (Lymph nodes examined positive)", 0, 50, 0)
                    mut = st.number_input("Số lượng đột biến (Mutation Count)", 0, 10000, 0)
                    npi = st.number_input("Chỉ số Nottingham (NPI)", 0.0, 10.0, 4.0)
                    os_m = st.number_input("Thời gian sống toàn bộ (Overall Survival, tháng)", 0.0, 300.0, 60.0)

                with col_b:
                    sx = st.selectbox("Loại phẫu thuật vú (Type of Breast Surgery)",
                                      ["Breast Conserving", "Mastectomy"])
                    grade = st.selectbox("Độ mô học (Neoplasm Histologic Grade)", [1, 2, 3])
                    stage = st.selectbox("Giai đoạn u (Tumor Stage)", [1, 2, 3, 4])
                    sex = st.selectbox("Giới tính (Sex)", ["Female", "Male"])
                    cell = st.selectbox("Cellularity", ["High", "Low", "Moderate"])
                    chemo = st.selectbox("Hóa trị (Chemotherapy)", ["No", "Yes"])
                    hormone = st.selectbox("Liệu pháp nội tiết (Hormone Therapy)", ["No", "Yes"])

                with col_c:
                    radio = st.selectbox("Xạ trị (Radio Therapy)", ["No", "Yes"])
                    er = st.selectbox("ER Status", ["Negative", "Positive"])
                    pr = st.selectbox("PR Status", ["Negative", "Positive"])
                    her2 = st.selectbox("HER2 Status", ["Negative", "Positive"])
                    gene = st.selectbox(
                        "3-Gene classifier subtype",
                        [
                            "ER+/HER2+",
                            "ER+/HER2- High Prolif",
                            "ER+/HER2- Low Prolif",
                            "ER-/HER2+",
                            "ER-/HER2-",
                        ],
                    )
                    pam50 = st.selectbox(
                        "Pam50 + Claudin-low subtype",
                        ["Basal-like", "Claudin-low", "HER2-enriched",
                         "Luminal A", "Luminal B", "Normal-like"],
                    )
                    relapse = st.selectbox("Trạng thái tái phát (Relapse Free Status)",
                                           ["Not Recurred", "Recurred"])

                    submit_clinical = st.form_submit_button("🔮 Dự đoán từ mô hình lâm sàng")

            if submit_clinical:
                row = {
                    "Age at Diagnosis": age,
                    "Tumor Size": size,
                    "Lymph nodes examined positive": lymph,
                    "Mutation Count": mut,
                    "Nottingham prognostic index": npi,
                    "Overall Survival (Months)": os_m,
                    "Type of Breast Surgery": sx,
                    "Neoplasm Histologic Grade": grade,
                    "Tumor Stage": stage,
                    "Sex": sex,
                    "Cellularity": cell,
                    "Chemotherapy": chemo,
                    "Hormone Therapy": hormone,
                    "Radio Therapy": radio,
                    "ER Status": er,
                    "PR Status": pr,
                    "HER2 Status": her2,
                    "3-Gene classifier subtype": gene,
                    "Pam50 + Claudin-low subtype": pam50,
                    "Relapse Free Status": relapse,
                }

                X = pd.DataFrame([row], columns=feature_names)

                y = int(clinical_model.predict(X)[0])
                pred_label = inv_label[y]
                clinical_pred_label = pred_label

                if "Deceased" in label_map:
                    prob_death = float(
                        clinical_model.predict_proba(X)[0][label_map["Deceased"]]
                    )
                else:
                    prob_death = float(np.max(clinical_model.predict_proba(X)[0]))
                clinical_prob_death = prob_death

                if pred_label == "Deceased":
                    st.error(f"🧬 Mô hình lâm sàng dự đoán kết cục: **{pred_label}**")
                else:
                    st.success(f"🧬 Mô hình lâm sàng dự đoán kết cục: **{pred_label}**")

                st.write(f"📈 Xác suất tử vong ước tính: **{prob_death*100:.1f}%**")

        # ---------------------------------------------
        # 7.3 ĐÁNH GIÁ TỔNG HỢP (ẢNH + LÂM SÀNG)
        # ---------------------------------------------
        st.markdown("---")
        st.subheader("🧠 Đánh giá tổng hợp từ hai mô hình")

        if (image_pred_probs is None) and (clinical_prob_death is None):
            st.info("Khi có cả **kết quả mô hình hình ảnh** và **kết quả mô hình lâm sàng**, "
                    "hệ thống sẽ hiển thị đánh giá tổng hợp tại đây.")
        else:
            if image_pred_probs is not None:
                p_malignant = float(image_pred_probs[labels_clf.index("malignant")])
                st.write("🔬 **Nhận định từ mô hình hình ảnh:**")
                st.write(
                    f"- Kết luận: **{image_pred_label_vi}** "
                    f"(xác suất ác tính ≈ {p_malignant*100:.1f}%)."
                )
            else:
                p_malignant = None

            if clinical_prob_death is not None:
                st.write("📋 **Nhận định từ mô hình lâm sàng:**")
                st.write(
                    f"- Kết cục dự đoán: **{clinical_pred_label}** "
                    f"(xác suất tử vong ≈ {clinical_prob_death*100:.1f}%)."
                )
            else:
                clinical_prob_death = None

            if (p_malignant is not None) and (clinical_prob_death is not None):
                combined_risk = 0.6 * p_malignant + 0.4 * clinical_prob_death

                if combined_risk < 0.3:
                    risk_group = "Nguy cơ thấp"
                elif combined_risk < 0.6:
                    risk_group = "Nguy cơ trung bình"
                else:
                    risk_group = "Nguy cơ cao"

                st.write("📎 **Chỉ số nguy cơ kết hợp (minh họa):**")
                st.write(
                    f"- Điểm nguy cơ ≈ **{combined_risk*100:.1f}%** → Nhóm: **{risk_group}**."
                )

                if risk_group == "Nguy cơ cao":
                    st.error(
                        "📌 Đánh giá tổng hợp: mô hình gợi ý **nguy cơ cao**. "
                        "Cần được bác sĩ chuyên khoa thăm khám và đánh giá trực tiếp."
                    )
                elif risk_group == "Nguy cơ trung bình":
                    st.warning(
                        "📌 Đánh giá tổng hợp: mô hình gợi ý **nguy cơ trung bình**. "
                        "Cần theo dõi sát, kết hợp thêm xét nghiệm và chẩn đoán hình ảnh khác."
                    )
                else:
                    st.success(
                        "📌 Đánh giá tổng hợp: mô hình gợi ý **nguy cơ thấp**. "
                        "Tuy nhiên, bệnh nhân vẫn cần tầm soát và khám định kỳ theo khuyến cáo."
                    )

                st.caption(
                    "⚠️ Lưu ý: Chỉ số nguy cơ kết hợp trên chỉ là **heuristic minh họa**, "
                    "chưa được hiệu chỉnh trên dữ liệu lâm sàng thật. "
                    "Không dùng để tự chẩn đoán hoặc thay thế ý kiến bác sĩ."
                )
            else:
                st.info("Cần có đủ cả **kết quả hình ảnh** và **kết quả lâm sàng** "
                        "để tính toán chỉ số nguy cơ kết hợp.")

# =====================================================
# 8) CHÂN TRANG (FOOTER)
# =====================================================
st.markdown("""
---
📘 **Tuyên bố miễn trừ trách nhiệm:**  
Ứng dụng này được phát triển phục vụ mục đích **nghiên cứu khoa học và giáo dục**.  
Không sử dụng cho **chẩn đoán, điều trị hoặc tư vấn y tế**.  

© 2025 – Dự án AI Siêu âm Vú.  
Tác giả minh họa: Lê Vũ Anh Tin – Trường THPT Chuyên Nguyễn Du.
""")
