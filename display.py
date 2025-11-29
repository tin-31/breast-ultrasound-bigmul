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

    # Mô hình lâm sàng METABRIC
    "model_cox.joblib": "1XtaTE_AjMAnNv5pO_u5Z3xC1PE_oYETq",
    "model_logistic.joblib": "1zdcXp1IvGXQT87XBTLUvyV0wmQFVFI4d",
    "model_xgb_recur.joblib": "1n_ntNn9qORqA0nZBbMNFOjOZVW9kaJfT",
    "model_rf_stage.joblib": "15A-fB9z2eUmKcg_UDqq8Zd1ttpTfMUY4",
    "model_xgb_stage.joblib": "19iu9b94IaLnXZyBiEidk0FNR4lthMChO",
    "preprocess.joblib": "1KU9NkpwCDvbTrOBONGQHjt2TzouCPfAv",
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
    """
    Load mô hình phân đoạn, phân loại và các mô hình lâm sàng METABRIC.
    clinical_models: dict gồm {cox, logistic, xgb_recur, rf_stage, xgb_stage}
    preprocess: thông tin tiền xử lý (features, encoders,...)
    """
    # Ảnh
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

    # Lâm sàng
    clinical_models = {}
    preprocess = None

    try:
        preprocess = joblib.load(os.path.join(MODEL_DIR, "preprocess.joblib"))

        clinical_models["cox"] = joblib.load(os.path.join(MODEL_DIR, "model_cox.joblib"))
        clinical_models["logistic"] = joblib.load(os.path.join(MODEL_DIR, "model_logistic.joblib"))
        clinical_models["xgb_recur"] = joblib.load(os.path.join(MODEL_DIR, "model_xgb_recur.joblib"))
        clinical_models["rf_stage"] = joblib.load(os.path.join(MODEL_DIR, "model_rf_stage.joblib"))
        clinical_models["xgb_stage"] = joblib.load(os.path.join(MODEL_DIR, "model_xgb_stage.joblib"))

    except Exception as e:
        st.error(f"❌ Không thể load đầy đủ mô hình lâm sàng METABRIC: {e}")

    return seg_model, class_model, clinical_models, preprocess

# ============================
# 3) HÀM XỬ LÝ ẢNH
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

# --- Hàm hỗ trợ đọc NIfTI ---
def load_nifti_slice(file, slice_strategy="middle"):
    img = nib.load(file)
    vol = img.get_fdata()
    mid = vol.shape[2] // 2
    if slice_strategy == "middle":
        slice_img = vol[:, :, mid]
    elif slice_strategy == "max_std":
        idx = np.argmax([np.std(vol[:, :, i]) for i in range(vol.shape[2])])
        slice_img = vol[:, :, idx]
    else:
        slice_img = vol[:, :, mid]
    return slice_img.astype(np.uint8)

# --- Hàm hỗ trợ đọc DICOM ---
def load_dicom_slice(file):
    ds = pydicom.dcmread(file)
    arr = apply_voi_lut(ds.pixel_array, ds)
    arr = arr.astype(np.float32)
    arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8) * 255
    return arr.astype(np.uint8)

# --- Tự động đọc ảnh 3D từ .nii/.gz hoặc DICOM .dcm ---
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
- Kết hợp thêm mô hình **dữ liệu lâm sàng** (METABRIC) để **hỗ trợ đánh giá nguy cơ tái phát và sống còn**.
- Đưa ra **nhận định tổng hợp** từ cả hai mô hình (hình ảnh + lâm sàng).

---

### ⚠️ Lưu ý quan trọng

- Đây **không phải** là công cụ chẩn đoán y khoa thực tế.  
- Kết quả từ mô hình chỉ mang tính **minh họa kỹ thuật** và **hỗ trợ học thuật**.  
- **Tuyệt đối không** sử dụng kết quả từ ứng dụng này để:
  - Tự chẩn đoán bệnh.
  - Tự ý điều trị.
  - Thay thế ý kiến hay chỉ định của bác sĩ chuyên khoa.
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
| **Breast Cancer Clinical Data / METABRIC** | Dữ liệu lâm sàng ung thư vú | Các kho dữ liệu công khai (TCGA, METABRIC, Mendeley, v.v.) |

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
2. 📊 Nhập **thông tin lâm sàng cơ bản** để mô hình METABRIC dự đoán:
   - Nguy cơ tử vong tương đối (Cox – risk score).
   - Xác suất tái phát (XGBoost).
   - Giai đoạn u dự đoán (RandomForest).
3. 🧠 Xem **đánh giá tổng hợp** kết hợp từ cả hai mô hình.

> ⚠️ Kết quả chỉ mang tính **minh họa học thuật**, không sử dụng cho chẩn đoán y khoa thực tế.
""")

    # Tải & load mô hình
    with st.spinner("🔧 Đang chuẩn bị mô hình..."):
        download_models()
        seg_model, class_model, clinical_models, preprocess = load_all_models()

    if not clinical_models or preprocess is None:
        st.warning("⚠️ Không tải được đầy đủ mô hình lâm sàng METABRIC. Chỉ sử dụng được phần hình ảnh.")

    # Biến lưu kết quả để dùng cho phần kết hợp
    image_pred_label_en = None
    image_pred_label_vi = None
    image_pred_probs = None

    clinical_risk_score = None
    clinical_prob_recur = None
    clinical_stage_pred = None

    labels_clf = ["benign", "malignant", "normal"]
    vi_map = {"benign": "U lành tính", "malignant": "U ác tính", "normal": "Bình thường"}

    # ---------------------------------------------
    # 7.1 PHÂN TÍCH ẢNH SIÊU ÂM (2D / 3D / DICOM)
    # ---------------------------------------------
    st.subheader("📷 Phân tích ảnh siêu âm vú")

    upload = st.file_uploader(
        "📤 Chọn ảnh siêu âm (PNG/JPG hoặc NIfTI .nii/.gz hoặc DICOM .dcm)",
        ["png", "jpg", "jpeg", "nii", "nii.gz", "dcm"]
    )

    if upload is not None:
        suffix = Path(upload.name).suffix.lower()
        if suffix in [".png", ".jpg", ".jpeg"]:
            arr = np.frombuffer(upload.read(), np.uint8)
            gray = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
            is_3d = False
        else:
            gray, dim = load_3d_slice(upload)
            is_3d = True

        if gray is not None:
            st.info(f"📁 Hệ thống phát hiện ảnh {'3D' if is_3d else '2D'} – đang xử lý...")
            gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)

            x_seg, g_seg = prep(gray, get_input_hwc(seg_model))
            x_clf, g_clf = prep(gray, get_input_hwc(class_model))

            seg_pred = seg_model.predict(x_seg, verbose=0)[0]
            mask = np.argmax(seg_pred, -1).astype(np.uint8)
            overlay_img = overlay(g_seg, mask)

            probs = class_model.predict(x_clf, verbose=0)[0]
            idx = int(np.argmax(probs))

            image_pred_label_en = labels_clf[idx]
            image_pred_label_vi = vi_map[image_pred_label_en]
            image_pred_probs = probs

            col1, col2 = st.columns(2)
            with col1:
                st.image(g_clf, caption="Ảnh đầu vào (chuẩn hóa)", use_column_width=True)
            with col2:
                st.image(overlay_img, caption="Kết quả phân đoạn", use_column_width=True)

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
    # 7.2 MÔ HÌNH LÂM SÀNG METABRIC
    # ---------------------------------------------
    st.subheader("📊 Thông tin lâm sàng (dựa trên mô hình METABRIC)")

    if not clinical_models or preprocess is None:
        st.warning("Không có mô hình lâm sàng METABRIC khả dụng, bỏ qua phần này.")
    else:
        num_features = preprocess["num_features"]
        cat_features = preprocess["cat_features"]
        encoders = preprocess["encoders"]
        stage_encoder = preprocess["stage_encoder"]
        features = preprocess["features"]

        with st.form("clinical_form_metabric"):
            col_a, col_b = st.columns(2)

            with col_a:
                age = st.number_input("Tuổi tại chẩn đoán (Age at Diagnosis)", 18, 100, 50)
                size = st.number_input("Kích thước u (Tumor Size, mm)", 0, 200, 20)
                lymph = st.number_input("Số hạch dương tính (Lymph nodes examined positive)", 0, 50, 0)
                npi = st.number_input("Nottingham prognostic index", 0.0, 10.0, 4.0)

            with col_b:
                er = st.selectbox("ER Status", ["Negative", "Positive"])
                pr = st.selectbox("PR Status", ["Negative", "Positive"])
                her2 = st.selectbox("HER2 Status", ["Negative", "Positive"])

            submit_clinical = st.form_submit_button("🔮 Dự đoán từ mô hình lâm sàng METABRIC")

        if submit_clinical:
            # Tạo row đúng tên biến
            row = {
                "Age at Diagnosis": age,
                "Tumor Size": size,
                "Lymph nodes examined positive": lymph,
                "Nottingham prognostic index": npi,
                "ER Status": er,
                "PR Status": pr,
                "HER2 Status": her2,
            }

            X = pd.DataFrame([row])

            # Áp encoder cho biến phân loại
            for col in cat_features:
                le = encoders[col]
                X[col] = le.transform(X[col].astype(str))

            # Đảm bảo đúng thứ tự features
            X = X[features]

            # 1) Cox – risk score
            try:
                risk = float(clinical_models["cox"].predict_partial_hazard(X)[0])
                clinical_risk_score = risk
                st.info(f"🕒 Mô hình sống còn (Cox) – risk score ≈ **{risk:.3f}** "
                        "(>1 nghĩa là nguy cơ cao hơn trung vị trong tập METABRIC).")
            except Exception as e:
                st.warning(f"Không tính được risk Cox: {e}")

            # 2) XGBoost – Recurrence
            try:
                prob_rec = float(clinical_models["xgb_recur"].predict_proba(X)[0, 1])
                clinical_prob_recur = prob_rec
                st.info(f"🔁 Xác suất tái phát (XGBoost) ≈ **{prob_rec*100:.1f}%**")
            except Exception as e:
                st.warning(f"Không tính được xác suất tái phát: {e}")

            # 3) RandomForest – Stage
            try:
                if stage_encoder is not None and clinical_models["rf_stage"] is not None:
                    code = int(clinical_models["rf_stage"].predict(X)[0])
                    label = stage_encoder.inverse_transform([code])[0]
                    clinical_stage_pred = label
                    st.info(f"📌 Giai đoạn u dự đoán (RF) trên dữ liệu METABRIC: **{label}**")
            except Exception as e:
                st.warning(f"Không dự đoán được giai đoạn: {e}")

    # ---------------------------------------------
    # 7.3 ĐÁNH GIÁ TỔNG HỢP (ẢNH + LÂM SÀNG)
    # ---------------------------------------------
    st.markdown("---")
    st.subheader("🧠 Đánh giá tổng hợp từ hai mô hình")

    if (image_pred_probs is None) and (clinical_prob_recur is None):
        st.info("Khi có cả **kết quả mô hình hình ảnh** và **kết quả mô hình lâm sàng**, "
                "hệ thống sẽ hiển thị đánh giá tổng hợp tại đây.")
    else:
        p_malignant = None

        if image_pred_probs is not None:
            p_malignant = float(image_pred_probs[labels_clf.index("malignant")])
            st.write("🔬 **Nhận định từ mô hình hình ảnh:**")
            st.write(
                f"- Kết luận: **{image_pred_label_vi}** "
                f"(xác suất ác tính ≈ {p_malignant*100:.1f}%)."
            )

        if clinical_prob_recur is not None:
            st.write("📋 **Nhận định từ mô hình lâm sàng (METABRIC):**")
            st.write(
                f"- Xác suất tái phát ước tính ≈ **{clinical_prob_recur*100:.1f}%**."
            )
            if clinical_stage_pred is not None:
                st.write(f"- Giai đoạn u dự đoán (RF): **{clinical_stage_pred}**.")
            if clinical_risk_score is not None:
                st.write(f"- Risk Cox ≈ **{clinical_risk_score:.3f}**.")
        # Nếu có đủ cả 2 → risk tổng hợp (minh họa)
        if (p_malignant is not None) and (clinical_prob_recur is not None):
            combined_risk = 0.6 * p_malignant + 0.4 * clinical_prob_recur

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
