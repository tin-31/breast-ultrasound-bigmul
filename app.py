import streamlit as st
import os
import io
import sys
import types
import pickle

import gdown
import numpy as np
import pandas as pd
import cv2
import joblib
import altair as alt
from PIL import Image

import keras  # Keras 3.4.1 (TF 2.16.1 backend)
from keras.models import load_model


# =========================
# 0) CONFIG & DRIVE FILES
# =========================
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

drive_files = {
    "Classifier_model_2.h5": "1fXPICuTkETep2oPiA56l0uMai2GusEJH",
    "best_model_cbam_attention_unet_fixed.keras": "1axOg7N5ssJrMec97eV-JMPzID26ynzN1",
    "clinical_epic_gb_model.pkl": "1z1wHVy9xyRXlRqxI8lYXMJhaJaUcKXnu",
    "clinical_epic_gb_metadata.pkl": "1WWlfeRqr99VL4nBQ-7eEptIxitKtXj6V",
}

# mapping màu overlay cho segmentation
SEG_CLASS_TO_COLOR = {
    1: np.array([0, 255, 0], dtype=np.float32),      # benign -> xanh
    2: np.array([255, 0, 0], dtype=np.float32),      # malignant -> đỏ
    3: np.array([255, 255, 0], dtype=np.float32),    # general lesion -> vàng
}


# ======================================================
# 1) PATCH khắc phục lỗi PCG64 khi load pickle (.pkl)
# ======================================================
def _patch_numpy_bitgenerator():
    """
    Chấp nhận các đường dẫn BitGenerator kiểu 'numpy.random._pcg64.PCG64'
    trong quá trình unpickle ở môi trường hiện tại.
    """
    try:
        from numpy.random import _pickle as _np_pickle
        _orig_ctor = _np_pickle.__bit_generator_ctor

        def _patched_ctor(bit_generator):
            if isinstance(bit_generator, str):
                if bit_generator.endswith(".PCG64"):
                    bit_generator = "PCG64"
                elif bit_generator.endswith(".PCG64DXSM"):
                    bit_generator = "PCG64DXSM"
                elif bit_generator.endswith(".MT19937"):
                    bit_generator = "MT19937"
                elif bit_generator.endswith(".Philox"):
                    bit_generator = "Philox"
                elif bit_generator.endswith(".SFC64"):
                    bit_generator = "SFC64"
            return _orig_ctor(bit_generator)

        _np_pickle.__bit_generator_ctor = _patched_ctor
    except Exception as e:
        # Không dừng app nếu patch không cần/không thành công
        print("Skip BitGenerator patch:", e)


class _RenameUnpickler(pickle.Unpickler):
    """Fallback: đổi tên module cũ -> mới khi unpickle (nếu cần)."""
    def find_class(self, module, name):
        if module == "numpy.random._pcg64" and name == "PCG64":
            module = "numpy.random._bit_generator"
        if module == "numpy.random.bit_generator" and name == "BitGenerator":
            module = "numpy.random._bit_generator"
        return super().find_class(module, name)


def _pickle_load_compat(path: str):
    with open(path, "rb") as f:
        data = f.read()
    bio = io.BytesIO(data)
    return _RenameUnpickler(bio).load()


def _joblib_load_robust(path: str):
    """Thử joblib.load; nếu lỗi BitGenerator thì patch + fallback pickle."""
    try:
        return joblib.load(path)
    except Exception as e:
        if "BitGenerator" in str(e) or "_pcg64" in str(e):
            _patch_numpy_bitgenerator()
            try:
                return joblib.load(path)
            except Exception:
                return _pickle_load_compat(path)
        raise


# ============================================
# 2) HÀM TẢI MODEL TỪ GOOGLE DRIVE (1 LẦN)
# ============================================
def _ensure_models():
    with st.spinner("Downloading model files (if not already cached)..."):
        for filename, file_id in drive_files.items():
            dest_path = os.path.join(MODEL_DIR, filename)
            if not os.path.exists(dest_path):
                url = f"https://drive.google.com/uc?id={file_id}"
                gdown.download(url, dest_path, quiet=False)
    # Files will be downloaded only once. Subsequent runs will skip this.


# ============================================
# 3) LOAD MODELS (có cache)
# ============================================
@st.cache_resource
def load_models():
    _ensure_models()

    seg_model = load_model(
        os.path.join(MODEL_DIR, "best_model_cbam_attention_unet_fixed.keras"),
        compile=False
    )
    class_model = load_model(
        os.path.join(MODEL_DIR, "Classifier_model_2.h5"),
        compile=False
    )

    # clinical .pkl + meta (dùng loader "chịu lỗi")
    try:
        gb_model = _joblib_load_robust(os.path.join(MODEL_DIR, "clinical_epic_gb_model.pkl"))
        gb_meta  = _joblib_load_robust(os.path.join(MODEL_DIR, "clinical_epic_gb_metadata.pkl"))
    except Exception as e:
        st.error(f"Lỗi khi tải mô hình lâm sàng: {e}")
        gb_model, gb_meta = None, None

    return seg_model, class_model, gb_model, gb_meta


# ============================================
# 4) TIỆN ÍCH XỬ LÝ ẢNH / OVERLAY
# ============================================
def _get_input_hw(model):
    """Lấy (H, W, C) từ input_shape của model Keras."""
    ishape = model.input_shape
    if isinstance(ishape, list):
        ishape = ishape[0]
    _, H, W, C = ishape
    H = 256 if H is None else int(H)
    W = 256 if W is None else int(W)
    C = 3   if C is None else int(C)
    return H, W, C


def _prep_for_classifier(img_gray_256: np.ndarray, target_hw: tuple):
    """Chuẩn hóa cho classifier: (1,H,W,3), EfficientNetV2 preprocess nếu có."""
    H, W = target_hw
    # img_gray_256 là (256,256) uint8 -> resize target nếu khác
    if (img_gray_256.shape[0], img_gray_256.shape[1]) != (H, W):
        img_resized = cv2.resize(img_gray_256, (W, H), interpolation=cv2.INTER_LINEAR)
    else:
        img_resized = img_gray_256

    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2RGB).astype(np.float32)

    try:
        from keras.applications.efficientnet_v2 import preprocess_input
        x = preprocess_input(img_rgb)
    except Exception:
        x = img_rgb / 255.0

    x = np.expand_dims(x, axis=0)  # (1,H,W,3)
    return x


def _prep_for_segmentation(img_gray_256: np.ndarray, target_hw: tuple, segC: int):
    """Chuẩn hóa cho segmentation: (1,H,W,C), scale về [0..1]."""
    H, W = target_hw
    if (img_gray_256.shape[0], img_gray_256.shape[1]) != (H, W):
        img_resized = cv2.resize(img_gray_256, (W, H), interpolation=cv2.INTER_LINEAR)
    else:
        img_resized = img_gray_256

    if segC == 1:
        x = img_resized.astype(np.float32) / 255.0
        x = np.expand_dims(x, axis=(0, -1))     # (1,H,W,1)
    else:
        x = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2RGB).astype(np.float32) / 255.0
        x = np.expand_dims(x, axis=0)           # (1,H,W,3)
    return x, img_resized


def _overlay_multiclass(base_gray_256: np.ndarray, mask: np.ndarray, alpha: float = 0.6) -> np.ndarray:
    """
    base_gray_256: (H,W) uint8
    mask: (H,W) uint8 -> 0 bg, 1 benign, 2 malignant, 3 general
    return: RGB uint8 với overlay
    """
    base_rgb = np.stack([base_gray_256]*3, axis=-1).astype(np.float32)
    over = base_rgb.copy()
    for cls_id, color in SEG_CLASS_TO_COLOR.items():
        m = (mask == cls_id)
        if np.any(m):
            over[m] = (1 - alpha) * over[m] + alpha * color
    over = np.clip(over, 0, 255).astype(np.uint8)
    return over


# ============================================
# 5) APP UI
# ============================================
st.title("Breast Cancer Prediction App")
st.markdown(
    "Web cho phép tải **ảnh siêu âm vú** để *phân loại* & *phân đoạn*, "
    "đồng thời nhập **thông tin lâm sàng** để *tiên lượng*. "
    "Các mô hình sẽ tự tải bằng **gdown** (chỉ 1 lần)."
)

# Load models (cached)
seg_model, class_model, gb_model, gb_meta = load_models()

# Extract metadata for clinical features (nếu có)
if gb_meta is not None:
    feature_names = gb_meta.get("feature_names", [])
    num_cols     = gb_meta.get("num_cols", [])
    cat_cols     = gb_meta.get("cat_cols", [])
    label_map    = gb_meta.get("label_map", {"Living": 0, "Deceased": 1})
    inv_label_map = {v: k for k, v in label_map.items()}
else:
    feature_names, num_cols, cat_cols, label_map = [], [], [], {}
    inv_label_map = {}

# Tabs
tab1, tab2 = st.tabs(["🔎 Ultrasound Image Analysis", "📊 Clinical Survival Prediction"])


# ---------------- Tab 1: Ảnh ----------------
with tab1:
    st.header("Ultrasound Image Analysis")
    uploaded_file = st.file_uploader("Choose an ultrasound image file (PNG/JPG)", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        # Đọc ảnh xám
        file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
        img_gray = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
        if img_gray is None:
            st.error("Could not read the image. Please upload a valid image file.")
        else:
            # Chuẩn hóa về 0..255 & resize 256 cho hiển thị
            img_gray = cv2.normalize(img_gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            img_gray_256 = cv2.resize(img_gray, (256, 256), interpolation=cv2.INTER_LINEAR)

            # Lấy input size của model
            segH, segW, segC = _get_input_hw(seg_model)
            clfH, clfW, clfC = _get_input_hw(class_model)

            # Chuẩn bị input riêng
            x_clf = _prep_for_classifier(img_gray_256, (clfH, clfW))
            x_seg, seg_show_base = _prep_for_segmentation(img_gray_256, (segH, segW), segC)

            # ---- Segmentation ----
            with st.spinner("Running segmentation..."):
                seg_pred = seg_model.predict(x_seg, verbose=0)[0]   # (H,W,C) hoặc (H,W,1)

            if seg_pred.ndim == 3 and seg_pred.shape[-1] >= 2:
                mask = np.argmax(seg_pred, axis=-1).astype(np.uint8)   # 0..C-1
            else:
                # Nếu là nhị phân (ít khả năng), coi 1 là general lesion để vẫn hiển thị
                mask = (seg_pred[..., 0] >= 0.5).astype(np.uint8)
                mask[mask == 1] = 3

            # Resize mask -> 256 cho overlay
            mask_256 = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_NEAREST)
            overlay_img = _overlay_multiclass(img_gray_256, mask_256, alpha=0.6)

            # ---- Classification ----
            with st.spinner("Running classification..."):
                class_probs = class_model.predict(x_clf, verbose=0)[0]  # array (3,)
            class_names = ['benign', 'malignant', 'normal']  # theo huấn luyện
            pred_idx = int(np.argmax(class_probs))
            pred_label = class_names[pred_idx]
            pred_prob = float(class_probs[pred_idx])

            col1, col2 = st.columns(2)
            with col1:
                st.image(img_gray_256, caption="Original Image (256×256)", use_column_width=True, clamp=True)
            with col2:
                st.image(overlay_img, caption="Segmentation Overlay (🟥 ác / 🟩 lành / 🟨 general)", use_column_width=True)

            vi_map = {"benign": "U lành tính", "malignant": "U ác tính", "normal": "Bình thường"}
            st.write(f"**Classification Result:** {vi_map.get(pred_label, pred_label)} (≈ {pred_prob:.2%})")

            # biểu đồ xác suất
            probs_df = pd.DataFrame({
                'Category': ['Benign', 'Malignant', 'Normal'],
                'Probability (%)': (class_probs * 100.0).round(2)
            })
            st.altair_chart(
                alt.Chart(probs_df).mark_bar().encode(
                    x=alt.X('Category', sort=None),
                    y=alt.Y('Probability (%)', scale=alt.Scale(domain=[0, 100]))
                ),
                use_container_width=True
            )
    else:
        st.info("Please upload a breast ultrasound image to analyze.")


# -------------- Tab 2: Lâm sàng --------------
with tab2:
    st.header("Clinical Survival Prediction")
    st.write("Enter the patient's clinical information below. On submission, the model will predict survival outcome.")

    if gb_model is None or gb_meta is None or not feature_names:
        st.warning("Clinical model/metadata not available. Please check model files.")
    else:
        with st.form("clinical_form"):
            # Numeric features
            age = st.number_input("Age at Diagnosis", min_value=0.0, max_value=120.0, value=50.0)
            tumor_size = st.number_input("Tumor Size (mm)", min_value=0.0, max_value=200.0, value=20.0)
            lymph_pos = st.number_input("Lymph nodes examined positive", min_value=0, max_value=50, value=0, step=1)
            mutation_count = st.number_input("Mutation Count", min_value=0, max_value=10000, value=0, step=1)
            npi = st.number_input("Nottingham Prognostic Index", min_value=0.0, max_value=10.0, value=4.0, format="%.2f")
            os_months = st.number_input("Overall Survival (Months)", min_value=0.0, max_value=300.0, value=60.0, format="%.2f")

            # Categorical features
            surgery_type = st.selectbox("Type of Breast Surgery", ["Breast Conserving", "Mastectomy"], index=0)
            hist_grade = st.selectbox("Neoplasm Histologic Grade", [1, 2, 3], index=0)
            tumor_stage = st.selectbox("Tumor Stage", [1, 2, 3, 4], index=0)
            sex = st.selectbox("Sex", ["Female", "Male"], index=0)
            cellularity = st.selectbox("Cellularity", ["High", "Low", "Moderate"], index=0)
            chemo = st.selectbox("Chemotherapy", ["No", "Yes"], index=0)
            hormone = st.selectbox("Hormone Therapy", ["No", "Yes"], index=0)
            radio = st.selectbox("Radio Therapy", ["No", "Yes"], index=0)
            er_status = st.selectbox("ER Status", ["Negative", "Positive"], index=0)
            pr_status = st.selectbox("PR Status", ["Negative", "Positive"], index=0)
            her2_status = st.selectbox("HER2 Status", ["Negative", "Positive"], index=0)
            gene_subtype = st.selectbox("3-Gene classifier subtype",
                                        ["ER+/HER2+", "ER+/HER2- High Prolif", "ER+/HER2- Low Prolif", "ER-/HER2+", "ER-/HER2-"], index=0)
            pam50_subtype = st.selectbox("Pam50 + Claudin-low subtype",
                                         ["Basal-like", "Claudin-low", "HER2-enriched", "Luminal A", "Luminal B", "Normal-like"], index=0)
            relapse_status = st.selectbox("Relapse Free Status", ["Not Recurred", "Recurred"], index=0)

            submit_btn = st.form_submit_button("Predict Survival")

        if submit_btn:
            # Vector đầu vào đúng thứ tự feature_names
            X_input = pd.DataFrame(data=[np.zeros(len(feature_names))], columns=feature_names)

            # Numeric
            X_input.at[0, "Age at Diagnosis"] = age
            X_input.at[0, "Tumor Size"] = tumor_size
            X_input.at[0, "Lymph nodes examined positive"] = lymph_pos
            X_input.at[0, "Mutation Count"] = mutation_count
            X_input.at[0, "Nottingham prognostic index"] = npi
            X_input.at[0, "Overall Survival (Months)"] = os_months

            # Helper set dummy
            def set_dummy(col_name, value):
                dummy = f"{col_name}_{value}"
                if isinstance(value, (int, float)):
                    if dummy not in feature_names and f"{col_name}_{value}.0" in feature_names:
                        dummy = f"{col_name}_{value}.0"
                if dummy in feature_names:
                    X_input.at[0, dummy] = 1

            # Categorical (chỉ đặt cột dummy tương ứng)
            set_dummy("Type of Breast Surgery", surgery_type)
            set_dummy("Neoplasm Histologic Grade", hist_grade)
            set_dummy("Tumor Stage", tumor_stage)
            set_dummy("Sex", sex)
            set_dummy("Cellularity", cellularity)
            set_dummy("Chemotherapy", chemo)
            set_dummy("Hormone Therapy", hormone)
            set_dummy("Radio Therapy", radio)
            set_dummy("ER Status", er_status)
            set_dummy("PR Status", pr_status)
            set_dummy("HER2 Status", her2_status)
            set_dummy("3-Gene classifier subtype", gene_subtype)
            set_dummy("Pam50 + Claudin-low subtype", pam50_subtype)
            set_dummy("Relapse Free Status", relapse_status)

            # Predict
            y_pred = gb_model.predict(X_input)[0]
            proba = None
            if hasattr(gb_model, "predict_proba"):
                proba = float(gb_model.predict_proba(X_input)[0, 1])  # class=1 (Deceased)

            outcome_label = inv_label_map.get(int(y_pred), str(y_pred))
            if str(outcome_label).lower().startswith("deceased"):
                st.error(f"**Predicted Outcome:** {outcome_label}")
            else:
                st.success(f"**Predicted Outcome:** {outcome_label}")

            if proba is not None:
                st.write(f"**Probability of death:** {proba*100:.1f}%")
            st.caption("(*Note:* Prediction is based on provided clinical inputs.)")
