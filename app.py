import os
import io
import platform
import pickle

import streamlit as st
import altair as alt
import gdown
import numpy as np
import pandas as pd
import cv2
import joblib

import tensorflow as tf
import keras
from keras.models import load_model
import sklearn

# ==============================
# Patch NumPy BitGenerator khi load .pkl (PCG64, v.v.)
# ==============================
def _patch_numpy_bitgenerator():
    try:
        from numpy.random import _pickle as _np_pickle
        _orig = _np_pickle.__bit_generator_ctor
        def _patched(bit_generator):
            if isinstance(bit_generator, str):
                if bit_generator.endswith(".PCG64"): bit_generator = "PCG64"
                elif bit_generator.endswith(".PCG64DXSM"): bit_generator = "PCG64DXSM"
                elif bit_generator.endswith(".MT19937"): bit_generator = "MT19937"
                elif bit_generator.endswith(".Philox"): bit_generator = "Philox"
                elif bit_generator.endswith(".SFC64"): bit_generator = "SFC64"
            return _orig(bit_generator)
        _np_pickle.__bit_generator_ctor = _patched
    except Exception as e:
        print("Skip BitGenerator patch:", e)

_patch_numpy_bitgenerator()

# ==============================
# CUSTOM OBJECTS cho các Lambda/CBAM dùng khi train
# ==============================
@keras.saving.register_keras_serializable(package="CBAM")
def spatial_mean(x):
    return tf.reduce_mean(x, axis=-1, keepdims=True)

@keras.saving.register_keras_serializable(package="CBAM")
def spatial_max(x):
    return tf.reduce_max(x, axis=-1, keepdims=True)

@keras.saving.register_keras_serializable(package="CBAM")
def cbam_mult(x):
    return x[0] * x[1]

@keras.saving.register_keras_serializable(package="CBAM")
def cbam_mult2(x):
    return x[0] * x[1]

# Hàm này xuất hiện trong config (.keras) của bạn như output_shape cho Lambda
@keras.saving.register_keras_serializable(package="CBAM")
def spatial_output_shape(input_shape):
    def _as_list(s):
        try:
            if isinstance(s, tf.TensorShape):
                return s.as_list()
        except Exception:
            pass
        return list(s) if isinstance(s, (tuple, list)) else s
    s = input_shape
    if isinstance(s, (list, tuple)) and len(s) > 0 and isinstance(s[0], (list, tuple, tf.TensorShape)):
        s = _as_list(s[0])
    else:
        s = _as_list(s)
    if isinstance(s, (list, tuple)):
        if len(s) == 4:  # (B,H,W,C)
            return (s[0], s[1], s[2], 1)
        if len(s) == 3:  # (H,W,C)
            return (s[0], s[1], 1)
    return s

CUSTOM_OBJECTS = {
    "spatial_mean": spatial_mean,
    "spatial_max": spatial_max,
    "cbam_mult": cbam_mult,
    "cbam_mult2": cbam_mult2,
    "spatial_output_shape": spatial_output_shape,
}

# ==============================
# Tải model từ Google Drive (1 lần)
# ==============================
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

drive_files = {
    "Classifier_model_2.h5": "1fXPICuTkETep2oPiA56l0uMai2GusEJH",
    "best_model_cbam_attention_unet_fixed.keras": "1axOg7N5ssJrMec97eV-JMPzID26ynzN1",
    "clinical_epic_gb_model.pkl": "1z1wHVy9xyRXlRqxI8lYXMJhaJaUcKXnu",
    "clinical_epic_gb_metadata.pkl": "1WWlfeRqr99VL4nBQ-7eEptIxitKtXj6V",
}

with st.spinner("Đang kiểm tra & tải mô hình (lần đầu có thể hơi lâu)…"):
    for fname, fid in drive_files.items():
        dst = os.path.join(MODEL_DIR, fname)
        if not os.path.exists(dst):
            gdown.download(f"https://drive.google.com/uc?id={fid}", dst, quiet=False)

# ==============================
# Load models (cache)
# ==============================
def _joblib_load_robust(path: str):
    try:
        return joblib.load(path)
    except Exception as e:
        # Remap các tên lớp generator cũ -> mới nếu file .pkl được tạo với NumPy cũ
        if "BitGenerator" in str(e) or "_pcg64" in str(e):
            with open(path, "rb") as f:
                data = f.read()
            bio = io.BytesIO(data)
            class _RenameUnpickler(pickle.Unpickler):
                def find_class(self, module, name):
                    if module == "numpy.random._pcg64" and name == "PCG64":
                        module = "numpy.random._bit_generator"
                    if module == "numpy.random.bit_generator" and name == "BitGenerator":
                        module = "numpy.random._bit_generator"
                    return super().find_class(module, name)
            return _RenameUnpickler(bio).load()
        raise

@st.cache_resource
def load_models():
    # Quan trọng: safe_mode=False để cho phép Lambda có function tùy biến
    seg_model = load_model(
        os.path.join(MODEL_DIR, "best_model_cbam_attention_unet_fixed.keras"),
        compile=False,
        custom_objects=CUSTOM_OBJECTS,
        safe_mode=False,
    )
    class_model = load_model(
        os.path.join(MODEL_DIR, "Classifier_model_2.h5"),
        compile=False,
        safe_mode=False,
    )
    gb_model = _joblib_load_robust(os.path.join(MODEL_DIR, "clinical_epic_gb_model.pkl"))
    gb_meta  = _joblib_load_robust(os.path.join(MODEL_DIR, "clinical_epic_gb_metadata.pkl"))
    return seg_model, class_model, gb_model, gb_meta

seg_model, class_model, gb_model, gb_meta = load_models()

# ==============================
# Utils xử lý ảnh/overlay
# ==============================
def get_input_hwc(model):
    shp = model.input_shape
    if isinstance(shp, list): shp = shp[0]
    _, H, W, C = shp
    H = 256 if H is None else int(H)
    W = 256 if W is None else int(W)
    C = 3   if C is None else int(C)
    return H, W, C

def prep_for_model(gray_uint8, target_hwc):
    H, W, C = target_hwc
    resized = cv2.resize(gray_uint8, (W, H), interpolation=cv2.INTER_LINEAR)
    if C == 1:
        x = resized.astype(np.float32) / 255.0
        x = np.expand_dims(x, axis=(0, -1))
    else:
        x = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB).astype(np.float32) / 255.0
        x = np.expand_dims(x, axis=0)
    return x, resized

SEG_COLORS = {  # 1=benign (green), 2=malignant (red), 3=general (yellow)
    1: np.array([0, 255, 0], dtype=np.float32),
    2: np.array([255, 0, 0], dtype=np.float32),
    3: np.array([255, 255, 0], dtype=np.float32),
}
def overlay_multiclass(base_gray_uint8, mask_uint8, alpha=0.6):
    base = np.stack([base_gray_uint8]*3, axis=-1).astype(np.float32)
    over = base.copy()
    for cls_id, color in SEG_COLORS.items():
        m = (mask_uint8 == cls_id)
        if np.any(m):
            over[m] = (1 - alpha) * over[m] + alpha * color
    return np.clip(over, 0, 255).astype(np.uint8)

# ==============================
# UI
# ==============================
st.set_page_config(page_title="Breast Cancer Prediction App", layout="wide")
st.title("Breast Cancer Prediction App")

st.sidebar.markdown(
    f"""
**Versions**
- Python: `{platform.python_version()}`
- NumPy: `{np.__version__}`
- scikit-learn: `{sklearn.__version__}`
- TensorFlow: `{tf.__version__}`
- Keras: `{keras.__version__}`
"""
)

tab1, tab2 = st.tabs(["🔎 Ultrasound Image Analysis", "📊 Clinical Survival Prediction"])

# ---- Tab 1
with tab1:
    st.header("Ultrasound Image Analysis")
    uploaded = st.file_uploader("Choose an ultrasound image (PNG/JPG)", type=["png", "jpg", "jpeg"])

    if uploaded is not None:
        arr = np.frombuffer(uploaded.read(), np.uint8)
        gray = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
        if gray is None:
            st.error("Could not read the image. Please upload a valid image file.")
        else:
            gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

            seg_hwc = get_input_hwc(seg_model)
            clf_hwc = get_input_hwc(class_model)
            x_seg, gray_seg = prep_for_model(gray, seg_hwc)
            x_clf, gray_clf = prep_for_model(gray, clf_hwc)

            with st.spinner("Running segmentation..."):
                seg_pred = seg_model.predict(x_seg, verbose=0)[0]

            if seg_pred.ndim == 3 and seg_pred.shape[-1] >= 3:
                seg_mask = np.argmax(seg_pred, axis=-1).astype(np.uint8)  # 0..K-1
            else:
                seg_mask = (seg_pred[..., 0] >= 0.5).astype(np.uint8)
                seg_mask[seg_mask == 1] = 3  # nếu chỉ nhị phân: dùng lớp 'general' (vàng)

            overlay_img = overlay_multiclass(gray_seg, seg_mask, alpha=0.6)

            with st.spinner("Running classification..."):
                class_probs = class_model.predict(x_clf, verbose=0)[0]
            class_names = ["benign", "malignant", "normal"]
            pred_idx = int(np.argmax(class_probs))
            pred_label = class_names[pred_idx]
            vi_map = {"benign": "U lành tính", "malignant": "U ác tính", "normal": "Bình thường"}

            c1, c2 = st.columns(2)
            with c1:
                st.image(np.stack([gray_clf]*3, axis=-1), caption="Original (resized for classifier)", use_column_width=True)
            with c2:
                st.image(overlay_img, caption="Segmentation overlay (🟩 lành / 🟥 ác / 🟨 general)", use_column_width=True)

            st.write(f"**Classification Result:** {vi_map.get(pred_label, pred_label)} (≈ {float(class_probs[pred_idx]):.2%})")

            probs_df = pd.DataFrame({
                "Category": ["Benign", "Malignant", "Normal"],
                "Probability (%)": (class_probs * 100).round(2)
            })
            st.altair_chart(
                alt.Chart(probs_df).mark_bar().encode(
                    x=alt.X("Category", sort=None),
                    y=alt.Y("Probability (%)", scale=alt.Scale(domain=[0, 100]))
                ),
                use_container_width=True
            )
    else:
        st.info("Please upload a breast ultrasound image to analyze.")

# ---- Tab 2
with tab2:
    st.header("Clinical Survival Prediction")
    try:
        feature_names = gb_meta["feature_names"]
        label_map     = gb_meta.get("label_map", {"Living": 0, "Deceased": 1})
    except Exception as e:
        st.warning(f"Clinical model metadata not available: {e}")
        feature_names = None

    if feature_names is not None:
        inv_label_map = {v: k for k, v in label_map.items()}

        with st.form("clinical_form"):
            # Numeric
            age           = st.number_input("Age at Diagnosis", min_value=0.0, max_value=120.0, value=50.0)
            tumor_size    = st.number_input("Tumor Size (mm)", min_value=0.0, max_value=200.0, value=20.0)
            lymph_pos     = st.number_input("Lymph nodes examined positive", min_value=0, max_value=50, value=0, step=1)
            mutation_cnt  = st.number_input("Mutation Count", min_value=0, max_value=10000, value=0, step=1)
            npi           = st.number_input("Nottingham prognostic index", min_value=0.0, max_value=10.0, value=4.0, format="%.2f")
            os_months     = st.number_input("Overall Survival (Months)", min_value=0.0, max_value=300.0, value=60.0, format="%.2f")

            # Categorical
            surgery_type  = st.selectbox("Type of Breast Surgery", ["Breast Conserving", "Mastectomy"], index=0)
            hist_grade    = st.selectbox("Neoplasm Histologic Grade", [1, 2, 3], index=0)
            tumor_stage   = st.selectbox("Tumor Stage", [1, 2, 3, 4], index=0)
            sex           = st.selectbox("Sex", ["Female", "Male"], index=0)
            cellularity   = st.selectbox("Cellularity", ["High", "Low", "Moderate"], index=0)
            chemo         = st.selectbox("Chemotherapy", ["No", "Yes"], index=0)
            hormone       = st.selectbox("Hormone Therapy", ["No", "Yes"], index=0)
            radio         = st.selectbox("Radio Therapy", ["No", "Yes"], index=0)
            er_status     = st.selectbox("ER Status", ["Negative", "Positive"], index=0)
            pr_status     = st.selectbox("PR Status", ["Negative", "Positive"], index=0)
            her2_status   = st.selectbox("HER2 Status", ["Negative", "Positive"], index=0)
            gene_subtype  = st.selectbox("3-Gene classifier subtype",
                                         ["ER+/HER2+", "ER+/HER2- High Prolif", "ER+/HER2- Low Prolif", "ER-/HER2+", "ER-/HER2-"], index=0)
            pam50_subtype = st.selectbox("Pam50 + Claudin-low subtype",
                                         ["Basal-like", "Claudin-low", "HER2-enriched", "Luminal A", "Luminal B", "Normal-like"], index=0)
            relapse_status = st.selectbox("Relapse Free Status", ["Not Recurred", "Recurred"], index=0)

            submit_btn = st.form_submit_button("Predict Survival")

        if submit_btn:
            X = pd.DataFrame([np.zeros(len(feature_names))], columns=feature_names)

            # Numeric
            for col, val in {
                "Age at Diagnosis": age,
                "Tumor Size": tumor_size,
                "Lymph nodes examined positive": lymph_pos,
                "Mutation Count": mutation_cnt,
                "Nottingham prognostic index": npi,
                "Overall Survival (Months)": os_months
            }.items():
                if col in X.columns:
                    X.at[0, col] = val

            # One-hot theo đúng tên cột
            def set_dummy(col, val):
                dummy = f"{col}_{val}"
                if isinstance(val, (int, float)) and dummy not in X.columns and f"{col}_{val}.0" in X.columns:
                    dummy = f"{col}_{val}.0"
                if dummy in X.columns:
                    X.at[0, dummy] = 1

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

            y_pred = gb_model.predict(X)[0]
            inv_label_map = {v: k for k, v in label_map.items()}
            outcome = inv_label_map.get(int(y_pred), str(y_pred))
            death_prob = None
            if hasattr(gb_model, "predict_proba"):
                death_prob = float(gb_model.predict_proba(X)[0, label_map.get("Deceased", 1)])

            if outcome == "Deceased":
                st.error(f"**Predicted Outcome:** {outcome}")
            else:
                st.success(f"**Predicted Outcome:** {outcome}")
            if death_prob is not None:
                st.write(f"**Probability of death:** {death_prob*100:.1f}%")
