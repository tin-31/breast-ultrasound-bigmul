import os
import json
import platform

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

from skops.io import load as skops_load   # ⭐ clinical model


# ============================================================
# 0) Custom objects cho CBAM / Lambda
# ============================================================
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
    if isinstance(shape, (list, tuple)) and len(shape) >= 3:
        if len(shape) >= 4:
            return (shape[0], shape[1], shape[2], 1)
        if len(shape) == 3:
            return (shape[0], shape[1], 1)
    return shape

CUSTOM_OBJECTS = {
    "spatial_mean": spatial_mean,
    "spatial_max": spatial_max,
    "spatial_output_shape": spatial_output_shape,
}


# ============================================================
# 1) Download model từ Google Drive
# ============================================================
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

drive_files = {
    # Vision models
    "Classifier_model_2.h5": "1fXPICuTkETep2oPiA56l0uMai2GusEJH",
    "best_model_cbam_attention_unet_fixed.keras": "1axOg7N5ssJrMec97eV-JMPzID26ynzN1",

    # Clinical models (V3 – RandomForest)
    "clinical_v3_model.skops": "1XVk4ZJv0qR3KEazzo-XKxjMo9_LHhal0",
    "clinical_v3_metadata.json": "1p2Dh8scfVECIEPv0Ed-5tq0aWEWWGDfr",
}

with st.spinner("Đang kiểm tra & tải mô hình (chỉ lần đầu)…"):
    for fname, fid in drive_files.items():
        dst = os.path.join(MODEL_DIR, fname)
        if not os.path.exists(dst):
            url = f"https://drive.google.com/uc?id={fid}"
            gdown.download(url, dst, quiet=False)


# ============================================================
# 2) Load tất cả models (vision + clinical_v3)
# ============================================================
@st.cache_resource
def load_all_models():

    # Vision segmentation
    seg_model = load_model(
        os.path.join(MODEL_DIR, "best_model_cbam_attention_unet_fixed.keras"),
        compile=False,
        custom_objects=CUSTOM_OBJECTS,
        safe_mode=False,
    )

    # Vision classifier
    class_model = load_model(
        os.path.join(MODEL_DIR, "Classifier_model_2.h5"),
        compile=False
    )

    # Clinical v3
    clinical_model = None
    clinical_meta = None

    skops_path = os.path.join(MODEL_DIR, "clinical_v3_model.skops")
    meta_path  = os.path.join(MODEL_DIR, "clinical_v3_metadata.json")

    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            clinical_meta = json.load(f)

        # ⭐ Không cần trusted=True (skops format chuẩn)
        clinical_model = skops_load(skops_path)

    except Exception as e:
        st.error(f"Không load được clinical model (.skops): {e}")

    return seg_model, class_model, clinical_model, clinical_meta


seg_model, class_model, clinical_model, clinical_meta = load_all_models()


# ============================================================
# 3) Utility xử lý ảnh
# ============================================================
def get_input_hwc(model):
    shape = model.input_shape
    if isinstance(shape, list):
        shape = shape[0]
    _, H, W, C = shape
    return int(H or 256), int(W or 256), int(C or 3)


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


COLOR_BENIGN     = np.array([0, 255, 0], dtype=np.float32)
COLOR_MALIGNANT  = np.array([255, 0, 0], dtype=np.float32)
COLOR_GENERAL    = (0, 255, 255)


def overlay_multiclass_with_general(base_gray_uint8, mask_uint8, alpha=0.6):
    base = np.stack([base_gray_uint8]*3, axis=-1).astype(np.float32)
    over = base.copy()

    m_b = mask_uint8 == 1
    m_m = mask_uint8 == 2

    if np.any(m_b):
        over[m_b] = (1-alpha)*over[m_b] + alpha*COLOR_BENIGN
    if np.any(m_m):
        over[m_m] = (1-alpha)*over[m_m] + alpha*COLOR_MALIGNANT

    general = (m_b | m_m).astype(np.uint8)*255
    if general.any():
        contours, _ = cv2.findContours(general, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        out = np.clip(over, 0, 255).astype(np.uint8)
        cv2.drawContours(out, contours, -1, COLOR_GENERAL, 2)
        return out

    return np.clip(over, 0, 255).astype(np.uint8)


# ============================================================
# 4) Streamlit UI
# ============================================================
st.set_page_config(page_title="Breast Cancer Prediction App", layout="wide")

st.title("Breast Cancer Prediction App")
st.caption("Ứng dụng phân tích ảnh siêu âm vú + tiên lượng lâm sàng (clinical survival).")

# Sidebar versions
st.sidebar.markdown(
    f"""
### Versions  
- Python: `{platform.python_version()}`
- NumPy: `{np.__version__}`
- scikit-learn: `{platform.python_version()}`
- TensorFlow: `{tf.__version__}`
- Keras: `{keras.__version__}`
"""
)


tab1, tab2 = st.tabs(["🔎 Ultrasound Image Analysis", "📊 Clinical Survival Prediction"])


# ============================================================
# TAB 1 — IMAGE ANALYSIS
# ============================================================
with tab1:
    st.header("Ultrasound Image Analysis")
    uploaded = st.file_uploader("Upload ultrasound image", type=["png", "jpg", "jpeg"])

    if uploaded:
        arr = np.frombuffer(uploaded.read(), np.uint8)
        gray = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)

        if gray is None:
            st.error("Invalid image!")
        else:
            gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

            seg_hwc = get_input_hwc(seg_model)
            clf_hwc = get_input_hwc(class_model)

            x_seg, gray_seg = prep_for_model(gray, seg_hwc)
            x_clf, gray_clf = prep_for_model(gray, clf_hwc)

            # Segmentation
            with st.spinner("Running segmentation..."):
                seg_pred = seg_model.predict(x_seg, verbose=0)[0]

            seg_mask = (
                np.argmax(seg_pred, axis=-1).astype(np.uint8)
                if seg_pred.ndim == 3 and seg_pred.shape[-1] >= 3
                else (seg_pred[..., 0] >= 0.5).astype(np.uint8)
            )

            overlay = overlay_multiclass_with_general(gray_seg, seg_mask)

            # Classification
            with st.spinner("Running classification..."):
                probs = class_model.predict(x_clf, verbose=0)[0]

            class_names = ["benign", "malignant", "normal"]
            vi_map = {"benign": "U lành tính", "malignant": "U ác tính", "normal": "Bình thường"}

            pred_idx = int(np.argmax(probs))
            pred_label = class_names[pred_idx]

            c1, c2 = st.columns(2)
            c1.image(np.stack([gray_clf]*3, axis=-1), caption="Original (resized)")
            c2.image(overlay, caption="Segmentation overlay")

            st.subheader(f"Classification: {vi_map[pred_label]} ({probs[pred_idx]:.2%})")

            dfp = pd.DataFrame({
                "Category": ["Benign", "Malignant", "Normal"],
                "Probability (%)": (probs*100).round(2)
            })
            st.altair_chart(
                alt.Chart(dfp).mark_bar().encode(
                    x=alt.X("Category"),
                    y=alt.Y("Probability (%)", scale=alt.Scale(domain=[0, 100]))
                )
            )


# ============================================================
# TAB 2 — CLINICAL MODEL
# ============================================================
with tab2:
    st.header("Clinical Survival Prediction (RandomForest Version)")

    if clinical_model is None or clinical_meta is None:
        st.error("Không load được clinical model!")
    else:

        feature_names = clinical_meta["feature_names"]
        label_map = clinical_meta["label_map"]
        inv_label_map = {v: k for k, v in label_map.items()}

        with st.form("clinical_form"):

            age = st.number_input("Age at Diagnosis", 0.0, 120.0, 50.0)
            tumor = st.number_input("Tumor Size", 0.0, 200.0, 20.0)
            lymph = st.number_input("Lymph nodes examined positive", 0, 50, 0)
            mut = st.number_input("Mutation Count", 0, 10000, 0)
            npi = st.number_input("Nottingham prognostic index", 0.0, 10.0, 4.0)
            os_m = st.number_input("Overall Survival (Months)", 0.0, 300.0, 60.0)

            # categoricals
            surgery = st.selectbox("Type of Breast Surgery", ["Breast Conserving", "Mastectomy"])
            grade = st.selectbox("Neoplasm Histologic Grade", [1, 2, 3])
            stage = st.selectbox("Tumor Stage", [1, 2, 3, 4])
            sex = st.selectbox("Sex", ["Female", "Male"])
            cell = st.selectbox("Cellularity", ["High", "Low", "Moderate"])
            chemo = st.selectbox("Chemotherapy", ["No", "Yes"])
            hormone = st.selectbox("Hormone Therapy", ["No", "Yes"])
            radio = st.selectbox("Radio Therapy", ["No", "Yes"])
            er = st.selectbox("ER Status", ["Negative", "Positive"])
            pr = st.selectbox("PR Status", ["Negative", "Positive"])
            her2 = st.selectbox("HER2 Status", ["Negative", "Positive"])
            subtype3 = st.selectbox("3-Gene subtype",
                                    ["ER+/HER2+", "ER+/HER2- High Prolif", "ER+/HER2- Low Prolif",
                                     "ER-/HER2+", "ER-/HER2-"])
            pam50 = st.selectbox("Pam50 + Claudin-low subtype",
                                 ["Basal-like", "Claudin-low", "HER2-enriched",
                                  "Luminal A", "Luminal B", "Normal-like"])
            relapse = st.selectbox("Relapse Free Status", ["Not Recurred", "Recurred"])

            submit = st.form_submit_button("Predict Survival")

        if submit:
            X = pd.DataFrame([np.zeros(len(feature_names))], columns=feature_names)

            # numeric
            numeric_vals = {
                "Age at Diagnosis": age,
                "Tumor Size": tumor,
                "Lymph nodes examined positive": lymph,
                "Mutation Count": mut,
                "Nottingham prognostic index": npi,
                "Overall Survival (Months)": os_m,
            }
            for k, v in numeric_vals.items():
                if k in X.columns: X.at[0, k] = v

            # 1-hot helper
            def set_dummy(col, val):
                d = f"{col}_{val}"
                if isinstance(val, (int, float)) and d not in X.columns and f"{col}_{val}.0" in X.columns:
                    d = f"{col}_{val}.0"
                if d in X.columns: X.at[0, d] = 1

            set_dummy("Type of Breast Surgery", surgery)
            set_dummy("Neoplasm Histologic Grade", grade)
            set_dummy("Tumor Stage", stage)
            set_dummy("Sex", sex)
            set_dummy("Cellularity", cell)
            set_dummy("Chemotherapy", chemo)
            set_dummy("Hormone Therapy", hormone)
            set_dummy("Radio Therapy", radio)
            set_dummy("ER Status", er)
            set_dummy("PR Status", pr)
            set_dummy("HER2 Status", her2)
            set_dummy("3-Gene classifier subtype", subtype3)
            set_dummy("Pam50 + Claudin-low subtype", pam50)
            set_dummy("Relapse Free Status", relapse)

            y_pred = int(clinical_model.predict(X)[0])
            y_label = inv_label_map[y_pred]

            prob = float(clinical_model.predict_proba(X)[0][ label_map["Deceased"] ])

            if y_label == "Deceased":
                st.error(f"Predicted Outcome: **{y_label}**")
            else:
                st.success(f"Predicted Outcome: **{y_label}**")

            st.write(f"**Probability of death:** {prob*100:.2f}%")
