# app.py
import os, io, platform
import numpy as np
import pandas as pd
import cv2
import joblib
import streamlit as st
import altair as alt

import tensorflow as tf
import keras
from keras.models import load_model
from keras.saving import register_keras_serializable

# ==============================
# 0) PATCH NumPy cho joblib: chấp nhận cả class/str BitGenerator
# ==============================
def patch_numpy_for_joblib():
    try:
        import numpy as _np
        from numpy.random import _pickle as _np_pickle
        _orig = _np_pickle.__bit_generator_ctor

        def _normalize_name(obj):
            # obj có thể là 'PCG64', <class '...PCG64'>, hay instance
            if isinstance(obj, str):
                s = obj
            elif hasattr(obj, "__name__"):
                s = obj.__name__
            else:
                s = str(obj)
            # lấy đuôi và quy chiếu các alias
            for key in ("PCG64DXSM", "PCG64", "Philox", "MT19937", "SFC64"):
                if key in s:
                    return key
            # nếu không khớp, cố tách token cuối cùng
            return s.split(".")[-1]

        def _patched_ctor(obj):
            name = _normalize_name(obj)
            try:
                # gọi ctor gốc với tên đã chuẩn hóa
                return _orig(name)
            except Exception:
                # fallback: tự tạo từ mapping NumPy hiện tại
                mapping = {
                    "PCG64": _np.random.PCG64,
                    "PCG64DXSM": getattr(_np.random, "PCG64DXSM", _np.random.PCG64),
                    "MT19937": _np.random.MT19937,
                    "Philox": getattr(_np.random, "Philox", _np.random.PCG64),
                    "SFC64": getattr(_np.random, "SFC64", _np.random.PCG64),
                }
                cls = mapping.get(name, _np.random.PCG64)
                return cls()

        _np_pickle.__bit_generator_ctor = _patched_ctor
    except Exception as e:
        print("Skip BitGenerator patch:", e)

patch_numpy_for_joblib()  # phải gọi TRƯỚC khi joblib.load()

# ==============================
# 1) Đăng ký các Lambda/CBAM cho Keras 3 (nạp .keras an toàn)
# ==============================
@register_keras_serializable(package="cbam", name="spatial_mean")
def spatial_mean(x):
    return tf.reduce_mean(x, axis=-1, keepdims=True)

@register_keras_serializable(package="cbam", name="spatial_max")
def spatial_max(x):
    return tf.reduce_max(x, axis=-1, keepdims=True)

@register_keras_serializable(package="cbam", name="spatial_output_shape")
def spatial_output_shape(input_shape):
    # Keras 3 có thể gọi lại output_shape function; trả về cùng H,W nhưng kênh = 1
    shape = tf.TensorShape(input_shape).as_list() if input_shape is not None else [None, None, None, None]
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

# ==============================
# 2) Tải file model từ Google Drive (chỉ 1 lần)
# ==============================
import gdown
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
            url = f"https://drive.google.com/uc?id={fid}"
            gdown.download(url, dst, quiet=False)

# ==============================
# 3) Nạp model (cache)
# ==============================
@st.cache_resource
def load_models():
    # Segmentation (CBAM Attention U‑Net, đa lớp)
    seg_model = load_model(
        os.path.join(MODEL_DIR, "best_model_cbam_attention_unet_fixed.keras"),
        compile=False, custom_objects=CUSTOM_OBJECTS, safe_mode=False
    )
    # Classifier (.h5)
    class_model = load_model(
        os.path.join(MODEL_DIR, "Classifier_model_2.h5"),
        compile=False
    )
    # Clinical models (.pkl) – sau khi đã patch NumPy ở trên
    gb_model  = joblib.load(os.path.join(MODEL_DIR, "clinical_epic_gb_model.pkl"))
    gb_meta   = joblib.load(os.path.join(MODEL_DIR, "clinical_epic_gb_metadata.pkl"))
    return seg_model, class_model, gb_model, gb_meta

seg_model, class_model, gb_model, gb_meta = load_models()

# ==============================
# 4) Tiện ích xử lý ảnh & overlay màu
# ==============================
def get_input_hwc(model):
    shape = model.input_shape
    if isinstance(shape, (list, tuple)) and isinstance(shape[0], (list, tuple)):
        shape = shape[0]
    _, H, W, C = shape
    return (H or 256, W or 256, C or 3)

def prep_for_model(gray_uint8, target_hwc):
    H, W, C = map(int, target_hwc)
    resized = cv2.resize(gray_uint8, (W, H), interpolation=cv2.INTER_LINEAR)
    if C == 1:
        x = resized.astype(np.float32) / 255.0
        x = np.expand_dims(x, axis=(0, -1))
    else:
        x = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB).astype(np.float32) / 255.0
        x = np.expand_dims(x, axis=0)
    return x, resized

# màu cho lớp cụ thể
COLOR_BENIGN = np.array([0, 255, 0], dtype=np.float32)      # xanh
COLOR_MALIGN = np.array([255, 0, 0], dtype=np.float32)      # đỏ
COLOR_GENERAL = np.array([255, 255, 0], dtype=np.float32)   # vàng

def overlay_segmentation(base_gray_uint8, mask_ids, alpha_general=0.25, alpha_specific=0.6):
    """
    mask_ids: uint8, 0=background, 1=benign, 2=malignant (theo model .keras của bạn).
    - Tô general-lesion (mask!=0) màu vàng nhạt trước.
    - Sau đó chồng cụ thể: 1= xanh, 2= đỏ.
    """
    base = np.stack([base_gray_uint8]*3, axis=-1).astype(np.float32)
    over = base.copy()
    lesion_union = (mask_ids != 0)
    if np.any(lesion_union):
        over[lesion_union] = (1 - alpha_general) * over[lesion_union] + alpha_general * COLOR_GENERAL
    benign = (mask_ids == 1)
    malign = (mask_ids == 2)
    if np.any(benign):
        over[benign] = (1 - alpha_specific) * over[benign] + alpha_specific * COLOR_BENIGN
    if np.any(malign):
        over[malign] = (1 - alpha_specific) * over[malign] + alpha_specific * COLOR_MALIGN
    return np.clip(over, 0, 255).astype(np.uint8)

# ==============================
# 5) Giao diện
# ==============================
st.set_page_config(page_title="Breast Cancer Prediction App", layout="wide")
st.title("Breast Cancer Prediction App")
st.caption("Web cho phép tải ảnh siêu âm vú để **phân loại** & **phân đoạn đa lớp**, đồng thời nhập thông tin lâm sàng để **tiên lượng**. Các mô hình sẽ tự tải bằng gdown (chỉ 1 lần).")

# Hiển thị version để tiện debug
import sklearn, tensorflow as tf
st.sidebar.markdown(
    f"**Versions**\n\n- Python: `{platform.python_version()}`\n- NumPy: `{np.__version__}`\n- scikit-learn: `{sklearn.__version__}`\n- TensorFlow: `{tf.__version__}`\n- Keras: `{keras.__version__}`"
)

tab1, tab2 = st.tabs(["🔎 Ultrasound Image Analysis", "📊 Clinical Survival Prediction"])

# ---- Tab 1: Ảnh ----
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

            seg_hwc = get_input_hwc(seg_model)      # model .keras của bạn: (256,256,3)
            clf_hwc = get_input_hwc(class_model)    # classifier .h5:  (256,256,3)
            x_seg, gray_seg = prep_for_model(gray, seg_hwc)
            x_clf, gray_clf = prep_for_model(gray, clf_hwc)

            with st.spinner("Running segmentation..."):
                seg_pred = seg_model.predict(x_seg, verbose=0)[0]  # (H,W,3) softmax
            seg_mask = np.argmax(seg_pred, axis=-1).astype(np.uint8)  # 0=bg,1=benign,2=malignant

            overlay_img = overlay_segmentation(gray_seg, seg_mask)

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
                st.image(overlay_img, caption="Segmentation overlay (🟨 general • 🟩 lành • 🟥 ác)", use_column_width=True)

            st.write(f"**Classification Result:** {vi_map.get(pred_label, pred_label)} (≈ {float(class_probs[pred_idx]):.2%})")

            probs_df = pd.DataFrame({"Category": ["Benign", "Malignant", "Normal"],
                                     "Probability (%)": (class_probs * 100).round(2)})
            st.altair_chart(
                alt.Chart(probs_df).mark_bar().encode(
                    x=alt.X("Category", sort=None),
                    y=alt.Y("Probability (%)", scale=alt.Scale(domain=[0, 100]))
                ),
                use_container_width=True
            )
    else:
        st.info("Please upload a breast ultrasound image to analyze.")

# ---- Tab 2: Lâm sàng ----
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
            mapping_num = {
                "Age at Diagnosis": age, "Tumor Size": tumor_size,
                "Lymph nodes examined positive": lymph_pos, "Mutation Count": mutation_cnt,
                "Nottingham prognostic index": npi, "Overall Survival (Months)": os_months
            }
            for col, val in mapping_num.items():
                if col in X.columns: X.at[0, col] = val

            # One-hot đúng tên cột
            def set_dummy(col, val):
                key = f"{col}_{val}"
                if isinstance(val, (int, float)) and key not in X.columns and f"{col}_{val}.0" in X.columns:
                    key = f"{col}_{val}.0"
                if key in X.columns: X.at[0, key] = 1

            for col, val in [
                ("Type of Breast Surgery", surgery_type),
                ("Neoplasm Histologic Grade", hist_grade),
                ("Tumor Stage", tumor_stage),
                ("Sex", sex),
                ("Cellularity", cellularity),
                ("Chemotherapy", chemo),
                ("Hormone Therapy", hormone),
                ("Radio Therapy", radio),
                ("ER Status", er_status),
                ("PR Status", pr_status),
                ("HER2 Status", her2_status),
                ("3-Gene classifier subtype", gene_subtype),
                ("Pam50 + Claudin-low subtype", pam50_subtype),
                ("Relapse Free Status", relapse_status),
            ]:
                set_dummy(col, val)

            y_pred = int(gb_model.predict(X)[0])
            inv = {v: k for k, v in label_map.items()}
            outcome = inv.get(y_pred, str(y_pred))
            prob = None
            if hasattr(gb_model, "predict_proba"):
                prob = float(gb_model.predict_proba(X)[0, label_map.get("Deceased", 1)])

            if outcome == "Deceased":
                st.error(f"**Predicted Outcome:** {outcome}")
            else:
                st.success(f"**Predicted Outcome:** {outcome}")
            if prob is not None:
                st.write(f"**Probability of death:** {prob*100:.1f}%")
