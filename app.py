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
import sklearn

# 👇 thêm skops để load model lâm sàng
from skops.io import load as skops_load


# ============ 0) Keras custom objects cho CBAM/Lambda ============
@register_keras_serializable(package="cbam", name="spatial_mean")
def spatial_mean(x):
    # (B, H, W, C) -> (B, H, W, 1)
    return tf.reduce_mean(x, axis=-1, keepdims=True)

@register_keras_serializable(package="cbam", name="spatial_max")
def spatial_max(x):
    # (B, H, W, C) -> (B, H, W, 1)
    return tf.reduce_max(x, axis=-1, keepdims=True)

@register_keras_serializable(package="cbam", name="spatial_output_shape")
def spatial_output_shape(input_shape):
    """
    Nhiều model Keras cũ lưu output_shape của Lambda dưới dạng hàm.
    Hàm này giúp Keras 3 deserialize được.
    """
    try:
        shape = tf.TensorShape(input_shape).as_list()
    except Exception:
        shape = list(input_shape) if isinstance(input_shape, (list, tuple)) else input_shape
    if isinstance(shape, (list, tuple)) and len(shape) >= 3:
        # Nếu có dạng (..., H, W, C) -> (..., H, W, 1)
        if len(shape) >= 4:
            return (shape[0], shape[1], shape[2], 1)
        # Nếu có dạng (H, W, C) -> (H, W, 1)
        if len(shape) == 3:
            return (shape[0], shape[1], 1)
    return shape

CUSTOM_OBJECTS = {
    "spatial_mean": spatial_mean,
    "spatial_max": spatial_max,
    "spatial_output_shape": spatial_output_shape,
}

# ============ 1) Tải model từ Google Drive (1 lần) ============
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# 👉 Thêm 2 file mới .skops + .json
drive_files = {
    # Keras models
    "Classifier_model_2.h5": "1fXPICuTkETep2oPiA56l0uMai2GusEJH",
    "best_model_cbam_attention_unet_fixed.keras": "1axOg7N5ssJrMec97eV-JMPzID26ynzN1",

    # Clinical model (phiên bản mới)
    # link bạn gửi:
    # https://drive.google.com/file/d/1j-6DIy4UQ6R-7TwEVlOLg0FldfAkzGnO/view?usp=sharing
    # https://drive.google.com/file/d/1i8er4zmW46bzY0wx4NzRq__06RWAuqvz/view?usp=sharing
    "clinical_epic_gb_model.skops": "1j-6DIy4UQ6R-7TwEVlOLg0FldfAkzGnO",
    "clinical_epic_gb_metadata.json": "1i8er4zmW46bzY0wx4NzRq__06RWAuqvz",
}

with st.spinner("Đang kiểm tra & tải mô hình (chỉ lần đầu)…"):
    for fname, fid in drive_files.items():
        dst = os.path.join(MODEL_DIR, fname)
        if not os.path.exists(dst):
            url = f"https://drive.google.com/uc?id={fid}"
            gdown.download(url, dst, quiet=False)

# ============ 2) Load models với cache ============
@st.cache_resource
def load_models():
    # ----- Segmentation (Keras 3) -----
    seg_model = load_model(
        os.path.join(MODEL_DIR, "best_model_cbam_attention_unet_fixed.keras"),
        compile=False,
        custom_objects=CUSTOM_OBJECTS,
        safe_mode=False,   # cần False để nhận Lambda custom
    )

    # ----- Classifier (h5) -----
    class_model = load_model(
        os.path.join(MODEL_DIR, "Classifier_model_2.h5"),
        compile=False
    )

    # ----- Clinical model (skops + json) -----
    gb_model = None
    gb_meta = None
    skops_path = os.path.join(MODEL_DIR, "clinical_epic_gb_model.skops")
    meta_path  = os.path.join(MODEL_DIR, "clinical_epic_gb_metadata.json")

    if os.path.exists(skops_path) and os.path.exists(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                gb_meta = json.load(f)

            # trusted=True vì đây là file do chính bạn train & upload
            gb_model = skops_load(skops_path, trusted=True)
        except Exception as e:
            st.warning(f"Không load được clinical model (.skops): {e}")
            gb_model = None
            gb_meta = None

    return seg_model, class_model, gb_model, gb_meta


seg_model, class_model, gb_model, gb_meta = load_models()

# ============ 3) Utils xử lý ảnh/overlay ============
def get_input_hwc(model):
    shape = model.input_shape
    if isinstance(shape, list):
        shape = shape[0]
    _, H, W, C = shape
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


# màu tô: 1= xanh (lành), 2= đỏ (ác)
COLOR_BENIGN     = np.array([0, 255, 0], dtype=np.float32)
COLOR_MALIGNANT  = np.array([255, 0, 0], dtype=np.float32)
COLOR_GENERAL    = (0, 255, 255)  # BGR của vàng khi vẽ contour bằng OpenCV


def overlay_multiclass_with_general(base_gray_uint8, mask_uint8, alpha=0.6):
    """
    - Nếu mask là softmax argmax (0..2): 0 nền, 1 lành, 2 ác
    - Tô vùng 1 (xanh), 2 (đỏ)
    - Viền vàng (general lesion) = (mask==1) ∪ (mask==2)
    """
    base = np.stack([base_gray_uint8]*3, axis=-1).astype(np.float32)
    over = base.copy()

    m_ben = (mask_uint8 == 1)
    m_mal = (mask_uint8 == 2)

    if np.any(m_ben):
        over[m_ben] = (1 - alpha) * over[m_ben] + alpha * COLOR_BENIGN
    if np.any(m_mal):
        over[m_mal] = (1 - alpha) * over[m_mal] + alpha * COLOR_MALIGNANT

    # Viền vàng cho tổng tổn thương
    general = (m_ben | m_mal).astype(np.uint8) * 255
    if general.any():
        contours, _ = cv2.findContours(general, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        over_uint8 = np.clip(over, 0, 255).astype(np.uint8)
        cv2.drawContours(over_uint8, contours, -1, COLOR_GENERAL, thickness=2, lineType=cv2.LINE_AA)
        return over_uint8
    return np.clip(over, 0, 255).astype(np.uint8)


# ============ 4) UI ============
st.set_page_config(page_title="Breast Cancer Prediction App", layout="wide")
st.title("Breast Cancer Prediction App")
st.caption(
    "Web cho phép tải ảnh siêu âm vú để **phân loại & phân đoạn đa lớp**, "
    "và nhập **thông tin lâm sàng** để tiên lượng sống còn."
)

# Hiển thị version để debug nhanh
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

            # Chuẩn bị input cho từng model
            seg_hwc = get_input_hwc(seg_model)
            clf_hwc = get_input_hwc(class_model)
            x_seg, gray_seg = prep_for_model(gray, seg_hwc)
            x_clf, gray_clf = prep_for_model(gray, clf_hwc)

            # Segmentation (đa lớp)
            with st.spinner("Running segmentation..."):
                seg_pred = seg_model.predict(x_seg, verbose=0)[0]  # (H,W,K) softmax hoặc (H,W,1)

            if seg_pred.ndim == 3 and seg_pred.shape[-1] >= 3:
                seg_mask = np.argmax(seg_pred, axis=-1).astype(np.uint8)  # 0..K-1
            else:
                # fallback nhị phân: tô vàng tổng tổn thương
                seg_mask = (seg_pred[..., 0] >= 0.5).astype(np.uint8) * 1  # 1 = benign giả
            overlay_img = overlay_multiclass_with_general(gray_seg, seg_mask, alpha=0.6)

            # Classification
            with st.spinner("Running classification..."):
                class_probs = class_model.predict(x_clf, verbose=0)[0]  # (3,)
            class_names = ["benign", "malignant", "normal"]
            vi_map = {"benign": "U lành tính", "malignant": "U ác tính", "normal": "Bình thường"}
            pred_idx = int(np.argmax(class_probs))
            pred_label = class_names[pred_idx]

            c1, c2 = st.columns(2)
            with c1:
                st.image(
                    np.stack([gray_clf]*3, axis=-1),
                    caption="Original (resized for classifier)",
                    use_column_width=True,
                )
            with c2:
                st.image(
                    overlay_img,
                    caption="Segmentation overlay (🟩 lành / 🟥 ác / viền 🟨 tổng tổn thương)",
                    use_column_width=True,
                )

            st.write(
                f"**Classification Result:** {vi_map.get(pred_label, pred_label)} "
                f"(≈ {float(class_probs[pred_idx]):.2%})"
            )

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

# ---- Tab 2: Lâm sàng ----
with tab2:
    st.header("Clinical Survival Prediction")

    if gb_model is None or gb_meta is None:
        st.warning(
            "Không load được clinical model (.skops). "
            "Vui lòng kiểm tra lại file trên Google Drive hoặc thư viện `skops`."
        )
    else:
        # Lấy metadata
        try:
            feature_names = gb_meta["feature_names"]
            label_map     = gb_meta.get("label_map", {"Living": 0, "Deceased": 1})
        except Exception as e:
            st.error(f"Lỗi đọc metadata clinical model: {e}")
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
                gene_subtype  = st.selectbox(
                    "3-Gene classifier subtype",
                    ["ER+/HER2+", "ER+/HER2- High Prolif", "ER+/HER2- Low Prolif", "ER-/HER2+", "ER-/HER2-"],
                    index=0
                )
                pam50_subtype = st.selectbox(
                    "Pam50 + Claudin-low subtype",
                    ["Basal-like", "Claudin-low", "HER2-enriched", "Luminal A", "Luminal B", "Normal-like"],
                    index=0
                )
                relapse_status = st.selectbox("Relapse Free Status", ["Not Recurred", "Recurred"], index=0)

                submit_btn = st.form_submit_button("Predict Survival")

            if submit_btn:
                # Tạo DataFrame 1 dòng, đúng thứ tự feature_names
                X = pd.DataFrame([np.zeros(len(feature_names))], columns=feature_names)

                # Numeric
                numeric_vals = {
                    "Age at Diagnosis": age,
                    "Tumor Size": tumor_size,
                    "Lymph nodes examined positive": lymph_pos,
                    "Mutation Count": mutation_cnt,
                    "Nottingham prognostic index": npi,
                    "Overall Survival (Months)": os_months
                }
                for col, val in numeric_vals.items():
                    if col in X.columns:
                        X.at[0, col] = val

                # One-hot helper
                def set_dummy(col, val):
                    dummy = f"{col}_{val}"
                    # trường hợp cột là số (1, 2, 3) nhưng trong feature_names là "x.0"
                    if isinstance(val, (int, float)) and dummy not in X.columns and f"{col}_{val}.0" in X.columns:
                        dummy = f"{col}_{val}.0"
                    if dummy in X.columns:
                        X.at[0, dummy] = 1

                # Categorical
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

                # Dự đoán
                y_pred = int(gb_model.predict(X)[0])
                inv_label_map = {v: k for k, v in label_map.items()}
                outcome = inv_label_map.get(y_pred, str(y_pred))

                death_prob = None
                if hasattr(gb_model, "predict_proba"):
                    # lấy prob của lớp "Deceased"
                    death_idx = label_map.get("Deceased", 1)
                    death_prob = float(gb_model.predict_proba(X)[0, death_idx])

                if outcome == "Deceased":
                    st.error(f"**Predicted Outcome:** {outcome}")
                else:
                    st.success(f"**Predicted Outcome:** {outcome}")
                if death_prob is not None:
                    st.write(f"**Probability of death:** {death_prob*100:.1f}%")
