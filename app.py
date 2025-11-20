import streamlit as st
import os
import sys
import io
import gdown
import numpy as np
import pandas as pd
import cv2
import joblib
import altair as alt
import keras
import tensorflow as tf
from PIL import Image
from keras.saving import register_keras_serializable


# =========================================================
# 0) Vá NumPy BitGenerator khi unpickle (.pkl) – fix PCG64
# =========================================================
def _patch_numpy_bitgenerator():
    # Alias module nếu _pcg64 không tồn tại
    try:
        import numpy.random._pcg64  # noqa: F401
    except Exception:
        try:
            import numpy.random._bit_generator as _bitgen
            sys.modules['numpy.random._pcg64'] = _bitgen
        except Exception:
            pass

    # Patch ctor để chấp nhận chuỗi module đầy đủ -> tên rút gọn
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
    except Exception:
        pass


# =========================================================
# 1) Đăng ký các hàm CBAM dùng trong Lambda layers của model
#    (đây là nguyên nhân lỗi "Could not locate function 'spatial_mean'")
# =========================================================
@register_keras_serializable(package="cbam", name="spatial_mean")
def spatial_mean(x):
    # (B,H,W,C) -> (B,H,W,1)
    return tf.reduce_mean(x, axis=-1, keepdims=True)

@register_keras_serializable(package="cbam", name="spatial_max")
def spatial_max(x):
    return tf.reduce_max(x, axis=-1, keepdims=True)

@register_keras_serializable(package="cbam", name="channel_mean")
def channel_mean(x):
    # (B,H,W,C) -> (B,1,1,C)
    return tf.reduce_mean(x, axis=[1, 2], keepdims=True)

@register_keras_serializable(package="cbam", name="channel_max")
def channel_max(x):
    return tf.reduce_max(x, axis=[1, 2], keepdims=True)


# =========================================================
# 2) Download model files (1 lần) bằng gdown
# =========================================================
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

drive_files = {
    "Classifier_model_2.h5": "1fXPICuTkETep2oPiA56l0uMai2GusEJH",
    "best_model_cbam_attention_unet_fixed.keras": "1axOg7N5ssJrMec97eV-JMPzID26ynzN1",
    "clinical_epic_gb_model.pkl": "1z1wHVy9xyRXlRqxI8lYXMJhaJaUcKXnu",
    "clinical_epic_gb_metadata.pkl": "1WWlfeRqr99VL4nBQ-7eEptIxitKtXj6V"
}

with st.spinner("Downloading model files (if not already cached)..."):
    for filename, file_id in drive_files.items():
        dest_path = os.path.join(MODEL_DIR, filename)
        if not os.path.exists(dest_path):
            url = f"https://drive.google.com/uc?id={file_id}"
            gdown.download(url, dest_path, quiet=False)


# =========================================================
# 3) Tiện ích ảnh
# =========================================================
def _get_input_hw(keras_model):
    shp = keras_model.input_shape
    if isinstance(shp, list):
        shp = shp[0]
    H, W, C = shp[1], shp[2], shp[3]
    if H is None or W is None:
        H = W = 256
    if C is None:
        C = 3
    return int(H), int(W), int(C)

def _preprocess_for_model(img_pil: Image.Image, target_hw, to_rgb: bool) -> np.ndarray:
    img = img_pil.convert("L")
    img = img.resize((target_hw[1], target_hw[0]))  # (W,H)
    arr = np.array(img).astype("float32") / 255.0
    if to_rgb:
        arr = np.stack([arr, arr, arr], axis=-1)  # (H,W,3)
    else:
        arr = np.expand_dims(arr, axis=-1)        # (H,W,1)
    return np.expand_dims(arr, axis=0)            # (1,H,W,C)

def _overlay_multiclass(orig_pil: Image.Image, mask: np.ndarray, alpha: float = 0.6) -> Image.Image:
    base = orig_pil.convert("RGB")
    base_np = np.array(base).astype(np.float32)

    mask_img = Image.fromarray(mask.astype(np.uint8))
    mask_resized = mask_img.resize(base.size, resample=Image.NEAREST)
    m = np.array(mask_resized)

    overlay = base_np.copy()
    colors = {
        1: np.array([0, 255, 0], dtype=np.float32),      # benign -> xanh
        2: np.array([255, 0, 0], dtype=np.float32),      # malignant -> đỏ
        3: np.array([255, 255, 0], dtype=np.float32),    # general lesion -> vàng
    }
    for cls_id, color in colors.items():
        region = (m == cls_id)
        if np.any(region):
            overlay[region] = (1 - alpha) * overlay[region] + alpha * color

    return Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8))


# =========================================================
# 4) Load models (có cache)
# =========================================================
@st.cache_resource
def load_models():
    # segmentation (CBAM + Lambda) cần custom_objects và safe_mode=False
    custom_objs = {
        "spatial_mean": spatial_mean,
        "spatial_max": spatial_max,
        "channel_mean": channel_mean,
        "channel_max": channel_max,
    }
    seg_model = keras.models.load_model(
        os.path.join(MODEL_DIR, "best_model_cbam_attention_unet_fixed.keras"),
        custom_objects=custom_objs,
        compile=False,
        safe_mode=False,     # cho phép load object/lambda đã đăng ký
    )

    # classifier
    class_model = keras.models.load_model(
        os.path.join(MODEL_DIR, "Classifier_model_2.h5"),
        compile=False
    )

    # clinical .pkl + meta (vá PCG64 trước khi load)
    _patch_numpy_bitgenerator()
    gb_model = joblib.load(os.path.join(MODEL_DIR, "clinical_epic_gb_model.pkl"))
    gb_meta  = joblib.load(os.path.join(MODEL_DIR, "clinical_epic_gb_metadata.pkl"))
    return seg_model, class_model, gb_model, gb_meta


try:
    seg_model, class_model, gb_model, gb_meta = load_models()
except Exception as e:
    st.error(f"Lỗi khi tải mô hình: {e}")
    seg_model = class_model = gb_model = gb_meta = None


# =========================================================
# 5) Meta lâm sàng
# =========================================================
feature_names = []
num_cols = []
cat_cols = []
label_map = {}
inv_label_map = {}
if gb_meta is not None:
    feature_names = gb_meta.get("feature_names", [])
    num_cols     = gb_meta.get("num_cols", [])
    cat_cols     = gb_meta.get("cat_cols", [])
    label_map    = gb_meta.get("label_map", {"Living": 0, "Deceased": 1})
    inv_label_map = {v: k for k, v in label_map.items()}


# =========================================================
# 6) UI
# =========================================================
st.title("Breast Cancer Prediction App")
st.markdown(
    "Web cho phép tải **ảnh siêu âm vú** để *phân loại* & *phân đoạn đa lớp*, "
    "và nhập **thông tin lâm sàng** để *tiên lượng*. "
    "Các mô hình sẽ tự tải bằng **gdown** (chỉ 1 lần)."
)

tab1, tab2 = st.tabs(["🔎 Ultrasound Image Analysis", "📊 Clinical Survival Prediction"])


# ---------------- Tab 1: ẢNH ----------------
with tab1:
    st.header("Ultrasound Image Analysis")
    uploaded_file = st.file_uploader("Choose an ultrasound image file (PNG/JPG)", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None and (seg_model is not None and class_model is not None):
        try:
            img_pil = Image.open(uploaded_file).convert("L")
        except Exception:
            file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
            img_cv = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
            if img_cv is None:
                st.error("Could not read the image. Please upload a valid image file.")
                st.stop()
            img_pil = Image.fromarray(img_cv)

        st.image(img_pil, caption="Original Image", use_column_width=True)

        # ----- Classification -----
        st.subheader("📌 Classification")
        Hc, Wc, Cc = _get_input_hw(class_model)
        x_cls = _preprocess_for_model(img_pil, (Hc, Wc), to_rgb=True)  # (1,H,W,3)
        with st.spinner("Predicting class..."):
            class_probs = class_model.predict(x_cls, verbose=0)[0]

        class_names = ["benign", "malignant", "normal"]
        if class_probs.shape[0] == 3:
            pred_idx = int(np.argmax(class_probs))
            pred_label = class_names[pred_idx]
            st.write(f"**Classification Result:** **{pred_label.upper()}** (≈ {class_probs[pred_idx]:.2%})")
            probs_df = pd.DataFrame({
                'Category': ['Benign', 'Malignant', 'Normal'],
                'Probability (%)': (class_probs * 100).round(2)
            })
            st.altair_chart(
                alt.Chart(probs_df).mark_bar().encode(
                    x=alt.X('Category', sort=None),
                    y=alt.Y('Probability (%)', scale=alt.Scale(domain=[0, 100]))
                ),
                use_container_width=True
            )
        else:
            st.warning(f"Classifier output has {class_probs.shape[0]} classes. Raw: {class_probs}")

        # ----- Segmentation (multiclass) -----
        st.subheader("🧩 Multiclass Segmentation (CBAM U-Net)")
        Hs, Ws, Cs = _get_input_hw(seg_model)
        # nếu model phân đoạn cần 3 kênh, to_rgb=True; nếu 1 kênh, to_rgb=False
        to_rgb_seg = (Cs == 3)
        x_seg = _preprocess_for_model(img_pil, (Hs, Ws), to_rgb=to_rgb_seg)
        with st.spinner("Segmenting..."):
            seg_pred = seg_model.predict(x_seg, verbose=0)[0]   # (H,W,K) hoặc (H,W)

        if seg_pred.ndim == 3 and seg_pred.shape[-1] >= 2:
            seg_mask = np.argmax(seg_pred, axis=-1).astype(np.uint8)   # 0/1/2/(3)
        elif seg_pred.ndim == 2:
            seg_mask = (seg_pred >= 0.5).astype(np.uint8)
        else:
            st.error(f"Unsupported segmentation output shape: {seg_pred.shape}")
            st.stop()

        overlay_img = _overlay_multiclass(img_pil, seg_mask, alpha=0.6)
        col1, col2 = st.columns(2)
        with col1:
            st.image(img_pil, caption="Original", use_column_width=True)
        with col2:
            st.image(overlay_img, caption="Overlay (🟥 ác / 🟩 lành / 🟨 general)", use_column_width=True)

    elif uploaded_file is None:
        st.info("Please upload a breast ultrasound image to analyze.")
    else:
        st.error("Models not loaded. Please check logs.")


# ---------------- Tab 2: LÂM SÀNG ----------------
with tab2:
    st.header("Clinical Survival Prediction")
    st.write("Enter the patient's clinical information below. On submission, the model predicts survival outcome.")

    if gb_model is None or gb_meta is None or not feature_names:
        st.warning("Clinical model/metadata not loaded. Please check the error above or your requirements.")
    else:
        with st.form("clinical_form"):
            numeric_vals = {}
            categorical_vals = {}

            st.markdown("**Numeric features**")
            for c in num_cols:
                numeric_vals[c] = st.text_input(c, value="")

            st.markdown("---")
            st.markdown("**Categorical features**")
            for c in cat_cols:
                categorical_vals[c] = st.text_input(c, value="")

            submit_btn = st.form_submit_button("Predict Survival")

        if submit_btn:
            def build_clinical_input_row(meta, user_num, user_cat):
                row = {}
                for col in meta.get("num_cols", []):
                    v = user_num.get(col, None)
                    if v is None or v == "":
                        row[col] = np.nan
                    else:
                        try:
                            row[col] = float(v)
                        except Exception:
                            row[col] = np.nan
                for col in meta.get("cat_cols", []):
                    v = user_cat.get(col, "")
                    row[col] = "Unknown" if (v is None or v == "") else str(v)

                df_raw = pd.DataFrame([row])
                df_ohe = pd.get_dummies(df_raw, columns=meta.get("cat_cols", []))
                X = df_ohe.reindex(columns=meta.get("feature_names", []), fill_value=0)
                return X

            try:
                X_input = build_clinical_input_row(gb_meta, numeric_vals, categorical_vals)
                y_pred = gb_model.predict(X_input)[0]
                outcome_label = inv_label_map.get(int(y_pred), str(y_pred))

                living_p = deceased_p = None
                if hasattr(gb_model, "predict_proba"):
                    proba = gb_model.predict_proba(X_input)[0]
                    living_p = float(proba[label_map.get("Living", 0)]) if proba.shape[0] == 2 else None
                    deceased_p = float(proba[label_map.get("Deceased", 1)]) if proba.shape[0] == 2 else None

                if outcome_label == "Deceased":
                    st.error(f"**Predicted Outcome:** {outcome_label}")
                else:
                    st.success(f"**Predicted Outcome:** {outcome_label}")

                if living_p is not None and deceased_p is not None:
                    st.write(f"- Probability **Living**: {living_p:.2%}")
                    st.write(f"- Probability **Deceased**: {deceased_p:.2%}")

            except Exception as e:
                st.error(f"Lỗi khi chạy tiên lượng: {e}")
