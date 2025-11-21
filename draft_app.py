import os
import json

import gdown
import numpy as np
import pandas as pd
import cv2
import streamlit as st
import altair as alt

import tensorflow as tf
from keras.models import load_model
from keras.saving import register_keras_serializable

import joblib


# ============================
# 0) STREAMLIT CONFIG
# ============================
st.set_page_config(page_title="Breast Cancer App", layout="wide")


# ============================
# 1) CUSTOM OBJECTS CBAM
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
# 2) DOWNLOAD MODELS
# ============================
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

drive_files = {
    # Segmentation + Classification
    "Classifier_model_2.h5": "1fXPICuTkETep2oPiA56l0uMai2GusEJH",
    "best_model_cbam_attention_unet_fixed.keras": "1axOg7N5ssJrMec97eV-JMPzID26ynzN1",

    # Clinical RandomForest
    "clinical_rf_model.joblib": "1zHBB05rVUK7H9eZ9y5N9stUZnhzYBafc",
    "clinical_rf_metadata.json": "1KHZWZXs8QV8jLNXBkAVsQa_DN3tHuXtx",
}

with st.spinner("⏳ Downloading models..."):
    for fname, fid in drive_files.items():
        path = os.path.join(MODEL_DIR, fname)
        if not os.path.exists(path):
            url = f"https://drive.google.com/uc?id={fid}"
            gdown.download(url, path, quiet=False)


# ============================
# 3) LOAD MODELS
# ============================
@st.cache_resource
def load_all_models():
    # Segmentation
    seg_model = load_model(
        os.path.join(MODEL_DIR, "best_model_cbam_attention_unet_fixed.keras"),
        compile=False,
        custom_objects=CUSTOM_OBJECTS,
        safe_mode=False,
    )

    # Classification
    class_model = load_model(
        os.path.join(MODEL_DIR, "Classifier_model_2.h5"),
        compile=False,
    )

    # Clinical (joblib)
    clinical_model = None
    clinical_meta = None

    try:
        clinical_model = joblib.load(os.path.join(MODEL_DIR, "clinical_rf_model.joblib"))
        with open(os.path.join(MODEL_DIR, "clinical_rf_metadata.json"), "r") as f:
            clinical_meta = json.load(f)
    except Exception as e:
        st.error(f"❌ Could not load clinical RF model: {e}")

    return seg_model, class_model, clinical_model, clinical_meta


seg_model, class_model, clinical_model, clinical_meta = load_all_models()


# ============================
# 4) IMAGE PROCESSING UTILS
# ============================
def get_input_hwc(model):
    shape = model.input_shape
    if isinstance(shape, list):
        shape = shape[0]
    _, H, W, C = shape
    return int(H), int(W), int(C)


def prep(gray, target_shape):
    H, W, C = target_shape
    resized = cv2.resize(gray, (W, H))
    if C == 1:
        x = resized.astype(np.float32) / 255.0
        x = np.expand_dims(x, (0, -1))
    else:
        x = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB).astype(np.float32) / 255.0
        x = np.expand_dims(x, 0)
    return x, resized


COLOR_B = np.array([0, 255, 0], np.float32)
COLOR_M = np.array([255, 0, 0], np.float32)
COLOR_G = (0, 255, 255)


def overlay(gray, mask, alpha=0.6):
    base = np.stack([gray] * 3, axis=-1).astype(np.float32)
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
# 5) MAIN APP UI (COMBINED)
# ============================
st.title("🩺 Breast Cancer Prediction App")
st.write(
    "Ứng dụng hỗ trợ bác sĩ: phân tích **siêu âm vú** + "
    "**dữ liệu lâm sàng** và hiển thị đánh giá tổng hợp (chỉ mang tính tham khảo, không thay thế chẩn đoán của bác sĩ)."
)

# Các biến để kết hợp kết quả
image_pred_label_en = None
image_pred_label_vi = None
image_pred_probs = None

clinical_pred_label = None
clinical_prob_death = None

# =====================================================
# 5.1 PHÂN TÍCH HÌNH ẢNH
# =====================================================
st.header("🔎 Ultrasound Image Analysis")

upload = st.file_uploader("Upload ảnh siêu âm (PNG/JPG)", ["png", "jpg", "jpeg"])

if upload:
    arr = np.frombuffer(upload.read(), np.uint8)
    gray = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)

    # Chuẩn bị input cho 2 model
    x_seg, g_seg = prep(gray, get_input_hwc(seg_model))
    x_clf, g_clf = prep(gray, get_input_hwc(class_model))

    # Segmentation
    seg_pred = seg_model.predict(x_seg, verbose=0)[0]
    mask = np.argmax(seg_pred, -1).astype(np.uint8)
    overlay_img = overlay(g_seg, mask)

    # Classification
    probs = class_model.predict(x_clf, verbose=0)[0]
    labels = ["benign", "malignant", "normal"]
    vi_map = {"benign": "U lành", "malignant": "U ác", "normal": "Bình thường"}
    idx = int(np.argmax(probs))

    image_pred_label_en = labels[idx]
    image_pred_label_vi = vi_map[image_pred_label_en]
    image_pred_probs = probs

    col1, col2 = st.columns(2)
    with col1:
        st.image(g_clf, caption="Input Ultrasound Image", use_column_width=True)
    with col2:
        st.image(overlay_img, caption="Segmentation Result", use_column_width=True)

    st.success(f"Kết quả mô hình hình ảnh: **{image_pred_label_vi}** ({probs[idx] * 100:.1f}%)")

    df_img = pd.DataFrame(
        {
            "Category": ["Benign", "Malignant", "Normal"],
            "Probability (%)": (probs * 100).round(2),
        }
    )

    st.altair_chart(
        alt.Chart(df_img)
        .mark_bar()
        .encode(
            x="Category",
            y="Probability (%)",
            tooltip=["Category", "Probability (%)"],
        ),
        use_container_width=True,
    )
else:
    st.info("Vui lòng tải ảnh siêu âm để mô hình xử lý.")


# =====================================================
# 5.2 DỰ ĐOÁN LÂM SÀNG
# =====================================================
st.header("📊 Clinical Survival Prediction")

if clinical_model is None or clinical_meta is None:
    st.error("❌ Clinical model not loaded – kiểm tra lại file joblib/json.")
else:
    feature_names = clinical_model.feature_names_in_
    label_map = clinical_meta["label_map"]  # ví dụ: {"Alive": 0, "Deceased": 1}
    inv_label = {v: k for k, v in label_map.items()}

    with st.form("clinical_form"):
        col_a, col_b, col_c = st.columns(3)

        with col_a:
            age = st.number_input("Age at Diagnosis", 0, 120, 50)
            size = st.number_input("Tumor Size", 0, 200, 20)
            lymph = st.number_input("Lymph nodes examined positive", 0, 50, 0)
            mut = st.number_input("Mutation Count", 0, 10000, 0)
            npi = st.number_input("Nottingham prognostic index", 0.0, 10.0, 4.0)
            os_m = st.number_input("Overall Survival (Months)", 0.0, 300.0, 60.0)

        with col_b:
            sx = st.selectbox("Type of Breast Surgery", ["Breast Conserving", "Mastectomy"])
            grade = st.selectbox("Neoplasm Histologic Grade", [1, 2, 3])
            stage = st.selectbox("Tumor Stage", [1, 2, 3, 4])
            sex = st.selectbox("Sex", ["Female", "Male"])
            cell = st.selectbox("Cellularity", ["High", "Low", "Moderate"])
            chemo = st.selectbox("Chemotherapy", ["No", "Yes"])
            hormone = st.selectbox("Hormone Therapy", ["No", "Yes"])

        with col_c:
            radio = st.selectbox("Radio Therapy", ["No", "Yes"])
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
                ["Basal-like", "Claudin-low", "HER2-enriched", "Luminal A", "Luminal B", "Normal-like"],
            )
            relapse = st.selectbox("Relapse Free Status", ["Not Recurred", "Recurred"])

        submit_clinical = st.form_submit_button("Predict Clinical Outcome")

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

        # Lấy xác suất tử vong nếu có key "Deceased" trong label_map
        if "Deceased" in label_map:
            prob = float(clinical_model.predict_proba(X)[0][label_map["Deceased"]])
        else:
            # fallback: lấy max probability (không chuẩn bằng)
            prob = float(np.max(clinical_model.predict_proba(X)[0]))

        clinical_prob_death = prob

        if pred_label == "Deceased":
            st.error(f"Predicted outcome: **{pred_label}**")
        else:
            st.success(f"Predicted outcome: **{pred_label}**")

        st.write(f"Estimated probability of death: **{prob * 100:.1f}%**")


# =====================================================
# 5.3 KẾT HỢP 2 KẾT QUẢ (IMAGE + CLINICAL)
# =====================================================
st.markdown("---")
st.header("🧠 Combined AI Assessment")

if image_pred_label_en is None and (clinical_pred_label is None or clinical_prob_death is None):
    st.info("Khi bạn đã có **kết quả hình ảnh** và **kết quả lâm sàng**, hệ thống sẽ đưa ra nhận định tổng hợp tại đây.")
else:
    # Mô tả từ phía hình ảnh
    img_text = None
    if image_pred_label_en is not None:
        img_text = f"Hình ảnh siêu âm được mô hình phân loại là: **{image_pred_label_vi}**."

    # Mô tả từ phía lâm sàng
    clin_text = None
    if clinical_pred_label is not None and clinical_prob_death is not None:
        clin_text = (
            f"Mô hình lâm sàng dự đoán kết cục: **{clinical_pred_label}** "
            f"với xác suất tử vong ước tính khoảng **{clinical_prob_death * 100:.1f}%**."
        )

    # Hiển thị riêng lẻ
    if img_text:
        st.write("🔬 **Nhận định từ hình ảnh:**")
        st.write(img_text)

    if clin_text:
        st.write("📋 **Nhận định từ dữ liệu lâm sàng:**")
        st.write(clin_text)

    # Tổng hợp định tính
    if image_pred_label_en is not None and clinical_pred_label is not None and clinical_prob_death is not None:
        if image_pred_label_en == "malignant" and clinical_prob_death >= 0.5:
            st.error(
                "📌 **Đánh giá tổng hợp:**\n\n"
                "- Hình ảnh gợi ý **tổn thương ác tính**.\n"
                "- Mô hình lâm sàng cho thấy **nguy cơ tử vong cao**.\n\n"
                "👉 Cần được bác sĩ chuyên khoa đánh giá khẩn và xem xét phác đồ điều trị phù hợp."
            )
        elif image_pred_label_en == "malignant" and clinical_prob_death < 0.5:
            st.warning(
                "📌 **Đánh giá tổng hợp:**\n\n"
                "- Hình ảnh gợi ý **tổn thương ác tính**.\n"
                "- Nguy cơ tử vong dự đoán **không quá cao**, nhưng vẫn cần theo dõi và điều trị sát.\n\n"
                "👉 Đề nghị trao đổi kết quả với bác sĩ chuyên khoa để có chỉ định tiếp theo."
            )
        elif image_pred_label_en in ["benign", "normal"] and clinical_prob_death < 0.5 and clinical_pred_label != "Deceased":
            st.success(
                "📌 **Đánh giá tổng hợp:**\n\n"
                "- Hình ảnh **không gợi ý tổn thương ác tính rõ ràng**.\n"
                "- Mô hình lâm sàng dự đoán **kết cục sống** với nguy cơ tử vong thấp.\n\n"
                "👉 Dù dấu hiệu hiện tại tương đối thuận lợi, bệnh nhân vẫn cần tái khám định kỳ theo chỉ định."
            )
        else:
            st.info(
                "📌 **Đánh giá tổng hợp:**\n\n"
                "- Kết quả mô hình hình ảnh và lâm sàng **chưa hoàn toàn đồng nhất** hoặc ở mức nguy cơ trung gian.\n"
                "- Cần **kết hợp thêm thông tin lâm sàng, xét nghiệm, sinh thiết** và đánh giá trực tiếp bởi bác sĩ.\n\n"
                "👉 Mô hình chỉ mang tính hỗ trợ, không thay thế quyết định chẩn đoán/điều trị."
            )

st.markdown(
    "> ⚠️ *Lưu ý: Tất cả kết quả trên chỉ có tính chất tham khảo, không dùng để tự chẩn đoán hay tự điều trị. "
    "Quyết định cuối cùng phải do bác sĩ lâm sàng đánh giá.*"
)
