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

import joblib


# ============================
# 0) CUSTOM OBJECTS CBAM
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
# 1) DOWNLOAD MODELS
# ============================
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

drive_files = {
    # Segmentation + Classification (giữ nguyên như cũ)
    "Classifier_model_2.h5": "1fXPICuTkETep2oPiA56l0uMai2GusEJH",
    "best_model_cbam_attention_unet_fixed.keras": "1axOg7N5ssJrMec97eV-JMPzID26ynzN1",

    # ⭐ Clinical RandomForest mới
    # 🔥 THAY 2 ID NÀY = ID DRIVE BẠN MỚI TẢI LÊN
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
# 2) LOAD MODELS
# ============================
@st.cache_resource
def load_all_models():

    # Segmentation
    seg_model = load_model(
        os.path.join(MODEL_DIR, "best_model_cbam_attention_unet_fixed.keras"),
        compile=False,
        custom_objects=CUSTOM_OBJECTS,
        safe_mode=False
    )

    # Classification
    class_model = load_model(
        os.path.join(MODEL_DIR, "Classifier_model_2.h5"),
        compile=False
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
# IMAGE PROCESSING UTILS
# ============================
def get_input_hwc(model):
    shape = model.input_shape
    if isinstance(shape, list): shape = shape[0]
    _, H, W, C = shape
    return int(H), int(W), int(C)

def prep(gray, target_shape):
    H, W, C = target_shape
    resized = cv2.resize(gray, (W, H))
    if C == 1:
        x = resized.astype(np.float32)/255.0
        x = np.expand_dims(x, (0, -1))
    else:
        x = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB).astype(np.float32)/255.0
        x = np.expand_dims(x, 0)
    return x, resized


COLOR_B = np.array([0, 255, 0], np.float32)
COLOR_M = np.array([255, 0, 0], np.float32)
COLOR_G = (0, 255, 255)

def overlay(gray, mask, alpha=0.6):
    base = np.stack([gray]*3, axis=-1).astype(np.float32)
    out = base.copy()

    ben = mask == 1
    mal = mask == 2

    if ben.any(): out[ben] = (1-alpha)*out[ben] + alpha*COLOR_B
    if mal.any(): out[mal] = (1-alpha)*out[mal] + alpha*COLOR_M

    general = ((ben | mal)*255).astype(np.uint8)
    if general.any():
        ct, _ = cv2.findContours(general, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        out_uint8 = out.clip(0,255).astype(np.uint8)
        cv2.drawContours(out_uint8, ct, -1, COLOR_G, 2)
        return out_uint8

    return out.clip(0,255).astype(np.uint8)


# ============================
# UI
# ============================
st.set_page_config(page_title="Breast Cancer App", layout="wide")
st.title("🩺 Breast Cancer Prediction App")


tab1, tab2 = st.tabs(["🔎 Image Analysis", "📊 Clinical Prediction"])


# =====================================================
# TAB 1 — IMAGE
# =====================================================
with tab1:
    st.header("Ultrasound Image Analysis")

    upload = st.file_uploader("Upload PNG/JPG", ["png","jpg","jpeg"])
    if upload:
        arr = np.frombuffer(upload.read(), np.uint8)
        gray = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)

        # Prepare
        x_seg, g_seg = prep(gray, get_input_hwc(seg_model))
        x_clf, g_clf = prep(gray, get_input_hwc(class_model))

        # Segmentation
        seg_pred = seg_model.predict(x_seg, verbose=0)[0]
        mask = np.argmax(seg_pred, -1).astype(np.uint8)

        overlay_img = overlay(g_seg, mask)

        # Classification
        probs = class_model.predict(x_clf, verbose=0)[0]
        labels = ["benign","malignant","normal"]
        vi = {"benign":"U lành", "malignant":"U ác", "normal":"Bình thường"}
        idx = int(np.argmax(probs))

        col1, col2 = st.columns(2)
        with col1: st.image(g_clf, caption="Input Image")
        with col2: st.image(overlay_img, caption="Segmentation Result")

        st.success(f"Kết quả: **{vi[labels[idx]]}** ({probs[idx]*100:.1f}%)")

        df = pd.DataFrame({
            "Category": ["Benign","Malignant","Normal"],
            "Probability (%)": (probs*100).round(2)
        })
        st.altair_chart(
            alt.Chart(df).mark_bar().encode(
                x="Category",
                y="Probability (%)"
            ),
            use_container_width=True
        )


# =====================================================
# TAB 2 — CLINICAL
# =====================================================
with tab2:
    st.header("Clinical Survival Prediction")

    if clinical_model is None:
        st.error("❌ Clinical model not loaded.")
    else:
        feature_names = clinical_meta["feature_names"]
        num_cols = clinical_meta["num_cols"]
        cat_cols = clinical_meta["cat_cols"]
        label_map = clinical_meta["label_map"]
        inv_label = {v:k for k,v in label_map.items()}

        # FORM
        with st.form("clinical"):

            # Numeric
            age = st.number_input("Age at Diagnosis", 0,120,50)
            size = st.number_input("Tumor Size",0,200,20)
            lymph = st.number_input("Lymph nodes +",0,50,0)
            mut = st.number_input("Mutation Count",0,10000,0)
            npi = st.number_input("Nottingham Prognostic Index",0.0,10.0,4.0)
            os_m = st.number_input("Overall Survival (Months)",0.0,300.0,60.0)

            # Categorical
            sx = st.selectbox("Type of Breast Surgery",["Breast Conserving","Mastectomy"])
            grade = st.selectbox("Histologic Grade",[1,2,3])
            stage = st.selectbox("Tumor Stage",[1,2,3,4])
            sex = st.selectbox("Sex",["Female","Male"])
            cell = st.selectbox("Cellularity",["High","Low","Moderate"])
            chemo = st.selectbox("Chemotherapy",["No","Yes"])
            hormone = st.selectbox("Hormone Therapy",["No","Yes"])
            radio = st.selectbox("Radio Therapy",["No","Yes"])
            er = st.selectbox("ER Status",["Negative","Positive"])
            pr = st.selectbox("PR Status",["Negative","Positive"])
            her2 = st.selectbox("HER2 Status",["Negative","Positive"])
            gene = st.selectbox("3-Gene subtype",
                ["ER+/HER2+","ER+/HER2- High Prolif","ER+/HER2- Low Prolif",
                 "ER-/HER2+","ER-/HER2-"])
            pam50 = st.selectbox("Pam50 subtype",
                ["Basal-like","Claudin-low","HER2-enriched","Luminal A","Luminal B","Normal-like"])
            relapse = st.selectbox("Relapse Status",["Not Recurred","Recurred"])

            submit = st.form_submit_button("Predict")

        if submit:

            # Create empty row
            X = pd.DataFrame([np.zeros(len(feature_names))], columns=feature_names)

            # Numeric
            values = {
                "Age at Diagnosis": age,
                "Tumor Size": size,
                "Lymph nodes examined positive": lymph,
                "Mutation Count": mut,
                "Nottingham prognostic index": npi,
                "Overall Survival (Months)": os_m
            }
            for k,v in values.items():
                if k in X.columns:
                    X.at[0,k] = v

            # One-hot helper
            def set_dummy(col,val):
                key = f"{col}_{val}"
                if key in X.columns:
                    X.at[0,key] = 1
                elif f"{key}.0" in X.columns:
                    X.at[0,f"{key}.0"] = 1

            # Categorical
            set_dummy("Type of Breast Surgery", sx)
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
            set_dummy("3-Gene classifier subtype", gene)
            set_dummy("Pam50 + Claudin-low subtype", pam50)
            set_dummy("Relapse Free Status", relapse)

            # Predict
            y = int(clinical_model.predict(X)[0])
            pred_label = inv_label[y]

            prob = float(clinical_model.predict_proba(X)[0][ label_map["Deceased"] ])

            if pred_label == "Deceased":
                st.error(f"Predicted outcome: **{pred_label}**")
            else:
                st.success(f"Predicted outcome: **{pred_label}**")

            st.write(f"Probability of death: **{prob*100:.1f}%**")
