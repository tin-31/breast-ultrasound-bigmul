# ==========================================
# 🩺 ỨNG DỤNG TRÍ TUỆ NHÂN TẠO HỖ TRỢ PHÂN TÍCH ẢNH SIÊU ÂM VÚ
# ==========================================
# ⚠️ Phiên bản dành cho nghiên cứu học thuật - Không sử dụng cho mục đích y tế thực tế.

import os
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import cv2
import streamlit as st
import altair as alt

import tensorflow as tf
import keras
from keras.models import load_model
from keras.saving import register_keras_serializable
from tensorflow.keras.layers import Conv2D

import joblib
import gdown

# EfficientNetV2 preprocessing
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input as eff_preprocess

# 3D / DICOM
import nibabel as nib
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut

# ------------------------------------------------------------
# STREAMLIT CONFIG
# ------------------------------------------------------------
st.set_page_config(
    page_title="AI Phân tích Siêu âm Vú",
    layout="wide",
    page_icon="🩺"
)

# Đảm bảo load model Keras cũ
try:
    keras.config.enable_unsafe_deserialization()
except:
    pass

# ------------------------------------------------------------
# CUSTOM OBJECTS DÙNG CHO U-NET CBAM
# ------------------------------------------------------------
@register_keras_serializable(package="cbam", name="spatial_mean")
def spatial_mean(x):
    return tf.reduce_mean(x, axis=-1, keepdims=True)

@register_keras_serializable(package="cbam", name="spatial_max")
def spatial_max(x):
    return tf.reduce_max(x, axis=-1, keepdims=True)

@register_keras_serializable(package="cbam", name="spatial_output_shape")
def spatial_output_shape(input_shape):
    shp = tf.TensorShape(input_shape).as_list()
    if len(shp) == 4:
        return (shp[0], shp[1], shp[2], 1)
    if len(shp) == 3:
        return (shp[0], shp[1], 1)
    return shp

CUSTOM_OBJECTS = {
    "spatial_mean": spatial_mean,
    "spatial_max": spatial_max,
    "spatial_output_shape": spatial_output_shape,
}

# ------------------------------------------------------------
# GOOGLE DRIVE MODEL FILES
# ------------------------------------------------------------
MODEL_DIR = "models"

drive_files = {
    "breast_ultrasound_classifier_ft.keras": "1IdGtQ1Sh1J1NC6acJ9mIkQdlq_xEeCJV",
    "best_model_cbam_attention_unet_fixed.keras": "1axOg7N5ssJrMec97eV-JMPzID26ynzN1",
    "clinical_rf_model.joblib": "1zHBB05rVUK7H9eZ9y5N9stUZnhzYBafc",
    "clinical_rf_metadata.json": "1KHZWZXs8QV8jLNXBkAVsQa_DN3tHuXtx",
}

def download_models():
    os.makedirs(MODEL_DIR, exist_ok=True)
    for fname, fid in drive_files.items():
        p = os.path.join(MODEL_DIR, fname)
        if not os.path.exists(p):
            url = f"https://drive.google.com/uc?id={fid}"
            st.info(f"📥 Đang tải mô hình: {fname}")
            gdown.download(url, p, quiet=False)
            st.success(f"✔ Đã tải xong {fname}")

@st.cache_resource
def load_all_models():
    seg = load_model(
        os.path.join(MODEL_DIR, "best_model_cbam_attention_unet_fixed.keras"),
        compile=False,
        custom_objects=CUSTOM_OBJECTS,
        safe_mode=False,
    )

    clf = load_model(
        os.path.join(MODEL_DIR, "breast_ultrasound_classifier_ft.keras"),
        compile=False,
    )

    clinical = None
    meta = None
    try:
        clinical = joblib.load(os.path.join(MODEL_DIR, "clinical_rf_model.joblib"))
        meta = json.load(open(os.path.join(MODEL_DIR, "clinical_rf_metadata.json")))
    except:
        pass

    return seg, clf, clinical, meta

# ------------------------------------------------------------
# TIỀN XỬ LÝ
# ------------------------------------------------------------
def get_input_hwc(model):
    shape = model.input_shape
    if isinstance(shape, list):
        shape = shape[0]
    _, H, W, C = shape
    return int(H), int(W), int(C)

def prep_seg(gray, target_shape):
    H, W, C = target_shape
    resized = cv2.resize(gray, (W, H))
    if C == 1:
        x = resized.astype(np.float32) / 255.0
        x = np.expand_dims(x, (0, -1))
    else:
        x = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB).astype(np.float32) / 255.0
        x = np.expand_dims(x, 0)
    return x, resized

def prep_classifier(gray, clf_model):
    _, H, W, C = clf_model.input_shape
    gray_resized = cv2.resize(gray, (W, H))
    rgb = cv2.cvtColor(gray_resized, cv2.COLOR_GRAY2RGB)
    rgb = rgb.astype(np.float32)
    rgb_pp = eff_preprocess(rgb)
    x = np.expand_dims(rgb_pp, axis=0)
    return x, gray_resized

# ------------------------------------------------------------
# SEGMENTATION OVERLAY
# ------------------------------------------------------------
COLOR_B = np.array([0, 255, 0])
COLOR_M = np.array([255, 0, 0])
COLOR_G = (0, 255, 255)

def overlay_segmentation(gray, mask, alpha=0.6):
    base = np.stack([gray]*3, axis=-1).astype(np.float32)
    out = base.copy()

    ben = (mask == 1)
    mal = (mask == 2)

    if ben.any():
        out[ben] = 0.4*out[ben] + 0.6*COLOR_B
    if mal.any():
        out[mal] = 0.4*out[mal] + 0.6*COLOR_M

    general = ((ben | mal)*255).astype(np.uint8)
    if general.any():
        ct, _ = cv2.findContours(general, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        out_uint8 = out.clip(0,255).astype(np.uint8)
        cv2.drawContours(out_uint8, ct, -1, COLOR_G, 2)
        return out_uint8

    return out.clip(0,255).astype(np.uint8)

# ------------------------------------------------------------
# TÌM LỚP CONV2D CUỐI CÙNG (CHO GRAD-CAM)
# ------------------------------------------------------------
def find_last_conv_layer(model):
    for layer in reversed(model.layers):
        if isinstance(layer, Conv2D):
            return layer.name
    raise ValueError("Không tìm thấy Conv2D layer để làm Grad-CAM.")

# ------------------------------------------------------------
# GRAD-CAM
# ------------------------------------------------------------
def make_gradcam_heatmap(img_array, model, layer_name, class_index=None):
    last_conv = model.get_layer(layer_name)
    grad_model = keras.Model([model.inputs], [last_conv.output, model.output])

    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(img_array)
        conv_out = conv_out[0]
        preds = preds[0]
        if class_index is None:
            class_index = tf.argmax(preds)
        class_score = preds[class_index]

    grads = tape.gradient(class_score, conv_out)
    pooled = tf.reduce_mean(grads, axis=(0,1))
    heat = tf.reduce_sum(conv_out * pooled, axis=-1)
    heat = np.maximum(heat, 0)
    heat /= (heat.max() + 1e-8)
    return heat

def mask_heatmap_with_segmentation(heatmap, mask_resized):
    H, W = mask_resized.shape[:2]
    heat_resized = cv2.resize(heatmap, (W, H))
    lesion = (mask_resized==1)|(mask_resized==2)
    masked = np.zeros_like(heat_resized)
    masked[lesion] = heat_resized[lesion]
    if masked.max()>0:
        masked/=masked.max()
    return masked

def apply_gradcam_on_gray(gray_resized, heatmap, alpha=0.5, gamma=0.7, thresh=0.15):
    heatmap = np.power(heatmap, gamma)
    heatmap[heatmap < thresh] = 0
    heat_uint8 = np.uint8(255*heatmap)
    heat_color = cv2.applyColorMap(heat_uint8, cv2.COLORMAP_JET)
    base = cv2.cvtColor(gray_resized, cv2.COLOR_GRAY2BGR)
    return cv2.addWeighted(heat_color, alpha, base, 1-alpha, 0)

def overlay_contour(cam_img, mask_resized):
    cnt_mask = ((mask_resized==1)|(mask_resized==2))*255
    cnt_mask = cnt_mask.astype(np.uint8)
    out = cam_img.copy()
    if cnt_mask.any():
        ct,_ = cv2.findContours(cnt_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, ct, -1, (0,255,255), 2)
    return out

# ------------------------------------------------------------
# ĐỌC DICOM / NIFTI
# ------------------------------------------------------------
def load_nifti_slice(file, slice_strategy="middle"):
    img = nib.load(file)
    vol = img.get_fdata()
    mid = vol.shape[2]//2
    return vol[:,:,mid].astype(np.uint8)

def load_dicom_slice(file):
    ds = pydicom.dcmread(file)
    arr = apply_voi_lut(ds.pixel_array, ds).astype(np.float32)
    arr = (arr-arr.min())/(arr.max()-arr.min())*255
    return arr.astype(np.uint8)

def load_3d_slice(upload):
    suffix = Path(upload.name).suffix.lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(upload.read())
        tmp_path = tmp.name
    try:
        if suffix in [".nii",".gz"]:
            return load_nifti_slice(tmp_path), "3D"
        elif suffix==".dcm":
            return load_dicom_slice(tmp_path), "DICOM"
        else:
            st.error("❌ Định dạng này không hỗ trợ 3D.")
            return None,None
    except Exception as e:
        st.error(f"Lỗi đọc ảnh: {e}")
        return None,None

# ------------------------------------------------------------
# SIDEBAR
# ------------------------------------------------------------
st.sidebar.title("📘 Danh mục")
page = st.sidebar.selectbox("Chọn nội dung", ["Ứng dụng","Giới thiệu","Nguồn dữ liệu"])

# ------------------------------------------------------------
# GIỚI THIỆU
# ------------------------------------------------------------
if page=="Giới thiệu":
    st.title("Ứng dụng AI phân tích siêu âm vú")

# ------------------------------------------------------------
# NGUỒN DỮ LIỆU
# ------------------------------------------------------------
elif page=="Nguồn dữ liệu":
    st.title("Nguồn dữ liệu & bản quyền")

# ------------------------------------------------------------
# ỨNG DỤNG
# ------------------------------------------------------------
else:
    st.title("🩺 ỨNG DỤNG AI PHÂN TÍCH SIÊU ÂM VÚ")

    with st.spinner("Đang chuẩn bị mô hình..."):
        download_models()
        seg_model, class_model, clinical_model, clinical_meta = load_all_models()

    labels_clf = ["benign", "malignant", "normal"]
    vi_map = {"benign":"U lành","malignant":"U ác","normal":"Bình thường"}

    uploaded = st.file_uploader(
    "Tải ảnh siêu âm (PNG/JPG/NIFTI/DICOM)",
    ["png", "jpg", "jpeg", "nii", "nii.gz", "dcm"]
)

image_pred_probs = None
clinical_prob_death = None
clinical_pred_label = None

if uploaded:
    suffix = Path(uploaded.name).suffix.lower()
    gray = None

    # 1) ẢNH 2D: PNG/JPG
    if suffix in [".png", ".jpg", ".jpeg"]:
        arr = np.frombuffer(uploaded.read(), np.uint8)
        gray = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)

    # 2) ẢNH 3D: NIfTI / DICOM
    elif suffix in [".nii", ".nii.gz", ".gz", ".dcm"]:
        gray, _ = load_3d_slice(uploaded)

    # 3) Định dạng không hỗ trợ
    else:
        st.error("❌ Định dạng ảnh không được hỗ trợ.")
        st.stop()

    # Nếu vì lý do nào đó vẫn không đọc được ảnh → dừng luôn
    if gray is None:
        st.error("❌ Không đọc được ảnh từ file. Vui lòng thử lại ảnh khác.")
        st.stop()

    # Chuẩn hóa về 0–255
    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)

    # ---- từ đây trở xuống giữ nguyên như bạn đang có ----
    x_seg, g_seg = prep_seg(gray, get_input_hwc(seg_model))
    seg_pred = seg_model.predict(x_seg, verbose=0)[0]
    mask = np.argmax(seg_pred, -1).astype(np.uint8)
    overlay_img = overlay_segmentation(g_seg, mask)

    x_clf, g_clf = prep_classifier(gray, class_model)
    probs = class_model.predict(x_clf, verbose=0)[0]
    idx = int(np.argmax(probs))
    image_pred_probs = probs
    pred_label_vi = vi_map[labels_clf[idx]]
    # … rồi đến phần Grad-CAM như bạn đã có

        # GRAD-CAM
        try:
            last_conv = find_last_conv_layer(class_model)
            class_idx_cam = labels_clf.index("malignant")

            heat_raw = make_gradcam_heatmap(x_clf, class_model, last_conv, class_idx_cam)

            mask_resized = cv2.resize(mask,(g_clf.shape[1], g_clf.shape[0]), 
                                      interpolation=cv2.INTER_NEAREST)

            heat_masked = mask_heatmap_with_segmentation(heat_raw, mask_resized)
            grad_img = apply_gradcam_on_gray(g_clf, heat_masked)
            grad_img = overlay_contour(grad_img, mask_resized)

        except Exception as e:
            grad_img=None
            st.warning(f"Không tạo được Grad-CAM: {e}")

        # HIỂN THỊ
        col1,col2,col3 = st.columns(3)
        with col1: st.image(g_clf, caption="Ảnh chuẩn hóa")
        with col2: st.image(overlay_img, caption="Phân đoạn")
        with col3:
            if grad_img is not None:
                st.image(grad_img, caption="Grad-CAM + Contour")
            else:
                st.info("Không có Grad-CAM.")

        st.success(f"🔍 Kết luận mô hình: **{pred_label_vi}** ({probs[idx]*100:.1f}%)")

    # LÂM SÀNG ------------------------------------------------
    st.markdown("---")
    st.subheader("📊 Mô hình lâm sàng")

    if clinical_model and clinical_meta:
        feature_names = clinical_model.feature_names_in_
        label_map = clinical_meta["label_map"]
        inv_label = {v:k for k,v in label_map.items()}

        with st.form("clinical"):
            col1,col2,col3 = st.columns(3)
            with col1:
                age = st.number_input("Age",0,120,50)
                size = st.number_input("Tumor Size",0,200,20)
                lymph = st.number_input("Positive Nodes",0,50,0)
                mut = st.number_input("Mutation Count",0,5000,0)
                npi = st.number_input("NPI",0.0,10.0,4.0)
                os_m = st.number_input("OS Months",0.0,300.0,60.0)
            with col2:
                sx = st.selectbox("Surgery",["Breast Conserving","Mastectomy"])
                grade = st.selectbox("Grade",[1,2,3])
                stage = st.selectbox("Stage",[1,2,3,4])
                sex = st.selectbox("Sex",["Female","Male"])
                cell = st.selectbox("Cellularity",["High","Low","Moderate"])
                chemo = st.selectbox("Chemotherapy",["No","Yes"])
            with col3:
                hormone = st.selectbox("Hormone",["No","Yes"])
                radio = st.selectbox("Radiotherapy",["No","Yes"])
                er = st.selectbox("ER",["Negative","Positive"])
                pr = st.selectbox("PR",["Negative","Positive"])
                her2 = st.selectbox("HER2",["Negative","Positive"])
                relapse = st.selectbox("Relapse",["Not Recurred","Recurred"])

            submit = st.form_submit_button("Dự đoán")

        if submit:
            row = {
                "Age at Diagnosis":age,
                "Tumor Size":size,
                "Lymph nodes examined positive":lymph,
                "Mutation Count":mut,
                "Nottingham prognostic index":npi,
                "Overall Survival (Months)":os_m,
                "Type of Breast Surgery":sx,
                "Neoplasm Histologic Grade":grade,
                "Tumor Stage":stage,
                "Sex":sex,
                "Cellularity":cell,
                "Chemotherapy":chemo,
                "Hormone Therapy":hormone,
                "Radio Therapy":radio,
                "ER Status":er,
                "PR Status":pr,
                "HER2 Status":her2,
                "Relapse Free Status":relapse,
            }

            X = pd.DataFrame([row], columns=feature_names)
            y = int(clinical_model.predict(X)[0])
            pred_label = inv_label[y]
            clinical_pred_label = pred_label

            prob = clinical_model.predict_proba(X)[0][label_map.get("Deceased",y)]
            clinical_prob_death = prob

            st.success(f"📋 Dự đoán lâm sàng: **{pred_label}**")
            st.write(f"Xác suất tử vong ~ {prob*100:.1f}%")

    # ĐÁNH GIÁ TỔNG HỢP ---------------------------------------
    st.markdown("---")
    st.subheader("🧠 Đánh giá tổng hợp")

    if image_pred_probs is not None and clinical_prob_death is not None:
        p_mal = float(image_pred_probs[labels_clf.index("malignant")])
        combined = 0.6*p_mal + 0.4*clinical_prob_death

        if combined < 0.3:
            grp="Nguy cơ thấp"
        elif combined < 0.6:
            grp="Nguy cơ trung bình"
        else:
            grp="Nguy cơ cao"

        st.write(f"Điểm nguy cơ: **{combined*100:.1f}%** → {grp}")

# ------------------------------------------------------------
# FOOTER
# ------------------------------------------------------------
st.markdown("""
---
Ứng dụng dành cho nghiên cứu học thuật, không dùng cho chẩn đoán thực tế.
""")

