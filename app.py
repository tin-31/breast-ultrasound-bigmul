# app.py
# ==========================================
# 🩺 ỨNG DỤNG TRÍ TUỆ NHÂN TẠO HỖ TRỢ PHÂN TÍCH ẢNH SIÊU ÂM VÚ
# ==========================================
# ⚠️ Phiên bản dành cho nghiên cứu học thuật - Không sử dụng cho mục đích y tế thực tế.

import os
import math
import gdown
import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import altair as alt
import plotly.graph_objects as go
import matplotlib.cm as mpl_cm
from PIL import Image
from io import BytesIO
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.efficientnet import preprocess_input

# -----------------------------
# Config mô hình + hằng số
# -----------------------------
SEG_MODEL_ID = "1axOg7N5ssJrMec97eV-JMPzID26ynzN1"
CLF_MODEL_ID = "1fXPICuTkETep2oPiA56l0uMai2GusEJH"
SEG_MODEL_PATH = "seg_model.keras"
CLF_MODEL_PATH = "clf_model.h5"

# Theo mapping giao diện mặc định: ["Lành tính","Ác tính","Bình thường"]
MALIGNANT_INDEX = 1

# -----------------------------
# Hàm hỗ trợ (CBAM placeholders)
# -----------------------------
def spatial_mean(t): return tf.reduce_mean(t, axis=-1, keepdims=True)
def spatial_max(t):  return tf.reduce_max(t, axis=-1, keepdims=True)
def spatial_output_shape(s): return (s[0], s[1], s[2], 1)

# -----------------------------
# Tải mô hình (từ Google Drive nếu cần)
# -----------------------------
def download_model_if_missing(model_id, output_path, model_name):
    if not os.path.exists(output_path):
        st.info(f"📥 Đang tải {model_name}...")
        gdown.download(f"https://drive.google.com/uc?id={model_id}", output_path, quiet=False)
        st.success(f"✅ {model_name} đã được tải!")

download_model_if_missing(SEG_MODEL_ID, SEG_MODEL_PATH, "Mô hình phân đoạn")
download_model_if_missing(CLF_MODEL_ID, CLF_MODEL_PATH, "Mô hình phân loại")

# -----------------------------
# Load models (cached)
# -----------------------------
@st.cache_resource
def load_models():
    custom_objects = {"spatial_mean": spatial_mean, "spatial_max": spatial_max, "spatial_output_shape": spatial_output_shape}
    from tensorflow import keras
    try:
        keras.config.enable_unsafe_deserialization()
    except Exception:
        pass
    classifier = tf.keras.models.load_model(CLF_MODEL_PATH, compile=False)
    segmentor  = tf.keras.models.load_model(SEG_MODEL_PATH, custom_objects=custom_objects, compile=False)
    return classifier, segmentor

# -----------------------------
# Preprocess / postprocess ảnh
# -----------------------------
def classify_preproc(image_bytes):
    pil = Image.open(BytesIO(image_bytes)).convert("RGB").resize((224,224))
    x = preprocess_input(np.expand_dims(img_to_array(pil), axis=0))
    return x, pil

def segment_preproc(image_bytes):
    pil = Image.open(BytesIO(image_bytes)).convert("RGB").resize((256,256))
    x = np.expand_dims(np.array(pil)/255.0, axis=0)
    return x, pil

def segment_postproc(img_batch_256, mask_softmax, alpha=0.5):
    base = np.squeeze(img_batch_256[0])  # (256,256,3) in [0,1]
    argmax = np.argmax(mask_softmax, axis=-1).astype(np.uint8)
    # If mask_softmax has at least 3 channels, channel 2 is malignant prob (index 2)
    p_malig_map = mask_softmax[..., 2] if mask_softmax.shape[-1] >= 3 else (argmax==2).astype(float)

    col = np.zeros_like(base, dtype=np.float32)
    col[argmax==1] = np.array([0.0,1.0,0.0])   # green for benign
    col[argmax==2] = np.array([1.0,0.0,0.0])   # red for malignant

    out = base.copy()
    mask_pixels = argmax > 0
    out[mask_pixels] = base[mask_pixels] * (1-alpha) + col[mask_pixels] * alpha
    overlay_pil = Image.fromarray((out*255).astype(np.uint8))
    return overlay_pil, argmax, p_malig_map

# -----------------------------
# Mask features + clinical risk + combiner
# -----------------------------
def compute_mask_features(mask_argmax):
    H,W = mask_argmax.shape
    total = float(H*W)
    lesion = mask_argmax > 0
    area_ratio = float(np.sum(lesion))/total
    malignant_area_ratio = float(np.sum(mask_argmax==2))/total
    ys,xs = np.where(lesion)
    if ys.size > 0:
        y1,y2 = int(ys.min()), int(ys.max())
        x1,x2 = int(xs.min()), int(xs.max())
        approx_diam_px = max(y2-y1+1, x2-x1+1)
    else:
        approx_diam_px = 0
    return {"area_ratio":area_ratio, "malignant_area_ratio":malignant_area_ratio, "approx_diam_px":int(approx_diam_px)}

def clinical_risk_points(form):
    pts = 0.0
    explain = []
    age = form.get("age",0)
    if age>=70: pts+=3; explain.append("Tuổi ≥70 (+3)")
    elif age>=50: pts+=2; explain.append("Tuổi 50–69 (+2)")
    elif age>=40: pts+=1; explain.append("Tuổi 40–49 (+1)")
    if form.get("sex")=="Nữ": pts+=1; explain.append("Giới nữ (+1)")
    fam = form.get("family_history","Không")
    if fam=="1 người": pts+=2; explain.append("Gia đình: 1 người (+2)")
    elif fam=="≥2 người": pts+=3; explain.append("Gia đình: ≥2 người (+3)")
    mut = form.get("genetic_mutation","Không/Không biết")
    if mut=="BRCA1": pts+=5; explain.append("BRCA1 (+5)")
    elif mut=="BRCA2": pts+=4; explain.append("BRCA2 (+4)")
    elif mut=="Khác": pts+=2; explain.append("Đột biến khác (+2)")
    if form.get("personal_cancer_history",False): pts+=4; explain.append("Từng ung thư vú (+4)")
    if form.get("high_risk_lesion",False): pts+=2; explain.append("Tổn thương nguy cơ cao (+2)")
    if form.get("chest_radiation_young",False): pts+=4; explain.append("Xạ trị ngực <30 tuổi (+4)")
    if form.get("early_menarche",False): pts+=1; explain.append("Kinh sớm (<12) (+1)")
    if form.get("late_menopause",False): pts+=1; explain.append("Mãn kinh muộn (>55) (+1)")
    if form.get("first_child_late_or_nulliparity",False): pts+=1; explain.append("Chưa sinh / con đầu >35 (+1)")
    if form.get("no_breastfeeding",False): pts+=1; explain.append("Không cho con bú (+1)")
    dens = form.get("breast_density","Không rõ")
    if dens=="B": pts+=1; explain.append("Mật độ B (+1)")
    elif dens=="C": pts+=2; explain.append("Mật độ C (+2)")
    elif dens=="D": pts+=3; explain.append("Mật độ D (+3)")
    if form.get("bmi_obese",False): pts+=1; explain.append("BMI ≥30 (+1)")
    if form.get("alcohol_high",False): pts+=1; explain.append("Rượu thường xuyên (+1)")
    if form.get("smoking",False): pts+=1; explain.append("Hút thuốc (+1)")
    if form.get("low_activity",False): pts+=1; explain.append("Ít vận động (+1)")
    return min(pts,20.0), explain

def _logit(p, eps=1e-6):
    p = float(np.clip(p, eps, 1-eps))
    return np.log(p/(1-p))

def _sigmoid(x):
    return 1.0/(1.0+math.exp(-x))

def combine_probabilities_rule_based(p_img_malignant, risk_points, mask_feats,
                                     w_clinical=1.0, w_area=0.3, w_size=0.1):
    risk_z = ((risk_points/20.0)-0.5)/0.25
    area_z = (mask_feats.get("area_ratio",0.0)-0.02)/0.03
    size_z = (mask_feats.get("approx_diam_px",0.0)-24.0)/16.0
    logit_final = _logit(p_img_malignant) + w_clinical*risk_z + w_area*area_z + w_size*size_z
    p_final = float(_sigmoid(logit_final))
    contrib = {"image_logit":float(_logit(p_img_malignant)), "risk_term":float(w_clinical*risk_z),
               "area_term":float(w_area*area_z), "size_term":float(w_size*size_z), "sum_logit":float(logit_final)}
    return p_final, contrib

# -----------------------------
# Explainable AI (Grad-CAM robust)
# -----------------------------
def _find_last_conv_layer(model):
    for layer in reversed(model.layers):
        # lựa chọn Conv2D types
        if isinstance(layer, (tf.keras.layers.Conv2D, tf.keras.layers.SeparableConv2D, tf.keras.layers.DepthwiseConv2D)):
            return layer.name
        # fallback: tìm output_shape có rank 4
        try:
            if hasattr(layer, "output_shape") and layer.output_shape and len(layer.output_shape)==4:
                return layer.name
        except Exception:
            pass
    return None

def make_gradcam_heatmap(img_for_cls, model, class_index=None, last_conv_layer_name=None):
    """
    img_for_cls: (1,224,224,3) preprocessed
    model: classifier
    class_index: int or None (auto argmax)
    returns heatmap normalized [0..1] (H,W)
    """
    last_conv_layer_name = last_conv_layer_name or _find_last_conv_layer(model)
    if last_conv_layer_name is None:
        raise RuntimeError("Không tìm thấy layer Conv phù hợp cho Grad-CAM.")
    # build grad model: conv_output + model.outputs
    outputs = model.outputs if isinstance(model.outputs, (list,tuple)) else [model.output]
    grad_model = tf.keras.models.Model([model.inputs], [model.get_layer(last_conv_layer_name).output, *outputs])

    with tf.GradientTape() as tape:
        outs = grad_model(img_for_cls, training=False)
        conv_out = outs[0]            # conv feature map
        preds = outs[1]               # may be tensor or list
        if isinstance(preds, (list,tuple)):
            preds = preds[0]
        preds = tf.convert_to_tensor(preds)
        if class_index is None:
            class_index = int(tf.argmax(preds[0]))
        loss = preds[:, class_index]

    grads = tape.gradient(loss, conv_out)
    if grads is None:
        raise RuntimeError("Gradients are None; cannot compute Grad-CAM.")
    pooled_grads = tf.reduce_mean(grads, axis=(0,1,2))
    conv_out = conv_out[0]
    heatmap = tf.reduce_sum(conv_out * pooled_grads, axis=-1)
    heatmap = tf.maximum(heatmap, 0)
    max_val = tf.reduce_max(heatmap)
    heatmap = heatmap / (max_val + 1e-6)
    return heatmap.numpy()

def overlay_gradcam_on_pil(pil_img, heatmap, alpha=0.35, colormap="jet"):
    # heatmap: [H,W] (0..1), pil_img size matches, else resize
    hm_img = Image.fromarray(np.uint8(255*heatmap)).resize(pil_img.size, Image.BILINEAR)
    hm_arr = np.asarray(hm_img)/255.0  # 0..1
    cmap = mpl_cm.get_cmap(colormap)
    colored = (cmap(hm_arr)[...,:3]*255).astype(np.uint8)
    overlay = Image.fromarray(colored).convert("RGBA")
    # alpha by heatmap
    alpha_mask = Image.fromarray((hm_arr*255).astype(np.uint8)).convert("L")
    overlay.putalpha(alpha_mask)
    base = pil_img.convert("RGBA")
    out = Image.alpha_composite(base, overlay).convert("RGB")
    return out

def malignant_prob_overlay_from_seg(base_pil_256, p_malig_map, alpha=0.65):
    heat = (np.clip(p_malig_map,0,1)*255).astype(np.uint8)
    heat_img = Image.fromarray(heat).resize(base_pil_256.size, Image.BILINEAR)
    overlay = Image.new("RGBA", base_pil_256.size, (255,0,0,0))
    overlay.putalpha(heat_img)
    return Image.alpha_composite(base_pil_256.convert("RGBA"), overlay).convert("RGB")

# -----------------------------
# Charts: Altair bar + Plotly gauge
# -----------------------------
def prob_bar_chart(p_vec):
    # p_vec order assumed [p_benign, p_malignant, p_normal]
    df = pd.DataFrame({"Lớp":["Bình thường","Lành tính","Ác tính"],
                       "Xác suất":[float(p_vec[2]), float(p_vec[0]), float(p_vec[1])]})
    chart = alt.Chart(df).mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
        x=alt.X("Lớp", sort=["Bình thường","Lành tính","Ác tính"]),
        y=alt.Y("Xác suất", scale=alt.Scale(domain=[0,1])),
        color=alt.Color("Lớp", scale=alt.Scale(range=["#9CA3AF","#10B981","#EF4444"])),
        tooltip=["Lớp","Xác suất"]
    ).properties(height=240)
    return chart

def gauge_pfinal_plotly(p_final):
    val = float(np.clip(p_final, 0, 1))*100.0
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=val,
        number={'suffix':'%','font':{'size':34,'color':'#e5e7eb'}},
        gauge={
            'axis': {'range':[0,100], 'tickcolor':'#9ca3af'},
            'bar': {'color':'#ef4444'},
            'bgcolor':'rgba(0,0,0,0)',
            'steps': [
                {'range':[0,15], 'color':'#064e3b'},
                {'range':[15,30], 'color':'#065f46'},
                {'range':[30,60], 'color':'#7c2d12'},
                {'range':[60,85], 'color':'#7f1d1d'},
                {'range':[85,100], 'color':'#991b1b'}
            ],
            'threshold': {'line':{'color':'#ffffff','width':3}, 'thickness':0.8, 'value': val}
        }
    ))
    fig.update_layout(height=260, margin=dict(l=10,r=10,t=8,b=8), paper_bgcolor='rgba(0,0,0,0)', font=dict(color='#e5e7eb'))
    return fig

# -----------------------------
# Prediction pipeline
# -----------------------------
def du_doan(file, classifier, segmentor):
    image_bytes = file.read()
    x_cls, pil224 = classify_preproc(image_bytes)
    x_seg, pil256 = segment_preproc(image_bytes)
    with tf.device("/CPU:0"):
        pred_class = classifier.predict(x_cls, verbose=0)[0]
        mask_soft = segmentor.predict(x_seg, verbose=0)[0]
    seg_overlay_pil, mask_argmax, p_malig_map = segment_postproc(x_seg, mask_soft)
    return pred_class, seg_overlay_pil, image_bytes, mask_argmax, pil224, p_malig_map, pil256, x_cls

# -----------------------------
# UI + Main
# -----------------------------
st.set_page_config(page_title="AI Phân tích Siêu âm Vú", layout="wide", page_icon="🩺")
st.markdown("""
<style>
.big-title {font-size:1.6rem; font-weight:700; color:#e5e7eb; margin-bottom:6px;}
.card {background:#0f172a; border:1px solid #1f2937; padding:1rem; border-radius:12px; color:#e5e7eb;}
.caption {color:#9CA3AF;}
</style>
""", unsafe_allow_html=True)

st.sidebar.title("📘 Danh mục")
chon_trang = st.sidebar.selectbox("Chọn nội dung hiển thị", ["Ứng dụng minh họa", "Giới thiệu", "Nguồn dữ liệu & Bản quyền"])

if chon_trang == "Giới thiệu":
    st.title("👩‍🔬 ỨNG DỤNG AI TRONG HỖ TRỢ CHẨN ĐOÁN SIÊU ÂM VÚ")
    st.markdown("Ứng dụng phục vụ **nghiên cứu học thuật**; không dùng cho chẩn đoán thực tế.")

elif chon_trang == "Ứng dụng minh họa":
    st.title("🩺 Minh họa mô hình AI trên ảnh siêu âm vú (kết hợp thông tin lâm sàng)")
    classifier, segmentor = load_models()

    with st.form("form_input"):
        colA, colB = st.columns([1.05,1.3])
        with colA:
            file = st.file_uploader("📤 Ảnh siêu âm (JPG/PNG)", type=["jpg","png"])
            st.caption("Minh họa kỹ thuật; không có giá trị chẩn đoán y tế.")
        with colB:
            st.markdown("### 🧍 Thông tin bệnh nhân")
            age = st.number_input("Tuổi", 15, 100, 45, step=1)
            sex = st.selectbox("Giới", ["Nữ","Nam"])
            family_history = st.selectbox("Người thân trực hệ mắc ung thư vú/buồng trứng", ["Không","1 người","≥2 người"])
            genetic_mutation = st.selectbox("Đột biến di truyền", ["Không/Không biết","BRCA1","BRCA2","Khác"])
            personal_cancer_history = st.checkbox("Từng mắc ung thư vú")
            high_risk_lesion = st.checkbox("Tổn thương nguy cơ cao (DCIS/LCIS/ADH)")
            chest_radiation_young = st.checkbox("Xạ trị vùng ngực (<30 tuổi)")
            early_menarche = st.checkbox("Có kinh sớm (<12)")
            late_menopause = st.checkbox("Mãn kinh muộn (>55)")
            first_child_late_or_nulliparity = st.checkbox("Chưa sinh / con đầu >35")
            no_breastfeeding = st.checkbox("Không cho con bú")
            density_sel = st.selectbox("Mật độ mô vú", ["Không rõ","A (thưa)","B","C","D (rất dày)"])
            density_norm = "Không rõ"
            if density_sel.startswith("A"): density_norm="A"
            elif density_sel.startswith("D"): density_norm="D"
            elif density_sel in ["B","C"]: density_norm=density_sel
            bmi_obese = st.checkbox("BMI ≥ 30")
            alcohol_high = st.checkbox("Uống rượu/bia thường xuyên")
            smoking = st.checkbox("Hút thuốc")
            low_activity = st.checkbox("Ít vận động")
            st.markdown("#### ⚖️ Tham số hợp nhất")
            w_clinical = st.slider("Trọng số nguy cơ lâm sàng", 0.0, 2.0, 1.0, 0.1)
            w_area = st.slider("Trọng số diện tích mask", 0.0, 1.0, 0.3, 0.05)
            w_size = st.slider("Trọng số kích thước ước lượng", 0.0, 1.0, 0.1, 0.05)
        submitted = st.form_submit_button("🚀 Phân tích")

    if not submitted:
        st.info("👆 Chọn ảnh và nhập thông tin, sau đó bấm Phân tích.")
    else:
        if file is None:
            st.warning("Vui lòng chọn một ảnh siêu âm.")
        else:
            with st.spinner("⏳ Đang suy luận..."):
                try:
                    pred_class, seg_overlay_pil, img_bytes, mask_argmax, pil224, p_malig_map, pil256, x_cls = du_doan(file, classifier, segmentor)
                except Exception as e:
                    st.error(f"Lỗi khi suy luận: {e}")
                    raise

            # Chuẩn hóa hiển thị ảnh (cùng kích thước)
            orig = Image.open(BytesIO(img_bytes)).convert("RGB")
            orig_show = orig.resize((512,512))
            seg_show  = seg_overlay_pil.resize((512,512))

            st.markdown("<div class='big-title'>🖼️ Ảnh & phân đoạn</div>", unsafe_allow_html=True)
            c1,c2 = st.columns(2)
            c1.image(orig_show, caption="Ảnh gốc (kích thước hiển thị 512×512)", use_container_width=True)
            c2.image(seg_show,  caption="Phân đoạn (Xanh: lành, Đỏ: ác)", use_container_width=True)

            # Prob bar
            st.markdown("<div class='big-title'>💡 Kết quả AI trên ảnh (softmax)</div>", unsafe_allow_html=True)
            st.altair_chart(prob_bar_chart(pred_class), use_container_width=True)

            # Mask features + risk points
            mask_feats = compute_mask_features(mask_argmax)
            form = {"age":age,"sex":sex,"family_history":family_history,"genetic_mutation":genetic_mutation,
                    "personal_cancer_history":personal_cancer_history,"high_risk_lesion":high_risk_lesion,
                    "chest_radiation_young":chest_radiation_young,"early_menarche":early_menarche,
                    "late_menopause":late_menopause,"first_child_late_or_nulliparity":first_child_late_or_nulliparity,
                    "no_breastfeeding":no_breastfeeding,"breast_density":density_norm,"bmi_obese":bmi_obese,
                    "alcohol_high":alcohol_high,"smoking":smoking,"low_activity":low_activity}
            risk_points, risk_explain = clinical_risk_points(form)

            st.markdown("<div class='big-title'>🧪 Đặc trưng từ phân đoạn</div>", unsafe_allow_html=True)
            a,b,c = st.columns(3)
            a.metric("Diện tích tổn thương (%)", f"{mask_feats['area_ratio']*100:.2f}%")
            b.metric("Tỉ lệ vùng ác tính (%)", f"{mask_feats['malignant_area_ratio']*100:.2f}%")
            c.metric("Đường kính ước lượng (px)", f"{mask_feats['approx_diam_px']}")

            st.markdown("<div class='big-title'>🧍 Điểm nguy cơ lâm sàng</div>", unsafe_allow_html=True)
            xcol,ycol = st.columns([1,2])
            xcol.metric("Risk Points (0–20)", f"{risk_points:.1f}")
            if risk_explain:
                ycol.caption("Các yếu tố đóng góp:")
                ycol.write("• " + "\n• ".join(risk_explain))
            else:
                ycol.caption("_Không có yếu tố nguy cơ nổi bật_")

            # Hợp nhất (Cách B)
            p_img_malig = float(pred_class[MALIGNANT_INDEX])
            p_final, contrib = combine_probabilities_rule_based(p_img_malig, risk_points, mask_feats, w_clinical, w_area, w_size)

            st.markdown("<div class='big-title'>🧮 Xác suất chẩn đoán cuối (kết hợp)</div>", unsafe_allow_html=True)
            gl, gr = st.columns([1,2])
            with gl:
                # Plotly gauge
                st.plotly_chart(gauge_pfinal_plotly(p_final), use_container_width=False)
            with gr:
                st.write(f"**p_img (ác tính, từ ảnh)** = `{p_img_malig:.3f}`")
                st.write(f"**p_final (ác tính, sau hợp nhất)** = **`{p_final:.3f}`**  (~ {p_final*100:.1f}%)")
                df_contrib = pd.DataFrame({
                    "Thành phần":["Ảnh (logit)","Lâm sàng (w*risk_z)","Diện tích (w*area_z)","Kích thước (w*size_z)"],
                    "Đóng góp":[contrib["image_logit"], contrib["risk_term"], contrib["area_term"], contrib["size_term"]]
                })
                bar_contrib = alt.Chart(df_contrib).mark_bar().encode(
                    x=alt.X("Thành phần", sort=None),
                    y=alt.Y("Đóng góp", scale=alt.Scale(domain=[min(-2, min(df_contrib["Đóng góp"])-0.2),
                                                                max( 2, max(df_contrib["Đóng góp"])+0.2)])),
                    color=alt.condition(alt.datum.Đóng góp>0, alt.value("#10B981"), alt.value("#EF4444"))
                ).properties(height=220)
                st.altair_chart(bar_contrib, use_container_width=True)
                st.caption("Các cột biểu diễn mức đóng góp (+/−) vào logit trước khi chuyển thành xác suất.")

            # Explainable AI (Grad-CAM & seg heatmap)
            st.markdown("<div class='big-title'>🧠 Explainable AI</div>", unsafe_allow_html=True)
            try:
                # Grad-CAM
                try:
                    heatmap = make_gradcam_heatmap(x_cls, classifier, class_index=MALIGNANT_INDEX)
                    gradcam_img = overlay_gradcam_on_pil(pil224, heatmap, alpha=0.35)
                except Exception as e:
                    gradcam_img = None
                    st.warning(f"Không tạo được Grad-CAM: {e}")
                # Malignant heatmap from seg
                malig_heat_img = malignant_prob_overlay_from_seg(pil256, p_malig_map)
                e1,e2 = st.columns(2)
                with e1:
                    if gradcam_img is not None:
                        e1.image(gradcam_img.resize((512,512)), caption="Grad-CAM (đỏ = ảnh quan trọng cho dự đoán ác tính)")
                    else:
                        e1.info("Grad-CAM không khả dụng.")
                with e2:
                    e2.image(malig_heat_img.resize((512,512)), caption="Heatmap xác suất ác tính từ phân đoạn")
            except Exception as e:
                st.error(f"Lỗi XAI: {e}")

            # Recommendation
            if p_final >= 0.85 or mask_feats["approx_diam_px"] >= 48:
                rec = "Nguy cơ **rất cao** → Khuyến cáo **tham vấn bác sĩ chuyên khoa + sinh thiết**."
            elif p_final >= 0.60:
                rec = "Nguy cơ **cao** → Tham vấn bác sĩ, cân nhắc sinh thiết."
            elif p_final >= 0.30:
                rec = "Nguy cơ **trung bình** → Chụp bổ sung / siêu âm lại và theo dõi sát."
            elif p_final >= 0.15:
                rec = "Nguy cơ **thấp–trung bình** → Theo dõi định kỳ."
            else:
                rec = "Nguy cơ **thấp** → Theo lịch tầm soát phù hợp tuổi/nguy cơ."
            st.info(rec)
            st.caption("Kết quả chỉ phục vụ nghiên cứu học thuật – không sử dụng cho chẩn đoán y tế thực tế.")

elif chon_trang == "Nguồn dữ liệu & Bản quyền":
    st.title("📊 Nguồn dữ liệu và bản quyền sử dụng")
    st.markdown("""
    | Nguồn | Giấy phép |
    |-------|-----------|
    | BUSI (Kaggle) | CC BY 4.0 |
    | BUS-UCLM (Kaggle) | CC BY-NC-SA 4.0 |
    | Breast Lesions USG (TCIA) | CC BY 3.0 |
    """)

st.markdown("""
---
📘 **Miễn trừ trách nhiệm:** Ứng dụng phục vụ **nghiên cứu – giáo dục**. Không sử dụng cho **chẩn đoán, điều trị**.
""")
