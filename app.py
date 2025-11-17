# ==========================================
# 🩺 ỨNG DỤNG TRÍ TUỆ NHÂN TẠO HỖ TRỢ PHÂN TÍCH ẢNH SIÊU ÂM VÚ
# ==========================================
# ⚠️ Phiên bản dành cho nghiên cứu học thuật - Không sử dụng cho mục đích y tế thực tế.
# ⚠️ Ứng dụng này chỉ mang tính minh họa kỹ thuật và học thuật.

import os, math
import gdown
import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import altair as alt
from PIL import Image, ImageDraw, ImageOps
from io import BytesIO
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.efficientnet import preprocess_input

# ==============================
# ⚙️ Cấu hình mô hình
# ==============================
SEG_MODEL_ID = "1axOg7N5ssJrMec97eV-JMPzID26ynzN1"
CLF_MODEL_ID = "1fXPICuTkETep2oPiA56l0uMai2GusEJH"
SEG_MODEL_PATH = "seg_model.keras"
CLF_MODEL_PATH = "clf_model.h5"

# Vị trí lớp "Ác tính" trong softmax (mapping: ["Lành tính","Ác tính","Bình thường"])
MALIGNANT_INDEX = 1

# ==============================
# 🔹 Hàm xử lý trung gian cho CBAM
# ==============================
def spatial_mean(t): return tf.reduce_mean(t, axis=-1, keepdims=True)
def spatial_max(t):  return tf.reduce_max(t, axis=-1, keepdims=True)
def spatial_output_shape(s): return (s[0], s[1], s[2], 1)

# ==============================
# 🔹 Tự động tải mô hình
# ==============================
def download_model(model_id, output_path, model_name):
    if not os.path.exists(output_path):
        st.info(f"📥 Đang tải {model_name} (ID: {model_id})...")
        gdown.download(f"https://drive.google.com/uc?id={model_id}", output_path, quiet=False)
        st.success(f"✅ {model_name} đã được tải xong!")

download_model(SEG_MODEL_ID, SEG_MODEL_PATH, "mô hình phân đoạn")
download_model(CLF_MODEL_ID, CLF_MODEL_PATH, "mô hình phân loại")

# ==============================
# 🔹 Tải mô hình an toàn
# ==============================
@st.cache_resource
def load_models():
    CUSTOM_OBJECTS = {"spatial_mean": spatial_mean, "spatial_max": spatial_max, "spatial_output_shape": spatial_output_shape}
    from tensorflow import keras
    try: keras.config.enable_unsafe_deserialization()
    except Exception: pass

    classifier = tf.keras.models.load_model(CLF_MODEL_PATH, compile=False)
    segmentor  = tf.keras.models.load_model(SEG_MODEL_PATH, custom_objects=CUSTOM_OBJECTS, compile=False)
    return classifier, segmentor

# ==============================
# 🔹 Tiền xử lý ảnh
# ==============================
def classify_preprop(image_bytes, return_pil=False):
    pil224 = Image.open(BytesIO(image_bytes)).convert("RGB").resize((224, 224))
    x = preprocess_input(np.expand_dims(img_to_array(pil224), axis=0))
    return (x, pil224) if return_pil else x

def segment_preprop(image_bytes, return_pil=False):
    pil256 = Image.open(BytesIO(image_bytes)).convert("RGB").resize((256, 256))
    x = np.expand_dims(np.array(pil256) / 255.0, axis=0)
    return (x, pil256) if return_pil else x

# ==============================
# 🔹 Hậu xử lý ảnh phân đoạn
# ==============================
def segment_postprop(img_batch_256, mask_softmax, alpha=0.5):
    """Trả về overlay (PIL) + mask argmax + map xác suất ác tính."""
    base = np.squeeze(img_batch_256[0])  # [0..1]
    argmax = np.argmax(mask_softmax, axis=-1)        # (256,256)
    p_malig_map = mask_softmax[..., 2] if mask_softmax.shape[-1] >= 3 else (argmax==2).astype(float)

    # tô màu
    col = np.zeros_like(base, dtype=np.float32)
    col[argmax==1] = np.array([0.0,1.0,0.0])  # xanh: lành
    col[argmax==2] = np.array([1.0,0.0,0.0])  # đỏ: ác

    out = base.copy()
    m = argmax > 0
    out[m] = base[m]*(1-alpha) + col[m]*alpha
    overlay = Image.fromarray((out*255).astype(np.uint8))
    return overlay, argmax.astype(np.uint8), p_malig_map.astype(np.float32)

# ==============================
# 🔹 Đặc trưng từ mask & hợp nhất (Cách B)
# ==============================
def compute_mask_features(mask_argmax):
    H, W = mask_argmax.shape
    total = float(H*W)
    lesion = mask_argmax>0
    area_ratio = float(np.sum(lesion))/total
    malignant_area_ratio = float(np.sum(mask_argmax==2))/total

    ys, xs = np.where(lesion)
    if ys.size>0:
        y1,y2 = int(ys.min()), int(ys.max()); x1,x2 = int(xs.min()), int(xs.max())
        approx_diam_px = max(y2-y1+1, x2-x1+1)
    else:
        approx_diam_px = 0
    return {"area_ratio":area_ratio, "malignant_area_ratio":malignant_area_ratio, "approx_diam_px":int(approx_diam_px)}

def clinical_risk_points(form):
    pts, explain = 0.0, []
    age = form.get("age",0)
    if age>=70: pts+=3; explain.append("Tuổi ≥70 (+3)")
    elif age>=50: pts+=2; explain.append("Tuổi 50–69 (+2)")
    elif age>=40: pts+=1; explain.append("Tuổi 40–49 (+1)")
    if form.get("sex")=="Nữ": pts+=1; explain.append("Giới nữ (+1)")
    fam = form.get("family_history","Không")
    if fam=="1 người": pts+=2; explain.append("Gia đình: 1 người (+2)")
    elif fam=="≥2 người": pts+=3; explain.append("Gia đình: ≥2 người (+3)")
    mut = form.get("genetic_mutation","Không/Không biết")
    if mut=="BRCA1": pts+=5; explain.append("Đột biến BRCA1 (+5)")
    elif mut=="BRCA2": pts+=4; explain.append("Đột biến BRCA2 (+4)")
    elif mut=="Khác": pts+=2; explain.append("Đột biến khác (+2)")
    if form.get("personal_cancer_history",False): pts+=4; explain.append("Từng ung thư vú (+4)")
    if form.get("high_risk_lesion",False): pts+=2; explain.append("Tổn thương nguy cơ cao (+2)")
    if form.get("chest_radiation_young",False): pts+=4; explain.append("Xạ trị ngực <30 tuổi (+4)")
    if form.get("early_menarche",False): pts+=1; explain.append("Có kinh sớm (<12) (+1)")
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

def _logit(p, eps=1e-6): p=float(np.clip(p,eps,1-eps)); return np.log(p/(1-p))
def _sigmoid(x): return 1/(1+np.exp(-x))

def combine_probabilities_rule_based(p_img_malignant, risk_points, mask_feats,
                                     w_clinical=1.0, w_area=0.3, w_size=0.1):
    risk_z = ((risk_points/20.0) - 0.5) / 0.25     # ~[-2,2]
    area_z = (mask_feats.get("area_ratio",0.0) - 0.02) / 0.03
    size_z = (mask_feats.get("approx_diam_px",0.0) - 24.0) / 16.0
    logit_final = _logit(p_img_malignant) + w_clinical*risk_z + w_area*area_z + w_size*size_z
    p_final = float(_sigmoid(logit_final))
    contrib = {
        "image_logit": float(_logit(p_img_malignant)),
        "risk_term": float(w_clinical*risk_z),
        "area_term": float(w_area*area_z),
        "size_term": float(w_size*size_z),
        "sum_logit": float(logit_final)
    }
    return p_final, contrib

# ==============================
# ⭐ NEW: Explainable AI (Grad‑CAM & Malignant heatmap)
# ==============================
def find_last_conv_layer_name(model):
    for layer in reversed(model.layers):
        if isinstance(layer, (tf.keras.layers.Conv2D,
                              tf.keras.layers.SeparableConv2D,
                              tf.keras.layers.DepthwiseConv2D)):
            return layer.name
    return None

def gradcam_overlay(model, x_preprocessed_224, base_pil_224, class_index=MALIGNANT_INDEX, alpha=0.55):
    """Tạo Grad‑CAM overlay cho lớp class_index."""
    last_conv = find_last_conv_layer_name(model)
    if last_conv is None:
        return base_pil_224  # fallback
    grad_model = tf.keras.models.Model([model.inputs],
                                       [model.get_layer(last_conv).output, model.output])
    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(x_preprocessed_224)
        loss = preds[:, class_index]
    grads = tape.gradient(loss, conv_out)                   # (1,H,W,C)
    pooled = tf.reduce_mean(grads, axis=(0,1,2))            # (C,)
    conv_out = conv_out[0]                                  # (H,W,C)
    heatmap = tf.reduce_sum(tf.multiply(pooled, conv_out), axis=-1)
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-6)
    heat = heatmap.numpy()

    # overlay (đỏ) theo alpha từ heat
    heat_L = Image.fromarray(np.uint8(255*heat)).resize(base_pil_224.size, Image.BILINEAR)
    overlay = Image.new("RGBA", base_pil_224.size, (255,0,0,0))
    overlay.putalpha(heat_L)  # alpha theo mức nóng
    base = base_pil_224.convert("RGBA")
    out = Image.alpha_composite(base, overlay).convert("RGB")
    return out

def malignant_prob_overlay_from_seg(base_pil_256, p_malig_map, alpha=0.65):
    """Heatmap xác suất ác tính từ phân đoạn (đỏ)."""
    heat = (np.clip(p_malig_map, 0, 1)*255).astype(np.uint8)
    heat_img = Image.fromarray(heat).resize(base_pil_256.size, Image.BILINEAR)
    overlay = Image.new("RGBA", base_pil_256.size, (255,0,0,0))
    overlay.putalpha(heat_img)
    out = Image.alpha_composite(base_pil_256.convert("RGBA"), overlay).convert("RGB")
    return out

# ==============================
# ⭐ NEW: Biểu đồ & Gauge
# ==============================
def prob_bar_chart(p_vec):
    df = pd.DataFrame({"Lớp":["Bình thường","Lành tính","Ác tính"],
                       "Xác suất":[float(p_vec[2]), float(p_vec[0]), float(p_vec[1])]})
    return alt.Chart(df).mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
        x=alt.X("Lớp", sort=["Bình thường","Lành tính","Ác tính"]),
        y=alt.Y("Xác suất", scale=alt.Scale(domain=[0,1])),
        color=alt.Color("Lớp", scale=alt.Scale(range=["#9CA3AF","#10B981","#EF4444"])),
        tooltip=["Lớp","Xác suất"]
    ).properties(height=240)

def gauge_chart(p_final):
    """Donut gauge Altair"""
    value = float(np.clip(p_final,0,1))
    df = pd.DataFrame({
        "label":["p_final","remainder"],
        "value":[value, 1-value],
        "color":["#DC2626","#E5E7EB"]  # đỏ & xám nhạt
    })
    ring = alt.Chart(df).mark_arc(outerRadius=110, innerRadius=70).encode(
        theta="value",
        color=alt.Color("color:N", scale=None, legend=None)
    )
    # Text trung tâm
    center = alt.Chart(pd.DataFrame({"text":[f"{value*100:.1f}%"]})).mark_text(size=28, fontWeight="bold").encode(
        text="text:N"
    )
    caption = alt.Chart(pd.DataFrame({"text":["Xác suất ác tính (kết hợp)"]})).mark_text(y=140, size=12).encode(text="text:N")
    return (ring + center + caption).properties(width=260, height=260)

# ==============================
# 🔹 Pipeline dự đoán
# ==============================
def du_doan(file, classifier, segmentor):
    image_bytes = file.read()
    x_cls, pil224 = classify_preprop(image_bytes, return_pil=True)
    x_seg, pil256 = segment_preprop(image_bytes, return_pil=True)

    with tf.device("/CPU:0"):
        pred_class = classifier.predict(x_cls, verbose=0)[0]     # (3,)
        mask_soft = segmentor.predict(x_seg, verbose=0)[0]       # (256,256,C)

    seg_overlay, mask_argmax, p_malig_map = segment_postprop(x_seg, mask_soft)
    return pred_class, seg_overlay, image_bytes, mask_argmax, pil224, p_malig_map, pil256, x_cls

# ==============================
# 🎨 UI tinh gọn (CSS nhẹ)
# ==============================
st.set_page_config(page_title="AI Phân tích Siêu âm Vú", layout="wide", page_icon="🩺")
st.markdown("""
<style>
.big-title {font-size:1.6rem; font-weight:700;}
.card {background:#0f172a; border:1px solid #1f2937; padding:1rem; border-radius:12px;}
.metric {font-size:1.6rem; font-weight:700;}
.caption {color:#9CA3AF;}
</style>
""", unsafe_allow_html=True)

st.sidebar.title("📘 Danh mục")
chon_trang = st.sidebar.selectbox("Chọn nội dung hiển thị", ["Ứng dụng minh họa", "Giới thiệu", "Nguồn dữ liệu & Bản quyền"])

# -----------------------------
# Trang Giới thiệu
# -----------------------------
if chon_trang == "Giới thiệu":
    st.title("👩‍🔬 ỨNG DỤNG AI TRONG HỖ TRỢ CHẨN ĐOÁN SIÊU ÂM VÚ")
    st.markdown("""
    Ứng dụng này phục vụ **nghiên cứu học thuật**; không dùng cho chẩn đoán y tế thực tế.
    """)

# -----------------------------
# Trang minh họa (nâng cấp UI + XAI + gauge)
# -----------------------------
elif chon_trang == "Ứng dụng minh họa":
    st.title("🩺 Minh họa mô hình AI trên ảnh siêu âm vú (kết hợp thông tin lâm sàng)")

    classifier, segmentor = load_models()

    # === Form nhập liệu
    with st.form("form_input"):
        colA, colB = st.columns([1.05, 1.3])
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
            if density_sel.startswith("A"): density_norm = "A"
            elif density_sel.startswith("D"): density_norm = "D"
            elif density_sel in ["B","C"]: density_norm = density_sel
            bmi_obese = st.checkbox("BMI ≥ 30")
            alcohol_high = st.checkbox("Uống rượu/bia thường xuyên")
            smoking = st.checkbox("Hút thuốc")
            low_activity = st.checkbox("Ít vận động")
            w_clinical = st.slider("Trọng số nguy cơ lâm sàng", 0.0, 2.0, 1.0, 0.1)
            w_area = st.slider("Trọng số diện tích mask", 0.0, 1.0, 0.3, 0.05)
            w_size = st.slider("Trọng số kích thước ước lượng", 0.0, 1.0, 0.1, 0.05)

        submitted = st.form_submit_button("🚀 Phân tích")

    if not submitted:
        st.info("👆 Hãy chọn ảnh và nhập thông tin, sau đó bấm **Phân tích**.")
    else:
        if file is None:
            st.warning("Vui lòng chọn một ảnh siêu âm.")
        else:
            with st.spinner("⏳ Đang suy luận..."):
                pred_class, seg_overlay_pil, img_bytes, mask_argmax, pil224, p_malig_map, pil256, x_cls = du_doan(file, classifier, segmentor)

            # ====== ẢNH ======
            orig = Image.open(BytesIO(img_bytes)).convert("RGB")
            orig_show = orig.resize((256,256))
            seg_show  = seg_overlay_pil.resize((256,256))

            st.markdown("<div class='big-title'>🖼️ Ảnh & kết quả phân đoạn</div>", unsafe_allow_html=True)
            c1,c2 = st.columns(2)
            c1.image(orig_show, caption="Ảnh gốc (256×256)", use_container_width=True)
            c2.image(seg_show,  caption="Phân đoạn (Xanh: lành, Đỏ: ác)", use_container_width=True)

            # ====== XÁC SUẤT ẢNH ======
            st.markdown("<div class='big-title'>💡 Kết quả AI trên ảnh (softmax)</div>", unsafe_allow_html=True)
            st.altair_chart(prob_bar_chart(pred_class), use_container_width=True)

            # ====== ĐẶC TRƯNG MASK + ĐIỂM RISK ======
            mask_feats = compute_mask_features(mask_argmax)
            form = {
                "age":age,"sex":sex,"family_history":family_history,"genetic_mutation":genetic_mutation,
                "personal_cancer_history":personal_cancer_history,"high_risk_lesion":high_risk_lesion,
                "chest_radiation_young":chest_radiation_young,"early_menarche":early_menarche,
                "late_menopause":late_menopause,"first_child_late_or_nulliparity":first_child_late_or_nulliparity,
                "no_breastfeeding":no_breastfeeding,"breast_density":density_norm,"bmi_obese":bmi_obese,
                "alcohol_high":alcohol_high,"smoking":smoking,"low_activity":low_activity
            }
            risk_points, risk_explain = clinical_risk_points(form)

            st.markdown("<div class='big-title'>🧪 Đặc trưng từ phân đoạn</div>", unsafe_allow_html=True)
            a,b,c = st.columns(3)
            a.metric("Diện tích tổn thương (%)", f"{mask_feats['area_ratio']*100:.2f}%")
            b.metric("Tỉ lệ vùng ác tính (%)", f"{mask_feats['malignant_area_ratio']*100:.2f}%")
            c.metric("Đường kính ước lượng (px)", f"{mask_feats['approx_diam_px']}")

            st.markdown("<div class='big-title'>🧍 Điểm nguy cơ lâm sàng</div>", unsafe_allow_html=True)
            x,y = st.columns([1,2])
            x.metric("Risk Points (0–20)", f"{risk_points:.1f}")
            if risk_explain:
                y.caption("Các yếu tố đóng góp:"); y.write("• " + "\n• ".join(risk_explain))
            else:
                y.caption("_Không có yếu tố nguy cơ nổi bật_")

            # ====== HỢP NHẤT (Cách B) + GAUGE ======
            p_img_malig = float(pred_class[MALIGNANT_INDEX])
            p_final, contrib = combine_probabilities_rule_based(
                p_img_malignant=p_img_malig, risk_points=risk_points, mask_feats=mask_feats,
                w_clinical=w_clinical, w_area=w_area, w_size=w_size
            )
            st.markdown("<div class='big-title'>🧮 Xác suất chẩn đoán cuối (kết hợp)</div>", unsafe_allow_html=True)
            gL, gR = st.columns([1,2])
            with gL:
                st.altair_chart(gauge_chart(p_final), use_container_width=False)
            with gR:
                st.write(f"**p_img (ác tính, từ ảnh)** = `{p_img_malig:.3f}`")
                st.write(f"**p_final (ác tính, sau hợp nhất)** = **`{p_final:.3f}`**  (~ {p_final*100:.1f}%)")

                # Giải thích định lượng đóng góp
                df_contrib = pd.DataFrame({
                    "Thành phần": ["Ảnh (logit)", "Lâm sàng (w*risk_z)", "Diện tích (w*area_z)", "Kích thước (w*size_z)"],
                    "Đóng góp vào logit": [contrib["image_logit"], contrib["risk_term"], contrib["area_term"], contrib["size_term"]]
                })
                bar_contrib = alt.Chart(df_contrib).mark_bar().encode(
                    x=alt.X("Thành phần", sort=None),
                    y=alt.Y("Đóng góp vào logit", scale=alt.Scale(domain=[min(-2,df_contrib["Đóng góp vào logit"].min()-0.2),
                                                                          max( 2,df_contrib["Đóng góp vào logit"].max()+0.2)])),
                    color=alt.condition("datum['Đóng góp vào logit']>0",
                                        alt.value("#10B981"), alt.value("#EF4444"))
                ).properties(height=220)
                st.altair_chart(bar_contrib, use_container_width=True)
                st.caption("Các cột thể hiện mức đóng góp (+/−) của từng nguồn thông tin vào **logit** trước khi chuyển sang xác suất.")

            # ====== XAI: GRAD‑CAM & MALIGNANT HEATMAP ======
            st.markdown("<div class='big-title'>🧠 Explainable AI</div>", unsafe_allow_html=True)
            cam = gradcam_overlay(classifier, x_cls, pil224, class_index=MALIGNANT_INDEX)
            malig_heat = malignant_prob_overlay_from_seg(pil256, p_malig_map)
            e1,e2 = st.columns(2)
            e1.image(cam, caption="Grad‑CAM (đỏ = vùng ảnh ảnh hưởng mạnh tới dự đoán ác tính)", use_container_width=True)
            e2.image(malig_heat, caption="Heatmap xác suất ác tính từ phân đoạn (đỏ = xác suất cao)", use_container_width=True)

            # ====== KHUYẾN NGHỊ ======
            if p_final >= 0.85 or mask_feats["approx_diam_px"] >= 48:
                rec = "Nguy cơ **rất cao** → Khuyến cáo **tham vấn bác sĩ chuyên khoa + sinh thiết**."
            elif p_final >= 0.60:
                rec = "Nguy cơ **cao** → Tham vấn bác sĩ, **cân nhắc sinh thiết** theo chỉ định."
            elif p_final >= 0.30:
                rec = "Nguy cơ **trung bình** → **Chụp bổ sung/siêu âm lại** và theo dõi sát."
            elif p_final >= 0.15:
                rec = "Nguy cơ **thấp–trung bình** → **Theo dõi** định kỳ, tái khám khi có triệu chứng."
            else:
                rec = "Nguy cơ **thấp** → Theo lịch **tầm soát** phù hợp tuổi/nguy cơ."
            st.info(rec)
            st.caption("Kết quả phục vụ nghiên cứu học thuật – không dùng cho chẩn đoán y tế thực tế.")

# -----------------------------
# Trang nguồn dữ liệu & bản quyền
# -----------------------------
elif chon_trang == "Nguồn dữ liệu & Bản quyền":
    st.title("📊 Nguồn dữ liệu và bản quyền sử dụng")
    st.markdown("""
    | Nguồn | Giấy phép | Liên kết |
    |-------|-----------|----------|
    | **BUSI (Kaggle)** | CC BY 4.0 | Kaggle |
    | **BUS‑UCLM (Kaggle)** | CC BY‑NC‑SA 4.0 | Kaggle |
    | **Breast Lesions USG (TCIA)** | CC BY 3.0 | TCIA |
    """)

st.markdown("""
---
📘 **Miễn trừ trách nhiệm:** Ứng dụng phục vụ **nghiên cứu – giáo dục**. Không sử dụng cho **chẩn đoán, điều trị**.
""")
