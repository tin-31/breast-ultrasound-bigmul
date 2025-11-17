# ==========================================
# 🩺 ỨNG DỤNG TRÍ TUỆ NHÂN TẠO HỖ TRỢ PHÂN TÍCH ẢNH SIÊU ÂM VÚ
# ==========================================
# ⚠️ Phiên bản dành cho nghiên cứu học thuật - Không sử dụng cho mục đích y tế thực tế.
# ⚠️ Ứng dụng này chỉ mang tính minh họa kỹ thuật và học thuật.

import os
import gdown
import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import altair as alt
from PIL import Image, ImageDraw
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

# Vị trí lớp "Ác tính" trong output softmax mô hình phân loại
# Theo mapping hiển thị: ["Lành tính", "Ác tính", "Bình thường"] -> index = 1
MALIGNANT_INDEX = 1

# ==============================
# 🔹 Hàm xử lý trung gian cho CBAM
# ==============================
def spatial_mean(t):
    return tf.reduce_mean(t, axis=-1, keepdims=True)

def spatial_max(t):
    return tf.reduce_max(t, axis=-1, keepdims=True)

def spatial_output_shape(s):
    return (s[0], s[1], s[2], 1)

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
    CUSTOM_OBJECTS = {
        "spatial_mean": spatial_mean,
        "spatial_max": spatial_max,
        "spatial_output_shape": spatial_output_shape
    }
    from tensorflow import keras
    try:
        keras.config.enable_unsafe_deserialization()
    except Exception:
        pass

    classifier = tf.keras.models.load_model(CLF_MODEL_PATH, compile=False)
    segmentor = tf.keras.models.load_model(SEG_MODEL_PATH, custom_objects=CUSTOM_OBJECTS, compile=False)
    return classifier, segmentor

# ==============================
# 🔹 Tiền xử lý ảnh
# ==============================
def classify_preprop(image_bytes):
    image = Image.open(BytesIO(image_bytes)).convert("RGB").resize((224, 224))
    image = preprocess_input(np.expand_dims(img_to_array(image), axis=0))
    return image

def segment_preprop(image_bytes):
    image = Image.open(BytesIO(image_bytes)).convert("RGB").resize((256, 256))
    image = np.expand_dims(np.array(image) / 255.0, axis=0)
    return image

# ==============================
# 🔹 Hậu xử lý ảnh phân đoạn
# ==============================
def segment_postprop(image, mask, alpha=0.5):
    """
    image: (1, 256, 256, 3) đã chuẩn hoá [0,1]
    mask : (256, 256, C) softmax
    """
    goc = np.squeeze(image[0])  # (256,256,3)
    chi_so = np.argmax(mask, axis=-1)  # (256,256)

    MAU_LANH = np.array([0.0, 1.0, 0.0])    # Xanh lá
    MAU_AC = np.array([1.0, 0.0, 0.0])      # Đỏ

    mau = np.zeros_like(goc, dtype=np.float32)
    mau[chi_so == 1] = MAU_LANH
    mau[chi_so == 2] = MAU_AC

    kq = goc.copy()
    vi_tri = chi_so > 0
    kq[vi_tri] = goc[vi_tri] * (1 - alpha) + mau[vi_tri] * alpha
    return kq, chi_so  # trả về overlay và mask argmax

# ==============================
# 🔹 Đặc trưng từ mask & kết hợp theo quy tắc (Cách B)
# ==============================
def compute_mask_features(mask_argmax):
    """
    mask_argmax: (H,W) với giá trị {0: nền, 1: lành, 2: ác}
    Trả về đặc trưng nhẹ dùng cho hợp nhất: tỉ lệ diện tích tổn thương, tỉ lệ ác tính, đường kính bbox ước lượng (px)
    """
    H, W = mask_argmax.shape
    total = float(H * W)
    lesion = mask_argmax > 0
    area_ratio = float(np.sum(lesion)) / total

    malignant_ratio = float(np.sum(mask_argmax == 2)) / total

    ys, xs = np.where(lesion)
    if ys.size > 0:
        y1, y2 = int(ys.min()), int(ys.max())
        x1, x2 = int(xs.min()), int(xs.max())
        approx_diam_px = max(y2 - y1 + 1, x2 - x1 + 1)
    else:
        approx_diam_px = 0

    return {
        "area_ratio": area_ratio,
        "malignant_area_ratio": malignant_ratio,
        "approx_diam_px": int(approx_diam_px),
    }

def clinical_risk_points(form):
    """
    Tính điểm nguy cơ lâm sàng (0..20) + diễn giải.
    Có thể tinh chỉnh hệ số theo dữ liệu thực tế.
    """
    pts = 0.0
    explain = []

    # Tuổi
    age = form.get("age", 0)
    if age >= 70: pts += 3; explain.append("Tuổi ≥70 (+3)")
    elif age >= 50: pts += 2; explain.append("Tuổi 50–69 (+2)")
    elif age >= 40: pts += 1; explain.append("Tuổi 40–49 (+1)")

    # Giới
    if form.get("sex") == "Nữ":
        pts += 1; explain.append("Giới nữ (+1)")

    # Gia đình & đột biến
    fam = form.get("family_history", "Không")
    if fam == "1 người": pts += 2; explain.append("Gia đình: 1 người trực hệ (+2)")
    elif fam == "≥2 người": pts += 3; explain.append("Gia đình: ≥2 người trực hệ (+3)")

    mut = form.get("genetic_mutation", "Không/Không biết")
    if mut == "BRCA1": pts += 5; explain.append("Đột biến BRCA1 (+5)")
    elif mut == "BRCA2": pts += 4; explain.append("Đột biến BRCA2 (+4)")
    elif mut == "Khác": pts += 2; explain.append("Đột biến khác (+2)")

    # Tiền sử bản thân
    if form.get("personal_cancer_history", False): pts += 4; explain.append("Từng ung thư vú (+4)")
    if form.get("high_risk_lesion", False): pts += 2; explain.append("Tổn thương nguy cơ cao (+2)")
    if form.get("chest_radiation_young", False): pts += 4; explain.append("Xạ trị ngực <30 tuổi (+4)")

    # Nội tiết – sinh sản
    if form.get("early_menarche", False): pts += 1; explain.append("Có kinh sớm (<12) (+1)")
    if form.get("late_menopause", False): pts += 1; explain.append("Mãn kinh muộn (>55) (+1)")
    if form.get("first_child_late_or_nulliparity", False): pts += 1; explain.append("Chưa sinh / con đầu >35 (+1)")
    if form.get("no_breastfeeding", False): pts += 1; explain.append("Không cho con bú (+1)")

    # Mật độ vú
    density = form.get("breast_density", "Không rõ")
    if density == "B": pts += 1; explain.append("Mật độ B (+1)")
    elif density == "C": pts += 2; explain.append("Mật độ C (+2)")
    elif density == "D": pts += 3; explain.append("Mật độ D (+3)")

    # Lối sống
    if form.get("bmi_obese", False): pts += 1; explain.append("BMI ≥30 (+1)")
    if form.get("alcohol_high", False): pts += 1; explain.append("Rượu thường xuyên (+1)")
    if form.get("smoking", False): pts += 1; explain.append("Hút thuốc (+1)")
    if form.get("low_activity", False): pts += 1; explain.append("Ít vận động (+1)")

    pts = min(pts, 20.0)
    return pts, explain

def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def _logit(p, eps=1e-6):
    p = float(np.clip(p, eps, 1 - eps))
    return np.log(p / (1 - p))

def combine_probabilities_rule_based(p_img_malignant, risk_points, mask_feats,
                                     w_clinical=1.0, w_area=0.3, w_size=0.1):
    """
    Hợp nhất theo quy tắc (Cách B):
    p_final = sigmoid( logit(p_img) + w_clinical*z(risk) + w_area*z(area) + w_size*z(size) )

    - risk_points: 0..20 -> z về khoảng ~[-2,2]
    - area_ratio: 0..1   -> z dịch quanh 0.02 (2%) để tránh phạt quá mức
    - approx_diam_px: ước lượng, chuẩn hoá đơn giản (bạn có thể tinh chỉnh)
    """
    # Chuẩn hoá về thang gần zero-mean
    risk_z = ((risk_points / 20.0) - 0.5) / 0.25    # ~[-2,2]
    area_z = (mask_feats.get("area_ratio", 0.0) - 0.02) / 0.03
    size_z = (mask_feats.get("approx_diam_px", 0.0) - 24.0) / 16.0

    # Tổ hợp trên logit
    logit_final = _logit(p_img_malignant) + w_clinical*risk_z + w_area*area_z + w_size*size_z
    p_final = float(_sigmoid(logit_final))

    debug = {"risk_z": float(risk_z), "area_z": float(area_z), "size_z": float(size_z),
             "w_clinical": float(w_clinical), "w_area": float(w_area), "w_size": float(w_size)}
    return p_final, debug

# ==============================
# 🔹 Pipeline dự đoán
# ==============================
def du_doan(file, classifier, segmentor):
    image_bytes = file.read()
    img_clf = classify_preprop(image_bytes)
    img_seg = segment_preprop(image_bytes)

    with tf.device("/CPU:0"):
        pred_class = classifier.predict(img_clf, verbose=0)        # (1,3)
        pred_mask = segmentor.predict(img_seg, verbose=0)[0]       # (256,256,C)

    seg_overlay, mask_argmax = segment_postprop(img_seg, pred_mask)
    return pred_class[0], seg_overlay, image_bytes, mask_argmax

# ==============================
# 🔹 Giao diện Streamlit (Chỉ tiếng Việt)
# ==============================
st.set_page_config(page_title="AI Phân tích Siêu âm Vú", layout="wide", page_icon="🩺")
st.sidebar.title("📘 Danh mục")

chon_trang = st.sidebar.selectbox(
    "Chọn nội dung hiển thị",
    ["Ứng dụng minh họa", "Giới thiệu", "Nguồn dữ liệu & Bản quyền"]
)

# -----------------------------
# Trang Giới thiệu
# -----------------------------
if chon_trang == "Giới thiệu":
    st.title("👩‍🔬 ỨNG DỤNG AI TRONG HỖ TRỢ CHẨN ĐOÁN SIÊU ÂM VÚ")
    st.markdown("""
    Dự án này được thực hiện với mục đích **nghiên cứu học thuật** trong lĩnh vực Trí tuệ nhân tạo và Y học hình ảnh.

    ⚠️ **Lưu ý quan trọng:**
    - Đây **không phải** là công cụ chẩn đoán y tế thật.
    - Ứng dụng chỉ dùng để **minh họa kỹ thuật xử lý ảnh và học sâu (Deep Learning)**.
    - Không nên sử dụng kết quả này để thay thế tư vấn hoặc chẩn đoán y tế từ bác sĩ.
    """)

# -----------------------------
# Trang minh họa chẩn đoán (Cách B – fusion theo quy tắc)
# -----------------------------
elif chon_trang == "Ứng dụng minh họa":
    st.title("🩺 Minh họa mô hình AI trên ảnh siêu âm vú (kết hợp thông tin lâm sàng)")

    classifier, segmentor = load_models()

    # Form nhập ảnh + thông tin bệnh nhân
    with st.form("form_input"):
        colA, colB = st.columns([1.1, 1.2])
        with colA:
            file = st.file_uploader("📤 Chọn ảnh siêu âm (JPG hoặc PNG)", type=["jpg", "png"])
            st.caption("Chỉ dùng minh họa, không có giá trị chẩn đoán y tế.")

        with colB:
            st.markdown("### 🧍 Thông tin bệnh nhân")
            age = st.number_input("Tuổi", min_value=15, max_value=100, value=45, step=1)
            sex = st.selectbox("Giới", ["Nữ", "Nam"])

            st.markdown("#### 👪 Tiền sử gia đình & di truyền")
            family_history = st.selectbox("Người thân trực hệ mắc ung thư vú/buồng trứng", ["Không", "1 người", "≥2 người"])
            genetic_mutation = st.selectbox("Đột biến di truyền", ["Không/Không biết", "BRCA1", "BRCA2", "Khác"])

            st.markdown("#### 🏥 Tiền sử bản thân")
            personal_cancer_history = st.checkbox("Từng mắc ung thư vú")
            high_risk_lesion = st.checkbox("Tổn thương nguy cơ cao (DCIS/LCIS/ADH)")
            chest_radiation_young = st.checkbox("Xạ trị vùng ngực (<30 tuổi)")

            st.markdown("#### 🧬 Nội tiết – sinh sản")
            early_menarche = st.checkbox("Có kinh sớm (<12)")
            late_menopause = st.checkbox("Mãn kinh muộn (>55)")
            first_child_late_or_nulliparity = st.checkbox("Chưa sinh / con đầu >35")
            no_breastfeeding = st.checkbox("Không cho con bú")

            st.markdown("#### 🧪 Mật độ mô vú")
            breast_density = st.selectbox("Mật độ", ["Không rõ", "A (thưa)", "B", "C", "D (rất dày)"])
            # Chuẩn hoá density về A/B/C/D
            density_norm = "Không rõ"
            if breast_density.startswith("A"): density_norm = "A"
            elif breast_density in ["B","C","D (rất dày)"]:
                density_norm = "D" if breast_density.startswith("D") else breast_density

            st.markdown("#### 🧠 Lối sống")
            bmi_obese = st.checkbox("BMI ≥ 30 (béo phì)")
            alcohol_high = st.checkbox("Uống rượu/bia thường xuyên")
            smoking = st.checkbox("Hút thuốc")
            low_activity = st.checkbox("Ít vận động")

            st.markdown("#### ⚖️ Tham số hợp nhất (có thể tinh chỉnh)")
            w_clinical = st.slider("Trọng số nguy cơ lâm sàng (w_clinical)", 0.0, 2.0, 1.0, 0.1)
            w_area = st.slider("Trọng số diện tích mask (w_area)", 0.0, 1.0, 0.3, 0.05)
            w_size = st.slider("Trọng số kích thước ước lượng (w_size)", 0.0, 1.0, 0.1, 0.05)

        submitted = st.form_submit_button("🚀 Phân tích")

    if not submitted:
        st.info("👆 Hãy chọn ảnh và nhập thông tin, sau đó bấm **Phân tích**.")
    else:
        if file is None:
            st.warning("Vui lòng chọn một ảnh siêu âm.")
        else:
            with st.spinner("⏳ Đang xử lý AI..."):
                pred_class, seg_image, img_bytes, mask_argmax = du_doan(file, classifier, segmentor)

            # Ảnh gốc
            anh_goc = Image.open(BytesIO(img_bytes)).convert("RGB")

            # Kết quả AI ảnh
            p_vec = pred_class.tolist()  # [p_benign, p_malignant, p_normal] nếu đúng mapping
            try:
                p_malignant = float(pred_class[MALIGNANT_INDEX])
            except Exception:
                p_malignant = float(np.max(pred_class))

            # Đặc trưng từ mask
            mask_feats = compute_mask_features(mask_argmax)

            # Điểm nguy cơ lâm sàng
            form = {
                "age": age, "sex": sex,
                "family_history": family_history,
                "genetic_mutation": genetic_mutation,
                "personal_cancer_history": personal_cancer_history,
                "high_risk_lesion": high_risk_lesion,
                "chest_radiation_young": chest_radiation_young,
                "early_menarche": early_menarche,
                "late_menopause": late_menopause,
                "first_child_late_or_nulliparity": first_child_late_or_nulliparity,
                "no_breastfeeding": no_breastfeeding,
                "breast_density": density_norm,
                "bmi_obese": bmi_obese,
                "alcohol_high": alcohol_high,
                "smoking": smoking,
                "low_activity": low_activity
            }
            risk_points, risk_explain = clinical_risk_points(form)

            # Hợp nhất theo quy tắc (Cách B)
            p_final, debug = combine_probabilities_rule_based(
                p_img_malignant=p_malignant,
                risk_points=risk_points,
                mask_feats=mask_feats,
                w_clinical=w_clinical, w_area=w_area, w_size=w_size
            )

            # Hiển thị
            st.markdown("---")
            col1, col2 = st.columns(2)
            with col1:
                st.image(anh_goc, caption="Ảnh gốc", use_container_width=True)
            with col2:
                st.image(seg_image, caption="Kết quả phân đoạn (Đỏ: Ác tính, Xanh: Lành tính)", use_container_width=True)

            st.markdown("### 💡 Kết quả AI trên ảnh (softmax)")
            df_prob = pd.DataFrame({
                "Lớp": ["Lành tính", "Ác tính", "Bình thường"],
                "Xác suất": p_vec
            })
            chart = alt.Chart(df_prob).mark_bar().encode(
                x=alt.X("Lớp", sort=["Bình thường","Lành tính","Ác tính"]),
                y=alt.Y("Xác suất", scale=alt.Scale(domain=[0,1])),
                tooltip=["Lớp","Xác suất"]
            ).properties(height=240)
            st.altair_chart(chart, use_container_width=True)

            st.markdown("### 🧪 Đặc trưng từ phân đoạn")
            cma, cmb, cmc = st.columns(3)
            cma.metric("Diện tích tổn thương (%)", f"{mask_feats['area_ratio']*100:.2f}%")
            cmb.metric("Tỉ lệ vùng ác tính (%)", f"{mask_feats['malignant_area_ratio']*100:.2f}%")
            cmc.metric("Đường kính ước lượng (px)", f"{mask_feats['approx_diam_px']}")

            st.markdown("### 🧍 Điểm nguy cơ lâm sàng")
            c1, c2 = st.columns([1,2])
            c1.metric("Risk Points (0–20)", f"{risk_points:.1f}")
            with c2:
                if risk_explain:
                    st.caption("Các yếu tố đóng góp:")
                    st.write("• " + "\n• ".join(risk_explain))
                else:
                    st.caption("_Không có yếu tố nguy cơ nổi bật_")

            st.markdown("### 🧮 Xác suất **kết hợp** (Cách B)")
            st.success(f"**p_final (ác tính, sau hợp nhất)** = **{p_final:.3f}**")
            with st.expander("Giải thích hợp nhất (debug)"):
                st.json(debug)

            # Khuyến nghị (có thể tinh chỉnh theo thực nghiệm)
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
            st.caption("Kết quả chỉ phục vụ nghiên cứu học thuật – không có giá trị chẩn đoán y tế.")

# -----------------------------
# Trang nguồn dữ liệu & bản quyền
# -----------------------------
elif chon_trang == "Nguồn dữ liệu & Bản quyền":
    st.title("📊 Nguồn dữ liệu và bản quyền sử dụng")
    st.markdown("""
    Ứng dụng sử dụng dữ liệu từ ba nguồn công khai, tuân thủ giấy phép phi thương mại (CC BY-NC-SA 4.0):

    | Nguồn | Giấy phép | Liên kết |
    |-------|------------|----------|
    | **BUSI (Arya Shah, Kaggle)** | CC BY 4.0 | [Link](https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset) |
    | **BUS-UCLM (Orvile, Kaggle)** | CC BY-NC-SA 4.0 | [Link](https://www.kaggle.com/datasets/orvile/bus-uclm-breast-ultrasound-dataset) |
    | **Breast Lesions USG (TCIA)** | CC BY 3.0 | [Link](https://www.cancerimagingarchive.net/collection/breast-lesions-usg/) |

    ---
    **Giấy phép sử dụng:**  
    - Phi thương mại (Non-Commercial).  
    - Phải trích dẫn nguồn dữ liệu gốc.  
    - Không sử dụng cho mục đích y tế hoặc thương mại.

    ---
    **Trích dẫn APA:**  
    - Shah, A. (2020). *Breast Ultrasound Images Dataset* [Dataset]. Kaggle.  
    - Orvile. (2023). *BUS-UCLM Breast Ultrasound Dataset* [Dataset]. Kaggle.  
    - The Cancer Imaging Archive. (2021). *Breast Lesions USG* [Dataset].
    """)

# -----------------------------
# Chân trang (footer)
# -----------------------------
st.markdown("""
---
📘 **Tuyên bố miễn trừ trách nhiệm:**  
Ứng dụng này được phát triển phục vụ mục đích **nghiên cứu khoa học và giáo dục**.  
Không sử dụng cho **chẩn đoán, điều trị hoặc tư vấn y tế**.  
© 2025 – Dự án AI Siêu âm Vú. Tác giả: Lê Vũ Anh Tin – Trường THPT Chuyên Nguyễn Du.
""")
