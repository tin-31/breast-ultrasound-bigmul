# ==========================================
# 🩺 ỨNG DỤNG TRÍ TUỆ NHÂN TẠO HỖ TRỢ PHÂN TÍCH ẢNH SIÊU ÂM VÚ
# ==========================================
# ⚠️ Phiên bản dành cho nghiên cứu học thuật - Không sử dụng cho mục đích y tế thực tế.
# ⚠️ Ứng dụng này chỉ mang tính minh họa kỹ thuật và học thuật.

import os
import gdown
import joblib
import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import altair as alt
from PIL import Image
from io import BytesIO
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.efficientnet import preprocess_input
import sklearn, joblib, numpy as np
st.sidebar.caption(f"sklearn={sklearn.__version__} | joblib={joblib.__version__} | numpy={np.__version__}")

# ==============================
# ⚙️ Cấu hình mô hình ẢNH
# ==============================
# Dùng đúng ID drive đã hoạt động trước đó
SEG_MODEL_ID = "1axOg7N5ssJrMec97eV-JMPzID26ynzN1"
CLF_MODEL_ID = "1fXPICuTkETep2oPiA56l0uMai2GusEJH"
SEG_MODEL_PATH = "seg_model.keras"     # giữ đúng tên cũ để load được custom_objects
CLF_MODEL_PATH = "clf_model.h5"

TEN_NHOM = ["Lành tính", "Ác tính", "Bình thường"]
MALIGNANT_INDEX = 1  # theo mapping ["Lành tính", "Ác tính", "Bình thường"]

# ==============================
# ⚙️ Cấu hình mô hình LÂM SÀNG (Epic)
# ==============================
CLINICAL_MODEL_PATH = "clinical_epic_gb_model.pkl"
CLINICAL_META_PATH  = "clinical_epic_gb_metadata.pkl"

# ==============================
# 🔹 Hàm xử lý trung gian cho CBAM (để load seg_model.keras)
# ==============================
def spatial_mean(t):
    return tf.reduce_mean(t, axis=-1, keepdims=True)

def spatial_max(t):
    return tf.reduce_max(t, axis=-1, keepdims=True)

def spatial_output_shape(s):
    return (s[0], s[1], s[2], 1)

# ==============================
# 🔹 Tải file mô hình (chỉ dùng gdown khi không có file cục bộ)
# ==============================
def ensure_model_file(path, gid, label):
    if os.path.exists(path):
        return
    try:
        st.info(f"📥 Đang tải {label} từ Google Drive…")
        # Cho phép đặt ID qua biến môi trường nếu muốn override
        gid = os.getenv(f"{label}_ID", gid)
        gdown.download(f"https://drive.google.com/uc?id={gid}", path, quiet=False)
        st.success(f"✅ Đã tải {label} xong.")
    except Exception as e:
        st.error(f"❌ Không tải được {label}: {e}. "
                 f"Vui lòng đẩy file '{path}' vào repo hoặc cung cấp ID Drive hợp lệ.")
        raise

# Chỉ tải khi file chưa có trong repo
ensure_model_file(SEG_MODEL_PATH, SEG_MODEL_ID, "SEG_MODEL")
ensure_model_file(CLF_MODEL_PATH, CLF_MODEL_ID, "CLF_MODEL")

# ==============================
# 🔹 Tải mô hình ẢNH an toàn
# ==============================
@st.cache_resource
def load_image_models():
    CUSTOM_OBJECTS = {
        "spatial_mean": spatial_mean,
        "spatial_max": spatial_max,
        "spatial_output_shape": spatial_output_shape
    }
    from tensorflow import keras
    try:
        # Một số phiên bản Keras yêu cầu bật để load custom layers/ops
        keras.config.enable_unsafe_deserialization()
    except Exception:
        pass

    classifier = tf.keras.models.load_model(CLF_MODEL_PATH, compile=False)
    segmentor  = tf.keras.models.load_model(SEG_MODEL_PATH, custom_objects=CUSTOM_OBJECTS, compile=False)
    return classifier, segmentor

# ==============================
# 🔹 Tải mô hình LÂM SÀNG (Epic)
# ==============================
@st.cache_resource
def load_clinical_model():
    try:
        model = joblib.load(CLINICAL_MODEL_PATH)   # Pipeline: OneHot + GB
        meta  = joblib.load(CLINICAL_META_PATH)    # {"num_cols": [...], "cat_cols": [...]}
        return model, meta
    except Exception as e:
        st.warning(f"⚠️ Không tải được mô hình lâm sàng Epic: {e}")
        return None, None

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
    image: (1,256,256,3) đã chuẩn hoá [0,1]
    mask : (256,256,C) softmax
    """
    goc = np.squeeze(image[0])
    chi_so = np.argmax(mask, axis=-1)

    MAU_LANH = np.array([0.0, 1.0, 0.0])  # xanh
    MAU_AC   = np.array([1.0, 0.0, 0.0])  # đỏ

    mau = np.zeros_like(goc, dtype=np.float32)
    mau[chi_so == 1] = MAU_LANH
    mau[chi_so == 2] = MAU_AC

    kq = goc.copy()
    vi_tri = chi_so > 0
    kq[vi_tri] = goc[vi_tri] * (1 - alpha) + mau[vi_tri] * alpha
    return kq, chi_so

def compute_mask_features(mask_argmax):
    """Trích một số đặc trưng đơn giản từ mask để hiển thị."""
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

# ==============================
# 🔹 Pipeline dự đoán ảnh
# ==============================
def predict_image(file, classifier, segmentor):
    image_bytes = file.read()
    img_clf = classify_preprop(image_bytes)
    img_seg = segment_preprop(image_bytes)

    with tf.device("/CPU:0"):
        pred_class = classifier.predict(img_clf, verbose=0)     # (1,3)
        pred_mask  = segmentor.predict(img_seg,  verbose=0)[0]  # (256,256,C)

    seg_overlay, mask_argmax = segment_postprop(img_seg, pred_mask)
    return pred_class, seg_overlay, image_bytes, mask_argmax

# ==============================
# 🔹 Page config & Sidebar
# ==============================
st.set_page_config(page_title="AI Phân tích Siêu âm Vú", layout="wide", page_icon="🩺")
st.sidebar.title("📘 Danh mục")
chon_trang = st.sidebar.selectbox("Chọn nội dung hiển thị",
                                  ["Ứng dụng minh họa", "Giới thiệu", "Nguồn dữ liệu & Bản quyền"])

# -----------------------------
# Trang Giới thiệu
# -----------------------------
if chon_trang == "Giới thiệu":
    st.title("👩‍🔬 ỨNG DỤNG AI TRONG HỖ TRỢ CHẨN ĐOÁN SIÊU ÂM VÚ")
    st.markdown("""
    Ứng dụng phục vụ **nghiên cứu học thuật** về Trí tuệ nhân tạo và Y học hình ảnh.
    **Không** dùng cho chẩn đoán hay điều trị thực tế.
    """)

# -----------------------------
# Trang minh họa chẩn đoán – TÍCH HỢP LÂM SÀNG
# -----------------------------
elif chon_trang == "Ứng dụng minh họa":
    st.title("🩺 Minh họa mô hình AI trên ảnh siêu âm vú (kèm thông tin lâm sàng)")

    # Load models
    classifier, segmentor = load_image_models()
    clinical_model, clinical_meta = load_clinical_model()

    # --- Giao diện nhập liệu ---
    colA, colB = st.columns([1.05, 1.0])

    with colA:
        file = st.file_uploader("📤 Chọn ảnh siêu âm (JPG hoặc PNG)", type=["jpg", "png"])
        st.caption("Kết quả chỉ mang tính minh họa, không có giá trị chẩn đoán y tế.")

    with colB:
        st.markdown("### 📋 Thông tin lâm sàng (Epic)")
        if clinical_model is None:
            st.info("Chưa tải được mô hình lâm sàng. Hãy kiểm tra các file `.pkl` trong repo.")
        else:
            # Lấy danh sách cột đã dùng khi train GB (đã lưu trong metadata)
            num_cols = clinical_meta.get("num_cols", [])
            cat_cols = clinical_meta.get("cat_cols", [])
            # Form đầu vào (khớp với các cột đã train trong script GB)
            with st.form("clinical_form"):
                c1, c2 = st.columns(2)

                with c1:
                    age = st.number_input("Age at Diagnosis", 20.0, 100.0, 50.0, 0.5)
                    tumor_size = st.number_input("Tumor Size (mm)", 0.0, 200.0, 20.0, 1.0)
                    ln_pos = st.number_input("Lymph nodes examined positive", 0, 50, 0, 1)
                    mut_count = st.number_input("Mutation Count", 0, 500, 10, 1)
                    npi = st.number_input("Nottingham prognostic index", 0.0, 10.0, 3.0, 0.1)
                    os_months = st.number_input("Overall Survival (Months)*", 0.0, 300.0, 60.0, 1.0)

                with c2:
                    surgery = st.selectbox("Type of Breast Surgery", ["Mastectomy", "Breast Conserving"])
                    grade = st.selectbox("Neoplasm Histologic Grade", ["1.0", "2.0", "3.0"])
                    stage = st.selectbox("Tumor Stage", ["1.0", "2.0", "3.0", "4.0"])
                    sex = st.selectbox("Sex", ["Female", "Male"])
                    cellularity = st.selectbox("Cellularity", ["Low", "Moderate", "High"])
                    chemo = st.selectbox("Chemotherapy", ["Yes", "No"])
                    horm = st.selectbox("Hormone Therapy", ["Yes", "No"])
                    radio = st.selectbox("Radio Therapy", ["Yes", "No"])
                    er = st.selectbox("ER Status", ["Positive", "Negative"])
                    pr = st.selectbox("PR Status", ["Positive", "Negative"])
                    her2 = st.selectbox("HER2 Status", ["Positive", "Negative"])
                    gene3 = st.selectbox("3-Gene classifier subtype",
                                         ["ER+/HER2- High Prolif", "ER+/HER2- Low Prolif", "HER2+", "Triple Neg"])
                    pam50 = st.selectbox("Pam50 + Claudin-low subtype",
                                         ["LumA", "LumB", "Basal", "Her2", "claudin-low", "Normal"])
                    rfs = st.selectbox("Relapse Free Status", ["Not Recurred", "Recurred"])

                submitted_clin = st.form_submit_button("📊 Dự đoán (Epic clinical)")

            p_deceased = None
            if submitted_clin:
                # Tạo 1 dòng DataFrame đúng thứ tự cột đã dùng khi train
                row = {
                    "Age at Diagnosis": age,
                    "Tumor Size": tumor_size,
                    "Lymph nodes examined positive": ln_pos,
                    "Mutation Count": mut_count,
                    "Nottingham prognostic index": npi,
                    "Overall Survival (Months)": os_months,
                    "Type of Breast Surgery": surgery,
                    "Neoplasm Histologic Grade": grade,
                    "Tumor Stage": stage,
                    "Sex": sex,
                    "Cellularity": cellularity,
                    "Chemotherapy": chemo,
                    "Hormone Therapy": horm,
                    "Radio Therapy": radio,
                    "ER Status": er,
                    "PR Status": pr,
                    "HER2 Status": her2,
                    "3-Gene classifier subtype": gene3,
                    "Pam50 + Claudin-low subtype": pam50,
                    "Relapse Free Status": rfs,
                }
                # Bảo đảm đủ cột như khi train
                for c in num_cols + cat_cols:
                    row.setdefault(c, "")

                input_df = pd.DataFrame([row])
                with st.spinner("⏳ Đang tính toán dựa trên mô hình Epic..."):
                    p_deceased = float(clinical_model.predict_proba(input_df)[0, 1])
                st.success(f"💀 Xác suất **Deceased** (Epic clinical): **{p_deceased:.3f}**")
                st.caption("(*) Biến 'Overall Survival (Months)' chỉ dùng minh hoạ cho pipeline huấn luyện.")

    st.markdown("---")

    # ---------- XỬ LÝ ẢNH ----------
    if file is None:
        st.info("👆 Hãy chọn một ảnh để mô hình tiến hành minh họa.")
    else:
        with st.spinner("⏳ Đang chạy mô hình ảnh..."):
            pred_class, seg_image, img_bytes, mask_argmax = predict_image(file, classifier, segmentor)
        anh_goc = Image.open(BytesIO(img_bytes)).convert("RGB")

        c1, c2 = st.columns(2)
        with c1:
            st.image(anh_goc, caption="Ảnh gốc", use_container_width=True)
        with c2:
            st.image(seg_image, caption="Kết quả phân đoạn (Đỏ: Ác tính, Xanh: Lành tính)", use_container_width=True)

        # Xác suất softmax 3 lớp
        prob_vec = pred_class[0].tolist()
        idx = int(np.argmax(pred_class))
        ket_qua = TEN_NHOM[idx]

        st.subheader("💡 Kết quả mô hình ẢNH")
        df_prob = pd.DataFrame({"Lớp": TEN_NHOM, "Xác suất": prob_vec})
        chart = alt.Chart(df_prob).mark_bar().encode(
            x=alt.X("Lớp", sort=TEN_NHOM),
            y=alt.Y("Xác suất", scale=alt.Scale(domain=[0, 1])),
            tooltip=["Lớp", "Xác suất"]
        ).properties(height=260)
        st.altair_chart(chart, use_container_width=True)

        if ket_qua == "Ác tính":
            st.error("🔴 Ảnh: mô hình dự đoán **Ác tính** (minh họa).")
        elif ket_qua == "Lành tính":
            st.success("🟢 Ảnh: mô hình dự đoán **Lành tính** (minh họa).")
        else:
            st.info("⚪ Ảnh: mô hình dự đoán **Bình thường** (minh họa).")

        try:
            p_malignant = float(pred_class[0, MALIGNANT_INDEX])
            st.caption(f"— Xác suất ác tính theo mô hình ảnh: **{p_malignant:.3f}**")
        except Exception:
            p_malignant = None

        # Một số đặc trưng đơn giản từ mask
        feats = compute_mask_features(mask_argmax)
        st.caption(f"— Diện tích tổn thương: **{feats['area_ratio']*100:.2f}%**, "
                   f"tỉ lệ vùng ác tính: **{feats['malignant_area_ratio']*100:.2f}%**, "
                   f"đường kính ước lượng: **{feats['approx_diam_px']} px**.")

        # --------- ĐÁNH GIÁ TỔNG QUAN (minh hoạ, KHÔNG cộng xác suất) ----------
        st.markdown("### 🧮 Đánh giá tổng quan (minh hoạ)")
        if p_malignant is None and p_deceased is None:
            st.info("Hãy nhập **thông tin lâm sàng** và **chọn ảnh** để xem đánh giá tổng quan.")
        else:
            bullets = []
            if p_malignant is not None:
                bullets.append(f"- Ảnh → xác suất **ác tính**: **{p_malignant:.2f}**")
            if p_deceased is not None:
                bullets.append(f"- Lâm sàng (Epic) → xác suất **deceased**: **{p_deceased:.2f}**")
            st.write("\n".join(bullets) if bullets else "_Chưa có đủ thông tin_.")
            # Quy tắc gợi ý minh hoạ:
            risk_note = "Tổng quan: "
            if (p_malignant is not None and p_malignant >= 0.60) and (p_deceased is not None and p_deceased >= 0.60):
                risk_note += "⚠️ **Nguy cơ cao** ở cả hai chiều (ác tính & tiên lượng xấu)."
            elif (p_malignant is not None and p_malignant >= 0.60) or (p_deceased is not None and p_deceased >= 0.60):
                risk_note += "ℹ️ **Nguy cơ tăng** ở một trong hai chiều."
            else:
                risk_note += "✅ **Thấp–trung bình** theo dữ liệu hiện có."
            st.info(risk_note)
            st.caption("Đây **không** phải là phép cộng xác suất. Hai mô hình giải quyết **hai câu hỏi khác nhau**.")

# -----------------------------
# Trang nguồn dữ liệu & bản quyền
# -----------------------------
elif chon_trang == "Nguồn dữ liệu & Bản quyền":
    st.title("📊 Nguồn dữ liệu và bản quyền sử dụng")
    st.markdown("""
    Ứng dụng sử dụng dữ liệu ảnh từ:
    - **BUSI (Kaggle)** – CC BY 4.0  
    - **BUS-UCLM (Kaggle)** – CC BY-NC-SA 4.0  
    - **Breast Lesions USG (TCIA)** – CC BY 3.0  

    Mô hình lâm sàng:
    - **Breast_Cancer_METABRIC_Epic_Hospital** (Mendeley Data, CC BY 4.0) – dùng để huấn luyện mô hình lâm sàng tham khảo.

    **Chỉ dùng cho nghiên cứu/giáo dục, không dùng cho mục đích y tế thực tế.**
    """)

# -----------------------------
# Chân trang
# -----------------------------
st.markdown("""
---
📘 **Tuyên bố miễn trừ trách nhiệm:**  
Ứng dụng này phục vụ mục đích **nghiên cứu khoa học và giáo dục**.  
Không sử dụng cho **chẩn đoán, điều trị hoặc tư vấn y tế**.  

🧪 Mô hình lâm sàng Epic là mô phỏng dựa trên dữ liệu quốc tế,  
không đại diện cho dân số Việt Nam và không dùng trong quyết định lâm sàng.  
© 2025 – Dự án AI Siêu âm Vú.
""")
