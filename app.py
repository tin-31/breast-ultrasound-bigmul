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

# ==============================
# ⚙️ Cấu hình mô hình
# ==============================
SEG_MODEL_ID = "1axOg7N5ssJrMec97eV-JMPzID26ynzN1"
CLF_MODEL_ID = "1fXPICuTkETep2oPiA56l0uMai2GusEJH"
SEG_MODEL_PATH = "seg_model.keras"
CLF_MODEL_PATH = "clf_model.h5"

# Mô hình lâm sàng (Epic Hospital – Gradient Boosting)
CLINICAL_MODEL_PATH = "clinical_epic_gb_model.pkl"
CLINICAL_META_PATH = "clinical_epic_gb_metadata.pkl"

# Vị trí lớp "Ác tính" trong output softmax mô hình phân loại ảnh
TEN_NHOM = ["Lành tính", "Ác tính", "Bình thường"]
MALIGNANT_INDEX = 1  # ["Lành tính", "Ác tính", "Bình thường"]


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
# 🔹 Tự động tải mô hình ảnh
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


@st.cache_resource
def load_clinical_model():
    """
    Tải mô hình lâm sàng Epic Hospital (Gradient Boosting).
    Nếu không có file .pkl thì trả về (None, None).
    """
    try:
        model = joblib.load(CLINICAL_MODEL_PATH)
        meta = joblib.load(CLINICAL_META_PATH)
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
    goc = np.squeeze(image[0])
    chi_so = np.argmax(mask, axis=-1)

    MAU_LANH = np.array([0.0, 1.0, 0.0])    # Xanh lá
    MAU_AC = np.array([1.0, 0.0, 0.0])      # Đỏ

    mau = np.zeros_like(goc, dtype=np.float32)
    mau[chi_so == 1] = MAU_LANH
    mau[chi_so == 2] = MAU_AC

    kq = goc.copy()
    vi_tri = chi_so > 0
    kq[vi_tri] = goc[vi_tri] * (1 - alpha) + mau[vi_tri] * alpha
    return kq


# ==============================
# 🔹 Pipeline dự đoán ảnh
# ==============================
def du_doan(file, classifier, segmentor):
    image_bytes = file.read()
    img_clf = classify_preprop(image_bytes)
    img_seg = segment_preprop(image_bytes)

    with tf.device("/CPU:0"):
        pred_class = classifier.predict(img_clf, verbose=0)        # (1,3)
        pred_mask = segmentor.predict(img_seg, verbose=0)[0]       # (256,256,C)

    seg_image = segment_postprop(img_seg, pred_mask)
    return pred_class, seg_image, image_bytes


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
# Trang minh họa chẩn đoán
# -----------------------------
elif chon_trang == "Ứng dụng minh họa":
    st.title("🩺 Minh họa mô hình AI trên ảnh siêu âm vú")

    classifier, segmentor = load_models()
    clinical_model, clinical_meta = load_clinical_model()

    file = st.file_uploader("📤 Chọn ảnh siêu âm (JPG hoặc PNG)", type=["jpg", "png"])

    # Tạo 2 tab: Ảnh siêu âm & Mô hình lâm sàng Epic
    tab_img, tab_clin = st.tabs(["🖼 Phân tích ảnh siêu âm", "📋 Mô hình lâm sàng (Epic Hospital)"])

    # ----- TAB 1: Ảnh siêu âm -----
    with tab_img:
        if file is None:
            st.info("👆 Hãy chọn một ảnh để mô hình tiến hành minh họa.")
        else:
            slot = st.empty()
            slot.text("⏳ Đang xử lý ảnh...")

            pred_class, seg_image, img_bytes = du_doan(file, classifier, segmentor)
            anh_goc = Image.open(BytesIO(img_bytes))

            cot1, cot2 = st.columns(2)
            with cot1:
                st.image(anh_goc, caption="Ảnh gốc", use_container_width=True)
            with cot2:
                st.image(seg_image, caption="Kết quả phân đoạn (Đỏ: Ác tính, Xanh: Lành tính)", use_container_width=True)

            # Xác suất softmax
            prob_vec = pred_class[0].tolist()   # [p_benign, p_malignant, p_normal]
            idx = int(np.argmax(pred_class))
            ket_qua = TEN_NHOM[idx]

            st.markdown("---")
            st.subheader("💡 Kết quả minh họa trên ảnh")

            if ket_qua == "Lành tính":
                st.success("🟢 Mô hình dự đoán: Khối u **lành tính** (chỉ mang tính minh họa).")
            elif ket_qua == "Ác tính":
                st.error("🔴 Mô hình dự đoán: Khối u **ác tính** (chỉ mang tính minh họa).")
            else:
                st.info("⚪ Mô hình dự đoán: **Không phát hiện bất thường rõ rệt** (chỉ mang tính minh họa).")

            # Biểu đồ xác suất
            df_prob = pd.DataFrame({
                "Lớp": TEN_NHOM,
                "Xác suất": prob_vec
            })
            chart = alt.Chart(df_prob).mark_bar().encode(
                x=alt.X("Lớp", sort=TEN_NHOM),
                y=alt.Y("Xác suất", scale=alt.Scale(domain=[0, 1])),
                tooltip=["Lớp", "Xác suất"]
            ).properties(height=280)
            st.altair_chart(chart, use_container_width=True)

            try:
                p_malignant = float(pred_class[0, MALIGNANT_INDEX])
                st.caption(f"Xác suất mô hình ảnh đánh giá là **ác tính**: {p_malignant:.3f}")
            except Exception:
                pass

            st.caption("Kết quả chỉ mang tính nghiên cứu học thuật, không có giá trị chẩn đoán y tế.")

    # ----- TAB 2: Mô hình lâm sàng Epic -----
    with tab_clin:
        st.subheader("📋 Mô phỏng mô hình lâm sàng từ dữ liệu Epic Hospital")
        st.caption("""
        Mô hình này được huấn luyện trên bộ dữ liệu quốc tế Breast_Cancer_METABRIC_Epic_Hospital,
        dự đoán xác suất bệnh nhân **đã tử vong vào thời điểm cuối theo dõi** (Overall Survival Status = Deceased).
        Kết quả chỉ mang tính minh họa, không sử dụng trong tiên lượng hay điều trị thực tế.
        """)

        if clinical_model is None:
            st.info("⚠️ Chưa có hoặc chưa tải được file `clinical_epic_gb_model.pkl` / metadata. "
                    "Hãy chắc chắn rằng các file này nằm cùng thư mục với app.")
        else:
            num_cols = clinical_meta["num_cols"]
            cat_cols = clinical_meta["cat_cols"]

            with st.form("clinical_form"):
                col1, col2 = st.columns(2)

                # --------- CÁC TRƯỜNG SỐ ----------
                with col1:
                    age = st.number_input("Tuổi lúc chẩn đoán (Age at Diagnosis)", 20.0, 100.0, 50.0, 0.5)
                    tumor_size = st.number_input("Kích thước khối u (Tumor Size, mm)", 0.0, 200.0, 20.0, 1.0)
                    ln_pos = st.number_input("Số hạch dương tính (Lymph nodes examined positive)", 0, 50, 0, 1)
                    mut_count = st.number_input("Số đột biến (Mutation Count)", 0, 500, 10, 1)
                    npi = st.number_input("Chỉ số Nottingham (NPI)", 0.0, 10.0, 3.0, 0.1)
                    os_months = st.number_input("Thời gian theo dõi (Overall Survival – Months)*", 0.0, 300.0, 60.0, 1.0)

                with col2:
                    # --------- CÁC TRƯỜNG PHÂN LOẠI ----------
                    surgery = st.selectbox("Loại phẫu thuật vú (Type of Breast Surgery)",
                                           ["Mastectomy", "Breast Conserving"])
                    grade = st.selectbox("Độ mô học (Neoplasm Histologic Grade)", ["1.0", "2.0", "3.0"])
                    stage = st.selectbox("Giai đoạn khối u (Tumor Stage)", ["1.0", "2.0", "3.0", "4.0"])
                    sex = st.selectbox("Giới (Sex)", ["Female", "Male"])
                    cellularity = st.selectbox("Cellularity", ["Low", "Moderate", "High"])
                    chemo = st.selectbox("Hóa trị (Chemotherapy)", ["Yes", "No"])
                    horm = st.selectbox("Nội tiết (Hormone Therapy)", ["Yes", "No"])
                    radio = st.selectbox("Xạ trị (Radio Therapy)", ["Yes", "No"])
                    er = st.selectbox("ER Status", ["Positive", "Negative"])
                    pr = st.selectbox("PR Status", ["Positive", "Negative"])
                    her2 = st.selectbox("HER2 Status", ["Positive", "Negative"])
                    gene3 = st.selectbox("3-Gene classifier subtype",
                                         ["ER+/HER2- High Prolif", "ER+/HER2- Low Prolif",
                                          "HER2+", "Triple Neg", "Khác"])
                    pam50 = st.selectbox("Pam50 + Claudin-low subtype",
                                         ["LumA", "LumB", "Basal", "Her2", "claudin-low", "Normal"])
                    rfs = st.selectbox("Tình trạng tái phát (Relapse Free Status)",
                                       ["Not Recurred", "Recurred"])

                submitted = st.form_submit_button("🚀 Dự đoán nguy cơ tử vong (Deceased)")

            if submitted:
                # Tạo 1 dòng DataFrame đúng thứ tự cột đã dùng khi train
                row = {}

                # điền cột số
                row["Age at Diagnosis"] = age
                row["Tumor Size"] = tumor_size
                row["Lymph nodes examined positive"] = ln_pos
                row["Mutation Count"] = mut_count
                row["Nottingham prognostic index"] = npi
                row["Overall Survival (Months)"] = os_months

                # điền cột phân loại
                row["Type of Breast Surgery"] = surgery
                row["Neoplasm Histologic Grade"] = grade
                row["Tumor Stage"] = stage
                row["Sex"] = sex
                row["Cellularity"] = cellularity
                row["Chemotherapy"] = chemo
                row["Hormone Therapy"] = horm
                row["Radio Therapy"] = radio
                row["ER Status"] = er
                row["PR Status"] = pr
                row["HER2 Status"] = her2
                row["3-Gene classifier subtype"] = gene3
                row["Pam50 + Claudin-low subtype"] = pam50
                row["Relapse Free Status"] = rfs

                input_df = pd.DataFrame([row])

                # Dự đoán
                with st.spinner("⏳ Đang tính toán dựa trên mô hình lâm sàng Epic..."):
                    p_deceased = float(clinical_model.predict_proba(input_df)[0, 1])

                st.success(f"💀 Xác suất bệnh nhân **tử vong** theo mô hình Epic: **{p_deceased:.3f}**")
                st.caption("(*) Một số biến như thời gian theo dõi chỉ mang tính mô phỏng, "
                           "trong thực tế không thể biết trước tại thời điểm chẩn đoán.")

                # Gợi ý chữ nghĩa (hoàn toàn phi lâm sàng, chỉ để minh họa)
                if p_deceased >= 0.8:
                    st.warning("Nguy cơ tiên lượng xấu **rất cao** (theo mô hình Epic, chỉ mang tính minh họa).")
                elif p_deceased >= 0.6:
                    st.warning("Nguy cơ tiên lượng xấu **cao** (theo mô hình Epic, chỉ mang tính minh họa).")
                elif p_deceased >= 0.4:
                    st.info("Nguy cơ tiên lượng xấu **trung bình** (theo mô hình Epic, chỉ mang tính minh họa).")
                else:
                    st.info("Nguy cơ tiên lượng xấu **thấp** (theo mô hình Epic, chỉ mang tính minh họa).")

                st.caption("""
                Kết quả trên được suy ra từ một mô hình học máy huấn luyện trên bộ dữ liệu nghiên cứu quốc tế,
                không đại diện cho bệnh nhân tại từng cơ sở cụ thể và **không dùng để thay thế quyết định của bác sĩ**.
                """)

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

    Ngoài ra, mô hình lâm sàng được huấn luyện trên bộ dữ liệu:
    - **Breast_Cancer_METABRIC_Epic_Hospital.csv** (METABRIC + Epic Hospital), công bố trên nền tảng Mendeley Data (giấy phép CC BY 4.0).

    ---
    **Giấy phép sử dụng:**  
    - Phi thương mại (Non-Commercial).  
    - Phải trích dẫn nguồn dữ liệu gốc.  
    - Không sử dụng cho mục đích y tế hoặc thương mại.
    """)

# -----------------------------
# Chân trang (footer)
# -----------------------------
st.markdown("""
---
📘 **Tuyên bố miễn trừ trách nhiệm:**  
Ứng dụng này được phát triển phục vụ mục đích **nghiên cứu khoa học và giáo dục**.  
Không sử dụng cho **chẩn đoán, điều trị hoặc tư vấn y tế**.  

🧪 Mô hình lâm sàng Epic chỉ là mô phỏng dựa trên dữ liệu quốc tế,  
không đại diện cho dân số Việt Nam và không dùng trong quyết định lâm sàng.  

© 2025 – Dự án AI Siêu âm Vú. Tác giả: Lê Vũ Anh Tin – Trường THPT Chuyên Nguyễn Du.
""")
