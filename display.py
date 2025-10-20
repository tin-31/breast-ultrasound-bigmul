# ==============================
# 🔹 Streamlit UI (ĐÃ CẬP NHẬT: Nút chuyển đổi ngôn ngữ)
# ==============================
st.set_page_config(page_title="Breast Ultrasound AI", layout="wide", page_icon="🩺")

# ---------------------------------------
# 🌐 Nút chuyển đổi ngôn ngữ toàn trang
# ---------------------------------------
if "lang" not in st.session_state:
    st.session_state.lang = "vi"  # Mặc định là Tiếng Việt

# CSS để chỉnh vị trí và kích thước nút
lang_button_css = """
    <style>
        div[data-testid="stToolbar"] {
            right: 120px !important;
        }
        #lang-toggle {
            position: fixed;
            top: 10px;
            right: 70px;
            z-index: 1000;
        }
        div[data-testid="stToolbarActions"] button {
            transform: scale(2.0) !important; /* Gấp đôi kích thước nút GitHub */
        }
    </style>
"""
st.markdown(lang_button_css, unsafe_allow_html=True)

# Hiển thị nút chuyển ngôn ngữ
lang_label = "🌏 English" if st.session_state.lang == "vi" else "🇻🇳 Tiếng Việt"
if st.button(lang_label, key="lang-btn"):
    st.session_state.lang = "en" if st.session_state.lang == "vi" else "vi"
    st.rerun()

# ---------------------------------------
# 🔸 Sidebar song ngữ
# ---------------------------------------
if st.session_state.lang == "vi":
    st.sidebar.title("📘 Điều hướng")
    app_mode = st.sidebar.selectbox("Chọn trang", ["Ứng dụng chẩn đoán", "Thông tin chung", "Thống kê về dữ liệu huấn luyện"])
else:
    st.sidebar.title("📘 Navigation")
    app_mode = st.sidebar.selectbox("Select page", ["Diagnostic App", "About", "Training Data Statistics"])

# ---------------------------------------
# 🔸 Trang Thông tin chung / About
# ---------------------------------------
if (st.session_state.lang == "vi" and app_mode == "Thông tin chung") or (st.session_state.lang == "en" and app_mode == "About"):
    if st.session_state.lang == "vi":
        st.title("👨‍🎓 Giới thiệu về thành viên")
        st.markdown("<h4>Lê Vũ Anh Tin - 11TH</h4>", unsafe_allow_html=True)
        try:
            st.image("Tin.jpg", width=500)
            st.markdown("<h4>Trường THPT Chuyên Nguyễn Du</h4>", unsafe_allow_html=True)
            st.image("school.jpg", width=500)
        except:
            st.info("🖼️ Ảnh giới thiệu chưa được tải lên.")
    else:
        st.title("👨‍🎓 Team Member Introduction")
        st.markdown("<h4>Lê Vũ Anh Tin - Grade 11TH</h4>", unsafe_allow_html=True)
        try:
            st.image("Tin.jpg", width=500)
            st.markdown("<h4>Nguyen Du High School for the Gifted</h4>", unsafe_allow_html=True)
            st.image("school.jpg", width=500)
        except:
            st.info("🖼️ Introduction images not uploaded yet.")

# ---------------------------------------
# 🔸 Trang thống kê dữ liệu / Statistics
# ---------------------------------------
elif (st.session_state.lang == "vi" and app_mode == "Thống kê về dữ liệu huấn luyện") or (st.session_state.lang == "en" and app_mode == "Training Data Statistics"):
    if st.session_state.lang == "vi":
        st.title("📊 Thống kê tổng quan về tập dữ liệu")
        st.caption("""
        Tập dữ liệu **Breast Ultrasound Images (BUI)** được kết hợp từ ba nguồn chính...
        """)
    else:
        st.title("📊 Overview of the Training Dataset")
        st.caption("""
        The **Breast Ultrasound Images (BUI)** dataset combines data from three main sources...
        """)

# ---------------------------------------
# 🔸 Trang ứng dụng chẩn đoán / Diagnostic App
# ---------------------------------------
elif (st.session_state.lang == "vi" and app_mode == "Ứng dụng chẩn đoán") or (st.session_state.lang == "en" and app_mode == "Diagnostic App"):
    if st.session_state.lang == "vi":
        st.title("🩺 Ứng dụng chẩn đoán bệnh ung thư vú từ hình ảnh siêu âm")
        file_label = "📤 Tải ảnh siêu âm (JPG hoặc PNG)"
        info_text = "👆 Vui lòng tải ảnh lên để bắt đầu chẩn đoán."
        seg_caption = "Kết quả phân đoạn (Đỏ: Ác tính, Xanh: Lành tính)"
        result_labels = ["Lành tính", "Ác tính", "Bình thường"]
    else:
        st.title("🩺 Breast Cancer Diagnostic App from Ultrasound Images")
        file_label = "📤 Upload ultrasound image (JPG or PNG)"
        info_text = "👆 Please upload an image to begin diagnosis."
        seg_caption = "Segmentation Result (Red: Malignant, Green: Benign)"
        result_labels = ["Benign", "Malignant", "Normal"]

    classifier, segmentor = load_models()
    file = st.file_uploader(file_label, type=["jpg", "png"])

    if file is None:
        st.info(info_text)
    else:
        slot = st.empty()
        slot.text("⏳ Analyzing image..." if st.session_state.lang == "en" else "⏳ Đang phân tích ảnh...")

        pred_class, seg_image, img_bytes = predict_pipeline(file, classifier, segmentor)
        input_image = Image.open(BytesIO(img_bytes))

        col1, col2 = st.columns(2)
        with col1:
            st.image(input_image, caption="Original Image" if st.session_state.lang == "en" else "Ảnh gốc", use_container_width=True)
        with col2:
            st.image(seg_image, caption=seg_caption, use_container_width=True)

        class_names = ["benign", "malignant", "normal"]
        result_index = np.argmax(pred_class)
        result = class_names[result_index]

        st.markdown("---")
        st.subheader("💡 Diagnostic Result" if st.session_state.lang == "en" else "💡 Kết quả chẩn đoán")

        if result == "benign":
            st.success("🟢 Benign tumor detected." if st.session_state.lang == "en" else "🟢 Kết luận: Khối u lành tính.")
        elif result == "malignant":
            st.error("🔴 Malignant breast cancer detected." if st.session_state.lang == "en" else "🔴 Kết luận: Ung thư vú ác tính.")
        else:
            st.info("⚪ No tumor detected (Normal)." if st.session_state.lang == "en" else "⚪ Kết luận: Không phát hiện khối u (Bình thường).")

        st.markdown("---")
        st.subheader("📈 Probability Details" if st.session_state.lang == "en" else "📈 Chi tiết xác suất")

        chart_df = pd.DataFrame({
            "Diagnosis Type" if st.session_state.lang == "en" else "Loại chẩn đoán": result_labels,
            "Probability (%)": [pred_class[0,0]*100, pred_class[0,1]*100, pred_class[0,2]*100]
        })

        chart = alt.Chart(chart_df).mark_bar().encode(
            x=alt.X("Diagnosis Type" if st.session_state.lang == "en" else "Loại chẩn đoán", sort=None),
            y=alt.Y("Probability (%)", scale=alt.Scale(domain=[0, 100])),
            color=alt.Color("Diagnosis Type" if st.session_state.lang == "en" else "Loại chẩn đoán",
                            scale=alt.Scale(
                                domain=result_labels,
                                range=["#10B981", "#EF4444", "#9CA3AF"]
                            )),
            tooltip=[alt.Tooltip("Probability (%)", format=".2f")]
        ).properties(
            title="Diagnosis Probability Chart" if st.session_state.lang == "en" else "Biểu đồ Xác suất Chẩn đoán"
        )
        st.altair_chart(chart, use_container_width=True)

        slot.success("✅ Diagnosis completed!" if st.session_state.lang == "en" else "✅ Hoàn tất chẩn đoán!")
