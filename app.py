import streamlit as st
import os
import gdown
import numpy as np
import pandas as pd
import cv2
import joblib
import keras  # Keras 3.4.1 (with TensorFlow 2.16.1 backend) for model loading
# (Ensure scikit-learn 1.2.2 is installed for unpickling the classifier model.)

# --- 1. Download model files from Google Drive if not already present ---
# Create a directory for models
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# Google Drive file IDs for the large files
drive_files = {
    "Classifier_model_2.h5": "1fXPICuTkETep2oPiA56l0uMai2GusEJH",
    "best_model_cbam_attention_unet_fixed.keras": "1axOg7N5ssJrMec97eV-JMPzID26ynzN1",
    "clinical_epic_gb_model.pkl": "1z1wHVy9xyRXlRqxI8lYXMJhaJaUcKXnu",
    "clinical_epic_gb_metadata.pkl": "1WWlfeRqr99VL4nBQ-7eEptIxitKtXj6V"
}

# Download each file if not already in MODEL_DIR
with st.spinner("Downloading model files (if not already cached)..."):
    for filename, file_id in drive_files.items():
        dest_path = os.path.join(MODEL_DIR, filename)
        if not os.path.exists(dest_path):
            # Construct the gdown URL and download
            url = f"https://drive.google.com/uc?id={file_id}"
            gdown.download(url, dest_path, quiet=False)
# Files will be downloaded only once. Subsequent runs will skip this.

# --- 2. Load models and metadata (with caching to avoid re-loading on each run) ---
@st.cache_resource  # Cache the loaded models across runs (Streamlit v1.18+)
def load_models():
    # Load the CBAM U-Net segmentation model
    seg_model = keras.models.load_model(
        os.path.join(MODEL_DIR, "best_model_cbam_attention_unet_fixed.keras"), 
        compile=False  # no need to compile for inference
    )
    # Load the ultrasound image classification model
    class_model = keras.models.load_model(
        os.path.join(MODEL_DIR, "Classifier_model_2.h5"), 
        compile=False
    )
    # Load the clinical gradient boosting model and metadata
    gb_model = joblib.load(os.path.join(MODEL_DIR, "clinical_epic_gb_model.pkl"))
    gb_meta = joblib.load(os.path.join(MODEL_DIR, "clinical_epic_gb_metadata.pkl"))
    return seg_model, class_model, gb_model, gb_meta

# Load all models (this will run only once thanks to caching)
seg_model, class_model, gb_model, gb_meta = load_models()

# Extract metadata for clinical features
feature_names = gb_meta["feature_names"]    # list of all feature column names used in the model
num_cols     = gb_meta["num_cols"]          # original numerical columns
cat_cols     = gb_meta["cat_cols"]          # original categorical columns
label_map    = gb_meta["label_map"]         # {'Living': 0, 'Deceased': 1}

# Prepare inverse label map for output
inv_label_map = {v: k for k, v in label_map.items()}

# --- 3. Set up Streamlit app UI ---
st.title("Breast Cancer Prediction App")
st.markdown("This web application allows you to analyze a breast ultrasound image (for tumor segmentation and classification) and input clinical data to predict patient survival. The models will run with the specified versions of TensorFlow/Keras, and all necessary files will be downloaded automatically using **gdown**.")

# Create two tabs: one for Image Analysis, one for Clinical Prediction
tab1, tab2 = st.tabs(["🔎 Ultrasound Image Analysis", "📊 Clinical Survival Prediction"])

# Tab 1: Ultrasound image segmentation and classification
with tab1:
    st.header("Ultrasound Image Analysis")
    st.write("Upload a breast ultrasound image. The app will segment the tumor (if present) and classify the lesion as benign, malignant, or normal.")
    
    # File uploader for ultrasound image
    uploaded_file = st.file_uploader("Choose an ultrasound image file (PNG or JPG)", type=["png", "jpg", "jpeg"])
    
    if uploaded_file is not None:
        # Read image data from the uploaded file
        file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
        if img is None:
            st.error("Could not read the image. Please upload a valid image file.")
        else:
            # Preprocess the image: normalize intensity and resize to 256x256
            img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            orig_shape = img.shape  # original image shape (h, w)
            img_resized = cv2.resize(img, (256, 256), interpolation=cv2.INTER_LINEAR)
            
            # Prepare image for models
            # Segmentation model expects shape (1, 256, 256, 1) with intensity 0-255
            img_for_seg = img_resized.astype(np.float32)
            img_for_seg = np.expand_dims(img_for_seg, axis=(0, -1))  # add batch and channel dims
            
            # Classification model expects shape (1, 256, 256, 3).
            # Convert grayscale to 3-channel (RGB) and cast to float32.
            img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2RGB).astype(np.float32)
            img_for_class = np.expand_dims(img_rgb, axis=0)  # shape (1,256,256,3)
            # (The EfficientNetV2 model has preprocessing included; the pixel values 0-255 float are acceptable.)
            
            # Run the segmentation model to get mask prediction
            mask_pred = seg_model.predict(img_for_seg)[0]  # shape (256,256,1)
            mask_pred = mask_pred[..., 0]  # reshape to (256,256)
            # Threshold the predicted mask (assuming binary segmentation with sigmoid output)
            mask_bin = (mask_pred >= 0.5).astype(np.uint8)
            
            # Create an overlay of the mask on the image for visualization
            base_img = np.stack([img_resized]*3, axis=-1)  # make 3-channel grayscale image for display
            mask_color = np.zeros_like(base_img)
            # Mark mask area in red color
            mask_color[:, :, 0] = mask_bin * 255  # Red channel
            mask_color[:, :, 1] = 0               # Green channel
            mask_color[:, :, 2] = 0               # Blue channel
            alpha = 0.4  # transparency for mask overlay
            overlay_img = cv2.addWeighted(base_img, 1-alpha, mask_color, alpha, 0)
            
            # Run the classification model to predict benign/malignant/normal
            class_probs = class_model.predict(img_for_class)[0]  # prediction probabilities for 3 classes
            class_names = ['benign', 'malignant', 'normal']      # class order as per training
            pred_idx = int(np.argmax(class_probs))
            pred_label = class_names[pred_idx]
            
            # Display results
            col1, col2 = st.columns(2)
            with col1:
                st.image(base_img, caption="Original Image (rescaled to 256x256)", use_column_width=True)
            with col2:
                st.image(overlay_img, caption="Tumor Segmentation Overlay", use_column_width=True)
            
            st.write(f"**Classification Result:** The model predicts this lesion is **{pred_label.upper()}**.")
            # Show probabilities for each class in a bar chart
            prob_percent = (class_probs * 100).round(2)
            probs_df = pd.DataFrame({
                'Category': ['Benign', 'Malignant', 'Normal'],
                'Probability (%)': prob_percent
            })
            st.altair_chart(
                alt.Chart(probs_df).mark_bar(color='#4E79A7').encode(
                    x=alt.X('Category', sort=None), 
                    y=alt.Y('Probability (%)', scale=alt.Scale(domain=[0,100]))
                ),
                use_container_width=True
            )
            st.caption("Predicted probability for each class (benign vs malignant vs normal).")
    else:
        st.info("Please upload a breast ultrasound image to analyze.")
        
# Tab 2: Clinical data input for survival prediction
with tab2:
    st.header("Clinical Survival Prediction")
    st.write("Enter the patient's clinical information below. Upon submission, the model will predict the patient's 5-year survival outcome (Living or Deceased).")
    # Create a form for input fields
    with st.form("clinical_form"):
        # Numeric features
        age = st.number_input("Age at Diagnosis", min_value=0.0, max_value=120.0, value=50.0)
        tumor_size = st.number_input("Tumor Size (mm)", min_value=0.0, max_value=200.0, value=20.0)
        lymph_pos = st.number_input("Lymph nodes examined positive", min_value=0, max_value=50, value=0, step=1)
        mutation_count = st.number_input("Mutation Count", min_value=0, max_value=10000, value=0, step=1)
        npi = st.number_input("Nottingham Prognostic Index", min_value=0.0, max_value=10.0, value=4.0, format="%.2f")
        os_months = st.number_input("Overall Survival (Months)", min_value=0.0, max_value=300.0, value=60.0, format="%.2f")
        
        # Categorical features (with baseline category as first option by default)
        surgery_type = st.selectbox("Type of Breast Surgery", ["Breast Conserving", "Mastectomy"], index=0)
        hist_grade = st.selectbox("Neoplasm Histologic Grade", [1, 2, 3], index=0)
        tumor_stage = st.selectbox("Tumor Stage", [1, 2, 3, 4], index=0)
        sex = st.selectbox("Sex", ["Female", "Male"], index=0)
        cellularity = st.selectbox("Cellularity", ["High", "Low", "Moderate"], index=0)
        chemo = st.selectbox("Chemotherapy", ["No", "Yes"], index=0)
        hormone = st.selectbox("Hormone Therapy", ["No", "Yes"], index=0)
        radio = st.selectbox("Radio Therapy", ["No", "Yes"], index=0)
        er_status = st.selectbox("ER Status", ["Negative", "Positive"], index=0)
        pr_status = st.selectbox("PR Status", ["Negative", "Positive"], index=0)
        her2_status = st.selectbox("HER2 Status", ["Negative", "Positive"], index=0)
        gene_subtype = st.selectbox("3-Gene classifier subtype", 
                                    ["ER+/HER2+", "ER+/HER2- High Prolif", "ER+/HER2- Low Prolif", "ER-/HER2+", "ER-/HER2-"], index=0)
        pam50_subtype = st.selectbox("Pam50 + Claudin-low subtype", 
                                     ["Basal-like", "Claudin-low", "HER2-enriched", "Luminal A", "Luminal B", "Normal-like"], index=0)
        relapse_status = st.selectbox("Relapse Free Status", ["Not Recurred", "Recurred"], index=0)
        
        submit_btn = st.form_submit_button("Predict Survival")
    
    if submit_btn:
        # When the user submits the form, construct the feature vector
        # Initialize a dataframe with one row, all features set to 0
        X_input = pd.DataFrame(data=[np.zeros(len(feature_names))], columns=feature_names)
        
        # Set numeric features
        X_input.at[0, "Age at Diagnosis"] = age
        X_input.at[0, "Tumor Size"] = tumor_size
        X_input.at[0, "Lymph nodes examined positive"] = lymph_pos
        X_input.at[0, "Mutation Count"] = mutation_count
        X_input.at[0, "Nottingham prognostic index"] = npi
        X_input.at[0, "Overall Survival (Months)"] = os_months
        
        # Helper function to set dummy column for a categorical value (if not baseline)
        def set_dummy(col_name, value):
            dummy_col = f"{col_name}_{value}"
            # If the value is numeric (e.g., 2.0), ensure format matches the column name
            if isinstance(value, (int, float)):
                # For numeric categories, match the exact format in feature_names (e.g., "2.0")
                # (If value has no decimal and baseline was float, add .0)
                if f"{col_name}_{value}" not in feature_names and f"{col_name}_{value}.0" in feature_names:
                    dummy_col = f"{col_name}_{value}.0"
                else:
                    dummy_col = f"{col_name}_{value}"
            # Set dummy column to 1 if it exists (means the selected category is not the baseline)
            if dummy_col in feature_names:
                X_input.at[0, dummy_col] = 1
        
        # Set categorical features (only the dummy corresponding to the chosen category)
        set_dummy("Type of Breast Surgery", surgery_type)
        set_dummy("Neoplasm Histologic Grade", hist_grade)
        set_dummy("Tumor Stage", tumor_stage)
        set_dummy("Sex", sex)
        set_dummy("Cellularity", cellularity)
        set_dummy("Chemotherapy", chemo)
        set_dummy("Hormone Therapy", hormone)
        set_dummy("Radio Therapy", radio)
        set_dummy("ER Status", er_status)
        set_dummy("PR Status", pr_status)
        set_dummy("HER2 Status", her2_status)
        set_dummy("3-Gene classifier subtype", gene_subtype)
        set_dummy("Pam50 + Claudin-low subtype", pam50_subtype)
        set_dummy("Relapse Free Status", relapse_status)
        
        # Make prediction using the gradient boosting model
        y_pred = gb_model.predict(X_input)[0]              # predicted class (0 or 1)
        y_proba = gb_model.predict_proba(X_input)[0, 1]    # probability of class 1 ("Deceased")
        outcome_label = inv_label_map.get(y_pred, "Unknown")
        
        # Display the prediction result
        if outcome_label == "Deceased":
            st.error(f"**Predicted Outcome:** {outcome_label} (high risk).")
        else:
            st.success(f"**Predicted Outcome:** {outcome_label} (likely survivor).")
        st.write(f"**Probability of death:** {y_proba*100:.1f}%")
        st.caption("(*Note:* The model prediction is based on the provided data and assumes a 5-year follow-up period.)")
