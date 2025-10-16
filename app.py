import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image
import gdown

# ===== DOWNLOAD MODELS =====
SEG_MODEL_ID = "1CYBZRssHYWNErdU0SbcYdhzGIwHIL2ra"
CLF_MODEL_ID = "1fXPICuTkETep2oPiA56l0uMai2GusEJH"
SEG_MODEL_PATH = "seg_model.keras"
CLF_MODEL_PATH = "clf_model.h5"

for url, path in [
    (f"https://drive.google.com/uc?id={SEG_MODEL_ID}", SEG_MODEL_PATH),
    (f"https://drive.google.com/uc?id={CLF_MODEL_ID}", CLF_MODEL_PATH)
]:
    if not tf.io.gfile.exists(path):
        gdown.download(url, path, quiet=False)

# ===== LOAD MODELS =====
import keras
keras.config.enable_unsafe_deserialization()
seg_model = tf.keras.models.load_model(SEG_MODEL_PATH, compile=False)
clf_model = tf.keras.models.load_model(CLF_MODEL_PATH, compile=False)

# ===== FUNCTIONS =====
def process_image(image):
    # Preprocess for classification
    img_clf = image.resize((224, 224))
    arr_clf = np.expand_dims(np.array(img_clf) / 255.0, 0)
    # Preprocess for segmentation
    img_seg = image.resize((256, 256))
    arr_seg = np.expand_dims(np.array(img_seg) / 255.0, 0)
    # Predict
    clf_pred = clf_model.predict(arr_clf)[0]
    seg_pred = seg_model.predict(arr_seg)[0]
    seg_mask = np.argmax(seg_pred, axis=-1)[..., None] * 127
    seg_mask_rgb = np.repeat(seg_mask, 3, axis=-1).astype(np.uint8)
    # Class result
    classes = ["Benign (Lành tính)", "Malignant (Ác tính)", "Normal"]
    result = classes[np.argmax(clf_pred)]
    return Image.fromarray(seg_mask_rgb), {classes[i]: float(clf_pred[i]) for i in range(3)}, result

# ===== GRADIO UI =====
demo = gr.Interface(
    fn=process_image,
    inputs=gr.Image(type="pil", label="Upload ảnh siêu âm"),
    outputs=[
        gr.Image(label="Phân đoạn khối u"),
        gr.Label(label="Xác suất chẩn đoán"),
        gr.Textbox(label="Kết luận")
    ],
    title="🩺 Chẩn đoán ung thư vú từ ảnh siêu âm",
    description="Model AI phát hiện và phân loại khối u (lành tính / ác tính / bình thường)"
)

demo.launch()
