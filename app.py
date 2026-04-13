import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import io
import matplotlib.pyplot as plt
import os


st.set_page_config(page_title="Brain Tumor Classification", layout="wide")

# --- LOAD MODEL (CACHED) ---
# Menggunakan cache agar model tidak diload berulang kali setiap refresh
@st.cache_resource
def get_model():
    model_path = "best_vgg16.h5"
    if not os.path.exists(model_path):
        return None
    return load_model(model_path)

model = get_model()
CLASS_LABELS = ["Glioma", "Meningioma", "Pituitary", "No Tumor"]
IMAGE_SIZE = (224, 224)

# --- FUNGSI PREDIKSI ---
def predict_class(img_array, model):
    img_scaled = img_array / 255.0
    img_scaled = np.expand_dims(img_scaled, axis=0)
    
    preds = model.predict(img_scaled)[0]
    class_index = np.argmax(preds)
    confidence = float(np.max(preds))
    
    return CLASS_LABELS[class_index], confidence, preds

# --- HEADER ---
st.title("🧠 Brain Tumor Classification")
st.markdown("---")

# Cek apakah model berhasil di-load
if model is None:
    st.error("File model 'best_vgg16.h5' tidak ditemukan. Pastikan file berada di direktori yang sama.")
    st.stop()

# --- LAYOUT UTAMA ---
# Col 1: Kiri (Upload & Hasil Teks)
# Spacer: Spasi tengah
# Col 2: Kanan (Gambar & Bar Chart)
col1, spacer, col2 = st.columns([1, 0.1, 1], gap="large")

with col1:
    st.subheader("Upload Citra")
    uploaded_file = st.file_uploader("Pilih file MRI (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Proses Gambar
    img = Image.open(uploaded_file).convert("RGB")
    img_resized = img.resize(IMAGE_SIZE)
    img_array = image.img_to_array(img_resized)
    
    # Lakukan Prediksi
    label, confidence, probas = predict_class(img_array, model)

    # --- KOLOM KANAN (TAMPILAN GAMBAR) ---
    with col2:
        st.subheader("Preview Citra")
        # Menampilkan gambar tepat di sebelah kanan kolom upload
        st.image(img, caption="Citra MRI yang diupload", width=300)

    # --- KOLOM KIRI BAWAH (HASIL PREDIKSI) ---
    with col1:
        st.write("") # Spasi vertikal
        st.divider() # Garis pembatas
        st.subheader("Hasil Diagnosa")
        
        # Penentuan warna untuk UI
        if label == "No Tumor":
            status_color = "normal" # Hijau di st.metric
            msg_type = st.success
        else:
            status_color = "off"
            msg_type = st.error

        # Tampilan Hasil Utama
        msg_type(f"**Kelas Terdeteksi: {label}**")
        st.metric(label="Confidence Score", value=f"{confidence:.2%}")
        
        st.info("Hasil ini digenerate menggunakan model Deep Learning VGG16.")

    # --- KOLOM KANAN BAWAH (BAR PROBABILITAS) ---
    with col2:
        st.write("") # Spasi vertikal
        st.divider() # Garis pembatas
        st.subheader("Detail Probabilitas")
        
        # Menampilkan Bar Chart untuk setiap kelas
        for cls, prob in zip(CLASS_LABELS, probas):
            st.write(f"**{cls}** ({prob:.2%})")
            st.progress(float(prob))

    
    with col1:
         # Membuat plot untuk didownload
        fig = plt.figure(figsize=(4, 4))
        plt.imshow(img_resized)
        plt.title(f"Pred: {label}\nConf: {confidence:.2%}")
        plt.axis("off")

        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches='tight')
        buf.seek(0)
        
        st.download_button(
            label="⬇️ Download Hasil Analisa",
            data=buf,
            file_name=f"prediksi_{label}.png",
            mime="image/png",
            use_container_width=True
        )

else:
    # Tampilan placeholder jika belum ada gambar
    with col2:
        st.info("Silakan upload gambar MRI di kolom sebelah kiri untuk melihat hasil prediksi.")