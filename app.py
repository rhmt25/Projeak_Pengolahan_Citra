import streamlit as st
import numpy as np
import cv2
from PIL import Image
from KenaliWajahHaar import kenali_wajah
import os
from BuatSignatureHaar2 import buat_signature_from_folder

# ✅ Jalankan hanya sekali saat data.pkl belum ada
if not os.path.exists("data.pkl"):
    buat_signature_from_folder("data_foto/")
    print("✅ Signature database dibuat otomatis.")

st.set_page_config("Aplikasi Pengolahan Citra", layout="wide")
st.title("PROYEK APLIKASI PENGENALAN WAJAH")

st.sidebar.title("🔧 Pilih Metode")
method = st.sidebar.selectbox("Metode Pengolahan", [
    "Tambah Data Wajah",
    "Deteksi Wajah"
])

if method == "Tambah Data Wajah":
    st.subheader("📸 Pilih metode input:")
    input_mode = st.radio("Metode Input", ["Upload Gambar", "Gunakan Kamera"])

    nama = st.text_input("Nama orang: ")

    # Inisialisasi variabel agar tidak error
    camera_file = None
    uploaded_file = None
    image = None

    if input_mode == "Upload Gambar":
        uploaded_file = st.file_uploader("Unggah satu foto wajah", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")

    elif input_mode == "Gunakan Kamera":
        camera_file = st.camera_input("Ambil gambar dari kamera")
        if camera_file is not None:
            image = Image.open(camera_file).convert("RGB")

    if image is not None and nama:
        if st.button("Simpan & Update Signature"):
            os.makedirs("data_foto", exist_ok=True)
            save_path = os.path.join("data_foto", f"{nama}.jpg")
            image.save(save_path)

            buat_signature_from_folder("data_foto/")
            st.success(f"Wajah atas nama '{nama}' telah ditambahkan dan database diperbarui.")
    elif (uploaded_file or camera_file) and not nama:
        st.warning("Masukkan nama terlebih dahulu sebelum menyimpan.")



elif method == "Deteksi Wajah":
    st.subheader("📸 Pilih metode input:")
    input_mode = st.radio("Metode Input", ["Upload Gambar", "Gunakan Kamera"])

    image = None

    if input_mode == "Upload Gambar":
        uploaded_file = st.file_uploader("Unggah gambar wajah", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")
            if st.button("Deteksi Wajah"):
                image_np = np.array(image)
                image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
                name, result_image = kenali_wajah(image_bgr)

                st.subheader(f"Hasil Deteksi: {name}")
                st.image(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB), use_column_width=True)

    elif input_mode == "Gunakan Kamera":
        camera_file = st.camera_input("Ambil gambar dari kamera")

        if camera_file is not None:
            image = Image.open(camera_file).convert("RGB")
            if st.button("Deteksi Wajah"):
                image_np = np.array(image)
                image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
                name, result_image = kenali_wajah(image_bgr)

                st.subheader(f"Hasil Deteksi: {name}")
                st.image(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB), use_column_width=True)
