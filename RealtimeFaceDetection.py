import cv2
import numpy as np
from KenaliWajahHaar import kenali_wajah
import streamlit as st
from PIL import Image

def realtime_face_detection():
    st.title("🔴 Deteksi Wajah Real-Time")
    
    # State untuk kontrol deteksi
    if 'detection_active' not in st.session_state:
        st.session_state.detection_active = False
    
    # Tombol Start/Stop
    col1, col2 = st.columns(2)
    with col1:
        if not st.session_state.detection_active:
            if st.button("▶ Mulai Deteksi", type="primary"):
                st.session_state.detection_active = True
                st.rerun()
    
    with col2:
        if st.session_state.detection_active:
            if st.button("⏹ Hentikan Deteksi", type="secondary"):
                st.session_state.detection_active = False
                st.rerun()
    
    # Placeholder untuk video stream
    frame_placeholder = st.empty()
    status_text = st.empty()
    
    if st.session_state.detection_active:
        status_text.info("🟢 Deteksi aktif - Menghadap kamera...")
        cap = cv2.VideoCapture(0)
        
        # Pengaturan kamera
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 24)
        
        try:
            while st.session_state.detection_active and cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    status_text.error("Gagal mengambil frame dari kamera")
                    break
                
                # Konversi warna dan deteksi wajah
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                names, result_frame = kenali_wajah(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
                result_frame_rgb = cv2.cvtColor(result_frame, cv2.COLOR_BGR2RGB)
                
                # Tampilkan frame dengan deteksi
                frame_placeholder.image(result_frame_rgb, channels="RGB", use_column_width=True)
                
                # Update status
                if names[0] != "Wajah tidak terdeteksi":
                    status_text.success(f"👥 {len(names)} wajah terdeteksi | {' | '.join(names)}")
                else:
                    status_text.warning("❌ Tidak ada wajah terdeteksi")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
    
    else:
        # Buat placeholder hitam sederhana
        placeholder_image = np.zeros((480, 640, 3), dtype=np.uint8)
        frame_placeholder.image(placeholder_image, caption="Kamera tidak aktif", use_column_width=True)
        status_text.info("⏸️ Deteksi dihentikan - Tekan Mulai Deteksi untuk memulai")