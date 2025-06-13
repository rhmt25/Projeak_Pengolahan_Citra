import cv2
import numpy as np
from KenaliWajahHaar import kenali_wajah
import streamlit as st
from PIL import Image

import cv2
import numpy as np
from KenaliWajahHaar import kenali_wajah
import streamlit as st

def realtime_face_detection():
    st.title("🔴 Deteksi Wajah Real-Time")
    
    # State management
    if 'detection_active' not in st.session_state:
        st.session_state.detection_active = False
    
    # Tombol kontrol
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
    
    frame_placeholder = st.empty()
    status_text = st.empty()
    
    if st.session_state.detection_active:
        # Coba akses kamera
        cap = None
        for i in range(4):  # Coba indeks 0-3
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)  # Gunakan CAP_DSHOW untuk Windows
            if cap.isOpened():
                break
            if cap:
                cap.release()
        
        if not cap or not cap.isOpened():
            status_text.error("""
            ❌ Kamera tidak dapat diakses. 
            - Pastikan kamera terhubung
            - Tidak digunakan aplikasi lain
            - Coba refresh halaman
            """)
            st.session_state.detection_active = False
            return
        
        try:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            
            while st.session_state.detection_active:
                ret, frame = cap.read()
                if not ret:
                    status_text.warning("Gagal membaca frame kamera")
                    continue
                
                # Proses deteksi
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                names, result_frame = kenali_wajah(frame)
                result_frame_rgb = cv2.cvtColor(result_frame, cv2.COLOR_BGR2RGB)
                
                # Tampilkan hasil
                frame_placeholder.image(result_frame_rgb, channels="RGB", use_container_width=True)
                
                if names[0] == "Wajah tidak terdeteksi":
                    status_text.warning("👀 Mencari wajah...")
                else:
                    status_text.success(f"👥 Terdeteksi: {', '.join(names)}")
                
        except Exception as e:
            status_text.error(f"Error: {str(e)}")
        
        finally:
            if 'cap' in locals() and cap.isOpened():
                cap.release()
            cv2.destroyAllWindows()
    
    else:
        # Tampilkan placeholder saat tidak aktif
        placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(placeholder, "Kamera tidak aktif", (50, 240), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        frame_placeholder.image(placeholder, use_container_width=True)
        status_text.info("⏸️ Tekan 'Mulai Deteksi' untuk memulai")