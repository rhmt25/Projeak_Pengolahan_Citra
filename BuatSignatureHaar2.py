# BuatSignatureHaar2.py (versi perbaikan)
print(">> Script BuatSignatureHaar2.py dimulai <<")
import os
import pickle
import cv2
from PIL import Image
import numpy as np
from keras_facenet import FaceNet

def buat_signature_from_folder(folder='\data_foto', output_file='data.pkl'):
    model = FaceNet()
    haar = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    database = {}

    if not os.path.exists(folder):
        print(f"Folder '{folder}' tidak ditemukan.")
        return

    file_list = [f for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if not file_list:
        print(f"Tidak ada gambar di folder '{folder}'.")
        return

    for filename in file_list:
        path = os.path.join(folder, filename)
        image = cv2.imread(path)

        if image is None:
            print(f"Gagal membaca gambar: {filename}")
            continue

        faces = haar.detectMultiScale(image, 1.1, 4)
        if len(faces) == 0:
            print(f"Tidak ditemukan wajah pada gambar: {filename}")
            continue

        x, y, w, h = faces[0]
        face = image[y:y+h, x:x+w]
        face = cv2.resize(face, (160, 160))
        face = np.expand_dims(face, axis=0)
        signature = model.embeddings(face)

        name = os.path.splitext(filename)[0]
        database[name] = signature
        print(f"✔ Wajah '{name}' diproses.")

    if database:
        with open(output_file, 'wb') as f:
            pickle.dump(database, f)
        print(f"✅ Signature disimpan ke {output_file} ({len(database)} data).")
    else:
        print("⚠ Tidak ada wajah yang berhasil diproses. File data.pkl tidak dibuat.")

# if __name__ == "__main__":
#     buat_signature_from_folder()
