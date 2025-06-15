print(">> Script KenaliWajahHaar.py dimulai <<")
import os
def kenali_wajah(image_bgr, data_path=os.path.join(os.getcwd(), "data.pkl")):
    from keras_facenet import FaceNet
    import pickle, cv2
    from numpy import asarray, expand_dims
    import numpy as np

    model = FaceNet()
    with open(data_path, 'rb') as f:
        database = pickle.load(f)

    HaarCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = HaarCascade.detectMultiScale(image_bgr, 1.1, 4)
    
    # Jika tidak ada wajah terdeteksi
    if len(faces) == 0:
        return ["Wajah tidak terdeteksi"], image_bgr

    identities = []
    threshold = 0.8  # Threshold untuk menentukan apakah wajah dikenali atau tidak
    
    for (x1, y1, w, h) in faces:
        x2, y2 = x1 + w, y1 + h
        face = image_bgr[y1:y2, x1:x2]
        face = cv2.resize(face, (160, 160))
        face = expand_dims(face, axis=0)
        signature = model.embeddings(face)

        min_dist = 100
        identity = 'Tidak dikenali'
        
        for key, value in database.items():
            dist = np.linalg.norm(value - signature)
            if dist < min_dist:
                min_dist = dist
                identity = key

        # Jika jarak terdekat masih di atas threshold, tetap dianggap tidak dikenali
        if min_dist > threshold:
            identity = 'Tidak dikenali'

        # Tambahkan anotasi ke gambar
        cv2.rectangle(image_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image_bgr, identity, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,0), 2)
        
        identities.append(identity)

    return identities, image_bgr