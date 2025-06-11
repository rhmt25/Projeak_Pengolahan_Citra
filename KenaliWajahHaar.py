print(">> Script KenaliWajahHaar.py dimulai <<")
import os
# dalam file KenaliWajahHaar.py
def kenali_wajah(image_bgr, data_path = os.path.join(os.getcwd(), "data.pkl")):
    from keras_facenet import FaceNet
    import pickle, cv2
    from numpy import asarray, expand_dims
    from PIL import Image
    import numpy as np

    model = FaceNet()
    with open(data_path, 'rb') as f:
        database = pickle.load(f)

    HaarCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    wajah = HaarCascade.detectMultiScale(image_bgr, 1.1, 4)
    identity = 'Tidak dikenali'

    if len(wajah) > 0:
        x1, y1, w, h = wajah[0]
        x2, y2 = x1 + w, y1 + h
        face = image_bgr[y1:y2, x1:x2]
        face = cv2.resize(face, (160, 160))
        face = expand_dims(face, axis=0)
        signature = model.embeddings(face)

        min_dist = 100
        for key, value in database.items():
            dist = np.linalg.norm(value - signature)
            if dist < min_dist:
                min_dist = dist
                identity = key

        # Tambahkan anotasi
        cv2.rectangle(image_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image_bgr, identity, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2)

    return identity, image_bgr
