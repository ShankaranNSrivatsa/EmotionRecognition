import cv2
from keras.models import load_model
import numpy as np
#model = load_model('Emotional_detection.h5')

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def show_webcam():
    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        print("Webcam couldn't be opened ERROR!")
        return
    while True:
        ret, frame = cam.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face = cv2.resize(gray, (48, 48))
        face = face.astype("float32") / 255.0
        face = np.reshape(face, (1, 48, 48, 1))

        cv2.imshow('Live Webcam',frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cam.release()
    cv2.destroyAllWindows()

show_webcam()