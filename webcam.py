import cv2
from keras.models import load_model
import numpy as np
model = load_model('Emotional_detection.h5')

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

        # Detect faces in the frame
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        for (x, y, w, h) in faces:
            face_roi = gray[y:y+h, x:x+w]

            face_resized = cv2.resize(face_roi, (48, 48))

            face_normalized = face_resized.astype("float32") / 255.0

            face_input = np.reshape(face_normalized, (1, 48, 48, 1))

            predictions = model.predict(face_input)
            emotion_index = np.argmax(predictions)
            confidence = np.max(predictions)

            emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
            emotion = emotion_labels[emotion_index]

            cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
            cv2.putText(frame, f"{emotion} ({confidence*100:.1f}%)", (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

        cv2.imshow('Live Webcam',frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cam.release()
    cv2.destroyAllWindows()

show_webcam()