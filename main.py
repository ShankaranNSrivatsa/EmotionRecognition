import cv2

def show_webcam():
    cam = cv2.VideoCapture(1)

    if not cam.isOpened():
        print("Webcam couldn't be opened ERROR!")
        return
    while True:
        ret, frame = cam.read()
        if not ret:
            break
        
        cv2.imshow('Live Webcam',frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cam.release()
    cv2.destroyAllWindows()

show_webcam()