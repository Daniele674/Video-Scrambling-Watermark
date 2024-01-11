#Detection del volto con OpenCV e scrambling della ROI con scramblery

import cv2
from scramblery import scramblery

scramble_settings = {
    'splits': 25,
    'type': 'pixel',
    'bg': True,
    'seed': None,
    'write': False  # Should always be False for video processing
}

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

video_path = 'testSet/Human_safari.mp4'

cap = cv2.VideoCapture(video_path)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # Estrai il volto
        face = frame[y:y+h, x:x+w]

        # Applica lo scrambling solo alla regione del volto
        scrambled_face = scramblery.scrambleface(face, splits=10, type='pixel', seamless=False, bg=True, seed=None, write=False)

        # Sostituisci il volto scramblato nell'immagine originale
        frame[y:y+h, x:x+w] = scrambled_face

    cv2.imshow('Face Scrambling', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()




"""
------------------------- SCRAMBLING DIRETTAMENTE CON OPENCV ----------------------------

import cv2


face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

video_path = '/home/vboxuser/Desktop/Compressione/Progetto_Compressione(Python)/Progetto_Compressione/Human_safari_cps_.mp4'
# video_path = '/home/vboxuser/Desktop/Compressione/Progetto_Compressione(Python)/Progetto_Compressione/testSet/sample-mpg-file.mpg'

cap = cv2.VideoCapture(video_path)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]

        small_face = cv2.resize(face, (0, 0), fx=0.05, fy=0.05)
        pixelated_face = cv2.resize(small_face, (w, h), interpolation=cv2.INTER_NEAREST)
        frame[y:y+h, x:x+w] = pixelated_face

    cv2.imshow('Pixelated Face', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

"""