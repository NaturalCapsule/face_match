from deepface import DeepFace
import cv2
import threading

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

count = 0

face_match = False

reference_img = cv2.imread('PUT_YOUR_IMAGE')

def check_face(frame):
    global face_match
    try:
        if DeepFace.verify(frame, reference_img.copy())['verified']:
            face_match = True
        else:
            face_match = False

    except ValueError:
        pass

while True:
    ret, frame = cap.read()

    if ret:
        if count % 60 == 0:
            try:
                threading.Thread(target = check_face(frame), args = (frame.copy(),)).start()
            except ValueError:
                pass
        count += 1

        if face_match:
            cv2.putText(frame, "MATCH", (20, 450), cv2.FONT_HERSHEY_TRIPLEX, 2, (0, 255, 0), 3)
        else:
            cv2.putText(frame, "NO MATCH", (20, 450), cv2.FONT_HERSHEY_TRIPLEX, 2, (0, 0, 255), 3)

    cv2.imshow('Face Match', frame)

    if cv2.waitKey(0) == ord('q'):
        break

cv2.destroyAllWindows()