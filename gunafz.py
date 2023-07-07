from ultralytics import YOLO
import cv2
import cvzone
import math
import imutils
from imutils.video import VideoStream

cap = cv2.VideoCapture(0)#'rtsp://admin:passw0rd@192.168.1.64:554/Streaming/Channels/1')
cap.set(3, 640)
cap.set(4, 128)

#cap = cv2.VideoCapture("C:/Users/Moh/PycharmProjects/pythonProject/video/GVD.mp4")

model = YOLO("besti.pt")

classNames = ['person', 'Gun' ]

while True:
    success, frame = cap.read()

    result = model(frame, stream=True)

    for r in result:
        boxes = r.boxes
        for box in boxes:
            # bounding box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1

            #confidence

            conf = math.ceil((box.conf[0] * 100)) / 100

            #className

            cls = int(box.cls[0])
            currentClass = classNames[cls]

            if currentClass == "Gun"  and conf > 0.4:
                print(conf)

                cvzone.cornerRect(frame, (x1, y1, w, h), l=9, rt=5, colorR=(255, 0, 0))
                cvzone.putTextRect(frame, f'{conf}{currentClass}', (max(0, x1), max(0, y1)))
                print(conf)

    frame = imutils.resize(frame, width=1200)
    cv2.imshow('AsimCodeCam', frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cv2.destroyAllWindows()
video_stream.stop()
