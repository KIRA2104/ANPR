import cv2

rtsp = "rtsp://admin:admin@123@192.168.1.250:554/cam/realmonitor?channel=1&subtype=0"
cap = cv2.VideoCapture(rtsp)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    cv2.imshow("Camera", frame)
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
