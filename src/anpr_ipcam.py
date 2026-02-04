from flask import Flask, render_template, Response
import cv2
from ultralytics import YOLO
import easyocr
import re

app = Flask(__name__)

# =====================
# LOAD MODELS
# =====================
vehicle_model = YOLO("src/assets/Vehicle.pt")
plate_model = YOLO("src/assets/Plate.pt")
reader = easyocr.Reader(['en'], gpu=False)

camera = cv2.VideoCapture(0)  # laptop webcam OR DroidCam

def normalize_plate(text):
    text = text.upper().replace(" ", "")
    text = text.replace("O", "0").replace("I", "1").replace("L", "1")
    match = re.search(r"[A-Z]{2}[0-9]{2}[A-Z]{1,2}[0-9]{3,4}", text)
    return match.group(0) if match else text

def generate_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break

        # Vehicle detection
        v_results = vehicle_model(frame, conf=0.4, verbose=False)[0]

        for vbox in v_results.boxes:
            x1, y1, x2, y2 = map(int, vbox.xyxy[0])
            vehicle_crop = frame[y1:y2, x1:x2]

            # Plate detection
            p_results = plate_model(vehicle_crop, conf=0.4, verbose=False)[0]

            for pbox in p_results.boxes:
                px1, py1, px2, py2 = map(int, pbox.xyxy[0])
                plate_crop = vehicle_crop[py1:py2, px1:px2]

                gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
                gray = cv2.resize(gray, None, fx=2, fy=2)
                _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

                ocr = reader.readtext(thresh, allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
                text = "".join([r[1] for r in ocr])
                plate = normalize_plate(text)

                cv2.rectangle(frame,
                              (x1+px1, y1+py1),
                              (x1+px2, y1+py2),
                              (0,255,0), 2)

                cv2.putText(frame, plate,
                            (x1+px1, y1+py1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)
