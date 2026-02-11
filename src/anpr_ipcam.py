from flask import Flask, Response, jsonify, request, render_template_string
import cv2
import threading
import queue
import time
import os
import numpy as np
from datetime import datetime
from ultralytics import YOLO
import easyocr
import tensorflow as tf

CLASSIFIER_PATH = "src/assets/vehicle_classifier.keras"
vehicle_classifier = tf.keras.models.load_model(CLASSIFIER_PATH)


CLASS_NAMES = ["bus", "car", "motorcycle", "truck"]

# =========================
# CONFIG
# =========================
RTSP_URL = os.environ.get("CAMERA_URL", "http://10.122.112.4:8080/video")  # IP Webcam default
VEHICLE_MODEL_PATH = "src/assets/Vehicle.pt"
PLATE_MODEL_PATH = "src/assets/Plate.pt"
OCR_CONF = 0.30
VEHICLE_CONF = 0.40
PLATE_CONF = 0.40
# =========================
# LOAD MODELS
# =========================
print("Loading models...")
vehicle_model = YOLO(VEHICLE_MODEL_PATH)
plate_model = YOLO(PLATE_MODEL_PATH)
reader = easyocr.Reader(['en'], gpu=False)
print("Models loaded.")
# =========================
# FLASK
# =========================
app = Flask(__name__)

# =========================
# GLOBAL STATE
# =========================
state_lock = threading.Lock()
global_state = {
    "latest_vehicle": "None",
    "latest_plate": "None",
    "recording": False,
    "video_writer": None,
    "video_path": None
}

# =========================
# VIDEO STREAM THREAD
# =========================
class VideoStream:
    def __init__(self, src):
        self.src = src
        self.cap = cv2.VideoCapture(self.src)
        self.q = queue.Queue(maxsize=1)
        self.stopped = False

    def start(self):
        threading.Thread(target=self.update, daemon=True).start()
        return self

    def update(self):
        while not self.stopped:
            if not self.cap.isOpened():
                self.cap.open(self.src)
                time.sleep(1)
                continue

            ret, frame = self.cap.read()
            if not ret:
                continue

            if not self.q.empty():
                try:
                    self.q.get_nowait()
                except:
                    pass

            self.q.put(frame)

    def read(self):
        try:
            return self.q.get_nowait()
        except:
            return None

    def stop(self):
        self.stopped = True
        self.cap.release()


stream = VideoStream(RTSP_URL).start()

# =========================
# INFERENCE FUNCTION
# =========================
def process_frame(frame):
    results_out = []

    try:
        # -------- VEHICLE DETECTION --------
        v_results = vehicle_model(frame, conf=VEHICLE_CONF, verbose=False)[0]

        for box in v_results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls = int(box.cls[0])
            v_type = v_results.names[cls]

            results_out.append({
                "type": "vehicle",
                "coords": [x1, y1, x2, y2],
                "label": v_type,
                "color": [0, 255, 0]
            })

        # -------- PLATE DETECTION FULL FRAME --------
        p_results = plate_model(frame, conf=PLATE_CONF, verbose=False)[0]

        for p in p_results.boxes:
            px1, py1, px2, py2 = map(int, p.xyxy[0])

            results_out.append({
                "type": "plate",
                "coords": [px1, py1, px2, py2],
                "label": None,
                "color": [0, 165, 255]
            })

            crop = frame[py1:py2, px1:px2]
            if crop.size == 0:
                continue

            ocr = reader.readtext(crop)
            if not ocr:
                continue

            best = max(ocr, key=lambda x: x[2])
            text, conf = best[1], best[2]

            if conf > OCR_CONF:
                results_out.append({
                    "type": "text",
                    "coords": [px1, py1 - 10],
                    "label": text,
                    "color": [0, 255, 255]
                })

                with state_lock:
                    global_state["latest_plate"] = text

    except Exception as e:
        print("Inference error:", e)

    return results_out


# =========================
# STREAM GENERATOR
# =========================
def generate_frames():
    while True:
        frame = stream.read()
        if frame is None:
            time.sleep(0.01)
            continue

        results = process_frame(frame)

        for r in results:
            if r["type"] in ["vehicle", "plate"]:
                x1, y1, x2, y2 = r["coords"]
                cv2.rectangle(frame, (x1, y1), (x2, y2), r["color"], 2)

            if r["type"] == "text":
                x, y = r["coords"]
                cv2.putText(frame, r["label"], (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, r["color"], 2)

        small = cv2.resize(frame, (640, 480))
        ret, buffer = cv2.imencode('.jpg', small)

        if not ret:
            continue

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')


# =========================
# ROUTES
# =========================
@app.route('/')
def index():
    return render_template_string("""
    <html>
    <body style='background:black;text-align:center'>
        <h2 style='color:white'>ANPR Live</h2>
        <img src='/video_feed' width='800'>
    </body>
    </html>
    """)


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/api/stats')
def stats():
    with state_lock:
        return jsonify(global_state)


# =========================
# MAIN
# =========================
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5005, debug=False, threaded=True)