import cv2
import os
import asyncio
import threading
import queue
import time
from datetime import datetime
from ultralytics import YOLO
import easyocr

from db import MongoLogger
from ocr_utils import PlateTracker, clean_plate
from draw import draw_info_panel

# =========================
# ENV + RTSP CONFIG
# =========================
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
os.environ["YOLO_VERBOSE"] = "False"

# FIX: Switched to subtype=0 for HD resolution
RTSP_URL = "rtsp://admin:admin%40123@192.168.1.250:554/cam/realmonitor?channel=1&subtype=0"
DETECT_EVERY_N_FRAMES = 5
OCR_CONF_THRESHOLD = 0.6

# =========================
# ABSOLUTE MODEL PATHS
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VEHICLE_MODEL = os.path.join(BASE_DIR, "assets", "Vehicle.pt")
PLATE_MODEL = os.path.join(BASE_DIR, "assets", "Plate.pt")

# =========================
# LOAD MODELS
# =========================
print("Loading models...")
vehicle_model = YOLO(VEHICLE_MODEL)
plate_model = YOLO(PLATE_MODEL)
reader = easyocr.Reader(['en'], gpu=False)
print("Models loaded successfully!")

db = MongoLogger()
tracker = PlateTracker(cooldown=60)

# =========================
# LOW-LATENCY VIDEO STREAM
# =========================
class VideoStream:
    def __init__(self, src):
        self.src = src
        self.cap = None
        self.queue = queue.Queue(maxsize=1) # maxsize=1 prevents lag buildup
        self.stopped = False
        self._connect()

    def _connect(self):
        if self.cap: self.cap.release()
        self.cap = cv2.VideoCapture(self.src, cv2.CAP_FFMPEG)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    def start(self):
        threading.Thread(target=self._update, daemon=True).start()
        return self

    def _update(self):
        while not self.stopped:
            if not self.cap or not self.cap.isOpened():
                self._connect()
                time.sleep(1)
                continue
            ret, frame = self.cap.read()
            if not ret or frame is None: continue
            
            if not self.queue.empty():
                try: self.queue.get_nowait()
                except queue.Empty: pass
            self.queue.put(frame)

    def read(self):
        try: return self.queue.get_nowait()
        except queue.Empty: return None

    def stop(self):
        self.stopped = True
        if self.cap: self.cap.release()

# =========================
# MAIN LOOP
# =========================
async def main():
    stream = VideoStream(RTSP_URL).start()
    await asyncio.sleep(2)
    
    frame_id = 0
    current_v_type = "Scanning..."
    current_plate = "Waiting..."
    last_ocr_plate = None
    last_box = None

    print("🚗 ANPR Started - HD Stream")

    try:
        while True:
            frame = stream.read()
            if frame is None: continue

            frame = cv2.resize(frame, (1280, 720), interpolation=cv2.INTER_CUBIC)
            frame_id += 1

            # Step A: DETECTION (THROTTLED)
            if frame_id % DETECT_EVERY_N_FRAMES == 0:
                results = vehicle_model(frame, conf=0.4, verbose=False)[0]

                if results and len(results.boxes) > 0:
                    box = results.boxes[0]
                    last_box = tuple(map(int, box.xyxy[0]))
                    current_v_type = results.names[int(box.cls[0])]
                    
                    x1, y1, x2, y2 = last_box
                    v_crop = frame[y1:y2, x1:x2]
                    p_results = plate_model(v_crop, conf=0.5, verbose=False)[0]

                    for p in p_results.boxes:
                        px1, py1, px2, py2 = map(int, p.xyxy[0])
                        p_img = v_crop[py1:py2, px1:px2]

                        ocr = reader.readtext(p_img)
                        if ocr:
                            text, conf = ocr[0][1], ocr[0][2]
                            plate = clean_plate(text)
                            if plate and conf >= OCR_CONF_THRESHOLD:
                                if plate != last_ocr_plate:
                                    current_plate = plate
                                    last_ocr_plate = plate
                                    if tracker.should_log(plate):
                                        asyncio.create_task(db.save_vehicle(current_v_type, plate, conf))
                else:
                    last_box = None
                    current_v_type = "Scanning..."

            # Step B: ALWAYS Draw the box if it exists
            if last_box:
                lx1, ly1, lx2, ly2 = last_box
                cv2.rectangle(frame, (lx1, ly1), (lx2, ly2), (0, 255, 0), 2)
                cv2.putText(frame, current_v_type, (lx1, ly1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Step C: UI Panel
            draw_info_panel(frame, vehicle_type=current_v_type, plate=current_plate)

            cv2.imshow("ANPR Live", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        stream.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    asyncio.run(main())