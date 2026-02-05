from flask import Flask, render_template, Response, jsonify, request
import cv2
import threading
import queue
import time
import yaml
import os
import json
from datetime import datetime
from ultralytics import YOLO
import easyocr
 
from db import MongoLogger
from ocr_utils import PlateTracker, clean_plate

app = Flask(__name__)

# =========================
# CONFIGURATION
# =========================
def load_config():
    possible_paths = ["config.yaml", "../config.yaml"]
    for path in possible_paths:
        if os.path.exists(path):
            print(f"Loading configuration from {path}...")
            with open(path, "r") as f:
                return yaml.safe_load(f)
    print("⚠️ Config file not found, using defaults.")
    return {}

CONFIG = load_config()

# Camera & Inference Settings
RTSP_URL = CONFIG.get("camera", {}).get("url", 0)  # Default to webcam if no URL
DETECT_EVERY_N_FRAMES = 2
OCR_CONF_THRESHOLD = 0.25
VEHICLE_CONF = CONFIG.get("models", {}).get("vehicle_conf", 0.25)
PLATE_CONF = CONFIG.get("models", {}).get("plate_conf", 0.25)

# Models
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
vehicle_filename = os.path.basename(CONFIG.get("models", {}).get("vehicle_path", "Vehicle.pt"))
plate_filename = os.path.basename(CONFIG.get("models", {}).get("plate_path", "Plate.pt"))

VEHICLE_MODEL_PATH = os.path.join(BASE_DIR, "assets", vehicle_filename)
PLATE_MODEL_PATH = os.path.join(BASE_DIR, "assets", plate_filename)

print(f"Loading Vehicle Model: {VEHICLE_MODEL_PATH}")
vehicle_model = YOLO(VEHICLE_MODEL_PATH)
print(f"Loading Plate Model: {PLATE_MODEL_PATH}")
plate_model = YOLO(PLATE_MODEL_PATH)
reader = easyocr.Reader(CONFIG.get("ocr", {}).get("lang", ['en']), gpu=False)

# Database
mongo_uri = CONFIG.get("mongodb", {}).get("uri", "mongodb://localhost:27017")
if "${MONGODB_URI}" in mongo_uri:
    mongo_uri = os.environ.get("MONGODB_URI", "mongodb://localhost:27017")
db = MongoLogger(uri=mongo_uri, db=CONFIG.get("mongodb", {}).get("database", "traffic"))
tracker = PlateTracker(cooldown=300)

# Global State
global_state = {
    "latest_vehicle": "None",
    "latest_plate": "None",
    "total_vehicles": 0,
    "total_plates": 0,
    "unique_plates": 0,
    "last_n_logs": [],
    "is_recording": False,
    "video_path": None,
    "video_writer": None
}
state_lock = threading.Lock()

# =========================
# VIDEO STREAMING CLASS
# =========================
class VideoStream:
    def __init__(self, src):
        self.src = src
        self.cap = cv2.VideoCapture(self.src)
        self.queue = queue.Queue(maxsize=2)
        self.stopped = False
        self.lock = threading.Lock()
        
    def start(self):
        t = threading.Thread(target=self.update, args=())
        t.daemon = True
        t.start()
        return self
        
    def update(self):
        while not self.stopped:
            if not self.cap.isOpened():
                time.sleep(1)
                self.cap.open(self.src)
                continue
                
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.1)
                continue
            
            with self.lock:
                if self.queue.full():
                    try:
                        self.queue.get_nowait()
                    except queue.Empty:
                        pass
                self.queue.put(frame)
                
    def read(self):
        with self.lock:
            return self.queue.get() if not self.queue.empty() else None
            
    def stop(self):
        self.stopped = True
        self.cap.release()

# Start Stream
# Use webcam (0) if config url is placeholder or failover
camera_url = RTSP_URL
if "YOUR_CAMERA_URL_HERE" in str(camera_url):
    print("⚠️ Using webcam fallback due to placeholder URL")
    camera_url = 0

stream = VideoStream(camera_url).start()

# =========================
# ASYNC INFERENCE WORKER
# =========================
class InferenceWorker:
    def __init__(self, vehicle_model, plate_model, reader, tracker, db):
        self.vehicle_model = vehicle_model
        self.plate_model = plate_model
        self.reader = reader
        self.tracker = tracker
        self.db = db
        self.stopped = False
        self.latest_frame = None
        self.frame_lock = threading.Lock()
        self.latest_results = []  # List of dicts: {box, label, color, type}
        self.results_lock = threading.Lock()
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def update_frame(self, frame):
        with self.frame_lock:
            self.latest_frame = frame.copy()

    def get_results(self):
        with self.results_lock:
            return list(self.latest_results)

    def _loop(self):
        while not self.stopped:
            frame = None
            with self.frame_lock:
                if self.latest_frame is not None:
                    frame = self.latest_frame.copy()
            
            if frame is None:
                time.sleep(0.01)
                continue

            # Run Inference
            results_to_store = self.process_image(frame)

            # Update shared results (for the server-side stream)
            with self.results_lock:
                self.latest_results = results_to_store
            
            # Small sleep to yield CPU
            time.sleep(0.001)

    def process_image(self, frame):
        """Run detection on a single frame and return results list."""
        results_to_store = []
        try:
            # 1. Vehicle Detection
            # Increase threshold to avoid "ghost" vehicles (e.g. wall patterns)
            effective_conf = max(VEHICLE_CONF, 0.45)
            results = self.vehicle_model(frame, conf=effective_conf, verbose=False)[0]
            
            h, w = frame.shape[:2]
            
            for box in v_results := results.boxes:
                vx1, vy1, vx2, vy2 = map(int, box.xyxy[0])
                v_conf = float(box.conf[0])
                cls = int(box.cls[0])
                v_type = results.names[cls]
                
                # Store Vehicle Box
                results_to_store.append({
                    "type": "vehicle",
                    "coords": [vx1, vy1, vx2, vy2],
                    "label": f"{v_type} {v_conf:.2f}",
                    "color": [0, 255, 0]
                })
                
                # 2. Plate Detection
                vx1, vy1 = max(0, vx1), max(0, vy1)
                vx2, vy2 = min(w, vx2), min(h, vy2)
                
                if vx2 > vx1 and vy2 > vy1:
                    v_crop = frame[vy1:vy2, vx1:vx2]
                    p_results = self.plate_model(v_crop, conf=PLATE_CONF, verbose=False)[0]
                    
                    for p in p_results.boxes:
                        px1, py1, px2, py2 = map(int, p.xyxy[0])
                        
                        # Global coords
                        gx1, gy1 = vx1 + px1, vy1 + py1
                        gx2, gy2 = vx1 + px2, vy1 + py2
                        
                        results_to_store.append({
                            "type": "plate_box",
                            "coords": [gx1, gy1, gx2, gy2],
                            "label": None,
                            "color": [0, 165, 255]
                        })
                        
                        # 3. OCR
                        ph, pw = v_crop.shape[:2]
                        px1, py1 = max(0, px1), max(0, py1)
                        px2, py2 = min(pw, px2), min(ph, py2)
                        
                        if px2 > px1 and py2 > py1:
                            p_crop = v_crop[py1:py2, px1:px2]
                            ocr = self.reader.readtext(p_crop)
                            if ocr:
                                best = max(ocr, key=lambda x: x[2])
                                text, conf = best[1], best[2]
                                
                                if conf >= OCR_CONF_THRESHOLD and len(text) > 4:
                                    cleaned = clean_plate(text)
                                    if cleaned:
                                        # Store OCR Result
                                        results_to_store.append({
                                            "type": "text",
                                            "coords": [gx1, gy1 - 5],
                                            "label": cleaned,
                                            "color": [0, 255, 255]
                                        })
                                        
                                        # LOGGING LOGIC
                                        with state_lock:
                                            global_state["latest_vehicle"] = v_type
                                            global_state["latest_plate"] = cleaned
                                            
                                            if self.tracker.should_log(cleaned):
                                                self.db.save_vehicle(v_type, cleaned, float(conf))
                                                global_state["unique_plates"] += 1
                                                global_state["total_plates"] += 1
                                                global_state["total_vehicles"] = global_state["unique_plates"]
                                                
                                                log_entry = {
                                                    "timestamp": datetime.now().strftime("%H:%M:%S"),
                                                    "plate": cleaned,
                                                    "type": v_type,
                                                    "conf": f"{conf:.2f}"
                                                }
                                                global_state["last_n_logs"].insert(0, log_entry)
                                                global_state["last_n_logs"] = global_state["last_n_logs"][:10]
        except Exception as e:
            print(f"Inference Error: {e}")
            
        return results_to_store

# Start Inference Worker
inference_worker = InferenceWorker(vehicle_model, plate_model, reader, tracker, db)
import numpy as np

# ... (VideoStream and generate_frames kept same) ...

# =========================
# FLASK ROUTES
# =========================
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/detect_frame', methods=['POST'])
def detect_frame():
    """Receives a frame from client, runs detection, returns JSON results."""
    file = request.files.get('frame')
    if not file:
        return jsonify({"error": "No frame provided"}), 400
        
    # Decode image
    npimg = np.frombuffer(file.read(), np.uint8)
    frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    
    if frame is None:
        return jsonify({"error": "Invalid image"}), 400
        
    # Run processing
    results = inference_worker.process_image(frame)
    
    return jsonify({"results": results})

@app.route('/api/stats')
def get_stats():
    with state_lock:
        return jsonify({
            "latest_vehicle": global_state["latest_vehicle"],
            "latest_plate": global_state["latest_plate"],
            "total_vehicles": global_state["total_vehicles"],
            "unique_plates": global_state["unique_plates"],
            "logs": global_state["last_n_logs"],
            "recording": global_state["is_recording"]
        })

@app.route('/api/record', methods=['POST'])
def toggle_record():
    with state_lock:
        if not global_state["is_recording"]:
            # Start Recording
            os.makedirs("recordings", exist_ok=True)
            path = f"recordings/REC_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            # Assuming 720p roughly, ideally get from frame
            global_state["video_writer"] = cv2.VideoWriter(path, fourcc, 20.0, (640, 480)) # Default resolution, might adjust
            global_state["video_path"] = path
            global_state["is_recording"] = True
            return jsonify({"status": "started", "path": path})
        else:
            # Stop Recording
            if global_state["video_writer"]:
                global_state["video_writer"].release()
                global_state["video_writer"] = None
            global_state["is_recording"] = False
            
            # Save to DB
            if global_state["video_path"]:
                db.save_video(global_state["video_path"], "manual_web_recording")
                
            return jsonify({"status": "stopped"})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5005, debug=True, threaded=True)
