import cv2
import os
import threading
import queue
import time
from datetime import datetime
import numpy as np
from ultralytics import YOLO
import easyocr
import tensorflow as tf
from db import MongoLogger
from ocr_utils import PlateTracker, clean_plate
from draw import draw_info_panel


os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
os.environ["YOLO_VERBOSE"] = "False"

RTSP_URL = "rtsp://admin:admin%40123@192.168.1.250:554/cam/realmonitor?channel=1&subtype=0"
DETECT_EVERY_N_FRAMES = 2
OCR_CONF_THRESHOLD = 0.25  
DEBUG_MODE = True
VEHICLE_CONF = 0.25 
PLATE_CONF = 0.25   


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VEHICLE_MODEL = os.path.join(BASE_DIR, "assets", "newvehicle.pt")
PLATE_MODEL = os.path.join(BASE_DIR, "assets", "Plate.pt")  




CLASS_NAMES = ["bus", "car", "motorcycle", "truck"]


print("Loading models...")
if not os.path.exists(VEHICLE_MODEL):
    print(f" ERROR: Vehicle model not found at {VEHICLE_MODEL}")
    exit(1)
if not os.path.exists(PLATE_MODEL):
    print(f" ERROR: Plate model not found at {PLATE_MODEL}")
    exit(1)
    
vehicle_model = YOLO(VEHICLE_MODEL)
plate_model = YOLO(PLATE_MODEL)
reader = easyocr.Reader(['en'], gpu=False)
print(f" Models loaded successfully!")
print(f"   Vehicle model: {VEHICLE_MODEL}")
print(f"   Plate model: {PLATE_MODEL}")

db = MongoLogger()
tracker = PlateTracker(cooldown=60)

class VideoStream:
    def __init__(self, src):
        self.src = src
        self.cap = None
        self.queue = queue.Queue(maxsize=1) 
        self.stopped = False
        self._connect()

    def _connect(self):
        if self.cap:
            self.cap.release()
        self.cap = cv2.VideoCapture(self.src, cv2.CAP_FFMPEG)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 20.0
        if self.fps <= 0:
            self.fps = 20.0

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
            if not ret or frame is None:
                continue

            if not self.queue.empty():
                try:
                    self.queue.get_nowait()
                except queue.Empty:
                    pass
            self.queue.put(frame)

    def read(self):
        try:
            return self.queue.get_nowait()
        except queue.Empty:
            
            time.sleep(0.01)
            return None

    def get_fps(self):
        """Get the camera FPS"""
        return self.fps if hasattr(self, 'fps') else 20.0

    def stop(self):
        self.stopped = True
        if self.cap:
            self.cap.release()


def classify_vehicle(crop):
    try:
        img = cv2.resize(crop, (224, 224))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img / 255.0
        img = np.expand_dims(img, axis=0)

        pred = vehicle_classifier.predict(img, verbose=0)
        idx = int(np.argmax(pred))
        conf = float(pred[0][idx])

        return f"{CLASS_NAMES[idx]} {conf:.2f}"
    except Exception as e:
        if DEBUG_MODE:
            print("Classifier error:", e)
        return "unknown"


def process_vehicle_detection(frame, results, db_instance, tracker_instance, last_ocr_plate_ref, last_vehicle_logged_ref):
    """
    Process vehicle detection and extract/recognize plates.
    Returns: (vehicle_type, plate, last_ocr_plate, last_box)
    """
    last_box = None
    current_v_type = "Scanning..."
    current_plate = "Waiting..."
    last_ocr_plate = last_ocr_plate_ref
    last_vehicle_logged = last_vehicle_logged_ref

    if results and len(results.boxes) > 0:
        
        box = max(results.boxes, key=lambda b: float(b.conf[0]))
        last_box = tuple(map(int, box.xyxy[0]))
        current_v_type = "unknown"
        
        vehicle_key = (current_v_type, tuple(last_box))
        if DEBUG_MODE and vehicle_key != last_vehicle_logged:
            print(f"🚗 Vehicle detected: {current_v_type} (conf: {box.conf[0]:.2f})")
            last_vehicle_logged = vehicle_key

        x1, y1, x2, y2 = last_box
       
        if x2 > x1 and y2 > y1:
           
            h, w = frame.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
           
            if x2 > x1 and y2 > y1:
                v_crop = frame[y1:y2, x1:x2]
                current_v_type = classify_vehicle(v_crop)
            else:
                if DEBUG_MODE:
                    print(" Invalid vehicle crop coordinates after clamping")
                v_crop = None
            
            try:
                if v_crop is None:
                    raise ValueError("Empty vehicle crop")

                p_results = plate_model(v_crop, conf=PLATE_CONF, verbose=False)[0]

                if DEBUG_MODE:
                    print(f"   Found {len(p_results.boxes)} plate(s) in vehicle")

              
                for p in p_results.boxes:
                    px1, py1, px2, py2 = map(int, p.xyxy[0])
                    
                  
                    h, w = v_crop.shape[:2]
                    px1, py1 = max(0, px1), max(0, py1)
                    px2, py2 = min(w, px2), min(h, py2)
                    
                    
                    if px2 > px1 and py2 > py1:
                        p_img = v_crop[py1:py2, px1:px2]

                       
                        if DEBUG_MODE and p_img.size > 0:
                            scaled_plate = cv2.resize(p_img, (300, 100))
                            cv2.imshow("OCR_CROP", scaled_plate)

                        try:
                            ocr = reader.readtext(p_img)
                            
                            if ocr and len(ocr) > 0:
                                
                                best = max(ocr, key=lambda x: x[2])
                                text, conf = best[1], best[2]
                                
                                if DEBUG_MODE:
                                    print(f" OCR Raw: '{text}' | Confidence: {conf:.2f}")
                                
                                plate = clean_plate(text)
                                
                                if DEBUG_MODE and plate:
                                    print(f" Cleaned Plate: {plate}")
                                
        
                                if plate and conf >= OCR_CONF_THRESHOLD:
                                    if plate != last_ocr_plate:
                                        current_plate = plate
                                        last_ocr_plate = plate
                                        print(f" NEW PLATE: {plate} (conf: {conf:.2f})")
                                        
                                        if tracker_instance.should_log(plate):
                                            print(f" Saving to DB: {plate}")
                                            
                                            db_instance.save_vehicle(current_v_type, plate, conf)
                                elif DEBUG_MODE and plate:
                                    print(f" Plate '{plate}' rejected - Confidence {conf:.2f} < {OCR_CONF_THRESHOLD}")
                            elif DEBUG_MODE:
                                print(f"   No OCR results for this crop")
                                
                        except Exception as ocr_err:
                            if DEBUG_MODE:
                                print(f" OCR error: {ocr_err}")
                    else:
                        if DEBUG_MODE:
                            print(f" Invalid plate region after clamping")
                            
            except Exception as plate_det_err:
                if DEBUG_MODE:
                    print(f" Plate detection error: {plate_det_err}")
        else:
            if DEBUG_MODE:
                print(f" Invalid vehicle crop coordinates")

    if not results or len(results.boxes) == 0:
        last_vehicle_logged = None

    return current_v_type, current_plate, last_ocr_plate, last_box, last_vehicle_logged


def main():
    recording = False
    video_writer = None
    video_path = None
    camera_fps = 20.0

    stream = VideoStream(RTSP_URL).start()
    time.sleep(2)

    frame_id = 0
    current_v_type = "Scanning..."
    current_plate = "Waiting..."
    last_ocr_plate = None
    last_box = None
    last_vehicle_logged = None

    print(" ANPR Started - HD Stream")
    print(" r = record | 1 = save | 2 = discard | q = quit")
    print(" t = test plate detection (mock) | Debug mode: ON")

    try:
        while True:
            frame = stream.read()
            if frame is None:
                continue

            frame = cv2.resize(frame, (1280, 720), interpolation=cv2.INTER_CUBIC)
            frame_id += 1
            if frame_id % DETECT_EVERY_N_FRAMES == 0:
                try:
                    results = vehicle_model(frame, conf=VEHICLE_CONF, verbose=False)[0]
                    current_v_type, current_plate, last_ocr_plate, last_box, last_vehicle_logged = process_vehicle_detection(
                        frame, results, db, tracker, last_ocr_plate, last_vehicle_logged
                    )
                except Exception as detection_err:
                    print(f" Detection error: {detection_err}")
                    last_box = None
                    current_v_type = "Scanning..."


            if last_box:
                lx1, ly1, lx2, ly2 = last_box
                cv2.rectangle(frame, (lx1, ly1), (lx2, ly2), (0, 255, 0), 2)
                cv2.putText(
                    frame, current_v_type, (lx1, ly1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
                )


            draw_info_panel(frame, vehicle_type=current_v_type, plate=current_plate)

            if recording and video_writer:
                video_writer.write(frame)

            cv2.imshow("ANPR Live", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('t'):
                test_plate = "MH12AB1234"
                current_plate = test_plate
                current_v_type = "Test-Car"
                print(f"🧪 [TEST MODE] Simulating detection: {test_plate}")
                try:
                    db.save_vehicle(current_v_type, test_plate, 0.99)
                except Exception as e:
                    print(f"⚠️ Test mode DB error: {e}")

            if key == ord('r') and not recording:
                os.makedirs("recordings", exist_ok=True)
                video_path = f"recordings/{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
                h, w = frame.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                camera_fps = stream.get_fps()
                video_writer = cv2.VideoWriter(video_path, fourcc, camera_fps, (w, h))
                recording = True
                print(f"Recording started at {camera_fps:.1f} FPS")

            elif key == ord('1') and recording:
                print("Stopping & saving")
                if video_writer:
                    video_writer.release()
                video_writer = None
                recording = False
                try:
                    db.save_video(video_path, reason="manual_save")
                except Exception as e:
                    print(f" Video save error: {e}")

            # STOP + DISCARD
            elif key == ord('2') and recording:
                print(" Stopping & discarding")
                if video_writer:
                    video_writer.release()
                video_writer = None
                recording = False
                if video_path and os.path.exists(video_path):
                    os.remove(video_path)

            elif key == ord('q'):
                break

    finally:
      
        if video_writer:
            video_writer.release()
        stream.stop()
        cv2.destroyAllWindows()
        try:
            cv2.destroyWindow("OCR_CROP")
        except cv2.error:
            pass
        stats = tracker.get_stats()
        print(f"\n Session Stats: {stats}")

if __name__ == "__main__":
    main()