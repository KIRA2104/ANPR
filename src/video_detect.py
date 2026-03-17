import cv2
from ultralytics import YOLO
import easyocr
import numpy as np
import re
import os
import json
from datetime import datetime
from contextlib import suppress
from difflib import get_close_matches
from db import MongoLogger


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)

PLATE_MODEL_PATH = os.path.join(BASE_DIR, "assets", "Plate.pt")
VEHICLE_MODEL_PATH = os.path.join(BASE_DIR, "assets", "Vehicle.pt")
VIDEO_OUTPUT_DIR = os.path.join(BASE_DIR, "output")
LOG_FILE_PATH = os.path.join(ROOT_DIR, "video_detections_log.jsonl")

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".m4v"}

if not os.path.exists(PLATE_MODEL_PATH):
    raise FileNotFoundError(f"Plate model not found at: {PLATE_MODEL_PATH}")
if not os.path.exists(VEHICLE_MODEL_PATH):
    raise FileNotFoundError(f"Vehicle model not found at: {VEHICLE_MODEL_PATH}")

print(f"Loading plate model from: {PLATE_MODEL_PATH}")
plate_model = YOLO(PLATE_MODEL_PATH)

print(f"Loading vehicle model from: {VEHICLE_MODEL_PATH}")
vehicle_model = YOLO(VEHICLE_MODEL_PATH)

print("Initializing EasyOCR reader...")
reader = easyocr.Reader(['en'], gpu=False)

# ===============================
# CONSTANTS
# ===============================
DETECTION_CONF = 0.25
PROCESS_EVERY_N_FRAMES = 2
MIN_PLATE_CONFIDENCE = 0.30
IMAGE_DETECTION_CONF = 0.15
IMAGE_MIN_PLATE_CONFIDENCE = 0.20
CONFIRMATION_THRESHOLD = 3      # Must see same plate N times before logging
MIN_FINAL_CONFIDENCE = 0.25
COOLDOWN_FRAMES = 100
VEHICLE_STALE_FRAMES = 20       # Frames after which a vehicle is considered gone
DEBUG_OCR = False

# ===============================
# ALL VALID INDIAN STATE/UT CODES
# ===============================
VALID_STATE_CODES = {
    "AN", "AP", "AR", "AS", "BR", "CG", "CH", "DD", "DL", "DN",
    "GA", "GJ", "HP", "HR", "JH", "JK", "KA", "KL", "LA", "LD",
    "MH", "ML", "MN", "MP", "MZ", "NL", "OD", "PB", "PY", "RJ",
    "SK", "TN", "TR", "TS", "UK", "UP", "WB"
}

def correct_state_code(code: str) -> str:
    """
    If the first 2 characters form a valid state code, return as-is.
    Otherwise, find the closest valid state code using difflib.
    Returns empty string if no close match found (cutoff 0.6).
    """
    if code in VALID_STATE_CODES:
        return code
    matches = get_close_matches(code, VALID_STATE_CODES, n=1, cutoff=0.6)
    if matches:
        print(f"   [State Correction] '{code}' → '{matches[0]}'")
        return matches[0]
    return ""  # Reject completely unrecognisable state codes

# ===============================
# LOGGING FUNCTION
# ===============================
def log_plate_detection(vehicle_type, plate_text, confidence,
                        video_path, frame_number, timestamp, raw_ocr_text=""):
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "video_file": os.path.basename(video_path),
        "frame_number": frame_number,
        "video_timestamp": timestamp,
        "vehicle_type": vehicle_type,
        "plate_number": plate_text,
        "ocr_confidence": confidence,
        "raw_ocr_text": raw_ocr_text
    }
    try:
        with open(LOG_FILE_PATH, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
    except Exception as e:
        print(f"⚠️ File logging error: {e}")

# ===============================
# PLATE CLEANING
# ===============================
def clean_plate_robust(text: str) -> str:
    """
    Clean and validate Indian license plate text.
    Format: XX 00 XX 0000  (e.g. MH12DE1433)
    Also validates/corrects the state code against known Indian RTO codes.
    """
    if not text:
        return ""

    text = text.upper()
    text = re.sub(r"[^A-Z0-9]", "", text)

    if len(text) < 8 or len(text) > 10:
        return ""

    dict_char_to_int = {
        'O': '0', 'I': '1', 'J': '3', 'A': '4', 'G': '6',
        'S': '5', 'Z': '2', 'B': '8', 'T': '7', 'Q': '0', 'D': '0'
    }
    dict_int_to_char = {
        '0': 'O', '1': 'I', '2': 'Z', '3': 'J', '4': 'A',
        '6': 'G', '5': 'S', '8': 'B', '7': 'T'
    }

    text_list = list(text)

    # Positions 0-1: state code — must be letters
    for i in [0, 1]:
        if text_list[i] in dict_int_to_char:
            text_list[i] = dict_int_to_char[text_list[i]]

    # Positions 2-3: district code — must be digits
    for i in [2, 3]:
        if text_list[i] in dict_char_to_int:
            text_list[i] = dict_char_to_int[text_list[i]]

    # Last 4 chars: serial digits
    suffix_start = len(text_list) - 4
    for i in range(suffix_start, len(text_list)):
        if text_list[i] in dict_char_to_int:
            text_list[i] = dict_char_to_int[text_list[i]]

    result = "".join(text_list)

    # Basic format check
    if not re.match(r"^[A-Z]{2}[0-9]{2}[A-Z]{0,3}[0-9]{3,4}$", result):
        return ""

    # --- STATE CODE VALIDATION & CORRECTION ---
    raw_state = result[:2]
    corrected_state = correct_state_code(raw_state)
    if not corrected_state:
        if DEBUG_OCR:
            print(f"   [Rejected] Unknown state code '{raw_state}' in '{result}'")
        return ""

    result = corrected_state + result[2:]
    return result


def extract_best_plate_from_ocr(ocr_result):
    """
    Pick best valid plate candidate from OCR output.
    Strategy:
    1) Try each OCR segment independently
    2) Try concatenated OCR text as fallback
    Returns (plate_text, confidence, raw_text)
    """
    if not ocr_result:
        return "", 0.0, ""

    best_plate = ""
    best_conf = 0.0
    best_raw = ""

    for item in ocr_result:
        if len(item) < 3:
            continue
        raw_text = str(item[1])
        conf = float(item[2])
        cleaned = clean_plate_robust(raw_text)
        if cleaned and conf > best_conf:
            best_plate = cleaned
            best_conf = conf
            best_raw = raw_text

    if best_plate:
        return best_plate, best_conf, best_raw

    combined_raw = "".join([str(r[1]) for r in ocr_result if len(r) >= 2])
    combined_plate = clean_plate_robust(combined_raw)
    combined_conf = max([float(r[2]) for r in ocr_result if len(r) >= 3], default=0.0)
    if combined_plate:
        return combined_plate, combined_conf, combined_raw

    return "", 0.0, combined_raw

# ===============================
# DRAW DETECTIONS
# ===============================
def draw_detection(frame, x1, y1, x2, y2, plate_text, confidence, vehicle_type):
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    label = f"{plate_text} ({confidence:.2f})"
    vehicle_label = f"Vehicle: {vehicle_type}"
    (w1, h1), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    (w2, h2), _ = cv2.getTextSize(vehicle_label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(frame, (x1, y1 - h1 - h2 - 10), (x1 + max(w1, w2) + 10, y1), (0, 255, 0), -1)
    cv2.putText(frame, label, (x1 + 5, y1 - h2 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    cv2.putText(frame, vehicle_label, (x1 + 5, y1 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    return frame


def detect_media_type(file_path: str) -> str:
    ext = os.path.splitext(file_path)[1].lower()
    if ext in VIDEO_EXTENSIONS:
        return "video"
    if ext in IMAGE_EXTENSIONS:
        return "image"
    return "unknown"


def process_image(image_path: str, save_output: bool = True, show_image: bool = True):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")

    frame = cv2.imread(image_path)
    if frame is None:
        raise ValueError(f"Cannot read image file: {image_path}")

    print(f"\n🖼️ Processing image: {image_path}")
    print(f"📁 Detections will be logged to: {LOG_FILE_PATH}\n")

    db = None
    try:
        db = MongoLogger()
        print(f"✅ Connected to MongoDB — DB: {db.db.name}, Collection: {db.logs.name}\n")
    except Exception as e:
        print(f"⚠️ MongoDB connection failed: {e}  (File logging still active)\n")

    h, w = frame.shape[:2]
    vehicle_type = "Unknown"
    vehicle_results = vehicle_model.predict(frame, conf=0.4, verbose=False)
    if vehicle_results and vehicle_results[0].boxes:
        box = vehicle_results[0].boxes[0]
        class_id = int(box.cls[0])
        vehicle_type = vehicle_model.names.get(class_id, "Unknown")

    results = plate_model.predict(frame, conf=IMAGE_DETECTION_CONF, verbose=False)
    detections = []

    for result in results:
        if result.boxes is None:
            continue
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w - 1, x2), min(h - 1, y2)
            if x2 <= x1 or y2 <= y1:
                continue

            pad_x = int(0.08 * (x2 - x1))
            pad_y = int(0.18 * (y2 - y1))
            px1 = max(0, x1 + pad_x)
            py1 = max(0, y1 + pad_y)
            px2 = min(w - 1, x2 - pad_x)
            py2 = min(h - 1, y2 - pad_y)
            if px2 <= px1 or py2 <= py1:
                continue

            plate_crop = frame[py1:py2, px1:px2]
            if plate_crop.size == 0:
                continue

            gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)
            gray = cv2.GaussianBlur(gray, (5, 5), 0)
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

            ocr_result = reader.readtext(
                thresh,
                allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
                detail=1
            )

            plate_text, conf, raw_text = extract_best_plate_from_ocr(ocr_result)
            if not plate_text:
                continue

            if conf < IMAGE_MIN_PLATE_CONFIDENCE:
                continue

            detections.append({
                "plate": plate_text,
                "confidence": conf,
                "vehicle_type": vehicle_type,
                "raw_text": raw_text,
                "bbox": (x1, y1, x2, y2)
            })

    if not detections:
        print("⚠️ No valid plate detected in image")
    else:
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            draw_detection(frame, x1, y1, x2, y2,
                           det["plate"], det["confidence"], det["vehicle_type"])

            log_plate_detection(
                det["vehicle_type"], det["plate"], float(det["confidence"]),
                image_path, 1, 0.0, det["raw_text"]
            )
            if db:
                with suppress(Exception):
                    db.save_vehicle(det["vehicle_type"], det["plate"], float(det["confidence"]))

            print(f"[✅ LOGGED] {det['vehicle_type']:15s} | {det['plate']} | Conf: {det['confidence']:.2f}")

    output_path = None
    if save_output:
        os.makedirs(VIDEO_OUTPUT_DIR, exist_ok=True)
        output_path = os.path.join(VIDEO_OUTPUT_DIR, f"output_{os.path.basename(image_path)}")
        cv2.imwrite(output_path, frame)
        print(f"🖼️ Output: {output_path}")

    if show_image:
        cv2.imshow('License Plate Detection - Image', frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

# ===============================
# MAIN PROCESSING
# ===============================
def process_video(video_path: str, save_output: bool = True, show_video: bool = True):
    """
    Process video and detect license plates.

    Logging rules:
    - A plate must be detected CONFIRMATION_THRESHOLD times before being logged.
    - Each unique vehicle (tracked by spatial proximity) is logged only ONCE.
    - After logging, the vehicle enters a cooldown of COOLDOWN_FRAMES frames.
    - Vehicles with no detections for VEHICLE_STALE_FRAMES are finalised/removed.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {video_path}")

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = frame_count / fps if fps > 0 else 0

    print(f"\n📹 Video Info:")
    print(f"   Resolution: {width}x{height}")
    print(f"   FPS: {fps}")
    print(f"   Total Frames: {frame_count}")
    print(f"   Duration: {duration:.2f}s")

    out = None
    output_path = None
    if save_output:
        os.makedirs(VIDEO_OUTPUT_DIR, exist_ok=True)
        output_path = os.path.join(VIDEO_OUTPUT_DIR, f"output_{os.path.basename(video_path)}")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print(f"📁 Detections will be logged to: {LOG_FILE_PATH}\n")

    db = None
    try:
        db = MongoLogger()
        print(f"✅ Connected to MongoDB — DB: {db.db.name}, Collection: {db.logs.name}\n")
    except Exception as e:
        print(f"⚠️ MongoDB connection failed: {e}  (File logging still active)\n")

    frame_num = 0
    processed_frames = 0

    # vehicle_id -> {
    #   plate, confidence, vehicle_type, frame_num, raw_text,
    #   last_center, detection_count, logged, cooldown_until,
    #   plate_counts: {plate_text: count}  ← confirmation tracking per plate
    # }
    vehicle_plate_tracker = {}
    logged_vehicles = set()

    PROXIMITY_THRESHOLD = 80  # pixels — tune if needed

    def get_vehicle_id(center):
        """Return (vehicle_id, is_new) based on proximity to existing tracks."""
        x, y = center
        closest_id = None
        closest_dist = PROXIMITY_THRESHOLD

        for vid, data in vehicle_plate_tracker.items():
            px, py = data['last_center']
            dist = ((x - px) ** 2 + (y - py) ** 2) ** 0.5
            if dist < closest_dist:
                closest_dist = dist
                closest_id = vid

        if closest_id is None:
            new_id = f"veh_{frame_num}_{int(x)}_{int(y)}"
            return new_id, True
        return closest_id, False

    print("🎬 Processing video...")
    print("-" * 60)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_num += 1
            current_time = frame_num / fps if fps > 0 else 0

            if frame_num % PROCESS_EVERY_N_FRAMES != 0:
                if save_output and out:
                    out.write(frame)
                continue

            processed_frames += 1

            # --- Vehicle type detection ---
            vehicle_type = "Unknown"
            vehicle_results = vehicle_model.predict(frame, conf=0.4, verbose=False)
            if vehicle_results and vehicle_results[0].boxes:
                box = vehicle_results[0].boxes[0]
                class_id = int(box.cls[0])
                vehicle_type = vehicle_model.names.get(class_id, "Unknown")

            # --- Plate detection ---
            results = plate_model.predict(frame, conf=DETECTION_CONF, verbose=False)

            for result in results:
                if result.boxes is None:
                    continue
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    h, w = frame.shape[:2]
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w - 1, x2), min(h - 1, y2)
                    if x2 <= x1 or y2 <= y1:
                        continue

                    pad_x = int(0.08 * (x2 - x1))
                    pad_y = int(0.18 * (y2 - y1))
                    px1 = max(0, x1 + pad_x)
                    py1 = max(0, y1 + pad_y)
                    px2 = min(w - 1, x2 - pad_x)
                    py2 = min(h - 1, y2 - pad_y)
                    if px2 <= px1 or py2 <= py1:
                        continue

                    plate_crop = frame[py1:py2, px1:px2]
                    if plate_crop.size == 0:
                        continue

                    gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
                    gray = cv2.resize(gray, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)
                    gray = cv2.GaussianBlur(gray, (5, 5), 0)
                    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
                    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

                    ocr_result = reader.readtext(
                        thresh,
                        allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
                        detail=1
                    )

                    plate_text, conf, raw_text = extract_best_plate_from_ocr(ocr_result)

                    if not plate_text:
                        continue

                    if conf < MIN_PLATE_CONFIDENCE:
                        continue

                    plate_center = ((x1 + x2) / 2, (y1 + y2) / 2)
                    vehicle_id, is_new = get_vehicle_id(plate_center)

                    if is_new:
                        vehicle_plate_tracker[vehicle_id] = {
                            'plate': plate_text,
                            'confidence': conf,
                            'vehicle_type': vehicle_type,
                            'frame_num': frame_num,
                            'raw_text': raw_text,
                            'last_center': plate_center,
                            'detection_count': 1,
                            'logged': False,
                            'cooldown_until': 0,
                            'plate_counts': {plate_text: 1}
                        }
                    else:
                        data = vehicle_plate_tracker[vehicle_id]
                        data['last_center'] = plate_center
                        data['frame_num'] = frame_num
                        data['detection_count'] += 1

                        # Track per-plate confirmation counts
                        data['plate_counts'][plate_text] = data['plate_counts'].get(plate_text, 0) + 1

                        # Pick the plate text with the highest confirmation count
                        best_plate = max(data['plate_counts'], key=data['plate_counts'].get)

                        # Update best confidence if this reading is better
                        if conf > data['confidence'] or data['plate'] != best_plate:
                            data['plate'] = best_plate
                            if conf > data['confidence']:
                                data['confidence'] = conf
                                data['raw_text'] = raw_text
                        data['vehicle_type'] = vehicle_type

            # --- Decide what to log ---
            for vehicle_id, data in list(vehicle_plate_tracker.items()):
                # Skip if already logged and still in cooldown
                if data['logged'] and frame_num < data['cooldown_until']:
                    continue

                # Only log if we've confirmed the plate CONFIRMATION_THRESHOLD times
                best_plate = data['plate']
                confirmation_count = data['plate_counts'].get(best_plate, 0)
                if confirmation_count < CONFIRMATION_THRESHOLD:
                    continue

                # Only log if vehicle hasn't been seen recently (likely left frame)
                # OR hasn't been logged yet but has enough confirmations
                frames_since_seen = frame_num - data['frame_num']
                should_log = (
                    not data['logged'] and
                    (frames_since_seen > VEHICLE_STALE_FRAMES or confirmation_count >= CONFIRMATION_THRESHOLD * 2)
                )

                if should_log and data['confidence'] >= MIN_FINAL_CONFIDENCE:
                    plate_text = data['plate']
                    confidence = data['confidence']
                    vtype = data['vehicle_type']

                    log_plate_detection(
                        vtype, plate_text, float(confidence),
                        video_path, data['frame_num'],
                        data['frame_num'] / fps if fps > 0 else 0,
                        data['raw_text']
                    )

                    if db:
                        with suppress(Exception):
                            db.save_vehicle(vtype, plate_text, float(confidence))

                    data['logged'] = True
                    data['cooldown_until'] = frame_num + COOLDOWN_FRAMES
                    logged_vehicles.add(vehicle_id)

                    print(f"[✅ LOGGED] {vtype:15s} | {plate_text} | "
                          f"Conf: {confidence:.2f} | Confirmed {confirmation_count}x")

            # --- Remove stale vehicles that were never logged and are gone ---
            stale_ids = [
                vid for vid, data in vehicle_plate_tracker.items()
                if frame_num - data['frame_num'] > COOLDOWN_FRAMES and data['logged']
            ]
            for vid in stale_ids:
                del vehicle_plate_tracker[vid]
                logged_vehicles.discard(vid)

            # --- Draw overlays ---
            for vehicle_id, data in vehicle_plate_tracker.items():
                if frame_num - data['frame_num'] > VEHICLE_STALE_FRAMES:
                    continue  # Don't draw stale vehicles

                cx, cy = data['last_center']
                offset = 60
                bx1, by1 = int(cx - offset), int(cy - offset)
                bx2, by2 = int(cx + offset), int(cy + offset)

                confirmation_count = data['plate_counts'].get(data['plate'], 0)

                if data['logged']:
                    color = (128, 128, 128)
                    status = "✅ LOGGED"
                elif confirmation_count >= CONFIRMATION_THRESHOLD:
                    color = (0, 255, 0)
                    status = f"Confirmed ({confirmation_count}x)"
                else:
                    color = (0, 255, 255)
                    status = f"Tracking ({confirmation_count}/{CONFIRMATION_THRESHOLD})"

                cv2.rectangle(frame, (bx1, by1), (bx2, by2), color, 2)

                label = f"{data['plate']} ({data['confidence']:.2f})"
                status_label = f"{status} | {data['vehicle_type']}"

                (w1, h1), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                (w2, h2), _ = cv2.getTextSize(status_label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(frame, (bx1, by1 - h1 - h2 - 10),
                              (bx1 + max(w1, w2) + 10, by1), color, -1)
                cv2.putText(frame, label, (bx1 + 5, by1 - h2 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                cv2.putText(frame, status_label, (bx1 + 5, by1 - 3),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

            if save_output and out:
                out.write(frame)

            if show_video:
                cv2.imshow('License Plate Detection', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("\n⏹️  Stopped by user")
                    break

            if processed_frames % 50 == 0:
                progress = (frame_num / frame_count) * 100 if frame_count > 0 else 0
                print(f"[{progress:.1f}%] Tracking {len(vehicle_plate_tracker)} | "
                      f"Logged {len(logged_vehicles)}")

    except KeyboardInterrupt:
        print("\n⏹️  Interrupted by user")

    finally:
        # Final pass — log any confirmed-but-not-yet-logged vehicles
        for vehicle_id, data in vehicle_plate_tracker.items():
            if data['logged']:
                continue
            best_plate = data['plate']
            confirmation_count = data['plate_counts'].get(best_plate, 0)
            if confirmation_count < CONFIRMATION_THRESHOLD:
                continue
            if data['confidence'] < MIN_FINAL_CONFIDENCE:
                continue

            log_plate_detection(
                data['vehicle_type'], best_plate, float(data['confidence']),
                video_path, data['frame_num'],
                data['frame_num'] / fps if fps > 0 else 0,
                data['raw_text']
            )
            if db:
                with suppress(Exception):
                    db.save_vehicle(data['vehicle_type'], best_plate, float(data['confidence']))

            data['logged'] = True
            logged_vehicles.add(vehicle_id)
            print(f"[✅ FINAL LOG] {data['vehicle_type']:15s} | {best_plate} | "
                  f"Conf: {data['confidence']:.2f} | Confirmed {confirmation_count}x")

        cap.release()
        if out:
            out.release()
        if show_video:
            cv2.destroyAllWindows()

    # --- Summary ---
    print("\n" + "=" * 60)
    print("📊 DETECTION SUMMARY")
    print("=" * 60)
    print(f"Total Frames:            {frame_num}")
    print(f"Processed Frames:        {processed_frames}")
    print(f"Unique Vehicles Tracked: {len(vehicle_plate_tracker)}")
    print(f"Unique Vehicles Logged:  {len(logged_vehicles)}")

    if logged_vehicles:
        print(f"\n✅ Logged Vehicles:")
        for vid in sorted(logged_vehicles):
            if vid not in vehicle_plate_tracker:
                continue
            d = vehicle_plate_tracker[vid]
            cnt = d['plate_counts'].get(d['plate'], 1)
            print(f"   • {d['plate']} ({d['vehicle_type']}) | "
                  f"Conf: {d['confidence']:.2f} | Confirmed {cnt}x")

    unlogged = [vid for vid, d in vehicle_plate_tracker.items() if not d['logged']]
    if unlogged:
        print(f"\n⚠️  Detected but not logged (below threshold or confidence):")
        for vid in sorted(unlogged):
            d = vehicle_plate_tracker[vid]
            cnt = d['plate_counts'].get(d['plate'], 1)
            print(f"   • {d['plate']} ({d['vehicle_type']}) | "
                  f"Conf: {d['confidence']:.2f} | Confirmed {cnt}x "
                  f"(need {CONFIRMATION_THRESHOLD})")

    print(f"\n📁 Log: {LOG_FILE_PATH}")
    if save_output and output_path:
        print(f"📹 Output: {output_path}")
    print("=" * 60)


# ===============================
# MAIN
# ===============================
if __name__ == "__main__":
    print("=" * 60)
    print("🚗 LICENSE PLATE DETECTION - MEDIA PROCESSOR")
    print("=" * 60)

    MEDIA_PATH = "/Users/gauravtalele/Downloads/LicensePlateDetection-AIML-main copy/recordings/test.mp4"
    save_output = True
    show_output = True

    try:
        selected_path = os.path.abspath(os.path.expanduser(MEDIA_PATH.strip().strip('"').strip("'")))
        if not os.path.isfile(selected_path):
            raise FileNotFoundError(f"Media file not found: {selected_path}")

        media_type = detect_media_type(selected_path)

        print(f"\n📌 Settings:")
        print(f"   Selected File:          {selected_path}")
        print(f"   Media Type:             {media_type}")
        print(f"   Save Output:            {save_output}")
        print(f"   Show Preview:           {show_output}")
        print(f"   Process Every N Frames: {PROCESS_EVERY_N_FRAMES}")
        print(f"   Min OCR Confidence:     {MIN_PLATE_CONFIDENCE}")
        print(f"   Confirmation Threshold: {CONFIRMATION_THRESHOLD} detections")
        print(f"   Min Final Confidence:   {MIN_FINAL_CONFIDENCE}")
        print(f"   Cooldown Frames:        {COOLDOWN_FRAMES}")
        print(f"   Valid State Codes:      {len(VALID_STATE_CODES)} Indian RTO codes")

        if media_type == "video":
            print("\n💡 Box colours:")
            print("   Yellow = Tracking (below confirmation threshold)")
            print("   Green  = Confirmed (≥ threshold, ready to log)")
            print("   Gray   = Already logged (in cooldown)")
            print("   Press 'q' to stop")
            process_video(selected_path, save_output=save_output, show_video=show_output)
        elif media_type == "image":
            process_image(selected_path, save_output=save_output, show_image=show_output)
        else:
            raise ValueError(
                f"Unsupported file type: {selected_path}. "
                f"Use one of: {sorted(IMAGE_EXTENSIONS | VIDEO_EXTENSIONS)}"
            )
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        raise SystemExit(1)