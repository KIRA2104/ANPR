import cv2
from ultralytics import YOLO
import easyocr
import numpy as np
import re
import os

# ===============================
# PATHS
# ===============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "assets", "best1.pt")
IMAGES_DIR = os.path.join(
    os.path.dirname(BASE_DIR),
    "recordings",
    "archive",
    "State-wise_OLX"
)
# ===============================
# LOAD MODEL & OCR
# ===============================
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Plate model not found at: {MODEL_PATH}")

print(f"Loading plate model from: {MODEL_PATH}")
model = YOLO(MODEL_PATH)

print("Initializing EasyOCR reader...")
reader = easyocr.Reader(['en'], gpu=False)

# ===============================
# CONSTANTS (RELAXED)
# ===============================
# Very light filters: only avoid obviously tiny or border-touching boxes.
DETECTION_CONF = 0.25          # lower conf to catch smaller/blurrier plates
MIN_PLATE_AREA_RATIO = 0.0003   # 0.03% of frame area – almost everything passes
MIN_PLATE_HEIGHT_RATIO = 0.015  # 1.5% of frame height – very small allowed
BORDER_MARGIN = 2               # only reject boxes that literally touch edges
PROCESS_EVERY_N_FRAMES = 1      # process every frame
DEBUG_REJECTIONS = False        # set True if you want to see why boxes are rejected
DEBUG_OCR = True                # enable OCR debug for now

# ===============================
# HELPERS
# ===============================
def is_plate_quality_good(x1, y1, x2, y2, frame_w, frame_h):
    """
    VERY relaxed quality check:
    - Box must not touch the frame border (so it's likely fully inside)
    - Box must not be absurdly tiny relative to the frame
    """
    width = x2 - x1
    height = y2 - y1

    if width <= 0 or height <= 0:
        return False

  
    if (
        x1 <= BORDER_MARGIN
        or y1 <= BORDER_MARGIN
        or x2 >= frame_w - BORDER_MARGIN
        or y2 >= frame_h - BORDER_MARGIN
    ):
        if DEBUG_REJECTIONS:
            print("border reject")
        return False

    # 2) Rough size check
    frame_area = frame_w * frame_h
    area_ratio = (width * height) / float(frame_area)
    height_ratio = height / float(frame_h)

    if area_ratio < MIN_PLATE_AREA_RATIO or height_ratio < MIN_PLATE_HEIGHT_RATIO:
        if DEBUG_REJECTIONS:
            print(f"size reject: area_ratio={area_ratio:.5f}, h_ratio={height_ratio:.5f}")
        return False

    return True


def normalize_plate(text: str) -> str:
    """
    Normalize OCR text into strict Indian plate format:

      XX00XX0000 or XX00X0000
      - First 2: letters (state)
      - Next 2: digits (district)
      - Next 1–2: letters (series)
      - Last 4: digits

    Returns cleaned plate or "" if not matching this pattern.
    """
    if not text:
        return ""

    # Basic cleanup
    text = text.upper()
    text = text.strip().replace(" ", "").replace("-", "")
    text = re.sub(r"[^A-Z0-9]", "", text)

    # Mild OCR character corrections
    text = text.replace("O", "0").replace("Q", "0")
    text = text.replace("I", "1").replace("L", "1")
    text = text.replace("S", "5").replace("Z", "2")
    text = text.replace("B", "8")

    # Accept plates with 3 or 4 digits at the end
    match = re.search(r"^[A-Z]{2}[0-9]{2}[A-Z]{1,2}[0-9]{3,4}$", text)
    return match.group(0) if match else ""


# ===============================
# READ IMAGES
# ===============================
if not os.path.isdir(IMAGES_DIR):
    raise FileNotFoundError(f"❌ Image folder not found: {IMAGES_DIR}")

state_dirs = [
    d for d in sorted(os.listdir(IMAGES_DIR))
    if os.path.isdir(os.path.join(IMAGES_DIR, d))
]

if not state_dirs:
    raise FileNotFoundError(f"❌ No state folders found in: {IMAGES_DIR}")

detected_texts = []  # store all accepted plates

print("Available states:")
print(", ".join(state_dirs))
print("\nType a state code to run (or 'all' to run every state, 'q' to quit).")

while True:
    selection = input("State> ").strip().upper()
    if selection in {"Q", "QUIT", "EXIT"}:
        break

    if selection == "ALL":
        selected_states = state_dirs
    elif selection in state_dirs:
        selected_states = [selection]
    else:
        print("Invalid state. Try again.")
        continue

    for state in selected_states:
        state_path = os.path.join(IMAGES_DIR, state)
        image_paths = []
        for root, _, files in os.walk(state_path):
            for filename in files:
                if filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
                    image_paths.append(os.path.join(root, filename))

        if not image_paths:
            print(f"\n=== {state} ===")
            print("No images found.")
            continue

        print(f"\n=== {state} ===")
        print(f"Found {len(image_paths)} images. Running detection...")

        for image_path in sorted(image_paths):
            img = cv2.imread(image_path)
            if img is None:
                print(f"[SKIP] Cannot read: {image_path}")
                continue

            results = model.predict(img, conf=DETECTION_CONF, verbose=False)
            found_in_image = []
            total_boxes = 0
            ocr_candidates = []

            for result in results:
                if result.boxes is None:
                    continue

                for box in result.boxes:
                    total_boxes += 1
                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    # Ensure bbox is within frame bounds
                    h, w = img.shape[:2]
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w - 1, x2), min(h - 1, y2)
                    if x2 <= x1 or y2 <= y1:
                        continue

                    # Debug: allow all boxes to reach OCR
                    # if not is_plate_quality_good(x1, y1, x2, y2, w, h):
                    #     continue

                    pad_x = int(0.08 * (x2 - x1))
                    pad_y = int(0.18 * (y2 - y1))

                    # Apply padding and clamp to image
                    px1 = max(0, x1 + pad_x)
                    py1 = max(0, y1 + pad_y)
                    px2 = min(w - 1, x2 - pad_x)
                    py2 = min(h - 1, y2 - pad_y)

                    if px2 <= px1 or py2 <= py1:
                        continue

                    plate_crop = img[py1:py2, px1:px2]
                    if plate_crop.size == 0:
                        continue

                    gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
                    gray = cv2.resize(gray, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)
                    gray = cv2.GaussianBlur(gray, (5, 5), 0)

                    _, thresh = cv2.threshold(
                        gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
                    )

                    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
                    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

                    ocr_result = reader.readtext(
                        thresh,
                        allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
                        detail=1
                    )

                    raw_text = "".join([r[1] for r in ocr_result]) if ocr_result else ""
                    plate_text = normalize_plate(raw_text)

                    if plate_text:
                        detected_texts.append(plate_text)
                        found_in_image.append(plate_text)
                    elif DEBUG_OCR and raw_text:
                        avg_conf = sum(r[2] for r in ocr_result) / len(ocr_result)
                        ocr_candidates.append(f"{raw_text} ({avg_conf:.2f})")

            if found_in_image:
                unique_found = sorted(set(found_in_image))
                print(f"[OK] {image_path}")
                print(f"     Plates: {', '.join(unique_found)}")
            else:
                if total_boxes == 0:
                    print(f"[NO DET] {image_path}")
                else:
                    print(f"[NONE] {image_path}")
                    if DEBUG_OCR and ocr_candidates:
                        print(f"     OCR: {', '.join(ocr_candidates[:5])}")

# Aggregate detections (simple frequency count)
from collections import Counter

counts = Counter(detected_texts)
print("\nUnique plates with counts:")
for plate, cnt in counts.items():
    print(f"  {plate}: {cnt} time(s)")
