import cv2
from ultralytics import YOLO
import easyocr
import numpy as np
import re

# ===============================
# PATHS
# ===============================
MODEL_PATH = "/Users/gauravtalele/Desktop/LicensePlateDetection-AIML-main copy/src/assets/Plate.pt"

IMAGE_PATH = "/Users/gauravtalele/Desktop/LicensePlateDetection-AIML-main copy/recordings/typesofcarnumberplates-02-01.jpg"

OUTPUT_PATH = "/Users/gauravtalele/Desktop/LicensePlateDetection-AIML-main copy/src/output/output_plate_result.jpg"

# ===============================
# LOAD MODEL & OCR
# ===============================
model = YOLO(MODEL_PATH)
reader = easyocr.Reader(['en'], gpu=False)

# ===============================
# HELPERS
# ===============================
def normalize_plate(text):
    text = text.upper().replace(" ", "")

    # Safe OCR corrections
    text = text.replace("O", "0").replace("Q", "0")
    text = text.replace("I", "1").replace("L", "1")
    text = text.replace("S", "5").replace("Z", "2")
    text = text.replace("B", "8")

    # Indian plate regex
    match = re.search(
        r"[A-Z]{2}[0-9]{2}[A-Z]{1,2}[0-9]{3,4}",
        text
    )

    return match.group(0) if match else text

# ===============================
# READ IMAGE
# ===============================
img = cv2.imread(IMAGE_PATH)
if img is None:
    raise FileNotFoundError("❌ Image not found")

# ===============================
# YOLO DETECTION
# ===============================
results = model.predict(img, conf=0.4, verbose=False)

detected_texts = []

for result in results:
    if result.boxes is None:
        continue

    for box in result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = box.conf[0].item()

        # ===============================
        # TIGHT PLATE CROP (CRITICAL)
        # ===============================
        pad_x = int(0.08 * (x2 - x1))
        pad_y = int(0.18 * (y2 - y1))

        plate_crop = img[
            y1 + pad_y : y2 - pad_y,
            x1 + pad_x : x2 - pad_x
        ]

        if plate_crop.size == 0:
            continue

        # ===============================
        # OCR PREPROCESSING (CLEAN)
        # ===============================
        gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        _, thresh = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        # ===============================
        # DEBUG WINDOWS (TEMP)
        # ===============================
        cv2.imshow("Plate Crop", plate_crop)
        cv2.imshow("OCR Input", thresh)
        cv2.waitKey(0)

        # ===============================
        # OCR (STRICT)
        # ===============================
        ocr_result = reader.readtext(
            thresh,
            allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
            detail=1
        )

        raw_text = "".join([r[1] for r in ocr_result])
        plate_text = normalize_plate(raw_text)
        detected_texts.append(plate_text)

        # ===============================
        # DRAW RESULT
        # ===============================
        label = f"{plate_text if plate_text else 'Plate'} ({conf:.2f})"

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            img,
            label,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

# ===============================
# SAVE & SHOW
# ===============================
cv2.imwrite(OUTPUT_PATH, img)
print("✅ Output saved to:", OUTPUT_PATH)
print("🔢 Detected Plate Numbers:", detected_texts)

cv2.imshow("ANPR Result", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
