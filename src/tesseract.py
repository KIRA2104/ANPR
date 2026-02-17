import cv2
import pytesseract
import numpy as np
import re
from ultralytics import YOLO
import os

# ==============================
# PATHS
# ==============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PLATE_MODEL_PATH = os.path.join(BASE_DIR, "assets", "Plate.pt")

IMAGE_PATH = "/Users/gauravtalele/Desktop/LicensePlateDetection-AIML-main copy/recordings/archive/State-wise_OLX/AR/AR1.jpg"

# ==============================
# LOAD MODEL
# ==============================
plate_model = YOLO(PLATE_MODEL_PATH)


# ==============================
# PREPROCESS
# ==============================
def preprocess_for_ocr(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)

    thresh = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31, 2
    )
    return thresh


def normalize_plate(text):
    text = text.upper()
    text = re.sub(r"[^A-Z0-9]", "", text)

    text = text.replace("O", "0").replace("I", "1").replace("Z", "2")
    text = text.replace("S", "5").replace("B", "8")

    match = re.search(r"[A-Z]{2}[0-9]{2}[A-Z]{1,2}[0-9]{3,4}", text)
    return match.group(0) if match else ""


img = cv2.imread(IMAGE_PATH)

results = plate_model(img, conf=0.25, verbose=False)[0]

for box in results.boxes:
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    crop = img[y1:y2, x1:x2]

    thresh = preprocess_for_ocr(crop)

    raw = pytesseract.image_to_string(
        thresh,
        config="--psm 8 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    )

    plate = normalize_plate(raw)

    print("RAW OCR:", raw.strip())
    print("FINAL PLATE:", plate)

    cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 2)

    cv2.imshow("Crop", crop)
    cv2.imshow("Thresh", thresh)

cv2.imshow("Result", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
