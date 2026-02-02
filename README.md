# License Plate and Vehicle Type Detector

Live IP camera inference that detects vehicles, finds license plates, reads text via OCR, overlays results on the stream, and logs entries to MongoDB.

## Features
- YOLO model for vehicle detection (`models/Vehicle.pt`)
- YOLO model for plate detection (`models/Plate.pt`)
- EasyOCR for plate text extraction
- On-frame overlays (boxes + labels)
- MongoDB logging with deduplication window
- Optional snapshots saved to disk

## Setup

1. Python env (recommended: use your existing venv)

```bash
python -m venv .venv
source .venv/bin/activate
```

2. Install dependencies

```bash
pip install -r requirements.txt
```

3. Place your trained models
- Vehicle detector at `models/Vehicle.pt`
- Plate detector at `models/Plate.pt`

4. Configure
- Copy `.env.example` to `.env` and set `MONGODB_URI`.
- Edit `config.yaml`:
  - `camera.url`: IP camera RTSP/HTTP URL
  - `models.vehicle_path` / `models.plate_path`
  - `mongodb` database/collection if desired

## Run

```bash
python src/app.py
```

- Press `q` to quit the live window.
- If the camera URL fails and `webcam_fallback` is true, it will open the default webcam.

## MongoDB Document
Each event document includes:
- `timestamp`, `camera_url`
- `vehicle_type`, `vehicle_conf`
- `plate_text`, `plate_conf`, `ocr_conf`
- `vehicle_bbox`, `plate_bbox`
- `snapshot_path` (if enabled)

## Notes
- OCR accuracy depends on plate crop quality. Tuning OCR pre-processing and detection thresholds can improve results.
- Deduplication uses plate text with a time window (`logging.dedup_interval_sec`). Adjust as needed.
- For GPU, set `models.device: "cuda"` in `config.yaml` if available.
