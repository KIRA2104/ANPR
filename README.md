# ANPR Guardian: License Plate & Vehicle Detection

A comprehensive web-based Automatic Number Plate Recognition (ANPR) system featuring live video detection, mobile camera integration, and a premium glassmorphism UI.

## Features

### 1. **Core ANPR Engine**
- **Vehicle Detection**: YOLOv8-based model (`yolov8n.pt`/`Vehicle.pt`) detects cars, motorcycles, buses, and trucks.
- **License Plate Detection**: Specialized YOLO model (`Plate.pt`) locates number plates.
- **OCR & Normalization**: EasyOCR with **smart post-processing** (`clean_plate_robust`) specifically tuned for Indian License Plates (Corrects `O`->`0`, `I`->`1`, etc.).
- **Vehicle Classification**: Optional integration with Keras classifier (`vehicle_classifier.keras`).

### 2. **Web Interface**
- **Premium UI**: Minimalist Dark Mode dashboard with glassmorphism effects.
- **Live Streaming**: Real-time video feed from IP Cameras (RTSP) or Webcams.
- **Mobile Camera Support**: Use your mobile phone as a network camera by scanning the QR code or accessing the local network IP.
- **Real-time Logs**: Displays latest detection (Vehicle Type + Plate Number) instantly.

### 3. **Backend & Data**
- **Flask Server**: Lightweight Python web server handling video streaming and API requests.
- **MongoDB Integration**: Auto-logs vehicle entries with timestamps and confidence scores.
- **Smart Deduplication**: Prevents spamming logs for the same vehicle within a configurable cooldown period.

## Installation

### Prerequisites
- Python 3.10+
- MongoDB (optional, for logging)

### Setup
1. **Clone & Install Dependencies**
   ```bash
   git clone <repo_url>
   cd LicensePlateDetection-AIML-main
   pip install -r requirements.txt
   ```

2. **Model Setup**
   Ensure your models are in `src/assets/`:
   - `Vehicle.pt` (or `best1.pt`)
   - `Plate.pt` (or `Traffic1.pt`)
   - `vehicle_classifier.keras` (Optional)

3. **Configuration**
   - Edit `config.yaml` to set your camera URL (`rtsp://...`) or use `0` for webcam.
   - Set environment variables if needed (`MONGODB_URI`).

## Usage

### 🚀 Run the Web App
Start the main application:
```bash
python src/anpr_ipcam.py
```
- Access the dashboard at: `http://localhost:5005`
- To use your mobile camera, ensure your phone and PC are on the same Wi-Fi, then visit `http://<YOUR_PC_IP>:5005` on your phone.

### 🧪 Batch Testing Script
To test detection accuracy on a folder of images without running the full web app:
```bash
python src/try.py
```
- This script provides **advanced debug logs** and visualizes OCR results in the terminal.
- Ideal for testing the `clean_plate_robust` logic.

## Project Structure
- `src/anpr_ipcam.py`: **Main Application Entry Point** (Flask + Inference Loop).
- `src/try.py`: Standalone batch testing script.
- `src/ocr_utils.py`: Helper functions for plate tracking and cleaning.
- `src/db.py`: Database connection handler.
- `src/templates/index.html`: Frontend UI.
- `src/assets/`: Directory for YOLO/Keras models.
- `recordings/`: Saved video clips (if recording is enabled).

## Troubleshooting
- **No Detection?** Check lighting conditions or adjust `confidence` thresholds in `config.yaml`.
- **OCR Errors?** The system is optimized for standard Indian fonts. Heavily stylized plates may fail. Use `src/try.py` to debug specific images.
- **Mobile Cam Lag?** Ensure a strong Wi-Fi connection.

---
**Author**: Gaurav Talele
**License**: MIT
