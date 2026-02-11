from typing import Tuple, Optional
import cv2

def draw_info_panel(frame, vehicle_type: Optional[str] = None, plate: Optional[str] = None):
   
    x, y = 30, 50
    
    v_text = f"Vehicle: {vehicle_type if vehicle_type else 'Scanning...'}"
    cv2.putText(frame, v_text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    
    y += 50
    p_text = f"Plate: {plate if plate else 'Waiting...'}"
    cv2.putText(frame, p_text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

def draw_box(img, xyxy, color: Tuple[int, int, int], label: Optional[str] = None):
    x1, y1, x2, y2 = map(int, xyxy)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    if label:
        cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)