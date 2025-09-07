from flask import Flask, request, jsonify, render_template
import cv2, base64, re
import numpy as np
from ultralytics import YOLO

app = Flask(__name__, template_folder='.')
model = YOLO("yolov8n.pt")

# Enhanced category mapping with colors (BGR format)
CATEGORY_MAP = {
    "person": {"category": "human", "color": (0, 0, 255)},    # Red for humans
    "cat": {"category": "animal", "color": (0, 255, 0)},      # Green for animals
    "dog": {"category": "animal", "color": (0, 255, 0)},
    "bird": {"category": "animal", "color": (0, 255, 0)},
    "cow": {"category": "animal", "color": (0, 255, 0)},
    "sheep": {"category": "animal", "color": (0, 255, 0)},
    "horse": {"category": "animal", "color": (0, 255, 0)},
    "car": {"category": "vehicle", "color": (255, 0, 0)},     # Blue for vehicles
    "bus": {"category": "vehicle", "color": (255, 0, 0)},
    "truck": {"category": "vehicle", "color": (255, 0, 0)},
    "motorbike": {"category": "vehicle", "color": (255, 0, 0)},
    "bicycle": {"category": "vehicle", "color": (255, 0, 0)},
    "potted plant": {"category": "plant", "color": (0, 255, 255)},  # Yellow for plants
    "tree": {"category": "plant", "color": (0, 255, 255)}
}

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/process_frame', methods=['POST'])
def process_frame():
    data = request.json['image']
    img_str = re.sub('^data:image/.+;base64,', '', data)
    img_bytes = base64.b64decode(img_str)
    img_np = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(img_np, cv2.IMREAD_COLOR)

    results = model.predict(
        frame,
        conf=0.6,        # Increased confidence threshold
        iou=0.7,         # Increased IOU threshold for better NMS
        imgsz=640,
        verbose=False,
        agnostic_nms=True  # Better NMS across classes
    )

    detected_objects = []
    min_box_size = 20
    distance_threshold = 4.0  # 4 meters threshold

    # Process detections
    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            label = r.names.get(cls_id, str(cls_id))
            
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Convert to int for drawing
            box_w = x2 - x1
            box_h = y2 - y1
            
            if box_w < min_box_size or box_h < min_box_size:
                continue

            # Get category info
            category_info = CATEGORY_MAP.get(label, {"category": "obstacle", "color": (128, 128, 128)})
            obj_type = category_info["category"]
            color = category_info["color"]

            # Calculate distance with improved formula
            box_area = box_w * box_h
            frame_area = frame.shape[0] * frame.shape[1]
            relative_size = np.sqrt(box_area / frame_area)
            distance = round(1.0 / relative_size, 1)  # More accurate distance estimation

            # Draw bounding box with thicker lines for better visibility
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
            
            # Add background to text for better readability
            text = f"{obj_type}: {distance}m"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7
            thickness = 2
            (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
            
            # Draw black background rectangle for text
            cv2.rectangle(frame, 
                        (x1, y1 - text_height - baseline - 5), 
                        (x1 + text_width, y1),
                        (0, 0, 0), 
                        -1)
            
            # Draw text
            cv2.putText(frame, text, 
                      (x1, y1 - baseline - 5), 
                      font, font_scale, color, thickness)
            detected_objects.append({
                        "object": obj_type,
                        "distance": distance,
                        "label": label,
                        "confidence": conf,
                        "box": {
                                 "x1": x1,
                                "y1": y1,
                                "x2": x2,
                                "y2": y2
                                }
                })

    # Encode the annotated frame
    _, buffer = cv2.imencode('.jpg', frame)
    encoded_image = base64.b64encode(buffer).decode('utf-8')

    return jsonify({
        "image": f"data:image/jpeg;base64,{encoded_image}",
        "detections": detected_objects
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
