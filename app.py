from flask import Flask, request, jsonify, render_template
import cv2, base64, re
import numpy as np
from ultralytics import YOLO   # use Ultralytics package instead of torch.hub

# Allow rendering index.html from project root
app = Flask(__name__, template_folder='.')

# Load YOLOv5 small model (fast) – auto-downloads once and caches
model = YOLO("yolov5s.pt")

# Map YOLO labels into categories
CATEGORY_MAP = {
    "person": "human",
    "cat": "animal",
    "dog": "animal",
    "bird": "animal",
    "cow": "animal",
    "sheep": "animal",
    "horse": "animal",
    "car": "vehicle",
    "bus": "vehicle",
    "truck": "vehicle",
    "motorbike": "vehicle",
    "bicycle": "vehicle",
    "potted plant": "plant",
    "tree": "plant"
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

    # Detect objects
    results = model(frame)

    warning_obj = None
    nearest_distance = None

    # Iterate over detections and pick the nearest relevant obstacle
    for r in results:
        for box in r.boxes:
            label = model.names[int(box.cls[0])]   # class name
            y1, y2 = float(box.xyxy[0][1]), float(box.xyxy[0][3])

            obj_type = CATEGORY_MAP.get(label, "obstacle")

            # Simple distance proxy from bounding box height
            box_h = max((y2 - y1), 1.0)
            est_distance_m = round(1000.0 / box_h, 1)

            if est_distance_m < 10:
                if nearest_distance is None or est_distance_m < nearest_distance:
                    nearest_distance = est_distance_m
                    warning_obj = obj_type

    return jsonify({"object": warning_obj, "distance": nearest_distance})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
