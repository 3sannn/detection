from flask import Flask, request, jsonify, render_template
import cv2, base64, re
import numpy as np
import torch

# Allow rendering index.html from project root
app = Flask(__name__, template_folder='.')

# Load YOLOv5 small model (fast)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

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
    detections = results.pandas().xyxy[0]

    warning_obj = None
    nearest_distance = None

    # Iterate over detections and pick the nearest relevant obstacle
    for _, row in detections.iterrows():
        label = row['name']
        y1 = float(row['ymin'])
        y2 = float(row['ymax'])

        obj_type = CATEGORY_MAP.get(label, "obstacle")

        # Simple distance proxy from bounding box height (pixel → meter calibration needed)
        box_h = max((y2 - y1), 1.0)
        est_distance_m = round(1000.0 / box_h, 1)

        if est_distance_m < 10:
            if nearest_distance is None or est_distance_m < nearest_distance:
                nearest_distance = est_distance_m
                warning_obj = obj_type

    return jsonify({"object": warning_obj, "distance": nearest_distance})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)