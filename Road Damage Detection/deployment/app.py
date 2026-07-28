# ===========================================================Libraries===========================================================

import cv2
import time 
import base64
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
from ultralytics import YOLO
from flask import Flask, jsonify, request, render_template

# ===========================================================Variables===========================================================

models_weights = {
    "YOLOv8n": "F:/AI/output/yolov8n/weights/best.pt",
    "YOLOv8s": "F:/AI/output/yolov8s/weights/best.pt",
    "YOLOv8m": "F:/AI/output/yolov8m/weights/best.pt",
}

models_display_info = {
    "YOLOv8n": {
        "display_name": "YOLOv8n",
        "description": "Fastest and smallest. Good for real-time use.",
        "tags": ["Fastest", "Smallest"],
    },
    "YOLOv8s": {
        "display_name": "YOLOv8s",
        "description": "Good balance of speed and accuracy.",
        "tags": ["Recommended"],
    },
    "YOLOv8m": {
        "display_name": "YOLOv8m",
        "description": "Most accurate, but slower and larger.",
        "tags": ["Most accurate"],
    },
}

comparison_csv = "F:/AI/output/model_comparison_results.csv"
allowed_extensions = ["jpg", "jpeg", "png", "jfif"]
default_conf = 0.25


# ===========================================================Flask===========================================================

app = Flask(__name__)


loaded_models = {}
def load_all_models():
    for name, weight in models_weights.items():
        if not Path(weight).exists():
            print("Couldn't load models")
            continue

        model = YOLO(weight)
        loaded_models[name] = model
        print(f"Loaded {name}")

load_all_models() 

def get_metrics():
    csv_path = Path(comparison_csv)
    if not csv_path.exists():
        return []

    df = pd.read_csv(csv_path)
    metrics_list = []

    for i, row in df.iterrows():
        model_name = row["Model"]
        display_info = models_display_info.get(model_name, {})

        metrics_list.append({
            "id" : model_name,
            "display_name" : display_info.get("display_name", model_name),
            "description" : display_info.get("description", ""),
            "tags" : display_info.get("tags", []),
            "map50" : round(float(row["mAP50"] * 100), 3),
            "precision" : round(float(row["Precision"] * 100), 3),
            "recall" : round(float(row["Recall"] * 100), 3),
            "fps" : round(float(row["FPS"]), 1),
            "size" : round(float(row["Size_MB"]), 1),
            "available" : model_name in loaded_models
        })

    return metrics_list


def is_allowed_extension(filename):
    if "." not in filename:
        return False

    extension = filename.rsplit(".", 1)[1].lower()
    return extension in allowed_extensions


def reading_image(file):
    img = Image.open(file.stream).convert("RGB")
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def img_to_base64(img):
    success, encoded = cv2.imencode(".jpg", img)
    return base64.b64encode(encoded).decode("utf-8")

# ===========================================================Routes===========================================================

@app.route('/')
def home_page():
    return render_template("index.html")


@app.route('/health')
def health_check():
    return jsonify({
        "status" : "ready" if loaded_models else "no models loaded",
        "models_loaded" : len(loaded_models)
    })


@app.route('/models')
def get_models():
    return jsonify(get_metrics())


@app.route('/predict', methods=["POST"])
def predict():
    model_name = request.form.get("model")
    if not model_name  or model_name not in loaded_models:
        return jsonify({"success" : False, "error" : "Please Choose a Valid Model"}), 400

    if "image" not in request.files:
        return jsonify({"success" : False, "error" : "No Image Was Uploaded"}), 400

    uploaded_file = request.files["image"] 
    if uploaded_file.filename == "":
        return jsonify({"success" : False, "error" : "No Image Was Selected"}), 400

    if not is_allowed_extension(uploaded_file.filename):
        return jsonify({"success" : False, "error" : "Invalid Extension"}), 400

    try:
        conf = float(request.form.get("confidence", default_conf))
    except ValueError:
        conf = default_conf

    try:
        img = reading_image(uploaded_file)
    except Exception:
        return jsonify({"success": False, "error": "Couldn't Read That Image File."}), 400        

    model = loaded_models[model_name]

    start = time.time()
    results = model.predict(img, conf= conf, verbose= False)[0]
    inference_ms = round((time.time() - start) * 1000, 1)

    detections = []
    for box in results.boxes:
        cls_id = int(box.cls[0])
        conf_percent = round(float(box.conf[0] * 100), 1)
        detections.append({
            "class" : model.names[cls_id],
            "confidence" : conf_percent
        })


    detections.sort(key=lambda d: d["confidence"], reverse= True)

    annotated_img = results.plot()
    annotated_img_base64 = "data:image/jpeg;base64," + img_to_base64(annotated_img)

    display_name = models_display_info.get(model_name, {}).get("display_name", model_name)

    return jsonify({
        "success": True,
        "model": display_name,
        "inference_ms": inference_ms,
        "object_count": len(detections),
        "detections": detections,
        "annotated_image": annotated_img_base64
    })



if  __name__ == "__main__":
    app.run(debug=True)