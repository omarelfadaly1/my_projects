# Road Damage Detection using YOLOv8

A deep learning project for automatic road damage detection using three YOLOv8 object detection models (YOLOv8n, YOLOv8s, and YOLOv8m).

The project includes:
- Dataset exploration and preprocessing
- Training multiple YOLOv8 models
- Model evaluation and comparison
- Interactive Flask web application for road damage detection

---

## Project Structure

```
Road-Damage-Detection/
│
├── notebooks/
├── deployment/
├── models/
├── results/
├── README.md
├── requirements.txt
└── .gitignore
```

---

## Models

The following models were trained and evaluated:

| Model | Description |
|--------|-------------|
| YOLOv8n | Fastest and smallest model |
| YOLOv8s | Best balance between speed and accuracy |
| YOLOv8m | Highest accuracy with larger model size |

---

## Evaluation Results

| Model | mAP50 | Precision | Recall | FPS | Size (MB) |
|--------|-------|-----------|--------|-----|-----------|
| YOLOv8n | 40.86% | 58.01% | 42.21% | 227.0 | 6.0 |
| YOLOv8s | **43.15%** | **62.04%** | 44.23% | 121.8 | 21.5 |
| YOLOv8m | 42.68% | 59.89% | **44.64%** | 51.8 | 49.6 |

---

## Technologies Used

- Python
- YOLOv8 (Ultralytics)
- PyTorch
- OpenCV
- Plotly
- Flask
- Pandas
- NumPy
- Pillow

---



## Features

- Compare three YOLOv8 models
- Adjustable confidence threshold
- Bounding-box visualization
- Detection summary
- Inference time measurement
- Responsive web interface

---

## Dataset

Road Damage Dataset in YOLO format.

**https://universe.roboflow.com/trafficsignssafeway/road-damage-l1ju7/browse**

---

## Author

**Omar Elfadaly**