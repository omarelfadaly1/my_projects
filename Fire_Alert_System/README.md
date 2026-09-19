# Fire Detection & Tracking System

A computer vision system that detects and tracks fire in video using a trained YOLO model and ByteTrack. The project extends model inference into a complete application with track history, fire-event handling, alerts, screenshot capture, database storage, background video processing, and a Tkinter desktop interface.

## Project Overview

The system processes video frame by frame and follows this general pipeline:

```text
Video
  ↓
YOLO Fire Detection
  ↓
ByteTrack Tracking
  ↓
Track History & Application Logic
  ↓
Fire Event / Alert
  ├── Screenshot
  ├── Database Record
  └── Notification
  ↓
Tkinter Application
```

The project was developed in stages, starting from fire detection and progressing toward a complete desktop application for fire monitoring.

## Dataset

Dataset used for training the fire detection model:

**Dataset:** [Smoke And Fire Detection](https://www.kaggle.com/datasets/sayedgamal99/smoke-fire-detection-yolo/data)

## Project Structure

```
fire_alert_system/
├── alerts/screenshots/               
├── config.py                 
├── src/
│   ├── detector.py           
│   ├── alerts.py              
│   ├── db.py            
│   └── video.py         
├── gui/
│   └── app.py                 
├── notebooks/
│   ├── train.ipynb
│   └── tracker.ipynb    
├── models/ best.pt                 
├── main.py         
└── data/ alerts.db
```

## Author

**Omar Elfadaly**
