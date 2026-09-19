# Libraries
import cv2
import time
import config as config
import numpy as np
from ultralytics import YOLO
from dataclasses import dataclass
from typing import Dict, List


@dataclass
class TrackState:
    consecutive_fire_frames: int = 0
    confirmed: bool = False
    last_alert_time: float = 0.0


@dataclass
class FireEvent:
    track_id: int 
    label: str
    confidence: float
    frame: np.ndarray


class FireDetector:
    def __init__(self, 
                 model_path: str = config.Model_path,
                 confidence_threshold: float = config.Conf_threshold,
                 confiramtion_frames: int = config.Confirmation_frames,
                 tracker: str = config.Tracker):

        self.model = YOLO(model_path)
        self.conf_thre = confidence_threshold
        self.confirmation = confiramtion_frames 
        self.Tracker = tracker
        self._tracks: Dict[int, TrackState] = {}

    def reset(self):
        self._tracks.clear()

    def processing(self, frame: np.ndarray):
        res = self.model.track(
            frame,
            persist=True,
            tracker= self.Tracker,
            conf= self.conf_thre,
            verbose= False
        )[0]

        annotated = frame.copy()
        new_events: List[FireEvent] = []

        boxes = res.boxes
        if boxes is None or boxes.id is None:
            return annotated, new_events

        xyxy = boxes.xyxy.cpu().numpy()
        ids = boxes.id.cpu().numpy().astype(int)
        cls_ids = boxes.cls.cpu().numpy().astype(int)
        confidence = boxes.conf.cpu().numpy()

        for box, id, cls, conf in zip(xyxy, ids, cls_ids, confidence):
            label = config.Class_names.get(cls, f"class_{cls}")
            state = self._tracks.setdefault(id, TrackState())  

            if label == "fire":
                state.consecutive_fire_frames += 1
                if state.consecutive_fire_frames >= self.confirmation:
                    state.confirmed = True
            else:
                state.consecutive_fire_frames = 0

            color, box_label = self._box_style(label, conf, state)                                 
            self._draw_box(annotated, box, id, box_label, color)

            if state.confirmed:
                now = time.time()
                if now - state.last_alert_time >= config.Alert_cooldown:
                    state.last_alert_time = now
                    new_events.append(FireEvent(track_id= id, label= label, confidence= float(conf), frame= annotated))
                    
        return annotated, new_events

    @staticmethod
    def _box_style(label: str, conf: float, state: TrackState):
        if label == "fire" and state.confirmed:
            return config.Color_fire, f"Fire Confirmed {conf:.2f}"
        if label == "smoke":
            return config.Color_smoke, f"Smoke {conf:.2f}"
        return config.Color_monitoring, f"{label} {conf:.2f}"

    @staticmethod
    def _draw_box(frame: np.ndarray, box, track_id: int, text: str, color) -> None:
        x1, y1, x2, y2 = box.astype(int)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label_text = f"ID {track_id} | {text}"
        (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
        cv2.putText(frame, label_text, (x1 + 2, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA,)                    