# Libraries
import cv2
import time
import queue
import threading
import numpy as np
from typing import Optional
from dataclasses import dataclass
from src.alert import AlertManager
from src.tracker import FireDetector


@dataclass
class FrameMessage:
    frame: np.ndarray
    finished: bool = False

class VideoWorker:
    def __init__(self, detector: FireDetector, alert: AlertManager):
        self.detector = detector
        self.alert = alert
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self.frame_queue: "queue.Queue[FrameMessage]" = queue.Queue(maxsize=2)

    def start(self, video_path: str):
        self.stop()
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, args=(video_path,), daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._thread = None        

    def _run(self, video_path: str):
        self.detector.reset()
        cap = cv2.VideoCapture(video_path)   
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        frame_interval = 1.0 / fps

        try:
            while not self._stop_event.is_set():
                start = time.time()
                success, frame = cap.read()
                if not success:
                    break

                annot_frames, events = self.detector.processing(frame)
                if events:
                    self.alert.handle_events(events)

                self._push_frame(FrameMessage(frame=annot_frames))
                elapsed = time.time() - start
                sleep = frame_interval - elapsed
                if sleep > 0:
                    time.sleep(sleep) 

        finally:
            cap.release()
            self._push_frame(FrameMessage(frame=None, finished=True))

    def _push_frame(self, message: FrameMessage):
        try:
            self.frame_queue.put_nowait(message)
        except queue.Full:
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self.frame_queue.put_nowait(message)
            except queue.Full:
                pass            
                                