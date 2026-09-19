# Libraries
import os
import cv2
import config as config
from src.db import Database
from datetime import datetime
from src.tracker import FireEvent
from typing import Callable, List, Optional

try:
    from plyer import notification as pl_n
except ImportError:
    pl_n = None


class AlertManager:
    def __init__(self,
                 database: Database,
                 screenshot_dir: str = config.Screenshot_dir,
                 on_notify: Optional[Callable[[str, str, str], None]] = None,):

        self.db = database
        self.screenshot_dir = screenshot_dir
        self.on_notify = on_notify
        os.makedirs(self.screenshot_dir, exist_ok=True)

    def handle_events(self, events: List[FireEvent]):
        for event in events:
            self._raise_alert(event)

    def _raise_alert(self, event: FireEvent):
        screenshot_path = self._save_screenshot(event)
        self.db.insert_alert(
            behavior = event.label,
            confidence= event.confidence,
            screenshot_path= screenshot_path
            )
        self._notify(event, screenshot_path)     

    def _save_screenshot(self, event: FireEvent):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{event.label}_track{event.track_id}_{timestamp}.jpg"
        path = os.path.join(self.screenshot_dir, filename)
        cv2. imwrite(path, event.frame)
        return path

    def _notify(self, event: FireEvent, screenshot_path: str):
        title = "🔥 Fire Alert Detected"
        message = (f"Behavior: {event.label.capitalize()} detected\n"
                   f"Confidence: {event.confidence:.0%}\n"
                   f"Screenshot saved successfully")

        if pl_n is not None:
            try:
                pl_n.notify(
                    title = title,
                    message = message,
                    app_name= config.App_title,
                    timeout  = 8,
                    )
            except Exception:
                pass

        if self.on_notify is not None:
            self.on_notify(title, message, screenshot_path)        
                 