import os

# Paths
Base_dir = os.path.dirname(os.path.abspath(__file__))
Model_path = os.path.join(Base_dir, "model", "best.pt")
Screenshot_dir = os.path.join(Base_dir, "alerts", "screenshots")
DB_path = os.path.join(Base_dir, "data", "alerts.db")

# Detection & Tracking

Class_names = {
    0: "smoke",
    1: "fire"
}

Conf_threshold = 0.45
Tracker = "bytetrack.yaml"
Confirmation_frames = 15
Alert_cooldown = 60
Color_monitoring = (0, 200, 0)
Color_smoke = (0, 165, 255)
Color_fire = (0, 0, 255)

# GUI

App_title = "Fire Tracking & Alert System"
Video_width = 860
Video_height = 480

def ensure_directories():

    os.makedirs(os.path.dirname(Model_path), exist_ok=True)
    os.makedirs(Screenshot_dir, exist_ok=True)
    os.makedirs(os.path.dirname(DB_path), exist_ok=True)