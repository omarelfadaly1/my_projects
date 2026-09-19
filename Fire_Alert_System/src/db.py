# Libraries
import sqlite3
from datetime import datetime
from dataclasses import dataclass


@dataclass
class AlertRecord:
    id: int
    timestamp: str
    behavior: str
    confidence: float
    screenshot_path: str


class Database:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init_schema()

    def _conncect(self):
        return sqlite3.connect(self.db_path)

    def _init_schema(self):
        with self._conncect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    behavior TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    screenshot_path TEXT NOT NULL
                )
                """
            )    
            conn.commit()

    def insert_alert(self, behavior: str, confidence: float, screenshot_path: str):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")  
        with self._conncect() as conn:
            insert = conn.execute(
                "INSERT INTO alerts (timestamp, behavior, confidence, screenshot_path)"
                "VALUES (?, ?, ?, ?)",
                (timestamp, behavior, confidence, screenshot_path),
            )      
            conn.commit()
            return insert.lastrowid

    def fetch_all_alerts(self):
        with self._conncect() as conn:
            rows = conn.execute(
                "SELECT id, timestamp, behavior, confidence ,screenshot_path "
                "FROM alerts ORDER BY id DESC"
            ).fetchall()
        return [AlertRecord(*row) for row in rows] 

    def clear_all_alerts(self):
        with self._conncect() as conn:
            conn.execute("DELETE FROM alerts")
            conn.commit()   