import sqlite3
import os
import time

class CalibrationDatabase:
    """
    Manages SQLite database storage for gaze calibration sessions.
    Allows saving rich feature vectors (eyes and 3D head coordinates)
    and loading the most recent session for regression training.
    """
    def __init__(self, db_dir="data", db_filename="calibration.db"):
        self.db_dir = db_dir
        self.db_path = os.path.join(db_dir, db_filename)
        os.makedirs(self.db_dir, exist_ok=True)
        self._init_db()

    def _init_db(self):
        """Initializes the SQLite database and creates the calibration table if missing."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS calibration_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                timestamp REAL,
                screen_x INTEGER,
                screen_y INTEGER,
                left_ratio_x REAL,
                left_ratio_y REAL,
                right_ratio_x REAL,
                right_ratio_y REAL,
                left_ear REAL,
                right_ear REAL,
                head_yaw REAL,
                head_pitch REAL,
                head_roll REAL,
                head_tx REAL,
                head_ty REAL,
                head_tz REAL
            )
        """)
        conn.commit()
        conn.close()

    def save_session(self, session_id, dataset):
        """
        Saves a list of calibration sample dictionaries to the database under session_id.
        """
        if not dataset:
            return

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        insert_query = """
            INSERT INTO calibration_data (
                session_id, timestamp, screen_x, screen_y,
                left_ratio_x, left_ratio_y, right_ratio_x, right_ratio_y,
                left_ear, right_ear, head_yaw, head_pitch, head_roll,
                head_tx, head_ty, head_tz
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        
        records = []
        for s in dataset:
            records.append((
                session_id,
                time.time(),
                s["screen_x"],
                s["screen_y"],
                s["left_ratio_x"],
                s["left_ratio_y"],
                s["right_ratio_x"],
                s["right_ratio_y"],
                s["left_ear"],
                s["right_ear"],
                s["head_yaw"],
                s["head_pitch"],
                s["head_roll"],
                s["head_tx"],
                s["head_ty"],
                s["head_tz"]
            ))

        cursor.executemany(insert_query, records)
        conn.commit()
        conn.close()
        print(f"[Database] Saved {len(dataset)} records to database under session: {session_id}")

    def load_latest_session(self):
        """
        Loads all records for the most recent session_id.
        Returns a list of tuples and column names.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get the latest session_id
        cursor.execute("SELECT MAX(session_id) FROM calibration_data")
        latest_session_id = cursor.fetchone()[0]
        
        if latest_session_id is None:
            conn.close()
            return [], []

        cursor.execute("""
            SELECT 
                screen_x, screen_y,
                left_ratio_x, left_ratio_y, right_ratio_x, right_ratio_y,
                left_ear, right_ear, head_yaw, head_pitch, head_roll,
                head_tx, head_ty, head_tz
            FROM calibration_data 
            WHERE session_id = ?
        """, (latest_session_id,))
        
        records = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]
        
        conn.close()
        return records, columns
