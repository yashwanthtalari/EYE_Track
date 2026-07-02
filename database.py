import sqlite3
import os
import time

class CalibrationDatabase:
    """
    Manages SQLite database storage for the Gaze Calibration "Golden Dataset".
    Allows saving highly detailed, multi-pass gaze tracking samples and loading
    them for training and analysis.
    """
    def __init__(self, db_dir="data", db_filename="calibration_golden.db"):
        self.db_dir = db_dir
        self.db_path = os.path.join(db_dir, db_filename)
        os.makedirs(self.db_dir, exist_ok=True)
        self._init_db()

    def _init_db(self):
        """Initializes database schema with the 'Golden Dataset' structure."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS golden_calibration_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                timestamp REAL,
                calibration_pass TEXT, -- 'coarse', 'test', 'refine'
                screen_x INTEGER,
                screen_y INTEGER,
                
                -- Left Eye Data
                left_ratio_x REAL,
                left_ratio_y REAL,
                left_ear REAL, -- openness
                left_confidence REAL,
                
                -- Right Eye Data
                right_ratio_x REAL,
                right_ratio_y REAL,
                right_ear REAL, -- openness
                right_confidence REAL,
                
                -- Head Pose Data
                head_yaw REAL,
                head_pitch REAL,
                head_roll REAL,
                head_tx REAL,
                head_ty REAL,
                head_tz REAL,
                head_confidence REAL,
                
                -- Face Geometry Data
                face_width REAL,
                face_height REAL,
                face_center_x REAL,
                face_center_y REAL
            )
        """)
        conn.commit()
        conn.close()

    def save_session_samples(self, session_id, dataset):
        """
        Saves a batch of rich calibration samples to the database under session_id.
        """
        if not dataset:
            return

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        insert_query = """
            INSERT INTO golden_calibration_data (
                session_id, timestamp, calibration_pass, screen_x, screen_y,
                left_ratio_x, left_ratio_y, left_ear, left_confidence,
                right_ratio_x, right_ratio_y, right_ear, right_confidence,
                head_yaw, head_pitch, head_roll, head_tx, head_ty, head_tz, head_confidence,
                face_width, face_height, face_center_x, face_center_y
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        
        records = []
        for s in dataset:
            records.append((
                session_id,
                time.time(),
                s.get("calibration_pass", "coarse"),
                s["screen_x"],
                s["screen_y"],
                s["left_ratio_x"],
                s["left_ratio_y"],
                s.get("left_ear", 0.0),
                s.get("left_confidence", 1.0),
                s["right_ratio_x"],
                s["right_ratio_y"],
                s.get("right_ear", 0.0),
                s.get("right_confidence", 1.0),
                s["head_yaw"],
                s["head_pitch"],
                s["head_roll"],
                s["head_tx"],
                s["head_ty"],
                s["head_tz"],
                s.get("head_confidence", 1.0),
                s.get("face_width", 0.0),
                s.get("face_height", 0.0),
                s.get("face_center_x", 0.0),
                s.get("face_center_y", 0.0)
            ))

        cursor.executemany(insert_query, records)
        conn.commit()
        conn.close()
        print(f"[Database] Saved {len(dataset)} Golden records to database under session: {session_id}")

    def load_latest_session(self, include_test_pass=False):
        """
        Loads records for the most recent session_id.
        Optional: can exclude the 'test' pass records to avoid training on validation data.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get latest session
        cursor.execute("SELECT MAX(session_id) FROM golden_calibration_data")
        latest_session_id = cursor.fetchone()[0]
        
        if latest_session_id is None:
            conn.close()
            return [], []

        query = """
            SELECT 
                screen_x, screen_y,
                left_ratio_x, left_ratio_y, right_ratio_x, right_ratio_y,
                left_ear, right_ear, head_yaw, head_pitch, head_roll,
                head_tx, head_ty, head_tz, face_width, face_height,
                face_center_x, face_center_y
            FROM golden_calibration_data 
            WHERE session_id = ?
        """
        
        if not include_test_pass:
            query += " AND calibration_pass != 'test'"
            
        cursor.execute(query, (latest_session_id,))
        records = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]
        
        conn.close()
        return records, columns
