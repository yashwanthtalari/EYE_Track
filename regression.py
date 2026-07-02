import os
import pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from database import CalibrationDatabase

class GazeRegressor:
    """
    Handles training, saving, loading, and predicting screen gaze coordinates
    using Ridge Regression with a 2nd-degree Polynomial Feature expansion.
    Trains on the rich 14-feature Golden Dataset (eyes, 3D head pose, and face geometry).
    """
    def __init__(self, models_dir="models", filename="regression.pkl", db_dir="data", db_filename="calibration_golden.db"):
        self.models_dir = models_dir
        self.filepath = os.path.join(models_dir, filename)
        self.db = CalibrationDatabase(db_dir=db_dir, db_filename=db_filename)
        
        self.model_x = None
        self.model_y = None

        os.makedirs(self.models_dir, exist_ok=True)

    def train_from_db(self):
        """
        Loads the latest calibration session from SQLite database and trains
        separate models for X and Y coordinates using a rich 14-feature vector.
        """
        records, columns = self.db.load_latest_session(include_test_pass=False)
        if not records:
            print("[Regression] No calibration records found in SQLite database. Cannot train.")
            return False

        # Load into DataFrame
        df = pd.DataFrame(records, columns=columns)
        if len(df) < 15:
            print("[Regression] Too few database records in the latest session to train.")
            return False

        # 14 features for maximum gaze prediction accuracy:
        # Left Eye (2), Right Eye (2), Head Rotation (3), Head Translation (3), Face Dimensions & Scale (4)
        feature_cols = [
            "left_ratio_x", "left_ratio_y",
            "right_ratio_x", "right_ratio_y",
            "head_yaw", "head_pitch", "head_roll",
            "head_tx", "head_ty", "head_tz",
            "face_width", "face_height", "face_center_x", "face_center_y"
        ]

        X = df[feature_cols].values
        y_x = df["screen_x"].values
        y_y = df["screen_y"].values

        # Build pipeline: Polynomial features -> Ridge regression
        # Ridge alpha=5.0 handles high dimensionality and prevents overfitting
        self.model_x = make_pipeline(PolynomialFeatures(degree=2), Ridge(alpha=5.0))
        self.model_y = make_pipeline(PolynomialFeatures(degree=2), Ridge(alpha=5.0))

        # Train models
        self.model_x.fit(X, y_x)
        self.model_y.fit(X, y_y)

        # Evaluate performance
        score_x = self.model_x.score(X, y_x)
        score_y = self.model_y.score(X, y_y)
        print(f"[Regression] Model trained from SQLite. X R^2: {score_x:.3f}, Y R^2: {score_y:.3f}")

        # Save model
        self.save_model()
        return True

    def save_model(self):
        """Saves current trained models to pickle file."""
        with open(self.filepath, "wb") as f:
            pickle.dump((self.model_x, self.model_y), f)
        print(f"[Regression] Saved models to {self.filepath}")

    def load_model(self):
        """Loads trained models from pickle file. Returns True if successful."""
        if not os.path.exists(self.filepath):
            print(f"[Regression] Model file {self.filepath} not found.")
            return False
        
        with open(self.filepath, "rb") as f:
            self.model_x, self.model_y = pickle.load(f)
        
        print("[Regression] Models loaded successfully.")
        return True

    def predict(self, left_x, left_y, right_x, right_y, yaw, pitch, roll, tx, ty, tz, face_w, face_h, face_cx, face_cy):
        """
        Predicts screen coordinate (screen_x, screen_y) using the 14-feature vector.
        """
        if self.model_x is None or self.model_y is None:
            raise ValueError("[Regression] Models are not loaded or trained yet.")

        features = np.array([[
            left_x, left_y,
            right_x, right_y,
            yaw, pitch, roll,
            tx, ty, tz,
            face_w, face_h, face_cx, face_cy
        ]])

        pred_x = self.model_x.predict(features)[0]
        pred_y = self.model_y.predict(features)[0]

        return pred_x, pred_y
