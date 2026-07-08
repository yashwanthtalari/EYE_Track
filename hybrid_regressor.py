import os
import pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler

from database import CalibrationDatabase

class GazeHybridRegressor:
    """
    High-accuracy Gaze Regressor utilizing a hybrid feature vector:
      - 12 geometric features (iris ratios, EAR, and head pose)
      - 1152 eye-strip pixel features (flattened grayscale eye patches)
    Trains a robust Ridge regression (or SVR) model online from database sessions.
    """
    def __init__(self, models_dir="models", filename="gaze_hybrid_online.pkl",
                 db_dir="data", db_filename="calibration_golden.db",
                 model_type="ridge"):
        self.models_dir = models_dir
        self.filepath = os.path.join(models_dir, filename)
        self.db = CalibrationDatabase(db_dir=db_dir, db_filename=db_filename)
        self.model_type = model_type
        
        self.model = None
        self.scaler = None
        
        os.makedirs(self.models_dir, exist_ok=True)

    def _build_features(self, df):
        """Reconstruct geometric and eye-patch features from database DataFrame."""
        X_geom = []
        X_eyes = []
        
        for row in df.itertuples(index=False):
            # 1. Reconstruct the 12 geometric features
            lh, lv, lear = row.left_ratio_x, row.left_ratio_y, row.left_ear
            rh, rv, rear = row.right_ratio_x, row.right_ratio_y, row.right_ear
            yaw, pitch, roll = row.head_yaw, row.head_pitch, row.head_roll
            
            geom = np.array([
                lh, lv, lear,
                rh, rv, rear,
                (lh + rh) * 0.5, (lv + rv) * 0.5,  # combined gaze
                (lh - rh),                          # vergence/asymmetry
                yaw / 90.0, pitch / 90.0, roll / 90.0  # head rotation
            ], dtype=np.float32)
            
            # 2. Extract deserialized eye features
            eyes = row.eye_features
            if eyes is None or len(eyes) != 1152:
                # Fallback to zero vector if empty or mismatch
                eyes = np.zeros(1152, dtype=np.float32)
                
            X_geom.append(geom)
            X_eyes.append(eyes)
            
        X_geom = np.array(X_geom, dtype=np.float32)
        X_eyes = np.array(X_eyes, dtype=np.float32)
        
        # Concatenate geometric features and eye-patch pixels
        return np.concatenate([X_geom, X_eyes], axis=1)

    def train_from_db(self, screen_w=1920, screen_h=1080, verbose=True):
        """Loads latest calibration session and trains Ridge or SVR model."""
        records, columns = self.db.load_latest_session(include_test_pass=False)
        if not records:
            print("[Hybrid Regressor] No calibration records found in database. Cannot train.")
            return False
            
        df = pd.DataFrame(records, columns=columns)
        if len(df) < 10:
            print(f"[Hybrid Regressor] Too few records ({len(df)}) to train model.")
            return False
            
        X = self._build_features(df)
        
        # Normalize target screen pixel coordinates to [0.0, 1.0] relative to screen dimensions
        Y = np.column_stack([
            df["screen_x"].values.astype(np.float64) / screen_w,
            df["screen_y"].values.astype(np.float64) / screen_h
        ])
        
        # Fit scaler
        self.scaler = StandardScaler().fit(X)
        Xs = self.scaler.transform(X)
        
        # Fit regression model
        if self.model_type == "ridge":
            self.model = MultiOutputRegressor(Ridge(alpha=5.0))
        elif self.model_type == "svr":
            self.model = MultiOutputRegressor(SVR(C=5.0, epsilon=0.01))
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")
            
        self.model.fit(Xs, Y)
        
        # Evaluate training error in pixels
        pred_norm = self.model.predict(Xs)
        pred_px = np.column_stack([pred_norm[:, 0] * screen_w, pred_norm[:, 1] * screen_h])
        true_px = np.column_stack([df["screen_x"].values, df["screen_y"].values])
        
        errors = np.sqrt(((true_px - pred_px) ** 2).sum(axis=1))
        
        if verbose:
            print(f"[Hybrid Regressor] Trained {self.model_type.upper()} model on {len(df)} samples.")
            print(f"[Hybrid Regressor] Training Pixel Error: mean={errors.mean():.1f}px, median={np.median(errors):.1f}px, max={errors.max():.1f}px")
            
        self.save_model()
        return True

    def save_model(self):
        """Saves current trained models and scalers to pickle file."""
        payload = {
            "model_type": self.model_type,
            "model": self.model,
            "scaler": self.scaler
        }
        with open(self.filepath, "wb") as f:
            pickle.dump(payload, f)
        print(f"[Hybrid Regressor] Saved models to {self.filepath}")

    def load_model(self):
        """Loads trained models from pickle file. Returns True if successful."""
        if not os.path.exists(self.filepath):
            print(f"[Hybrid Regressor] Model file {self.filepath} not found.")
            return False
            
        with open(self.filepath, "rb") as f:
            payload = pickle.load(f)
            
        self.model_type = payload["model_type"]
        self.model = payload["model"]
        self.scaler = payload["scaler"]
        
        print(f"[Hybrid Regressor] Model loaded successfully: {self.model_type.upper()}")
        return True

    def predict(self, geom, eyes):
        """
        Predicts screen normalized coordinates (screen_x_norm, screen_y_norm) in [0, 1].
        """
        if self.model is None or self.scaler is None:
            raise ValueError("[Hybrid Regressor] Model is not loaded or trained yet.")
            
        # Concatenate features into single row
        row = np.concatenate([geom, eyes]).reshape(1, -1)
        row_s = self.scaler.transform(row)
        
        # Predict normalized screen values
        pred = self.model.predict(row_s)[0]
        
        # Clip to screen boundary envelope [0, 1]
        nx = float(np.clip(pred[0], 0.0, 1.0))
        ny = float(np.clip(pred[1], 0.0, 1.0))
        
        return nx, ny
