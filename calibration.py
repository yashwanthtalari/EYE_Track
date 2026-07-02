import cv2
import numpy as np
import time
import os
import pyautogui

from camera import VideoCamera
from detector import FaceMeshDetector
from iris import EyeExtractor
from geometry import EyeGeometry
from head_pose import HeadPoseEstimator
from database import CalibrationDatabase

class GazeCalibrator:
    """
    Manages the 9-point calibration process using OpenCV full-screen visualization.
    Uses a state machine to enforce gaze stabilization before capture and analyzes
    movement variance post-capture. Stores rich datasets (eyes & 3D head coords) in SQLite.
    """
    def __init__(self, data_dir="data", db_filename="calibration.db"):
        self.db = CalibrationDatabase(db_dir=data_dir, db_filename=db_filename)
        
        # Grid positions (normalized coordinates 0.0 to 1.0)
        self.grid_coords = [
            (0.1, 0.1), (0.5, 0.1), (0.9, 0.1),
            (0.1, 0.5), (0.5, 0.5), (0.9, 0.5),
            (0.1, 0.9), (0.5, 0.9), (0.9, 0.9)
        ]
        
        # Screen dimensions
        self.screen_w, self.screen_h = pyautogui.size()

        # Gaze stability tracking history
        self.stability_window = 10
        self.ratio_history_x = []
        self.ratio_history_y = []

    def _reset_stability_buffer(self):
        self.ratio_history_x.clear()
        self.ratio_history_y.clear()

    def _add_to_stability_buffer(self, rx, ry):
        self.ratio_history_x.append(rx)
        self.ratio_history_y.append(ry)
        if len(self.ratio_history_x) > self.stability_window:
            self.ratio_history_x.pop(0)
            self.ratio_history_y.pop(0)

    def _is_gaze_stable(self, threshold=0.010):
        """
        Returns True if standard deviation of ratio histories is below the threshold.
        """
        if len(self.ratio_history_x) < self.stability_window:
            return False
        
        std_x = np.std(self.ratio_history_x)
        std_y = np.std(self.ratio_history_y)
        
        return std_x < threshold and std_y < threshold

    def run_calibration(self, camera, detector, estimator):
        """
        Runs the state-machine calibration GUI.
        Captures 40 samples per point slowly (20 Hz) for maximum accuracy.
        """
        window_name = "VisionPoint Gaze Calibration"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        dataset = []
        current_point_idx = 0
        samples_per_point = 40  # Increased for higher training volume & accuracy
        
        # Calibration states: "idle", "stabilizing", "capturing"
        state = "idle"
        captured_samples = 0
        current_point_dataset = []
        
        feedback_msg = ""
        feedback_color = (255, 255, 255)
        last_feedback_time = 0
        
        print("[Calibration] Press ESC to abort, Space to start point calibration.")

        while current_point_idx < len(self.grid_coords):
            success, bgr_frame, rgb_frame = camera.read_frame()
            if not success:
                continue

            landmarks = detector.find_face_landmarks(rgb_frame)
            pixel_landmarks = detector.get_pixel_landmarks(landmarks, camera.target_width, camera.target_height)

            eye_features = None
            yaw, pitch, roll = 0.0, 0.0, 0.0
            rvec, tvec = None, None
            
            if landmarks is not None:
                eye_features = EyeExtractor.extract_eye_features(landmarks)
                yaw, pitch, roll, rvec, tvec = estimator.estimate_pose(pixel_landmarks, camera.target_width, camera.target_height)

            # Create blank fullscreen frame
            ui_frame = np.zeros((self.screen_h, self.screen_w, 3), dtype=np.uint8)

            # Determine instructions based on state
            if state == "idle":
                inst_line1 = "Look at the RED dot and press SPACE to begin."
                inst_line2 = "Keep your gaze steady on the target."
            elif state == "stabilizing":
                inst_line1 = "Focusing... hold your gaze steady on the YELLOW dot."
                inst_line2 = "Analyzing eye movement stability..."
            elif state == "capturing":
                inst_line1 = "Recording calibration data (BLUE dot)..."
                inst_line2 = f"Captured: {captured_samples} / {samples_per_point}"

            # Render texts
            cv2.putText(ui_frame, inst_line1, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(ui_frame, inst_line2, (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2, cv2.LINE_AA)
            cv2.putText(ui_frame, f"Point {current_point_idx + 1} of {len(self.grid_coords)}", (50, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 250, 150), 2, cv2.LINE_AA)

            # Draw temporary flashing warnings/feedbacks
            if time.time() - last_feedback_time < 2.0:
                cv2.putText(ui_frame, feedback_msg, (50, self.screen_h - 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, feedback_color, 2, cv2.LINE_AA)

            # Target position
            target_norm_x, target_norm_y = self.grid_coords[current_point_idx]
            target_px_x = int(target_norm_x * self.screen_w)
            target_px_y = int(target_norm_y * self.screen_h)

            # Select target dot color based on state
            if state == "idle":
                dot_color = (0, 0, 255)       # Red
            elif state == "stabilizing":
                dot_color = (0, 255, 255)     # Yellow/Orange
            elif state == "capturing":
                dot_color = (255, 0, 0)       # Blue

            # Draw calibration circle
            cv2.circle(ui_frame, (target_px_x, target_px_y), 24, dot_color, 2)
            cv2.circle(ui_frame, (target_px_x, target_px_y), 8, dot_color, -1)

            # State Machine calculations
            if landmarks is not None and eye_features is not None:
                r_x, r_y, r_ear = EyeGeometry.calculate_ratios(eye_features["right_eye"], is_left_eye=False)
                l_x, l_y, l_ear = EyeGeometry.calculate_ratios(eye_features["left_eye"], is_left_eye=True)
                avg_rx = (r_x + l_x) / 2.0
                avg_ry = (r_y + l_y) / 2.0

                # Feed stability buffer
                self._add_to_stability_buffer(avg_rx, avg_ry)

                # Blink detection check
                is_blinking = (r_ear < 0.18 or l_ear < 0.18)

                if state == "stabilizing":
                    if is_blinking:
                        self._reset_stability_buffer()
                    elif self._is_gaze_stable(threshold=0.012):
                        state = "capturing"
                        captured_samples = 0
                        current_point_dataset.clear()

                elif state == "capturing":
                    if is_blinking:
                        state = "stabilizing"
                        captured_samples = 0
                        current_point_dataset.clear()
                        self._reset_stability_buffer()
                        feedback_msg = "Blink detected! Holding target to stabilize..."
                        feedback_color = (0, 165, 255)
                        last_feedback_time = time.time()
                    else:
                        # Extract 3D head position coordinates (tvec)
                        if tvec is not None:
                            tx = float(tvec[0][0])
                            ty = float(tvec[1][0])
                            tz = float(tvec[2][0])
                        else:
                            tx, ty, tz = 0.0, 0.0, 0.0

                        # Record rich gaze and head position coordinates
                        current_point_dataset.append({
                            "screen_x": target_px_x,
                            "screen_y": target_px_y,
                            "left_ratio_x": l_x,
                            "left_ratio_y": l_y,
                            "right_ratio_x": r_x,
                            "right_ratio_y": r_y,
                            "left_ear": l_ear,
                            "right_ear": r_ear,
                            "head_yaw": yaw,
                            "head_pitch": pitch,
                            "head_roll": roll,
                            "head_tx": tx,
                            "head_ty": ty,
                            "head_tz": tz
                        })
                        captured_samples += 1
                        
                        # Capturing slowly (50ms interval = 20 Hz) for maximum accuracy
                        time.sleep(0.05) 

                        if captured_samples >= samples_per_point:
                            # Verify eye ratio variance during capture
                            avg_x_vals = [(s["left_ratio_x"] + s["right_ratio_x"])/2.0 for s in current_point_dataset]
                            avg_y_vals = [(s["left_ratio_y"] + s["right_ratio_y"])/2.0 for s in current_point_dataset]
                            
                            std_cx = np.std(avg_x_vals)
                            std_cy = np.std(avg_y_vals)

                            if std_cx > 0.018 or std_cy > 0.018:
                                # High variance = movement detected! Recapture.
                                state = "stabilizing"
                                captured_samples = 0
                                current_point_dataset.clear()
                                self._reset_stability_buffer()
                                feedback_msg = "Movement detected! Retaking calibration..."
                                feedback_color = (0, 0, 255)
                                last_feedback_time = time.time()
                                print(f"[Calibration] Capture failed due to eye movement (std_x={std_cx:.3f}, std_y={std_cy:.3f}). Retrying...")
                            else:
                                # Successful capture! Add to master dataset
                                dataset.extend(current_point_dataset)
                                current_point_idx += 1
                                state = "idle"
                                self._reset_stability_buffer()
                                feedback_msg = f"Point {current_point_idx} captured successfully!"
                                feedback_color = (0, 255, 0)
                                last_feedback_time = time.time()
                                time.sleep(0.4) # Pause before next dot

            cv2.imshow(window_name, ui_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == 27: # ESC
                print("[Calibration] Aborted by user.")
                cv2.destroyWindow(window_name)
                return False
            elif key == 32: # SPACE
                if state == "idle":
                    state = "stabilizing"
                    self._reset_stability_buffer()

        cv2.destroyWindow(window_name)

        # Save session to SQLite database
        if dataset:
            session_id = f"session_{int(time.time())}"
            self.db.save_session(session_id, dataset)
            return True
        else:
            print("[Calibration] No data captured.")
            return False
