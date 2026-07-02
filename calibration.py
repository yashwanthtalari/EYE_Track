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
from regression import GazeRegressor

class GazeCalibrator:
    """
    Manages the Multi-Pass gaze calibration process using OpenCV full-screen visualization.
    - Pass 1 (Coarse): Calibrates 5 grid points to build an initial mapping.
    - Pass 2 (Verification): Measures Euclidean prediction errors at 4 test positions.
    - Pass 3 (Adaptive Refinement): Prompts the user to calibrate extra points in high-error zones.
    - Final Evaluation: Computes and reports overall screen accuracy statistics.
    Logs rich 'Golden Dataset' outputs directly to SQLite.
    """
    def __init__(self, data_dir="data", db_filename="calibration_golden.db"):
        self.db = CalibrationDatabase(db_dir=data_dir, db_filename=db_filename)
        self.screen_w, self.screen_h = pyautogui.size()

        # State machine settings
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

    def _is_gaze_stable(self, threshold=0.012):
        if len(self.ratio_history_x) < self.stability_window:
            return False
        std_x = np.std(self.ratio_history_x)
        std_y = np.std(self.ratio_history_y)
        return std_x < threshold and std_y < threshold

    def run_calibration(self, camera, detector, estimator):
        """
        Runs the multi-pass calibration workflow.
        """
        session_id = f"session_{int(time.time())}"
        
        # -------------------------------------------------------------
        # PASS 1: COARSE CALIBRATION
        # -------------------------------------------------------------
        coarse_points = [
            (0.1, 0.1), (0.9, 0.1), (0.5, 0.5), (0.1, 0.9), (0.9, 0.9)
        ]
        print(f"[Calibration] Starting Pass 1 (Coarse Mapping) on {len(coarse_points)} points...")
        success = self._run_calibration_pass(
            camera, detector, estimator, session_id,
            points=coarse_points, pass_name="coarse",
            instructions="PASS 1/3 (Coarse): Look at the RED dot, hold steady, and press SPACE.",
            samples_per_point=40, save_to_db=True
        )
        if not success:
            return False

        # Train initial coarse model to evaluate verification points
        coarse_regressor = GazeRegressor(db_filename="calibration_golden.db")
        trained_coarse = coarse_regressor.train_from_db()
        if not trained_coarse:
            print("[Calibration] Failed to train initial coarse model.")
            return False

        # -------------------------------------------------------------
        # PASS 2: VERIFICATION & ERROR TESTING
        # -------------------------------------------------------------
        test_points = [
            (0.5, 0.1),  # Top-Center
            (0.1, 0.5),  # Middle-Left
            (0.9, 0.5),  # Middle-Right
            (0.5, 0.9)   # Bottom-Center
        ]
        print(f"[Calibration] Starting Pass 2 (Verification Testing) on {len(test_points)} points...")
        
        # We run the verification pass, capturing test features and checking coarse prediction error
        test_success, verification_results = self._run_verification_pass(
            camera, detector, estimator, session_id,
            points=test_points, regressor=coarse_regressor
        )
        if not test_success:
            return False

        # Determine if any region has error above threshold (e.g. 100 pixels)
        refinement_points = []
        error_threshold_px = 100.0
        
        # Map high-error verification targets to adaptive refinement targets
        refinement_mapping = {
            (0.5, 0.1): [(0.5, 0.3)],  # Top-Center error -> Add Top refinement dot
            (0.1, 0.5): [(0.3, 0.5)],  # Left error -> Add Left refinement dot
            (0.9, 0.5): [(0.7, 0.5)],  # Right error -> Add Right refinement dot
            (0.5, 0.9): [(0.5, 0.7)]   # Bottom error -> Add Bottom refinement dot
        }

        for pt, avg_err in verification_results.items():
            if avg_err > error_threshold_px:
                print(f"[Calibration] High prediction error at {pt}: {avg_err:.1f}px (Threshold: {error_threshold_px}px)")
                refinement_points.extend(refinement_mapping[pt])

        # Remove duplicate refinement points (if any)
        refinement_points = list(set(refinement_points))

        # -------------------------------------------------------------
        # PASS 3: ADAPTIVE REFINEMENT (IF REQUIRED)
        # -------------------------------------------------------------
        if refinement_points:
            print(f"[Calibration] Starting Pass 3 (Adaptive Refinement) on {len(refinement_points)} high-error zones...")
            success = self._run_calibration_pass(
                camera, detector, estimator, session_id,
                points=refinement_points, pass_name="refine",
                instructions="PASS 3/3 (Refinement): Look at the YELLOW dot, hold steady, and press SPACE.",
                samples_per_point=40, save_to_db=True
            )
            if not success:
                return False
        else:
            print("[Calibration] Pass 2 verification succeeded with high accuracy. Skipping Pass 3.")

        # Show final verification results
        self._display_final_report(verification_results)
        return True

    def _run_calibration_pass(self, camera, detector, estimator, session_id, points, pass_name, instructions, samples_per_point, save_to_db):
        """Helper to run a standard calibration pass collecting gaze samples."""
        window_name = "VisionPoint Gaze Calibration"
        current_point_idx = 0
        state = "idle"
        captured_samples = 0
        current_point_dataset = []

        feedback_msg = ""
        feedback_color = (255, 255, 255)
        last_feedback_time = 0

        while current_point_idx < len(points):
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

            ui_frame = np.zeros((self.screen_h, self.screen_w, 3), dtype=np.uint8)

            # Draw targets and texts
            if state == "idle":
                inst_line1 = instructions
                inst_line2 = "Keep your gaze steady on the target."
            elif state == "stabilizing":
                inst_line1 = "Focusing... hold your gaze steady on the YELLOW dot."
                inst_line2 = "Analyzing eye movement stability..."
            elif state == "capturing":
                inst_line1 = f"Recording calibration data ({pass_name} pass)..."
                inst_line2 = f"Captured: {captured_samples} / {samples_per_point}"

            cv2.putText(ui_frame, inst_line1, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(ui_frame, inst_line2, (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2, cv2.LINE_AA)
            cv2.putText(ui_frame, f"Point {current_point_idx + 1} of {len(points)}", (50, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 250, 150), 2, cv2.LINE_AA)

            if time.time() - last_feedback_time < 2.0:
                cv2.putText(ui_frame, feedback_msg, (50, self.screen_h - 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, feedback_color, 2, cv2.LINE_AA)

            # Draw Target Dot
            target_norm_x, target_norm_y = points[current_point_idx]
            target_px_x = int(target_norm_x * self.screen_w)
            target_px_y = int(target_norm_y * self.screen_h)

            if state == "idle":
                dot_color = (0, 0, 255)       # Red
            elif state == "stabilizing":
                dot_color = (0, 255, 255)     # Yellow
            elif state == "capturing":
                dot_color = (255, 0, 0)       # Blue

            cv2.circle(ui_frame, (target_px_x, target_px_y), 24, dot_color, 2)
            cv2.circle(ui_frame, (target_px_x, target_px_y), 8, dot_color, -1)

            if landmarks is not None and eye_features is not None:
                r_x, r_y, r_ear = EyeGeometry.calculate_ratios(eye_features["right_eye"], is_left_eye=False)
                l_x, l_y, l_ear = EyeGeometry.calculate_ratios(eye_features["left_eye"], is_left_eye=True)
                avg_rx = (r_x + l_x) / 2.0
                avg_ry = (r_y + l_y) / 2.0

                self._add_to_stability_buffer(avg_rx, avg_ry)
                is_blinking = (r_ear < 0.18 or l_ear < 0.18)

                # Face boundaries for golden dataset
                x_coords = landmarks[:, 0]
                y_coords = landmarks[:, 1]
                min_x, max_x = np.min(x_coords), np.max(x_coords)
                min_y, max_y = np.min(y_coords), np.max(y_coords)
                face_w = max_x - min_x
                face_h = max_y - min_y
                face_cx = (max_x + min_x) / 2.0
                face_cy = (max_y + min_y) / 2.0

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
                        feedback_msg = "Blink detected! Stabilizing..."
                        feedback_color = (0, 165, 255)
                        last_feedback_time = time.time()
                    else:
                        if tvec is not None:
                            tx, ty, tz = float(tvec[0][0]), float(tvec[1][0]), float(tvec[2][0])
                        else:
                            tx, ty, tz = 0.0, 0.0, 0.0

                        current_point_dataset.append({
                            "calibration_pass": pass_name,
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
                            "head_tz": tz,
                            "face_width": face_w,
                            "face_height": face_h,
                            "face_center_x": face_cx,
                            "face_center_y": face_cy
                        })
                        captured_samples += 1
                        time.sleep(0.05)

                        if captured_samples >= samples_per_point:
                            # Validate variance
                            avg_x_vals = [(s["left_ratio_x"] + s["right_ratio_x"])/2.0 for s in current_point_dataset]
                            avg_y_vals = [(s["left_ratio_y"] + s["right_ratio_y"])/2.0 for s in current_point_dataset]
                            std_cx = np.std(avg_x_vals)
                            std_cy = np.std(avg_y_vals)

                            if std_cx > 0.018 or std_cy > 0.018:
                                state = "stabilizing"
                                captured_samples = 0
                                current_point_dataset.clear()
                                self._reset_stability_buffer()
                                feedback_msg = "Movement detected! Retaking..."
                                feedback_color = (0, 0, 255)
                                last_feedback_time = time.time()
                            else:
                                if save_to_db:
                                    self.db.save_session_samples(session_id, current_point_dataset)
                                current_point_idx += 1
                                state = "idle"
                                self._reset_stability_buffer()
                                feedback_msg = f"Target {current_point_idx} captured!"
                                feedback_color = (0, 255, 0)
                                last_feedback_time = time.time()
                                time.sleep(0.4)

            cv2.imshow(window_name, ui_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                cv2.destroyWindow(window_name)
                return False
            elif key == 32:
                if state == "idle":
                    state = "stabilizing"
                    self._reset_stability_buffer()

        cv2.destroyWindow(window_name)
        return True

    def _run_verification_pass(self, camera, detector, estimator, session_id, points, regressor):
        """Runs the verification pass and measures target coordinate prediction errors."""
        window_name = "VisionPoint Gaze Calibration"
        current_point_idx = 0
        state = "idle"
        captured_samples = 0
        current_point_dataset = []
        verification_results = {}

        feedback_msg = ""
        feedback_color = (255, 255, 255)
        last_feedback_time = 0

        # Number of testing verification samples
        test_samples = 15

        while current_point_idx < len(points):
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

            ui_frame = np.zeros((self.screen_h, self.screen_w, 3), dtype=np.uint8)

            if state == "idle":
                inst_line1 = "PASS 2/3 (Verification): Look at the GREEN dot and press SPACE."
                inst_line2 = "The system will check accuracy on intermediate points."
            elif state == "stabilizing":
                inst_line1 = "Focusing... hold steady on the YELLOW dot."
                inst_line2 = "Stabilizing gaze..."
            elif state == "capturing":
                inst_line1 = "Evaluating coordinate prediction accuracy..."
                inst_line2 = f"Testing sample: {captured_samples} / {test_samples}"

            cv2.putText(ui_frame, inst_line1, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(ui_frame, inst_line2, (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2, cv2.LINE_AA)
            cv2.putText(ui_frame, f"Test Target {current_point_idx + 1} of {len(points)}", (50, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (250, 150, 250), 2, cv2.LINE_AA)

            if time.time() - last_feedback_time < 2.0:
                cv2.putText(ui_frame, feedback_msg, (50, self.screen_h - 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, feedback_color, 2, cv2.LINE_AA)

            # Draw target dot (Green for testing pass)
            target_norm_x, target_norm_y = points[current_point_idx]
            target_px_x = int(target_norm_x * self.screen_w)
            target_px_y = int(target_norm_y * self.screen_h)

            if state == "idle":
                dot_color = (0, 255, 0)       # Green
            elif state == "stabilizing":
                dot_color = (0, 255, 255)     # Yellow
            elif state == "capturing":
                dot_color = (255, 255, 0)     # Cyan

            cv2.circle(ui_frame, (target_px_x, target_px_y), 24, dot_color, 2)
            cv2.circle(ui_frame, (target_px_x, target_px_y), 8, dot_color, -1)

            if landmarks is not None and eye_features is not None:
                r_x, r_y, r_ear = EyeGeometry.calculate_ratios(eye_features["right_eye"], is_left_eye=False)
                l_x, l_y, l_ear = EyeGeometry.calculate_ratios(eye_features["left_eye"], is_left_eye=True)
                avg_rx = (r_x + l_x) / 2.0
                avg_ry = (r_y + l_y) / 2.0

                self._add_to_stability_buffer(avg_rx, avg_ry)
                is_blinking = (r_ear < 0.18 or l_ear < 0.18)

                x_coords = landmarks[:, 0]
                y_coords = landmarks[:, 1]
                min_x, max_x = np.min(x_coords), np.max(x_coords)
                min_y, max_y = np.min(y_coords), np.max(y_coords)
                face_w = max_x - min_x
                face_h = max_y - min_y
                face_cx = (max_x + min_x) / 2.0
                face_cy = (max_y + min_y) / 2.0

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
                        feedback_msg = "Blink detected! Stabilizing..."
                        feedback_color = (0, 165, 255)
                        last_feedback_time = time.time()
                    else:
                        if tvec is not None:
                            tx, ty, tz = float(tvec[0][0]), float(tvec[1][0]), float(tvec[2][0])
                        else:
                            tx, ty, tz = 0.0, 0.0, 0.0

                        current_point_dataset.append({
                            "calibration_pass": "test",
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
                            "head_tz": tz,
                            "face_width": face_w,
                            "face_height": face_h,
                            "face_center_x": face_cx,
                            "face_center_y": face_cy
                        })
                        captured_samples += 1
                        time.sleep(0.04)

                        if captured_samples >= test_samples:
                            # Evaluate Euclidean error on the captured test features
                            errors = []
                            for s in current_point_dataset:
                                pred_x, pred_y = regressor.predict(
                                    s["left_ratio_x"], s["left_ratio_y"],
                                    s["right_ratio_x"], s["right_ratio_y"],
                                    s["head_yaw"], s["head_pitch"], s["head_roll"],
                                    s["head_tx"], s["head_ty"], s["head_tz"],
                                    s["face_width"], s["face_height"],
                                    s["face_center_x"], s["face_center_y"]
                                )
                                dist = np.sqrt((pred_x - target_px_x)**2 + (pred_y - target_px_y)**2)
                                errors.append(dist)

                            avg_error = np.mean(errors)
                            verification_results[points[current_point_idx]] = avg_error

                            # Save test records for validation logging
                            self.db.save_session_samples(session_id, current_point_dataset)

                            current_point_idx += 1
                            state = "idle"
                            self._reset_stability_buffer()
                            feedback_msg = f"Test point calculated. Error: {avg_error:.1f}px"
                            feedback_color = (150, 255, 150)
                            last_feedback_time = time.time()
                            time.sleep(0.4)

            cv2.imshow(window_name, ui_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                cv2.destroyWindow(window_name)
                return False, {}
            elif key == 32:
                if state == "idle":
                    state = "stabilizing"
                    self._reset_stability_buffer()

        cv2.destroyWindow(window_name)
        return True, verification_results

    def _display_final_report(self, verification_results):
        """Displays final accuracy analytics report on the screen."""
        window_name = "VisionPoint - Calibration Complete"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        avg_errors = list(verification_results.values())
        mean_err = np.mean(avg_errors)
        
        # Mapping mean error to a raw accuracy score (e.g. 150px error = 85% accuracy)
        max_acceptable_error = 300.0
        accuracy_score = max(0.0, min(100.0, 100.0 - (mean_err / max_acceptable_error) * 50.0))

        while True:
            ui_frame = np.zeros((self.screen_h, self.screen_w, 3), dtype=np.uint8)

            # Draw report layout
            cv2.putText(ui_frame, "VISIONPOINT CALIBRATION REPORT", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (100, 255, 100), 3, cv2.LINE_AA)
            cv2.putText(ui_frame, "==============================", (50, 140), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (150, 150, 150), 2, cv2.LINE_AA)

            cv2.putText(ui_frame, f"Gaze Accuracy Score: {accuracy_score:.1f}%", (50, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(ui_frame, f"Mean Gaze Coordinate Deviation: {mean_err:.1f} pixels", (50, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

            # Detail points
            y_offset = 350
            cv2.putText(ui_frame, "Verification Details:", (50, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2, cv2.LINE_AA)
            
            for pt, err in verification_results.items():
                y_offset += 40
                quadrant_name = "Region"
                if pt == (0.5, 0.1): quadrant_name = "Top-Center"
                elif pt == (0.1, 0.5): quadrant_name = "Middle-Left"
                elif pt == (0.9, 0.5): quadrant_name = "Middle-Right"
                elif pt == (0.5, 0.9): quadrant_name = "Bottom-Center"
                
                status = "PASS" if err < 120 else "ADAPTIVE REFINEMENT APPLIED"
                color = (0, 255, 0) if err < 120 else (0, 255, 255)
                
                cv2.putText(ui_frame, f"  * {quadrant_name}: error = {err:.1f}px ({status})", (50, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 1, cv2.LINE_AA)

            cv2.putText(ui_frame, "Press SPACE to save models and begin real-time eye tracking.", (50, self.screen_h - 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

            cv2.imshow(window_name, ui_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 32: # SPACE
                break

        cv2.destroyWindow(window_name)
