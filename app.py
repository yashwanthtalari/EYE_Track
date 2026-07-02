import cv2
import numpy as np
import argparse
import sys
import pyautogui
import os

from camera import VideoCamera
from detector import FaceMeshDetector
from iris import EyeExtractor
from geometry import EyeGeometry
from head_pose import HeadPoseEstimator
from calibration import GazeCalibrator
from regression import GazeRegressor
from smoothing import GazeSmoother
from overlay import CursorController

def parse_args():
    parser = argparse.ArgumentParser(description="VisionPoint - Webcam Eye Gaze Tracking Engine")
    parser.add_argument("--device", type=int, default=0, help="Camera device index (default: 0)")
    parser.add_argument("--width", type=int, default=1280, help="Camera width (default: 1280)")
    parser.add_argument("--height", type=int, default=720, help="Camera height (default: 720)")
    parser.add_argument("--alpha", type=float, default=0.15, help="Smoothing alpha (default: 0.15)")
    parser.add_argument("--no-adaptive", action="store_true", help="Disable adaptive smoothing (use fixed alpha)")
    parser.add_argument("--calibrate", action="store_true", help="Launch calibration on startup")
    return parser.parse_args()

def run_face_alignment(camera, detector, estimator):
    """
    Shows a camera preview to allow the user to align their face and verify tracking.
    Returns True to proceed to calibration, False to exit.
    """
    window_name = "VisionPoint - Align Your Face"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    print("[System] Opening face alignment preview. Please align your face in the camera frame.")
    print("[System] Press SPACE to start calibration, ESC to quit.")

    while True:
        success, bgr_frame, rgb_frame = camera.read_frame()
        if not success:
            continue

        landmarks = detector.find_face_landmarks(rgb_frame)
        pixel_landmarks = detector.get_pixel_landmarks(landmarks, camera.target_width, camera.target_height)

        # Draw visual guides
        cv2.putText(bgr_frame, "Face Alignment & Tracking Verification", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(bgr_frame, "Press SPACE to start calibration, ESC to exit", (15, camera.target_height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

        if landmarks is not None and pixel_landmarks is not None:
            # Head pose
            yaw, pitch, roll, rvec, tvec = estimator.estimate_pose(pixel_landmarks, camera.target_width, camera.target_height)
            estimator.draw_pose_axes(bgr_frame, rvec, tvec, camera.target_width, camera.target_height)

            # Extract eyes
            eye_features = EyeExtractor.extract_eye_features(landmarks)
            
            # Draw landmarks
            for eye_name, eye_data in eye_features.items():
                iris_norm = eye_data["iris_center"]
                iris_px = (int(iris_norm[0] * camera.target_width), int(iris_norm[1] * camera.target_height))
                cv2.circle(bgr_frame, iris_px, 4, (0, 255, 0), -1) # Green iris center

                for part, pt_norm in eye_data.items():
                    if part == "iris_center":
                        continue
                    pt_px = (int(pt_norm[0] * camera.target_width), int(pt_norm[1] * camera.target_height))
                    cv2.circle(bgr_frame, pt_px, 2, (0, 0, 255), -1) # Red eye corners

            cv2.putText(bgr_frame, "Status: FACE DETECTED (READY)", (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
        else:
            cv2.putText(bgr_frame, "Status: NO FACE DETECTED - Please align!", (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)

        cv2.imshow(window_name, bgr_frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27: # ESC
            cv2.destroyWindow(window_name)
            return False
        elif key == 32: # SPACE
            cv2.destroyWindow(window_name)
            return True

def main():
    args = parse_args()
    
    print("==========================================================")
    print("      VisionPoint: Webcam Gaze Estimation Engine v1.0     ")
    print("==========================================================")
    
    # Initialize components
    camera = VideoCamera(device_index=args.device, target_width=args.width, target_height=args.height)
    detector = FaceMeshDetector()
    estimator = HeadPoseEstimator()
    regressor = GazeRegressor()
    smoother = GazeSmoother(alpha=args.alpha, use_adaptive=not args.no_adaptive)
    controller = CursorController()
    calibrator = GazeCalibrator()

    # Try to load existing model
    model_loaded = regressor.load_model()
    control_mouse = False
    show_camera = True
    
    # If calibrate flag is set or no model exists, prompt for calibration
    if args.calibrate or not model_loaded:
        if not model_loaded:
            print("[System] No trained gaze regression model found. Starting calibration...")
        else:
            print("[System] --calibrate flag set. Starting calibration...")
        
        # Show alignment preview first
        proceed = run_face_alignment(camera, detector, estimator)
        if not proceed:
            print("[System] Exiting face alignment preview.")
            camera.release()
            cv2.destroyAllWindows()
            return
            
        success = calibrator.run_calibration(camera, detector, estimator)
        if success:
            print("[System] Calibration finished. Training regression model...")
            model_loaded = regressor.train_from_db()
        else:
            print("[System] Calibration aborted/failed. Gaze tracking might be unavailable.")

    print("\n--- Key Bindings in Camera Window ---")
    print("  ESC : Exit application")
    print("  C   : Recalibrate (runs 9-point calibration)")
    print("  M   : Toggle real-time cursor control (currently OFF)")
    print("  V   : Toggle show/hide camera video window")
    print("==========================================================\n")

    camera_window_name = "VisionPoint Gaze Tracking - Active"
    frame_count = 0
    
    while True:
        # 1. Read webcam frame
        success, bgr_frame, rgb_frame = camera.read_frame()
        if not success:
            print("[Error] Failed to read frame from webcam.")
            break

        frame_count += 1

        # 2. Run Face Mesh detector
        landmarks = detector.find_face_landmarks(rgb_frame)
        pixel_landmarks = detector.get_pixel_landmarks(landmarks, camera.target_width, camera.target_height)

        # Draw default interface header
        cv2.putText(bgr_frame, f"FPS: {camera.get_fps():.1f}", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
        status_text = f"Mouse Control: {'ON' if control_mouse else 'OFF'} (Press M) | Calibrate: Press C | Exit: ESC"
        cv2.putText(bgr_frame, status_text, (15, camera.target_height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

        if landmarks is not None and pixel_landmarks is not None:
            # 3. Extract eye landmarks and calculate features
            eye_features = EyeExtractor.extract_eye_features(landmarks)
            
            # Extract anatomical left & right eye ratios + blink checks
            r_ratio_x, r_ratio_y, r_ear = EyeGeometry.calculate_ratios(eye_features["right_eye"], is_left_eye=False)
            l_ratio_x, l_ratio_y, l_ear = EyeGeometry.calculate_ratios(eye_features["left_eye"], is_left_eye=True)

            # Average pupil ratios for a single simplified feature vector
            eye_ratio_x = (r_ratio_x + l_ratio_x) / 2.0
            eye_ratio_y = (r_ratio_y + l_ratio_y) / 2.0
            avg_ear = (r_ear + l_ear) / 2.0

            # 4. Estimate head pose rotation angles
            yaw, pitch, roll, rvec, tvec = estimator.estimate_pose(pixel_landmarks, camera.target_width, camera.target_height)

            # Extract 3D head translation components
            if tvec is not None:
                tx = float(tvec[0][0])
                ty = float(tvec[1][0])
                tz = float(tvec[2][0])
            else:
                tx, ty, tz = 0.0, 0.0, 0.0

            # Extract face geometry boundaries for golden feature vector
            x_coords = landmarks[:, 0]
            y_coords = landmarks[:, 1]
            min_x, max_x = np.min(x_coords), np.max(x_coords)
            min_y, max_y = np.min(y_coords), np.max(y_coords)
            face_w = max_x - min_x
            face_h = max_y - min_y
            face_cx = (max_x + min_x) / 2.0
            face_cy = (max_y + min_y) / 2.0

            # Draw visual debug overlays:
            # 3D Head Pose axes on Nose Tip
            estimator.draw_pose_axes(bgr_frame, rvec, tvec, camera.target_width, camera.target_height)

            # Draw eye contour indices & iris points on the screen BGR frame
            for eye_name, eye_data in eye_features.items():
                # Get iris point and convert to pixel coordinates
                iris_norm = eye_data["iris_center"]
                iris_px = (int(iris_norm[0] * camera.target_width), int(iris_norm[1] * camera.target_height))
                
                # Draw iris center in neon green
                cv2.circle(bgr_frame, iris_px, 3, (0, 255, 0), -1)

                # Draw eye eyelids and corners
                for part, pt_norm in eye_data.items():
                    if part == "iris_center":
                        continue
                    pt_px = (int(pt_norm[0] * camera.target_width), int(pt_norm[1] * camera.target_height))
                    cv2.circle(bgr_frame, pt_px, 2, (0, 0, 255), -1)

            # Print numerical metrics on screen for debug
            cv2.putText(bgr_frame, f"Head Pose - Yaw: {yaw:.1f} Pitch: {pitch:.1f} Z: {tz:.1f}", (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(bgr_frame, f"L Ratio - X: {l_ratio_x:.2f} Y: {l_ratio_y:.2f} | R Ratio - X: {r_ratio_x:.2f} Y: {r_ratio_y:.2f}", (15, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1, cv2.LINE_AA)

            # Throttled console logging to reduce tracing confusion (approx once every 30 frames)
            if frame_count % 30 == 0:
                print(f"[Gaze Trace] L-Ratio: ({l_ratio_x:.3f},{l_ratio_y:.3f}) | R-Ratio: ({r_ratio_x:.3f},{r_ratio_y:.3f}) | Head: Pos=({tx:.1f},{ty:.1f},{tz:.1f}) Rot=({yaw:.1f},{pitch:.1f})")

            # 5. Predict Gaze if model is trained
            if model_loaded:
                # Blink detection threshold: EAR lower than 0.18 on BOTH eyes
                # If blinking, skip prediction updating to avoid cursor glitching/jumping.
                is_blinking = (r_ear < 0.18 and l_ear < 0.18)

                if is_blinking:
                    cv2.putText(bgr_frame, "[BLINK DETECTED]", (camera.target_width // 2 - 80, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
                else:
                    try:
                        pred_x, pred_y = regressor.predict(
                            l_ratio_x, l_ratio_y,
                            r_ratio_x, r_ratio_y,
                            yaw, pitch, roll,
                            tx, ty, tz,
                            face_w, face_h, face_cx, face_cy
                        )

                        # Smooth prediction
                        smooth_x, smooth_y = smoother.smooth(pred_x, pred_y)

                        # Move physical cursor
                        if control_mouse:
                            controller.move_cursor(smooth_x, smooth_y)

                        # Draw gaze point projection on webcam preview
                        controller.draw_gaze_on_frame(bgr_frame, smooth_x, smooth_y, controller.screen_w, controller.screen_h)

                    except Exception as e:
                        cv2.putText(bgr_frame, f"Pred Error: {str(e)[:30]}", (15, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

        # 6. Render camera window (if enabled)
        if show_camera:
            cv2.imshow(camera_window_name, bgr_frame)
        else:
            # If camera window is hidden, destroy it so it does not capture cursor focus
            cv2.destroyWindow(camera_window_name)

        # 7. Keyboard control handler
        # Since waitKey is required for OpenCV windows, we run it even if show_camera is False
        key = cv2.waitKey(1) & 0xFF
        if key == 27: # ESC
            break
        elif key == ord('c') or key == ord('C'):
            print("[System] Recalibrating...")
            cv2.destroyAllWindows() # Clear existing windows
            success = calibrator.run_calibration(camera, detector, estimator)
            if success:
                model_loaded = regressor.train_from_db()
            # Recreate windows if camera is visible
            if show_camera:
                cv2.namedWindow(camera_window_name, cv2.WINDOW_NORMAL)
        elif key == ord('m') or key == ord('M'):
            control_mouse = not control_mouse
            print(f"[System] Gaze Mouse Control: {'ENABLED' if control_mouse else 'DISABLED'}")
        elif key == ord('v') or key == ord('V'):
            show_camera = not show_camera
            print(f"[System] Show Camera Feed: {show_camera}")

    # Cleanup
    print("[System] Exiting...")
    camera.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
