"""
Live gaze prediction using the hybrid model trained by train_hybrid.py.

Runs the webcam through the EXACT same pipeline as training:
    frame -> crop_face -> FaceMesh on crop -> geometric + eye-strip features
    -> StandardScaler -> regressor -> normalized (x, y) gaze point.

The point is smoothed (EMA) and drawn full-screen so you can see where the
model thinks you're looking.

Usage:
    python predict_live.py
    python predict_live.py --model models/gaze_hybrid_model.pkl --smooth 0.4

Controls:
    ESC : quit
"""
import cv2
import numpy as np
import argparse
import pickle
import os

from camera import VideoCamera
from detector import FaceMeshDetector
from collect_data import crop_face
from gaze_features import extract_from_crop


def parse_args():
    p = argparse.ArgumentParser(description="Live gaze prediction from the trained hybrid model.")
    p.add_argument("--model", default="models/gaze_hybrid_model.pkl", help="Trained model pickle")
    p.add_argument("--device", type=int, default=0, help="Camera device index")
    p.add_argument("--smooth", type=float, default=0.35,
                   help="EMA smoothing 0..1; lower = smoother/laggier (default: 0.35)")
    p.add_argument("--crop-pad", type=float, default=0.4, help="Face crop padding (match collect_data)")
    p.add_argument("--crop-size", type=int, default=224, help="Face crop size (match collect_data)")
    return p.parse_args()


def assemble_features(geom, eyes, feats):
    """Build the model input row from the two feature blocks, per the saved config."""
    if feats == "geom":
        return geom
    if feats == "eyes":
        return eyes
    return np.concatenate([geom, eyes])


def main():
    args = parse_args()
    if not os.path.exists(args.model):
        print(f"[Live] Model not found: {args.model}. Train it with train_hybrid.py first.")
        return

    with open(args.model, "rb") as f:
        payload = pickle.load(f)
    model = payload["model"]
    scaler = payload["scaler"]
    feats = payload["feats"]
    eye_size = tuple(payload["eye_size"])
    screen_w = payload.get("screen_w", 1920)
    screen_h = payload.get("screen_h", 1080)
    print(f"[Live] Loaded {payload.get('model_type','model')} | feats={feats} | "
          f"eye_size={eye_size} | target screen {screen_w}x{screen_h}")

    camera = VideoCamera(device_index=args.device)
    detector = FaceMeshDetector()

    window_name = "VisionPoint - Live Gaze"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    smooth_x, smooth_y = 0.5, 0.5
    have_pred = False
    a = args.smooth

    while True:
        success, bgr, rgb = camera.read_frame()
        if not success:
            continue

        landmarks = detector.find_face_landmarks(rgb)
        ui = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)
        feat_ok = False

        if landmarks is not None:
            crop = crop_face(bgr, landmarks, camera.target_width, camera.target_height,
                             args.crop_pad, args.crop_size)
            if crop is not None:
                geom, eyes = extract_from_crop(crop, detector, eye_size)
                if geom is not None and eyes is not None:
                    row = assemble_features(geom, eyes, feats).reshape(1, -1)
                    nx, ny = model.predict(scaler.transform(row))[0]
                    nx = float(np.clip(nx, 0.0, 1.0))
                    ny = float(np.clip(ny, 0.0, 1.0))
                    feat_ok = True
                    if not have_pred:
                        smooth_x, smooth_y = nx, ny
                        have_pred = True
                    else:
                        smooth_x = a * nx + (1 - a) * smooth_x
                        smooth_y = a * ny + (1 - a) * smooth_y

        gx = int(smooth_x * screen_w)
        gy = int(smooth_y * screen_h)
        cv2.circle(ui, (gx, gy), 40, (0, 200, 255), 3)
        cv2.circle(ui, (gx, gy), 6, (0, 200, 255), -1)
        cv2.line(ui, (gx - 55, gy), (gx - 20, gy), (0, 200, 255), 1)
        cv2.line(ui, (gx + 20, gy), (gx + 55, gy), (0, 200, 255), 1)
        cv2.line(ui, (gx, gy - 55), (gx, gy - 20), (0, 200, 255), 1)
        cv2.line(ui, (gx, gy + 20), (gx, gy + 55), (0, 200, 255), 1)

        status = f"gaze: ({smooth_x:.2f}, {smooth_y:.2f})   FPS: {camera.get_fps():.0f}"
        cv2.putText(ui, status, (40, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (180, 180, 180), 2, cv2.LINE_AA)
        cv2.putText(ui, "Move your EYES. ESC to quit.",
                    (40, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 150, 150), 2, cv2.LINE_AA)
        if not feat_ok:
            cv2.putText(ui, "NO FACE / FEATURES", (40, screen_h - 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 165, 255), 2, cv2.LINE_AA)

        cv2.imshow(window_name, ui)
        if (cv2.waitKey(1) & 0xFF) == 27:
            break

    camera.release()
    cv2.destroyAllWindows()
    print("[Live] Stopped.")


if __name__ == "__main__":
    main()
