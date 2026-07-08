"""
Live gaze prediction using the deep CNN + ANN model trained by train_cnn.py.
Processes raw webcam frames:
    frame -> crop_face -> grayscale -> equalize -> resize (32x32) -> PyTorch CNN -> normalized (x, y) -> screen coordinates.
"""
import os
import cv2
import numpy as np
import argparse
import torch
import torch.nn as nn

from camera import VideoCamera
from detector import FaceMeshDetector
from collect_data import crop_face

# Define identical model architecture
class GazeCNN(nn.Module):
    def __init__(self):
        super(GazeCNN, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 32x32 -> 16x16
            
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 16x16 -> 8x8
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)   # 8x8 -> 4x4
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )
        
    def forward(self, x):
        features = self.cnn(x)
        features = features.view(features.size(0), -1)
        return self.fc(features)

def parse_args():
    p = argparse.ArgumentParser(description="Live gaze prediction from PyTorch CNN+ANN model.")
    p.add_argument("--model", default="models/gaze_cnn_model.pth", help="Trained PyTorch model path")
    p.add_argument("--device", type=int, default=0, help="Camera device index")
    p.add_argument("--smooth", type=float, default=0.35, help="EMA smoothing factor 0..1 (lower = smoother)")
    p.add_argument("--crop-pad", type=float, default=0.4, help="Face crop padding (must match training)")
    return p.parse_args()

def main():
    args = parse_args()
    if not os.path.exists(args.model):
        print(f"[Live CNN] Error: Model file not found at {args.model}. Train it first using train_cnn.py.")
        return
        
    # Load model state and configuration
    payload = torch.load(args.model)
    img_size = payload.get("img_size", (32, 32))
    screen_w = payload.get("screen_w", 1920)
    screen_h = payload.get("screen_h", 1080)
    
    model = GazeCNN()
    model.load_state_dict(payload["model_state"])
    model.eval()
    
    print(f"[Live CNN] Loaded CNN model from {args.model}")
    print(f"[Live CNN] Input resolution: {img_size} | Active screen target: {screen_w}x{screen_h}")

    camera = VideoCamera(device_index=args.device)
    detector = FaceMeshDetector()

    window_name = "VisionPoint - Live Gaze (CNN + ANN)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    smooth_x, smooth_y = 0.5, 0.5
    have_pred = False
    a = args.smooth

    print("[Live CNN] Starting tracking loop. Press ESC inside the window to exit.", flush=True)

    while True:
        success, bgr, rgb = camera.read_frame()
        if not success:
            continue

        landmarks = detector.find_face_landmarks(rgb)
        ui = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)
        feat_ok = False

        if landmarks is not None:
            # Crop face using exact training padding
            crop = crop_face(bgr, landmarks, camera.target_width, camera.target_height, args.crop_pad, 224)
            if crop is not None:
                # Preprocess frame: grayscale -> equalize -> resize
                gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                gray = cv2.equalizeHist(gray)
                resized = cv2.resize(gray, img_size, interpolation=cv2.INTER_AREA)
                
                # Convert to normalized float32 tensor
                img_norm = resized.astype(np.float32) / 255.0
                img_tensor = torch.tensor(img_norm).unsqueeze(0).unsqueeze(0)  # Shape (1, 1, H, W)
                
                with torch.no_grad():
                    pred = model(img_tensor)[0].numpy()
                    
                nx = float(np.clip(pred[0], 0.0, 1.0))
                ny = float(np.clip(pred[1], 0.0, 1.0))
                feat_ok = True
                
                if not have_pred:
                    smooth_x, smooth_y = nx, ny
                    have_pred = True
                else:
                    smooth_x = a * nx + (1 - a) * smooth_x
                    smooth_y = a * ny + (1 - a) * smooth_y

        gx = int(smooth_x * screen_w)
        gy = int(smooth_y * screen_h)
        
        # Render a precision target circle and crosshairs
        cv2.circle(ui, (gx, gy), 40, (255, 100, 0), 3) # Blue-cyan color
        cv2.circle(ui, (gx, gy), 6, (255, 100, 0), -1)
        cv2.line(ui, (gx - 55, gy), (gx - 20, gy), (255, 100, 0), 1)
        cv2.line(ui, (gx + 20, gy), (gx + 55, gy), (255, 100, 0), 1)
        cv2.line(ui, (gx, gy - 55), (gx, gy - 20), (255, 100, 0), 1)
        cv2.line(ui, (gx, gy + 20), (gx, gy + 55), (255, 100, 0), 1)

        # Status text overlays
        status = f"Gaze: ({smooth_x:.2f}, {smooth_y:.2f})   FPS: {camera.get_fps():.0f}"
        cv2.putText(ui, status, (40, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (180, 180, 180), 2, cv2.LINE_AA)
        cv2.putText(ui, "Model: Deep CNN+ANN (Black-box Gaze Estimator) | ESC to quit",
                    (40, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 150, 150), 2, cv2.LINE_AA)
                    
        if not feat_ok:
            cv2.putText(ui, "ALIGNMENT LOST / NO FACE", (40, screen_h - 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 165, 255), 2, cv2.LINE_AA)

        cv2.imshow(window_name, ui)
        if (cv2.waitKey(1) & 0xFF) == 27:
            break

    camera.release()
    cv2.destroyAllWindows()
    print("[Live CNN] Stopped.")

if __name__ == "__main__":
    main()
