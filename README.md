# VisionPoint: Webcam-Based Eye Gaze Estimation Engine

VisionPoint is a modular, high-accuracy webcam-based gaze tracking engine built using **OpenCV**, **MediaPipe Face Mesh**, **SQLite**, and **Scikit-Learn**. It maps relative pupil movement and 3D head pose to coordinate locations on your monitor screen using a machine learning regression model.

---

## 🚀 Key Features

* **High-Accuracy Gaze Calibration State Machine**: 
  * Enforces **pre-capture stabilization** (checks variance of ratios over a sliding window before capturing to ensure focus).
  * Implements **blink detection** (aborts automatically if the user blinks mid-capture).
  * Validates **post-capture variance** (automatically rejects a target dot if standard deviation suggests eye movement or saccades).
* **Rich 10-Feature Mapping Profile**:
  * Predicts coordinate locations based on Left Pupil Ratio (2), Right Pupil Ratio (2), 3D Head rotation (3), and 3D Head translation coordinates (3).
* **Adaptive Coordinate Smoothing**:
  * Utilizes dynamic Exponential Moving Average (EMA) to eliminate jitter when looking at a single point, while allowing fast response tracking during quick eye movements.
* **Persistent SQLite database storage**:
  * Saves calibration sets under unique session identifiers in `data/calibration.db`.
* **Clean, Decoupled Software Architecture**:
  * Built as independent modules, making it easy to swap modules (e.g. replacing regression mapping with a custom neural network).

---

## 📂 File Layout

* **`app.py`**: App loop, coordinate bindings, visual previews, and interactive key commands.
* **`camera.py`**: Video capture stream wrapping, horizontal mirroring, and resizing.
* **`detector.py`**: MediaPipe Face Mesh model initiator with iris tracking refinement.
* **`iris.py`**: Landmark coordinate mappings for eyes contours and pupil points.
* **`geometry.py`**: Normalization ratios and EAR blink metrics.
* **`head_pose.py`**: Estimates 3D head orientation (Yaw, Pitch, Roll) and Translation vectors (`tx`, `ty`, `tz`).
* **`calibration.py`**: Fullscreen calibration GUI and state management.
* **`neural_regressor.py`**: From-scratch NumPy MLP (hand-written forward/backprop, He init, Adam, L2 + early stopping) mapping the 14-feature vector → screen (x, y). Drop-in for the legacy Ridge model.
* **`train.py`**: Offline trainer/evaluator — trains the MLP on the latest calibration session, reports held-out pixel error, and optionally compares against the Ridge baseline (`--compare`).
* **`regression.py`**: Legacy Ridge Regression baseline (2nd-degree Polynomial features).
* **`database.py`**: Local SQLite database interface.
* **`smoothing.py`**: Standard and adaptive EMA coordinate smoothing filters.
* **`overlay.py`**: PyAutoGUI cursor controller driver.

---

## 🛠️ Setup & Installation

Ensure you have **Python 3.10** installed. 

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yashwanthtalari/EYE_Track.git
   cd EYE_Track
   ```

2. **Install requirements**:
   ```bash
   pip install -r requirements.txt
   ```

---

## 🎮 How to Run

Launch the application:
```bash
python app.py
```

### Key Bindings (Webcam Window)
* **`SPACE`**: In the Calibration wizard, press space to calibrate the active target dot.
* **`M`**: Toggle cursor movement driver (moving your OS mouse cursor with your eyes).
* **`C`**: Recalibrate.
* **`V`**: Hide/Show webcam preview window.
* **`ESC`**: Exit the program.

---

## 🤝 Calibration Guide
1. Run the app. A window titled **"VisionPoint - Align Your Face"** will appear. Ensure your face is centered, and you can see green dot indicators on your eyes.
2. Press **`SPACE`** to enter fullscreen calibration.
3. Look steadily at the **RED** circle, and press **`SPACE`**. 
4. The dot turns **Yellow** (stabilizing) and then **Blue** (capturing). Keep looking at it until the capture concludes.
5. Repeat for all 9 points to compile your dataset and train the regressor.
