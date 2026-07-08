"""
Shared feature extraction for the image-based gaze model.

Given a single square face crop (BGR, e.g. the 224x224 images produced by
collect_data.py, or a live crop from crop_face), this module produces two
complementary feature blocks:

  1. GEOMETRIC features  -- iris horizontal/vertical ratios, eye-aspect-ratios,
     and head-pose angles. These are (largely) invariant to head *position* and
     scale because they are ratios relative to the eye corners, so they carry
     the gaze signal far more efficiently than raw pixels. Reuses the project's
     existing iris.py / geometry.py / head_pose.py.

  2. EYE-STRIP pixels    -- tight grayscale crops of the left and right eye
     regions, histogram-equalized and flattened. Unlike a whole-face 32x32
     image (where each iris is ~2 px), this spends the resolution budget where
     the gaze signal actually lives.

Both training (via enrich_dataset.py) and live inference (predict_live.py) call
the SAME function here, so the features are guaranteed identical end-to-end.
"""
import cv2
import numpy as np

from iris import EyeExtractor
from geometry import EyeGeometry
from head_pose import HeadPoseEstimator

# MediaPipe Face Mesh eye-contour rings, used to draw a tight bbox per eye.
RIGHT_EYE_RING = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
LEFT_EYE_RING = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]

# Default eye-strip size (width, height) in pixels, per eye.
DEFAULT_EYE_SIZE = (32, 18)

# Length of the geometric feature vector (see geometric_features()).
GEOM_DIM = 12

_head_pose = HeadPoseEstimator()


def geometric_features(landmarks, crop_w, crop_h):
    """
    Build the geometric feature vector from normalized face-mesh landmarks.

    Returns a float32 array of length GEOM_DIM, or None if landmarks are missing.
    All values are roughly in [-1, 1] / [0, 1] so they play nicely with scaling.
    """
    if landmarks is None:
        return None

    eyes = EyeExtractor.extract_eye_features(landmarks)
    if eyes is None:
        return None

    lh, lv, lear = EyeGeometry.calculate_ratios(eyes["left_eye"], is_left_eye=True)
    rh, rv, rear = EyeGeometry.calculate_ratios(eyes["right_eye"], is_left_eye=False)

    # Head pose (degrees) via solvePnP on pixel landmarks of the crop.
    px = landmarks.copy()
    px[:, 0] *= crop_w
    px[:, 1] *= crop_h
    yaw, pitch, roll, _, _ = _head_pose.estimate_pose(px, crop_w, crop_h)

    vec = np.array([
        lh, lv, lear,
        rh, rv, rear,
        (lh + rh) * 0.5, (lv + rv) * 0.5,  # combined gaze estimate
        (lh - rh),                          # vergence / asymmetry
        yaw / 90.0, pitch / 90.0, roll / 90.0,  # head pose, scaled ~[-1,1]
    ], dtype=np.float32)
    return vec


def _eye_patch(gray, landmarks, ring, crop_w, crop_h, out_size, pad=0.35):
    """Extract one equalized grayscale eye patch (out_size = (w, h))."""
    xs = landmarks[ring, 0] * crop_w
    ys = landmarks[ring, 1] * crop_h
    x1, x2 = float(np.min(xs)), float(np.max(xs))
    y1, y2 = float(np.min(ys)), float(np.max(ys))

    ew, eh = x2 - x1, y2 - y1
    if ew < 2 or eh < 2:
        return None

    # Pad the box; eyes are wide, so a little vertical breathing room helps.
    x1 -= ew * pad
    x2 += ew * pad
    y1 -= eh * pad
    y2 += eh * pad

    xi1, yi1 = max(0, int(round(x1))), max(0, int(round(y1)))
    xi2, yi2 = min(crop_w, int(round(x2))), min(crop_h, int(round(y2)))
    if xi2 - xi1 < 2 or yi2 - yi1 < 2:
        return None

    patch = gray[yi1:yi2, xi1:xi2]
    patch = cv2.resize(patch, out_size, interpolation=cv2.INTER_AREA)
    patch = cv2.equalizeHist(patch)
    return patch.astype(np.float32) / 255.0


def eye_features(bgr, landmarks, crop_w, crop_h, eye_size=DEFAULT_EYE_SIZE):
    """
    Concatenated flattened left+right eye patches, or None if either eye fails.
    Length = 2 * eye_size[0] * eye_size[1].
    """
    if landmarks is None:
        return None
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    right = _eye_patch(gray, landmarks, RIGHT_EYE_RING, crop_w, crop_h, eye_size)
    left = _eye_patch(gray, landmarks, LEFT_EYE_RING, crop_w, crop_h, eye_size)
    if right is None or left is None:
        return None
    return np.concatenate([right.flatten(), left.flatten()]).astype(np.float32)


def eye_dim(eye_size=DEFAULT_EYE_SIZE):
    return 2 * eye_size[0] * eye_size[1]


def extract_from_crop(bgr, detector, eye_size=DEFAULT_EYE_SIZE):
    """
    Run FaceMesh on a face crop and return (geom_vec, eye_vec).

    Either element may be None if detection or a sub-step fails. `detector` is a
    FaceMeshDetector instance (reused across calls for speed).
    """
    h, w = bgr.shape[:2]
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    landmarks = detector.find_face_landmarks(rgb)
    if landmarks is None:
        return None, None
    geom = geometric_features(landmarks, w, h)
    eyes = eye_features(bgr, landmarks, w, h, eye_size)
    return geom, eyes
