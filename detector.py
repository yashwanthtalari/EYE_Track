import cv2
import mediapipe as mp
import numpy as np

class FaceMeshDetector:
    """
    Wrapper around MediaPipe Face Mesh.
    Configures Face Mesh with refine_landmarks=True to get iris landmarks.
    """
    def __init__(self, max_num_faces=1, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=max_num_faces,
            refine_landmarks=True,  # Enables refined contours for eyes, lips, and irises
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )

    def find_face_landmarks(self, rgb_frame):
        """
        Processes RGB frame and returns normalized landmark array (shape: [N, 3]) if face is detected.
        Returns None if no face is detected.
        """
        results = self.face_mesh.process(rgb_frame)
        if not results.multi_face_landmarks:
            return None

        # Return landmarks for the first face detected
        face_landmarks = results.multi_face_landmarks[0]
        
        # Extract normalized coordinates (x, y, z)
        landmarks = np.array([
            [lm.x, lm.y, lm.z] for lm in face_landmarks.landmark
        ])
        
        return landmarks

    def get_pixel_landmarks(self, landmarks, frame_w, frame_h):
        """
        Converts normalized landmarks [0, 1] to pixel coordinates [0, width/height]
        """
        if landmarks is None:
            return None
        pixel_landmarks = landmarks.copy()
        pixel_landmarks[:, 0] *= frame_w
        pixel_landmarks[:, 1] *= frame_h
        return pixel_landmarks
