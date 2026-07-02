import numpy as np

# MediaPipe Face Mesh indices
# Anatomical RIGHT eye (looks like Left side on screen)
RIGHT_EYE_OUTER = 33
RIGHT_EYE_INNER = 133
RIGHT_EYE_TOP = 159
RIGHT_EYE_BOTTOM = 145
RIGHT_IRIS_CENTER = 468

# Anatomical LEFT eye (looks like Right side on screen)
LEFT_EYE_INNER = 362
LEFT_EYE_OUTER = 263
LEFT_EYE_TOP = 386
LEFT_EYE_BOTTOM = 374
LEFT_IRIS_CENTER = 473

class EyeExtractor:
    """
    Extracts key coordinates for the left and right eyes, including
    inner corners, outer corners, eyelid centers, and iris centers.
    Uses pixel-scaled or normalized landmarks.
    """
    @staticmethod
    def extract_eye_features(landmarks):
        """
        Extracts eye landmarks from the face mesh.
        Returns a dictionary containing left and right eye feature points.
        """
        if landmarks is None:
            return None

        # Helper to convert landmark [x, y, z] to np.array
        def get_point(idx):
            return landmarks[idx]

        features = {
            "right_eye": {
                "outer_corner": get_point(RIGHT_EYE_OUTER),
                "inner_corner": get_point(RIGHT_EYE_INNER),
                "top_eyelid": get_point(RIGHT_EYE_TOP),
                "bottom_eyelid": get_point(RIGHT_EYE_BOTTOM),
                "iris_center": get_point(RIGHT_IRIS_CENTER),
            },
            "left_eye": {
                "inner_corner": get_point(LEFT_EYE_INNER),
                "outer_corner": get_point(LEFT_EYE_OUTER),
                "top_eyelid": get_point(LEFT_EYE_TOP),
                "bottom_eyelid": get_point(LEFT_EYE_BOTTOM),
                "iris_center": get_point(LEFT_IRIS_CENTER),
            }
        }
        return features
