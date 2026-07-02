import numpy as np

class EyeGeometry:
    """
    Computes geometric properties of eyes, such as horizontal ratio,
    vertical ratio, and Eye Aspect Ratio (EAR) to detect blinks.
    """
    @staticmethod
    def distance(p1, p2):
        """Calculates 3D or 2D Euclidean distance between two points."""
        return np.linalg.norm(np.array(p1) - np.array(p2))

    @classmethod
    def calculate_eye_aspect_ratio(cls, top, bottom, outer, inner):
        """
        Calculates Eye Aspect Ratio (EAR).
        EAR = dist(top, bottom) / dist(outer, inner)
        """
        vertical_dist = cls.distance(top, bottom)
        horizontal_dist = cls.distance(outer, inner)
        if horizontal_dist == 0:
            return 0.0
        return vertical_dist / horizontal_dist

    @classmethod
    def calculate_ratios(cls, eye_data, is_left_eye=False):
        """
        Calculates:
        - horizontal_ratio: 0.0 (leftmost on screen) to 1.0 (rightmost on screen)
        - vertical_ratio: 0.0 (highest on screen/up) to 1.0 (lowest on screen/down)
        - ear: Eye Aspect Ratio (for blink detection)
        """
        outer = eye_data["outer_corner"]
        inner = eye_data["inner_corner"]
        top = eye_data["top_eyelid"]
        bottom = eye_data["bottom_eyelid"]
        iris = eye_data["iris_center"]

        # Calculate EAR
        ear = cls.calculate_eye_aspect_ratio(top, bottom, outer, inner)

        # Map to Screen orientation:
        # Since the webcam image is flipped horizontally (mirror),
        # larger x is towards the right side of the screen.
        #
        # For Right Eye (left on screen):
        # - Outer corner is on the screen-left (smaller x)
        # - Inner corner is on the screen-right (larger x)
        #
        # For Left Eye (right on screen):
        # - Inner corner is on the screen-left (smaller x)
        # - Outer corner is on the screen-right (larger x)
        
        if is_left_eye:
            screen_left_x = inner[0]
            screen_right_x = outer[0]
        else:
            screen_left_x = outer[0]
            screen_right_x = inner[0]

        width_x = screen_right_x - screen_left_x
        if width_x == 0:
            horizontal_ratio = 0.5
        else:
            horizontal_ratio = (iris[0] - screen_left_x) / width_x

        # Vertical ratio (smaller y is top, larger y is bottom)
        top_y = top[1]
        bottom_y = bottom[1]
        height_y = bottom_y - top_y
        if height_y == 0:
            vertical_ratio = 0.5
        else:
            vertical_ratio = (iris[1] - top_y) / height_y

        return horizontal_ratio, vertical_ratio, ear
