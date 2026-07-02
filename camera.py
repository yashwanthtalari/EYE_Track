import cv2
import time

class VideoCamera:
    """
    Wrapper for OpenCV VideoCapture to stream frames from laptop webcam.
    Flipped horizontally to act as a mirror, with helper to resize and convert to RGB.
    """
    def __init__(self, device_index=0, target_width=1280, target_height=720):
        self.cap = cv2.VideoCapture(device_index)
        # Set resolution hints
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, target_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, target_height)
        
        self.target_width = target_width
        self.target_height = target_height
        
        self.prev_time = time.time()
        self.fps = 0.0

        if not self.cap.isOpened():
            print(f"[Warning] Could not open video device index {device_index}")

    def read_frame(self):
        """
        Reads frame, flips horizontally, resizes if needed, and returns:
        - success: bool
        - bgr_frame: cv2 frame (BGR) for drawing/display
        - rgb_frame: cv2 frame (RGB) for MediaPipe processing
        """
        success, frame = self.cap.read()
        if not success or frame is None:
            return False, None, None

        # Flip horizontally (mirror effect)
        frame = cv2.flip(frame, 1)

        # Optional resize if dimensions don't match target
        h, w = frame.shape[:2]
        if w != self.target_width or h != self.target_height:
            frame = cv2.resize(frame, (self.target_width, self.target_height))

        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Update FPS
        curr_time = time.time()
        time_diff = curr_time - self.prev_time
        if time_diff > 0:
            self.fps = 0.9 * self.fps + 0.1 * (1.0 / time_diff)
        self.prev_time = curr_time

        return True, frame, rgb_frame

    def get_fps(self):
        return self.fps

    def release(self):
        if self.cap.isOpened():
            self.cap.release()
