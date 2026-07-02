import cv2
import pyautogui

# Set pyautogui fail-safe (moving mouse to corner raises fail-safe exception)
pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0.0 # Remove delay for smooth movement

class CursorController:
    """
    Handles moving the physical OS cursor or drawing the cursor debug overlay.
    """
    def __init__(self, screen_w=None, screen_h=None):
        self.screen_w, self.screen_h = pyautogui.size() if (screen_w is None) else (screen_w, screen_h)

    def move_cursor(self, screen_x, screen_y):
        """Moves physical mouse cursor to screen_x, screen_y, clamped to bounds."""
        # Clamp to screen bounds to avoid PyAutoGUI out of bounds crashes
        clamped_x = max(0, min(int(screen_x), self.screen_w - 1))
        clamped_y = max(0, min(int(screen_y), self.screen_h - 1))
        
        try:
            pyautogui.moveTo(clamped_x, clamped_y)
        except Exception as e:
            # Catch potential fail-safe triggers gracefully
            print(f"[Overlay] PyAutoGUI fail-safe triggered or error: {e}")

    @staticmethod
    def draw_gaze_on_frame(frame, screen_x, screen_y, screen_w, screen_h):
        """
        Projects predicted screen coordinates back onto the webcam frame dimension
        to show a debug visualization.
        """
        fh, fw = frame.shape[:2]
        if screen_w <= 0 or screen_h <= 0:
            return

        # Map screen position (0..screen_w, 0..screen_h) to frame position (0..fw, 0..fh)
        map_x = int(screen_x * fw / screen_w)
        map_y = int(screen_y * fh / screen_h)

        # Clamp to frame boundary
        map_x = max(0, min(map_x, fw - 1))
        map_y = max(0, min(map_y, fh - 1))

        # Draw crosshair and circle
        cv2.circle(frame, (map_x, map_y), 15, (0, 255, 0), 2)
        cv2.circle(frame, (map_x, map_y), 4, (0, 255, 0), -1)
        
        # Add text coordinate label
        cv2.putText(
            frame, 
            f"Gaze: ({int(screen_x)}, {int(screen_y)})", 
            (map_x + 20, map_y + 5), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.5, 
            (0, 255, 0), 
            1, 
            cv2.LINE_AA
        )
