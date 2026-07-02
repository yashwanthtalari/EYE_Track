import cv2
import numpy as np

class HeadPoseEstimator:
    """
    Estimates head orientation (Yaw, Pitch, Roll) using a 3D generic face model
    and OpenCV's SolvePnP.
    """
    def __init__(self):
        # 3D model points in coordinate system where:
        # - origin (0,0,0) is at the nose tip
        # - x-axis points to the anatomical left (screen right)
        # - y-axis points upwards
        # - z-axis points out of the face
        self.model_points = np.array([
            [0.0, 0.0, 0.0],             # Nose tip (index 1)
            [0.0, -330.0, -65.0],        # Chin (index 152)
            [-225.0, 170.0, -135.0],     # Right eye outer corner (index 33)
            [225.0, 170.0, -135.0],      # Left eye outer corner (index 263)
            [-150.0, -150.0, -125.0],    # Right mouth corner (index 61)
            [150.0, -150.0, -125.0]      # Left mouth corner (index 291)
        ], dtype=np.float32)

        # Corresponding MediaPipe Face Mesh indices
        self.landmarks_indices = [1, 152, 33, 263, 61, 291]

    def estimate_pose(self, pixel_landmarks, img_w, img_h):
        """
        Estimates Yaw, Pitch, Roll from pixel landmarks.
        Returns:
            yaw: head rotation around Y-axis (degrees) - looking left/right
            pitch: head rotation around X-axis (degrees) - looking up/down
            roll: head rotation around Z-axis (degrees) - tilt
            rvec, tvec: rotation and translation vectors
        """
        if pixel_landmarks is None:
            return 0.0, 0.0, 0.0, None, None

        # Extract the 6 key 2D points
        image_points = np.array([
            pixel_landmarks[idx][:2] for idx in self.landmarks_indices
        ], dtype=np.float32)

        # Camera intrinsics approximation
        focal_length = img_w
        center = (img_w / 2.0, img_h / 2.0)
        camera_matrix = np.array([
            [focal_length, 0.0, center[0]],
            [0.0, focal_length, center[1]],
            [0.0, 0.0, 1.0]
        ], dtype=np.float32)

        dist_coeffs = np.zeros((4, 1)) # Assuming no lens distortion

        # Solve PnP
        success, rvec, tvec = cv2.solvePnP(
            self.model_points,
            image_points,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )

        if not success:
            return 0.0, 0.0, 0.0, None, None

        # Convert rotation vector to rotation matrix
        R, _ = cv2.Rodrigues(rvec)

        # Calculate Euler angles from rotation matrix
        sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
        singular = sy < 1e-6

        if not singular:
            x = np.arctan2(R[2, 1], R[2, 2])
            y = np.arctan2(-R[2, 0], sy)
            z = np.arctan2(R[1, 0], R[0, 0])
        else:
            x = np.arctan2(-R[1, 2], R[1, 1])
            y = np.arctan2(-R[2, 0], sy)
            z = 0

        # Convert to degrees
        pitch = np.degrees(x)
        yaw = np.degrees(y)
        roll = np.degrees(z)

        return yaw, pitch, roll, rvec, tvec

    def draw_pose_axes(self, frame, rvec, tvec, img_w, img_h):
        """
        Draws 3D coordinate axes on the nose tip (origin) to visualize pitch, yaw, roll.
        Red = X-axis, Green = Y-axis, Blue = Z-axis.
        """
        if rvec is None or tvec is None:
            return

        focal_length = img_w
        center = (img_w / 2.0, img_h / 2.0)
        camera_matrix = np.array([
            [focal_length, 0.0, center[0]],
            [0.0, focal_length, center[1]],
            [0.0, 0.0, 1.0]
        ], dtype=np.float32)
        dist_coeffs = np.zeros((4, 1))

        # 3D points for axes (length 150 units)
        axis_3d = np.array([
            [150.0, 0.0, 0.0],  # X axis (Right/Left)
            [0.0, 150.0, 0.0],  # Y axis (Up/Down - positive Y is down in OpenCV, but let's invert for positive up)
            [0.0, 0.0, 150.0]   # Z axis (forward/backward)
        ], dtype=np.float32)

        # Invert model Y axis for display alignment (model coordinates have positive Y UP, OpenCV has positive Y DOWN)
        axis_3d_disp = axis_3d.copy()
        axis_3d_disp[1, 1] = -150.0 # Point green arrow UP visually

        # Project axes to 2D image coordinates
        projected_points, _ = cv2.projectPoints(
            np.vstack([np.zeros((1, 3)), axis_3d_disp]),
            rvec, tvec,
            camera_matrix,
            dist_coeffs
        )

        p_nose = tuple(projected_points[0].ravel().astype(int))
        p_x = tuple(projected_points[1].ravel().astype(int))
        p_y = tuple(projected_points[2].ravel().astype(int))
        p_z = tuple(projected_points[3].ravel().astype(int))

        # Draw red X-axis
        cv2.line(frame, p_nose, p_x, (0, 0, 255), 3)
        # Draw green Y-axis
        cv2.line(frame, p_nose, p_y, (0, 255, 0), 3)
        # Draw blue Z-axis
        cv2.line(frame, p_nose, p_z, (255, 0, 0), 3)
