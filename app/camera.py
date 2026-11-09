
"""Camera management using OpenCV."""

import cv2
import numpy as np

from app.config import CAMERA_INDEX, CAMERA_WIDTH, CAMERA_HEIGHT


class CameraManager:
    """Manage camera access and frame capture."""

    def __init__(self) -> None:
        """Initialize the camera manager."""
        self.capture: cv2.VideoCapture | None = None

    def open_camera(self) -> bool:
        """Open the default camera."""
        if self.is_open():
            return True

        capture = cv2.VideoCapture(
            CAMERA_INDEX,
            cv2.CAP_DSHOW,
        )

        if not capture.isOpened():
            capture.release()
            return False

        capture.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

        self.capture = capture
        return True

    def is_open(self) -> bool:
        """Check whether the camera is open."""
        return (
            self.capture is not None
            and self.capture.isOpened()
        )

    def read_frame(self) -> np.ndarray | None:
        """Read a frame and convert it from BGR to RGB."""
        if not self.is_open():
            return None

        success, frame = self.capture.read()

        if not success or frame is None:
            return None

        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def close_camera(self) -> None:
        """Release the camera safely."""
        if self.capture is not None:
            self.capture.release()
            self.capture = None
