# vision/camera.py - Camera capture module
import cv2
import base64
import logging
import numpy as np
from typing import Optional

logger = logging.getLogger(__name__)

class Camera:
    """Handles webcam capture for vision input."""

    def __init__(self, camera_index: int = 0):
        """
        Initialize the camera.

        Args:
            camera_index: Camera device index (0 for default webcam)
        """
        self.camera_index = camera_index
        self._cap = None

    def _ensure_camera(self) -> bool:
        """Ensure camera is open, open it if needed."""
        if self._cap is None or not self._cap.isOpened():
            self._cap = cv2.VideoCapture(self.camera_index)
            if not self._cap.isOpened():
                logger.error(f"Could not open camera {self.camera_index}")
                return False
            logger.info(f"Camera {self.camera_index} opened")
        return True

    def capture_frame(self) -> Optional[np.ndarray]:
        """
        Capture a single frame from the camera.

        Returns:
            BGR image as numpy array, or None if capture failed
        """
        if not self._ensure_camera():
            return None

        ret, frame = self._cap.read()
        if not ret:
            logger.error("Failed to capture frame")
            return None

        return frame

    def capture_base64(self, quality: int = 60, max_width: int = 640) -> Optional[str]:
        """
        Capture a frame and return as base64 encoded JPEG.

        Args:
            quality: JPEG quality (1-100)
            max_width: Resize image if wider than this

        Returns:
            Base64 encoded JPEG string, or None if capture failed
        """
        frame = self.capture_frame()
        if frame is None:
            return None

        # Resize if too large (speeds up vision processing)
        h, w = frame.shape[:2]
        if w > max_width:
            scale = max_width / w
            new_h = int(h * scale)
            frame = cv2.resize(frame, (max_width, new_h))

        # Encode as JPEG
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
        _, buffer = cv2.imencode('.jpg', frame, encode_params)

        # Convert to base64
        base64_image = base64.b64encode(buffer).decode('utf-8')
        return base64_image

    def release(self):
        """Release the camera resource."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None
            logger.info("Camera released")

    def __del__(self):
        self.release()


# Test the camera
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    cam = Camera()

    # Test capture
    frame = cam.capture_frame()
    if frame is not None:
        print(f"Captured frame: {frame.shape}")

        # Test base64
        b64 = cam.capture_base64()
        if b64:
            print(f"Base64 length: {len(b64)}")

    cam.release()
