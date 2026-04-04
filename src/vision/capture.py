from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from .config import VisionConfig


class FrameCapture:
    """OpenCV-based video frame capture with context manager support.

    Designed to be testable: the underlying cv2.VideoCapture can be
    replaced in tests by patching cv2.VideoCapture before calling open().
    """

    def __init__(self, config: VisionConfig | None = None) -> None:
        self._config = config or VisionConfig()
        self._cap: Optional[cv2.VideoCapture] = None

    def open(self) -> None:
        """Open the video capture device.

        Raises
        ------
        RuntimeError
            If the device cannot be opened.
        """
        self._cap = cv2.VideoCapture(self._config.capture_device)
        if not self._cap.isOpened():
            self._cap = None
            raise RuntimeError(
                f"Could not open video capture device {self._config.capture_device}"
            )
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._config.target_width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._config.target_height)
        self._cap.set(cv2.CAP_PROP_FPS, self._config.fps)

    def read(self) -> Optional[np.ndarray]:
        """Read the next frame.

        Returns
        -------
        np.ndarray or None
            BGR frame, or None if the device is not open or read failed.
        """
        if self._cap is None:
            return None
        ret, frame = self._cap.read()
        return frame if ret else None

    def close(self) -> None:
        """Release the capture device."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __enter__(self) -> "FrameCapture":
        self.open()
        return self

    def __exit__(self, *args) -> None:
        self.close()
