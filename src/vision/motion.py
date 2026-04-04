from __future__ import annotations

import time

import cv2
import numpy as np

from .config import VisionConfig
from .types import BoundingBox, MotionResult


class MotionDetector:
    """Detects motion in video frames using MOG2 background subtraction.

    Processing pipeline per frame:
    1. MOG2 background subtraction → foreground mask
    2. Morphological open (erosion + dilation) to remove salt-and-pepper noise
    3. Dilation to merge nearby regions into unified blobs
    4. findContours + area filtering to extract bounding boxes
    """

    def __init__(self, config: VisionConfig | None = None) -> None:
        self._config = config or VisionConfig()
        self._bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=self._config.mog2_history,
            varThreshold=self._config.mog2_var_threshold,
            detectShadows=False,
        )
        kernel_size = self._config.morph_kernel_size
        self._morph_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (kernel_size, kernel_size)
        )

    def detect(self, frame: np.ndarray) -> list[BoundingBox]:
        """Apply MOG2 and extract motion bounding boxes from a BGR frame.

        Parameters
        ----------
        frame : np.ndarray
            BGR frame, shape (H, W, 3), uint8.

        Returns
        -------
        list[BoundingBox]
            Detected motion regions sorted by area descending.
            Empty list when no significant motion is found.
        """
        mask = self._bg_subtractor.apply(frame)

        # Remove small noise before dilation to avoid merging unrelated regions
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._morph_kernel)
        mask = cv2.dilate(
            mask, self._morph_kernel, iterations=self._config.dilation_iterations
        )

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        boxes = []
        for contour in contours:
            area = int(cv2.contourArea(contour))
            if area < self._config.min_contour_area:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            boxes.append(BoundingBox(x=x, y=y, w=w, h=h, area=area))

        return sorted(boxes, key=lambda b: b.area, reverse=True)

    def detect_with_result(self, frame: np.ndarray) -> MotionResult:
        """Detect motion and return a MotionResult with annotated frame."""
        boxes = self.detect(frame)
        annotated = self.draw_boxes(frame, boxes)
        return MotionResult(
            frame=frame,
            annotated_frame=annotated,
            boxes=boxes,
            has_motion=len(boxes) > 0,
            timestamp=time.time(),
        )

    def draw_boxes(self, frame: np.ndarray, boxes: list[BoundingBox]) -> np.ndarray:
        """Return a copy of frame with bounding boxes drawn in green.

        Parameters
        ----------
        frame : np.ndarray
            BGR frame.
        boxes : list[BoundingBox]

        Returns
        -------
        np.ndarray
            New array (not in-place modification).
        """
        annotated = frame.copy()
        for box in boxes:
            cv2.rectangle(
                annotated,
                (box.x, box.y),
                (box.x + box.w, box.y + box.h),
                color=(0, 255, 0),
                thickness=2,
            )
        return annotated

    def reset(self) -> None:
        """Reinitialize the background model. Use after scene changes."""
        self._bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=self._config.mog2_history,
            varThreshold=self._config.mog2_var_threshold,
            detectShadows=False,
        )
