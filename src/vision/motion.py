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
        # IoU tracking for temporal novelty scoring
        self._prev_boxes: list[BoundingBox] = []

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

        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        # Raw boxes (no area filter yet — merging may grow them)
        raw: list[BoundingBox] = []
        for contour in contours:
            area = int(cv2.contourArea(contour))
            x, y, w, h = cv2.boundingRect(contour)
            raw.append(
                BoundingBox(
                    x=x, y=y, w=w, h=h, area=area,
                )
            )

        fh, fw = frame.shape[:2]
        max_area = int(
            fw * fh * self._config.max_box_ratio
        )
        merged = self._merge_nearby_boxes(
            raw, self._config.merge_gap, max_area,
        )

        boxes = [
            b for b in merged
            if b.area >= self._config.min_contour_area
        ]

        sorted_boxes = sorted(
            boxes, key=lambda b: b.area, reverse=True,
        )
        self.assign_temporal_weights(sorted_boxes)
        self._prev_boxes = sorted_boxes
        return sorted_boxes

    # ── Spatial merge ───────────────────────────────────

    @staticmethod
    def _edge_distance(
        a: BoundingBox, b: BoundingBox,
    ) -> float:
        """Min distance between the edges of two boxes.

        Returns 0 when boxes overlap or touch.
        """
        dx = max(0, max(a.x, b.x) - min(a.x + a.w, b.x + b.w))
        dy = max(0, max(a.y, b.y) - min(a.y + a.h, b.y + b.h))
        return float(dx + dy)

    @staticmethod
    def _union_box(
        a: BoundingBox, b: BoundingBox,
    ) -> BoundingBox:
        """Return the smallest box enclosing both *a* and *b*."""
        x1 = min(a.x, b.x)
        y1 = min(a.y, b.y)
        x2 = max(a.x + a.w, b.x + b.w)
        y2 = max(a.y + a.h, b.y + b.h)
        w, h = x2 - x1, y2 - y1
        return BoundingBox(
            x=x1, y=y1, w=w, h=h, area=w * h,
        )

    @classmethod
    def _merge_nearby_boxes(
        cls,
        boxes: list[BoundingBox],
        gap: int,
        max_area: int = 0,
    ) -> list[BoundingBox]:
        """Merge boxes whose edges are within *gap* px.

        Uses iterative greedy merging: on each pass the
        first overlapping/nearby pair is merged into its
        union box and the pass restarts.  Convergence is
        guaranteed because each merge reduces the list
        length by one.

        Parameters
        ----------
        gap : int
            Maximum edge distance for two boxes to be
            merged.  Negative values disable merging.
        max_area : int
            If > 0, a merge is skipped when the union
            box would exceed this area (prevents
            chain-merging into a frame-sized box).
        """
        if gap < 0:
            return list(boxes)
        result = list(boxes)
        changed = True
        while changed:
            changed = False
            n = len(result)
            for i in range(n):
                for j in range(i + 1, n):
                    d = cls._edge_distance(
                        result[i], result[j],
                    )
                    if d > gap:
                        continue
                    candidate = cls._union_box(
                        result[i], result[j],
                    )
                    if (
                        max_area > 0
                        and candidate.area > max_area
                    ):
                        continue
                    result = [
                        result[k]
                        for k in range(n)
                        if k != i and k != j
                    ] + [candidate]
                    changed = True
                    break
                if changed:
                    break
        return result

    # ── IoU-based temporal novelty ──────────────────────

    @staticmethod
    def _iou(a: BoundingBox, b: BoundingBox) -> float:
        """Compute Intersection-over-Union between two boxes."""
        x1 = max(a.x, b.x)
        y1 = max(a.y, b.y)
        x2 = min(a.x + a.w, b.x + b.w)
        y2 = min(a.y + a.h, b.y + b.h)
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        union = a.area + b.area - inter
        return inter / union if union > 0 else 0.0

    def assign_temporal_weights(
        self, boxes: list[BoundingBox],
    ) -> None:
        """Set ``source_score`` on each box to a temporal weight.

        A box that overlaps significantly (IoU > 0.3) with a
        previous-frame box is considered *persistent* and receives
        weight 0.5.  A new box gets weight 1.0.

        The pipeline will later multiply this weight by the area
        ratio and anomaly score to produce the final
        ``source_score``.
        """
        for box in boxes:
            best_iou = 0.0
            for prev in self._prev_boxes:
                iou = self._iou(box, prev)
                if iou > best_iou:
                    best_iou = iou
            # New box → 1.0; persistent → 0.5
            box.source_score = (
                0.5 if best_iou > 0.3 else 1.0
            )

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
        self._prev_boxes = []
