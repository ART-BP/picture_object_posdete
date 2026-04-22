#!/usr/bin/env python3
import math
import threading
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple


class RecoveryAction(str, Enum):
    MOVE_LAST_DIRECTION = "move_last_direction"
    ROTATE_IN_PLACE = "rotate_in_place"
    CANCEL_TASK = "cancel_task"


@dataclass(frozen=True)
class RecoveryEvent:
    action: RecoveryAction
    heading_rad: float
    lost_sec: float


class RecoveryController:
    """Track detection loss timeline and emit staged recovery actions."""

    def __init__(
        self,
        move_after_sec: float = 3.0,
        rotate_after_sec: float = 5.0,
        cancel_after_sec: float = 10.0,
    ) -> None:
        if move_after_sec <= 0.0:
            raise ValueError("move_after_sec must be > 0")
        if rotate_after_sec <= move_after_sec:
            raise ValueError("rotate_after_sec must be > move_after_sec")
        if cancel_after_sec <= rotate_after_sec:
            raise ValueError("cancel_after_sec must be > rotate_after_sec")

        self.move_after_sec = float(move_after_sec)
        self.rotate_after_sec = float(rotate_after_sec)
        self.cancel_after_sec = float(cancel_after_sec)

        self._lock = threading.Lock()
        self._active = False
        self._last_detect_sec = 0.0
        self._last_heading_rad = None
        self._move_sent = False
        self._rotate_sent = False
        self._cancel_sent = False
        self.last_pos = None

    def on_task(self, task: str, now_sec: float) -> None:
        task_norm = str(task).strip().lower()
        with self._lock:
            if task_norm == "follow":
                self._active = True
                self._last_detect_sec = float(now_sec)
                self._move_sent = False
                self._rotate_sent = False
                self._cancel_sent = False
                return

            # Any non-follow task disables recovery timeline.
            self._active = False
            self._move_sent = False
            self._rotate_sent = False
            self._cancel_sent = False

    def on_detection(self, x: float, y: float, now_sec: float) -> None:
        if x is None or y is None:
            return
        x = x
        y = y
        if not (math.isfinite(x) and math.isfinite(y)):
            return
        if self.last_pos is None:
            self.last_pos = (x, y)
            return
        heading = float(math.atan2(y - self.last_pos[1], x - self.last_pos[0]))
        self.last_pos = (x, y)
        with self._lock:
            if not self._active:
                return
            self._last_heading_rad = heading
            self._last_detect_sec = float(now_sec)
            # A fresh detection starts a new loss episode.
            self._move_sent = False
            self._rotate_sent = False
            self._cancel_sent = False

    def poll(self, now_sec: float) -> Optional[RecoveryEvent]:
        with self._lock:
            if not self._active:
                return None

            lost_sec = max(0.0, float(now_sec) - float(self._last_detect_sec))
            heading = float(self._last_heading_rad) if self._last_heading_rad is not None else None

            if lost_sec >= self.move_after_sec and not self._move_sent and heading is not None:
                self._move_sent = True
                return RecoveryEvent(
                    action=RecoveryAction.MOVE_LAST_DIRECTION,
                    heading_rad=heading,
                    lost_sec=lost_sec,
                )

            if (
                lost_sec >= self.rotate_after_sec
                and (self._move_sent or heading is None)
                and not self._rotate_sent
            ):
                self._move_sent = True
                self._rotate_sent = True
                return RecoveryEvent(
                    action=RecoveryAction.ROTATE_IN_PLACE,
                    heading_rad=heading,
                    lost_sec=lost_sec,
                )

            if (
                lost_sec >= self.cancel_after_sec
                and self._rotate_sent
                and not self._cancel_sent
            ):
                self._cancel_sent = True
                self._active = False
                return RecoveryEvent(
                    action=RecoveryAction.CANCEL_TASK,
                    heading_rad=heading,
                    lost_sec=lost_sec,
                )
            return None
