from collections import deque
from dataclasses import dataclass
import time

import cv2
import numpy as np


try:
    import mediapipe as mp
except ImportError:  # pragma: no cover - runtime fallback for missing optional dependency
    mp = None


@dataclass
class LandmarkFatigueConfig:
    ear_threshold: float = 0.21
    mar_threshold: float = 0.56
    window_seconds: float = 60.0
    min_blink_frames: int = 2
    min_yawn_frames: int = 6
    yawn_cooldown_seconds: float = 1.5
    max_blink_seconds: float = 0.8


class LandmarkFatigueTracker:
    """Track eye and mouth state from MediaPipe FaceMesh landmarks."""

    LEFT_EYE = (33, 160, 158, 133, 153, 144)
    RIGHT_EYE = (362, 385, 387, 263, 373, 380)
    MOUTH_HORIZONTAL = (61, 291)
    MOUTH_VERTICAL = ((13, 14), (81, 178), (312, 402))

    def __init__(self, config=None, max_faces=1):
        if mp is None:
            raise RuntimeError("mediapipe is not installed")
        self.config = config or LandmarkFatigueConfig()
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=max_faces,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.reset()

    def reset(self):
        self._start_time = time.time()
        self.total_frames = 0
        self.eye_closed_frames = 0
        self.blink_count = 0
        self.yawn_count = 0
        self.closed_streak = 0
        self.mouth_open_streak = 0
        self.eye_was_closed = False
        self.yawn_active = False
        self.last_yawn_time = None
        self._closed_start_time = None
        self._mouth_open_start_time = None
        self._last_longest_eye_closure = 0.0
        self._last_yawn_duration = 0.0
        self.frame_window = deque()
        self.blink_events = deque()
        self.yawn_events = deque()
        self.eye_closure_events = deque()
        self.yawn_duration_events = deque()

    def close(self):
        self.face_mesh.close()

    @staticmethod
    def _distance(a, b):
        return float(np.linalg.norm(np.array(a) - np.array(b)))

    @staticmethod
    def _point(landmarks, index, width, height):
        item = landmarks[index]
        return np.array([item.x * width, item.y * height], dtype=np.float32)

    def _eye_aspect_ratio(self, landmarks, indices, width, height):
        p1, p2, p3, p4, p5, p6 = [
            self._point(landmarks, idx, width, height) for idx in indices
        ]
        vertical = self._distance(p2, p6) + self._distance(p3, p5)
        horizontal = self._distance(p1, p4)
        return vertical / (2.0 * horizontal + 1e-6)

    def _mouth_aspect_ratio(self, landmarks, width, height):
        left = self._point(landmarks, self.MOUTH_HORIZONTAL[0], width, height)
        right = self._point(landmarks, self.MOUTH_HORIZONTAL[1], width, height)
        horizontal = self._distance(left, right)
        vertical_values = []
        for top_idx, bottom_idx in self.MOUTH_VERTICAL:
            top = self._point(landmarks, top_idx, width, height)
            bottom = self._point(landmarks, bottom_idx, width, height)
            vertical_values.append(self._distance(top, bottom))
        return float(np.mean(vertical_values) / (horizontal + 1e-6))

    def process_frame(self, frame, update=True, draw=True):
        height, width = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self.face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            window_features = self.window_features
            return {
                "face_found": False,
                "eye_state": "未检测到人脸",
                "mouth_state": "未检测到人脸",
                "ear": None,
                "mar": None,
                "perclos": window_features["perclos"],
                "blink_count": self.blink_count,
                "blink_rate": window_features["blink_rate"],
                "yawn_count": window_features["yawn_count"],
                "window_features": window_features,
            }

        landmarks = results.multi_face_landmarks[0].landmark
        left_ear = self._eye_aspect_ratio(landmarks, self.LEFT_EYE, width, height)
        right_ear = self._eye_aspect_ratio(landmarks, self.RIGHT_EYE, width, height)
        ear = (left_ear + right_ear) / 2.0
        mar = self._mouth_aspect_ratio(landmarks, width, height)

        eye_closed = ear < self.config.ear_threshold
        mouth_open = mar > self.config.mar_threshold

        if update:
            self._update_counts(eye_closed, mouth_open, ear, mar)

        window_features = self.window_features

        if draw:
            self._draw_status(frame, landmarks, width, height, ear, mar, eye_closed, mouth_open)

        return {
            "face_found": True,
            "eye_closed": eye_closed,
            "mouth_open": mouth_open,
            "eye_state": f"EAR: {ear:.2f} ({'闭合' if eye_closed else '正常'})",
            "mouth_state": f"MAR: {mar:.2f} ({'张开' if mouth_open else '闭合'})",
            "ear": ear,
            "mar": mar,
            "perclos": window_features["perclos"],
            "blink_count": self.blink_count,
            "blink_rate": window_features["blink_rate"],
            "yawn_count": window_features["yawn_count"],
            "window_features": window_features,
            "fatigue_features": [
                f"EAR={ear:.2f}",
                f"MAR={mar:.2f}",
                f"PERCLOS={window_features['perclos']:.2f}",
                f"LongestEyeClose={window_features['longest_eye_closure']:.1f}s",
                f"YawnDuration={window_features['max_yawn_duration']:.1f}s",
            ],
        }

    @property
    def start_time(self):
        if not hasattr(self, "_start_time"):
            self._start_time = time.time()
        return self._start_time

    @property
    def perclos(self):
        return self.window_features["perclos"]

    @property
    def window_features(self):
        self._prune_windows(time.time())
        frame_count = len(self.frame_window)
        closed_count = sum(1 for item in self.frame_window if item["eye_closed"])
        mouth_open_count = sum(1 for item in self.frame_window if item["mouth_open"])
        ears = [item["ear"] for item in self.frame_window if item["ear"] is not None]
        mars = [item["mar"] for item in self.frame_window if item["mar"] is not None]
        window_span = self._window_span_seconds()
        current_eye_closure = self._current_duration(self._closed_start_time)
        current_yawn_duration = self._current_duration(self._mouth_open_start_time)
        longest_eye_closure = max(
            [duration for _, duration in self.eye_closure_events] + [current_eye_closure, 0.0]
        )
        max_yawn_duration = max(
            [duration for _, duration in self.yawn_duration_events] + [current_yawn_duration, 0.0]
        )
        return {
            "window_seconds": self.config.window_seconds,
            "active_window_seconds": window_span,
            "frame_count": frame_count,
            "perclos": closed_count / max(1, frame_count),
            "blink_count": len(self.blink_events),
            "blink_rate": len(self.blink_events) / max(window_span, 1.0),
            "yawn_count": len(self.yawn_events),
            "mouth_open_ratio": mouth_open_count / max(1, frame_count),
            "longest_eye_closure": longest_eye_closure,
            "current_eye_closure": current_eye_closure,
            "max_yawn_duration": max_yawn_duration,
            "current_yawn_duration": current_yawn_duration,
            "ear_mean": float(np.mean(ears)) if ears else None,
            "ear_min": float(np.min(ears)) if ears else None,
            "mar_mean": float(np.mean(mars)) if mars else None,
            "mar_max": float(np.max(mars)) if mars else None,
        }

    def _update_counts(self, eye_closed, mouth_open, ear=None, mar=None):
        now = time.time()
        self.total_frames += 1
        self.frame_window.append({
            "time": now,
            "eye_closed": eye_closed,
            "mouth_open": mouth_open,
            "ear": ear,
            "mar": mar,
        })

        if eye_closed:
            self.eye_closed_frames += 1
            self.closed_streak += 1
            self.eye_was_closed = True
            if self._closed_start_time is None:
                self._closed_start_time = now
        else:
            if self._closed_start_time is not None:
                duration = now - self._closed_start_time
                self._last_longest_eye_closure = max(self._last_longest_eye_closure, duration)
                self.eye_closure_events.append((now, duration))
                self._closed_start_time = None
            if self.eye_was_closed and self.closed_streak >= self.config.min_blink_frames:
                self.blink_count += 1
                self.blink_events.append(now)
            self.closed_streak = 0
            self.eye_was_closed = False

        if mouth_open:
            self.mouth_open_streak += 1
            if self._mouth_open_start_time is None:
                self._mouth_open_start_time = now
            enough_frames = self.mouth_open_streak >= self.config.min_yawn_frames
            cooled_down = (
                self.last_yawn_time is None
                or now - self.last_yawn_time >= self.config.yawn_cooldown_seconds
            )
            if enough_frames and not self.yawn_active and cooled_down:
                self.yawn_count += 1
                self.last_yawn_time = now
                self.yawn_events.append(now)
                self.yawn_active = True
        else:
            if self._mouth_open_start_time is not None:
                duration = now - self._mouth_open_start_time
                self._last_yawn_duration = max(self._last_yawn_duration, duration)
                self.yawn_duration_events.append((now, duration))
                self._mouth_open_start_time = None
            self.mouth_open_streak = 0
            self.yawn_active = False
        self._prune_windows(now)

    def _prune_windows(self, now):
        cutoff = now - self.config.window_seconds
        while self.frame_window and self.frame_window[0]["time"] < cutoff:
            self.frame_window.popleft()
        while self.blink_events and self.blink_events[0] < cutoff:
            self.blink_events.popleft()
        while self.yawn_events and self.yawn_events[0] < cutoff:
            self.yawn_events.popleft()
        while self.eye_closure_events and self.eye_closure_events[0][0] < cutoff:
            self.eye_closure_events.popleft()
        while self.yawn_duration_events and self.yawn_duration_events[0][0] < cutoff:
            self.yawn_duration_events.popleft()

    def _window_span_seconds(self):
        if len(self.frame_window) < 2:
            return min(max(time.time() - self.start_time, 1.0), self.config.window_seconds)
        return max(self.frame_window[-1]["time"] - self.frame_window[0]["time"], 1.0)

    @staticmethod
    def _current_duration(start_time):
        return 0.0 if start_time is None else time.time() - start_time

    def _draw_status(self, frame, landmarks, width, height, ear, mar, eye_closed, mouth_open):
        for idx in self.LEFT_EYE + self.RIGHT_EYE + self.MOUTH_HORIZONTAL:
            point = self._point(landmarks, idx, width, height).astype(int)
            cv2.circle(frame, tuple(point), 2, (0, 255, 255), -1)
        for top_idx, bottom_idx in self.MOUTH_VERTICAL:
            for idx in (top_idx, bottom_idx):
                point = self._point(landmarks, idx, width, height).astype(int)
                cv2.circle(frame, tuple(point), 2, (255, 200, 0), -1)

        eye_color = (0, 0, 255) if eye_closed else (0, 220, 0)
        mouth_color = (0, 0, 255) if mouth_open else (0, 220, 0)
        cv2.putText(frame, f"EAR {ear:.2f}", (18, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, eye_color, 2)
        cv2.putText(frame, f"MAR {mar:.2f}", (18, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.7, mouth_color, 2)
        features = self.window_features
        cv2.putText(frame, f"PERCLOS {features['perclos']:.2f}", (18, 92), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(frame, f"Close {features['longest_eye_closure']:.1f}s", (18, 122), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
