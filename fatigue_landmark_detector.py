from collections import deque
from dataclasses import dataclass
import math
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
    eye_closed_probability_threshold: float = 0.80
    mouth_open_probability_threshold: float = 0.70
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

    # Crop geometry matches the FatigueSense classifier training pipeline.
    MODEL_LEFT_EYE = (33, 133, 160, 159, 158, 144, 145, 153)
    MODEL_RIGHT_EYE = (362, 263, 387, 386, 385, 373, 374, 380)
    MODEL_MOUTH = (61, 291, 81, 178, 13, 14, 402, 311, 308)
    LEFT_EYE_OUTER_CORNER = 33
    RIGHT_EYE_OUTER_CORNER = 263
    MODEL_INPUT_SIZE = (64, 64)
    EYE_BOX_W_RATIO = 0.50
    EYE_BOX_H_RATIO = 0.35
    MOUTH_LANDMARK_W_SCALE = 1.6
    MOUTH_LANDMARK_H_SCALE = 2.5
    MOUTH_BOX_H_RATIO = 0.70
    MOUTH_VERTICAL_OFFSET_RATIO = 0.20

    def __init__(self, config=None, max_faces=1, state_classifier=None):
        if mp is None:
            raise RuntimeError("mediapipe is not installed")
        self.config = config or LandmarkFatigueConfig()
        self.state_classifier = state_classifier
        self.classifier_error = None
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

    def _region_center(self, landmarks, indices, width, height):
        points = [
            self._point(landmarks, index, width, height).astype(np.int32)
            for index in indices
        ]
        center = np.mean(points, axis=0)
        return int(center[0]), int(center[1])

    @staticmethod
    def _crop_aligned_box(frame, center, angle_deg, box_width, box_height):
        """Rotate around a facial region and return a 64x64 BGR crop."""
        frame_height, frame_width = frame.shape[:2]
        center_x, center_y = center
        matrix = cv2.getRotationMatrix2D(
            (float(center_x), float(center_y)), -angle_deg, 1.0
        )
        rotated = cv2.warpAffine(
            frame,
            matrix,
            (frame_width, frame_height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )

        x1 = max(0, int(center_x - box_width / 2))
        y1 = max(0, int(center_y - box_height / 2))
        x2 = min(frame_width, int(center_x + box_width / 2))
        y2 = min(frame_height, int(center_y + box_height / 2))
        if x2 <= x1 or y2 <= y1:
            return None

        crop = rotated[y1:y2, x1:x2]
        if crop.size == 0:
            return None
        return cv2.resize(
            crop,
            LandmarkFatigueTracker.MODEL_INPUT_SIZE,
            interpolation=cv2.INTER_CUBIC,
        )

    def _extract_model_crops(self, frame, landmarks, width, height):
        left_outer = self._point(
            landmarks, self.LEFT_EYE_OUTER_CORNER, width, height
        ).astype(np.int32)
        right_outer = self._point(
            landmarks, self.RIGHT_EYE_OUTER_CORNER, width, height
        ).astype(np.int32)
        delta_x, delta_y = right_outer - left_outer
        inter_eye_distance = max(1, int(math.hypot(delta_x, delta_y)))
        roll_deg = float(np.degrees(np.arctan2(delta_y, delta_x)))

        eye_box_width = int(self.EYE_BOX_W_RATIO * inter_eye_distance)
        eye_box_height = int(self.EYE_BOX_H_RATIO * inter_eye_distance)
        left_center = self._region_center(
            landmarks, self.MODEL_LEFT_EYE, width, height
        )
        right_center = self._region_center(
            landmarks, self.MODEL_RIGHT_EYE, width, height
        )

        mouth_points = [
            self._point(landmarks, index, width, height).astype(np.int32)
            for index in self.MODEL_MOUTH
        ]
        mouth_x = [point[0] for point in mouth_points]
        mouth_y = [point[1] for point in mouth_points]
        mouth_landmark_width = max(mouth_x) - min(mouth_x)
        mouth_landmark_height = max(mouth_y) - min(mouth_y)
        mouth_box_width = max(
            inter_eye_distance,
            int(self.MOUTH_LANDMARK_W_SCALE * mouth_landmark_width),
        )
        mouth_box_height = max(
            int(self.MOUTH_BOX_H_RATIO * inter_eye_distance),
            int(self.MOUTH_LANDMARK_H_SCALE * mouth_landmark_height),
        )
        mouth_center_x, mouth_center_y = self._region_center(
            landmarks, self.MODEL_MOUTH, width, height
        )
        mouth_center_y += int(
            self.MOUTH_VERTICAL_OFFSET_RATIO * mouth_box_height
        )

        return (
            self._crop_aligned_box(
                frame,
                left_center,
                roll_deg,
                eye_box_width,
                eye_box_height,
            ),
            self._crop_aligned_box(
                frame,
                right_center,
                roll_deg,
                eye_box_width,
                eye_box_height,
            ),
            self._crop_aligned_box(
                frame,
                (mouth_center_x, mouth_center_y),
                roll_deg,
                mouth_box_width,
                mouth_box_height,
            ),
        )

    def _resolve_eye_mouth_state(
        self, frame, landmarks, width, height, ear, mar
    ):
        fallback = {
            "eye_closed": ear < self.config.ear_threshold,
            "mouth_open": mar > self.config.mar_threshold,
            "eye_closed_probability": None,
            "mouth_open_probability": None,
            "model_latency_ms": None,
            "detector": "landmark_rule",
        }
        if self.state_classifier is None:
            return fallback

        crops = self._extract_model_crops(frame, landmarks, width, height)
        if any(crop is None for crop in crops):
            return fallback

        try:
            probabilities = self.state_classifier.predict(*crops)
        except Exception as exc:
            # Disable a broken classifier so it cannot stall every later frame.
            self.classifier_error = str(exc)
            self.state_classifier = None
            print(f"预训练眼嘴模型推理失败，回退到EAR/MAR: {exc}")
            return fallback

        return {
            "eye_closed": (
                probabilities.mean_eye_closed
                >= self.config.eye_closed_probability_threshold
            ),
            "mouth_open": (
                probabilities.mouth_open
                >= self.config.mouth_open_probability_threshold
            ),
            "eye_closed_probability": probabilities.mean_eye_closed,
            "mouth_open_probability": probabilities.mouth_open,
            "model_latency_ms": (
                probabilities.eye_latency_ms + probabilities.mouth_latency_ms
            ),
            "detector": "pretrained_cnn",
        }

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
                "detector": "landmark_rule",
            }

        landmarks = results.multi_face_landmarks[0].landmark
        left_ear = self._eye_aspect_ratio(landmarks, self.LEFT_EYE, width, height)
        right_ear = self._eye_aspect_ratio(landmarks, self.RIGHT_EYE, width, height)
        ear = (left_ear + right_ear) / 2.0
        mar = self._mouth_aspect_ratio(landmarks, width, height)

        state = self._resolve_eye_mouth_state(
            frame, landmarks, width, height, ear, mar
        )
        eye_closed = state["eye_closed"]
        mouth_open = state["mouth_open"]

        if update:
            self._update_counts(eye_closed, mouth_open, ear, mar)

        window_features = self.window_features

        if draw:
            self._draw_status(
                frame,
                landmarks,
                width,
                height,
                ear,
                mar,
                eye_closed,
                mouth_open,
                state,
            )

        if state["detector"] == "pretrained_cnn":
            eye_state = (
                f"CNN: {state['eye_closed_probability']:.2f} "
                f"({'闭合' if eye_closed else '正常'})"
            )
            mouth_state = (
                f"CNN: {state['mouth_open_probability']:.2f} "
                f"({'张开' if mouth_open else '闭合'})"
            )
        else:
            eye_state = f"EAR: {ear:.2f} ({'闭合' if eye_closed else '正常'})"
            mouth_state = f"MAR: {mar:.2f} ({'张开' if mouth_open else '闭合'})"

        return {
            "face_found": True,
            "eye_closed": eye_closed,
            "mouth_open": mouth_open,
            "eye_state": eye_state,
            "mouth_state": mouth_state,
            "ear": ear,
            "mar": mar,
            "eye_closed_probability": state["eye_closed_probability"],
            "mouth_open_probability": state["mouth_open_probability"],
            "model_latency_ms": state["model_latency_ms"],
            "detector": state["detector"],
            "perclos": window_features["perclos"],
            "blink_count": self.blink_count,
            "blink_rate": window_features["blink_rate"],
            "yawn_count": window_features["yawn_count"],
            "window_features": window_features,
            "fatigue_features": [
                (
                    f"EyeCNN={state['eye_closed_probability']:.2f}"
                    if state["eye_closed_probability"] is not None
                    else f"EAR={ear:.2f}"
                ),
                (
                    f"MouthCNN={state['mouth_open_probability']:.2f}"
                    if state["mouth_open_probability"] is not None
                    else f"MAR={mar:.2f}"
                ),
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

    def _draw_status(
        self,
        frame,
        landmarks,
        width,
        height,
        ear,
        mar,
        eye_closed,
        mouth_open,
        state,
    ):
        for idx in self.LEFT_EYE + self.RIGHT_EYE + self.MOUTH_HORIZONTAL:
            point = self._point(landmarks, idx, width, height).astype(int)
            cv2.circle(frame, tuple(point), 2, (0, 255, 255), -1)
        for top_idx, bottom_idx in self.MOUTH_VERTICAL:
            for idx in (top_idx, bottom_idx):
                point = self._point(landmarks, idx, width, height).astype(int)
                cv2.circle(frame, tuple(point), 2, (255, 200, 0), -1)

        eye_color = (0, 0, 255) if eye_closed else (0, 220, 0)
        mouth_color = (0, 0, 255) if mouth_open else (0, 220, 0)
        if state["detector"] == "pretrained_cnn":
            eye_label = f"Eye CNN {state['eye_closed_probability']:.2f}"
            mouth_label = f"Mouth CNN {state['mouth_open_probability']:.2f}"
        else:
            eye_label = f"EAR {ear:.2f}"
            mouth_label = f"MAR {mar:.2f}"
        cv2.putText(frame, eye_label, (18, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, eye_color, 2)
        cv2.putText(frame, mouth_label, (18, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.7, mouth_color, 2)
        features = self.window_features
        cv2.putText(frame, f"PERCLOS {features['perclos']:.2f}", (18, 92), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(frame, f"Close {features['longest_eye_closure']:.1f}s", (18, 122), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
