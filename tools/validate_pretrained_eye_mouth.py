"""Validate FatigueSense eye and mouth checkpoints in isolation.

This script intentionally avoids importing the upstream FatigueSense package,
which requires Python 3.10+. It reproduces only the published MobileNetV3-Small
architecture and preprocessing needed for checkpoint compatibility testing.
"""

import argparse
import json
import math
from pathlib import Path
import time
from typing import Dict, Iterable, Optional, Sequence, Tuple

import cv2
import mediapipe as mp
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import mobilenet_v3_small


LEFT_EYE_INDICES = (33, 133, 160, 159, 158, 144, 145, 153)
RIGHT_EYE_INDICES = (362, 263, 387, 386, 385, 373, 374, 380)
MOUTH_INDICES = (61, 291, 81, 178, 13, 14, 402, 311, 308)
LEFT_EYE_OUTER_CORNER = 33
RIGHT_EYE_OUTER_CORNER = 263

INPUT_SIZE = (64, 64)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

EYE_BOX_W_RATIO = 0.50
EYE_BOX_H_RATIO = 0.35
MOUTH_LANDMARK_W_SCALE = 1.6
MOUTH_LANDMARK_H_SCALE = 2.5
MOUTH_BOX_H_RATIO = 0.70
MOUTH_VERTICAL_OFFSET_RATIO = 0.20


class BinaryROIClassifier(nn.Module):
    """Published FatigueSense MobileNetV3-Small binary ROI classifier."""

    BACKBONE_OUT_FEATURES = 576

    def __init__(self):
        super().__init__()
        base = mobilenet_v3_small(weights=None)
        self.backbone = base.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Linear(self.BACKBONE_OUT_FEATURES, 128),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(128, 2),
        )

    def forward(self, image):
        features = self.backbone(image)
        features = self.pool(features)
        return self.head(features.flatten(1))


def load_model(checkpoint_path: Path, device: torch.device):
    """Load a tensor-only checkpoint and require an exact architecture match."""
    model = BinaryROIClassifier()
    state = torch.load(str(checkpoint_path), map_location="cpu", weights_only=True)
    if not isinstance(state, dict):
        raise TypeError(f"Checkpoint must contain a state dict: {checkpoint_path}")
    if "state_dict" in state and isinstance(state["state_dict"], dict):
        state = state["state_dict"]
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    return model


def point(landmarks, index: int, width: int, height: int):
    item = landmarks[index]
    return int(item.x * width), int(item.y * height)


def region_center(landmarks, indices: Iterable[int], width: int, height: int):
    points = [point(landmarks, index, width, height) for index in indices]
    return (
        int(sum(item[0] for item in points) / len(points)),
        int(sum(item[1] for item in points) / len(points)),
    )


def crop_aligned_box(
    frame: np.ndarray,
    center: Tuple[int, int],
    angle_deg: float,
    box_width: int,
    box_height: int,
):
    """Rotate the face around a region center and return a 64x64 crop."""
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
    return cv2.resize(crop, INPUT_SIZE, interpolation=cv2.INTER_CUBIC)


def extract_crops(frame: np.ndarray):
    """Extract left eye, right eye, and mouth crops with MediaPipe FaceMesh."""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    with mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
    ) as face_mesh:
        result = face_mesh.process(rgb)
    if not result.multi_face_landmarks:
        raise RuntimeError("MediaPipe did not find a face in the validation image")

    landmarks = result.multi_face_landmarks[0].landmark
    height, width = frame.shape[:2]
    left_outer = point(landmarks, LEFT_EYE_OUTER_CORNER, width, height)
    right_outer = point(landmarks, RIGHT_EYE_OUTER_CORNER, width, height)
    delta_x = right_outer[0] - left_outer[0]
    delta_y = right_outer[1] - left_outer[1]
    inter_eye_distance = max(1, int(math.hypot(delta_x, delta_y)))
    roll_deg = float(np.degrees(np.arctan2(delta_y, delta_x)))

    eye_box_width = int(EYE_BOX_W_RATIO * inter_eye_distance)
    eye_box_height = int(EYE_BOX_H_RATIO * inter_eye_distance)
    left_center = region_center(landmarks, LEFT_EYE_INDICES, width, height)
    right_center = region_center(landmarks, RIGHT_EYE_INDICES, width, height)

    mouth_points = [point(landmarks, index, width, height) for index in MOUTH_INDICES]
    mouth_width = max(item[0] for item in mouth_points) - min(
        item[0] for item in mouth_points
    )
    mouth_height = max(item[1] for item in mouth_points) - min(
        item[1] for item in mouth_points
    )
    mouth_box_width = max(
        inter_eye_distance, int(MOUTH_LANDMARK_W_SCALE * mouth_width)
    )
    mouth_box_height = max(
        int(MOUTH_BOX_H_RATIO * inter_eye_distance),
        int(MOUTH_LANDMARK_H_SCALE * mouth_height),
    )
    mouth_center_x, mouth_center_y = region_center(
        landmarks, MOUTH_INDICES, width, height
    )
    mouth_center_y += int(MOUTH_VERTICAL_OFFSET_RATIO * mouth_box_height)

    crops = {
        "left_eye": crop_aligned_box(
            frame,
            left_center,
            roll_deg,
            eye_box_width,
            eye_box_height,
        ),
        "right_eye": crop_aligned_box(
            frame,
            right_center,
            roll_deg,
            eye_box_width,
            eye_box_height,
        ),
        "mouth": crop_aligned_box(
            frame,
            (mouth_center_x, mouth_center_y),
            roll_deg,
            mouth_box_width,
            mouth_box_height,
        ),
    }
    if any(crop is None for crop in crops.values()):
        raise RuntimeError("One or more validation crops are empty")
    return crops


def make_transform():
    return transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize(INPUT_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def predict_probabilities(
    model: nn.Module,
    crops: Sequence[np.ndarray],
    device: torch.device,
):
    """Return two-class probabilities and measured CPU inference latency."""
    transform = make_transform()
    tensors = []
    for crop in crops:
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        tensors.append(transform(Image.fromarray(rgb)))
    batch = torch.stack(tensors).to(device)

    started = time.perf_counter()
    with torch.no_grad():
        probabilities = torch.softmax(model(batch), dim=-1).cpu().numpy()
    latency_ms = (time.perf_counter() - started) * 1000.0
    return probabilities, latency_ms


def validate_probability_pair(values: np.ndarray, name: str):
    if values.shape != (2,):
        raise ValueError(f"{name} output must contain exactly two probabilities")
    if not np.isfinite(values).all():
        raise ValueError(f"{name} output contains non-finite probabilities")
    if np.any(values < 0.0) or np.any(values > 1.0):
        raise ValueError(f"{name} probabilities must be in [0, 1]")
    if not math.isclose(float(values.sum()), 1.0, abs_tol=1e-5):
        raise ValueError(f"{name} probabilities do not sum to 1")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eye-weights", type=Path, required=True)
    parser.add_argument("--mouth-weights", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    for path in (args.eye_weights, args.mouth_weights, args.image):
        if not path.is_file():
            raise FileNotFoundError(path)

    device = torch.device(args.device)
    frame = cv2.imread(str(args.image))
    if frame is None:
        raise ValueError(f"OpenCV could not read validation image: {args.image}")

    eye_model = load_model(args.eye_weights, device)
    mouth_model = load_model(args.mouth_weights, device)
    crops = extract_crops(frame)
    eye_probs, eye_latency_ms = predict_probabilities(
        eye_model, [crops["left_eye"], crops["right_eye"]], device
    )
    mouth_probs, mouth_latency_ms = predict_probabilities(
        mouth_model, [crops["mouth"]], device
    )

    validate_probability_pair(eye_probs[0], "left_eye")
    validate_probability_pair(eye_probs[1], "right_eye")
    validate_probability_pair(mouth_probs[0], "mouth")

    report: Dict[str, object] = {
        "status": "passed",
        "device": str(device),
        "image": str(args.image.resolve()),
        "eye_weights": str(args.eye_weights.resolve()),
        "mouth_weights": str(args.mouth_weights.resolve()),
        "probabilities": {
            "left_eye": {
                "closed": float(eye_probs[0, 0]),
                "open": float(eye_probs[0, 1]),
            },
            "right_eye": {
                "closed": float(eye_probs[1, 0]),
                "open": float(eye_probs[1, 1]),
            },
            "mouth": {
                "closed": float(mouth_probs[0, 0]),
                "open": float(mouth_probs[0, 1]),
            },
        },
        "latency_ms": {
            "two_eyes": eye_latency_ms,
            "mouth": mouth_latency_ms,
        },
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
