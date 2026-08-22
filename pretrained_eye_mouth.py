"""Pretrained eye and mouth state classifiers used by the fatigue pipeline.

The architecture and preprocessing follow the MIT-licensed FatigueSense
project. Checkpoints contain only model parameters and are loaded with
``weights_only=True`` to avoid executing serialized Python objects.
"""

from dataclasses import asdict, dataclass
import hashlib
import math
from pathlib import Path
import time
from typing import Dict, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import mobilenet_v3_small


PathLike = Union[str, Path]

INPUT_SIZE = (64, 64)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# These hashes identify the FatigueSense checkpoints validated for this project.
EXPECTED_EYE_SHA256 = (
    "d62b57da4500a7748571e8c3adc378990dc99fd3c28c48cef97bd97bef9940f3"
)
EXPECTED_MOUTH_SHA256 = (
    "104492bde316d9b4bcef1efafcb8e1374798bafc1b85d3e6f914427e3f42af84"
)


class BinaryROIClassifier(nn.Module):
    """MobileNetV3-Small classifier compatible with FatigueSense weights."""

    BACKBONE_OUT_FEATURES = 576

    def __init__(self) -> None:
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

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        features = self.backbone(image)
        features = self.pool(features)
        return self.head(features.flatten(1))


@dataclass(frozen=True)
class EyeMouthProbabilities:
    """State probabilities from one frame.

    Eye class 0 means closed and mouth class 1 means open. The corresponding
    complementary probability can therefore be calculated as ``1 - value``.
    """

    left_eye_closed: float
    right_eye_closed: float
    mouth_open: float
    eye_latency_ms: float
    mouth_latency_ms: float

    def __post_init__(self) -> None:
        probability_fields = (
            "left_eye_closed",
            "right_eye_closed",
            "mouth_open",
        )
        for field_name in probability_fields:
            value = getattr(self, field_name)
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{field_name} must be a finite value in [0, 1]")

        for field_name in ("eye_latency_ms", "mouth_latency_ms"):
            value = getattr(self, field_name)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{field_name} must be a finite non-negative value")

    @property
    def mean_eye_closed(self) -> float:
        return (self.left_eye_closed + self.right_eye_closed) / 2.0

    def as_dict(self) -> Dict[str, float]:
        return asdict(self)


def sha256_file(path: PathLike, chunk_size: int = 1024 * 1024) -> str:
    """Calculate a checkpoint digest without loading the entire file at once."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as checkpoint:
        while True:
            chunk = checkpoint.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def verify_checkpoint(path: PathLike, expected_sha256: str) -> Path:
    """Require a readable checkpoint whose contents match the known digest."""
    checkpoint_path = Path(path).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)

    actual_sha256 = sha256_file(checkpoint_path)
    if actual_sha256.lower() != expected_sha256.lower():
        raise ValueError(
            f"Checkpoint hash mismatch for {checkpoint_path}: "
            f"expected {expected_sha256}, got {actual_sha256}"
        )
    return checkpoint_path


def _load_model(checkpoint_path: Path, device: torch.device) -> nn.Module:
    model = BinaryROIClassifier()
    state = torch.load(
        str(checkpoint_path),
        map_location="cpu",
        weights_only=True,
    )
    if not isinstance(state, dict):
        raise TypeError(f"Checkpoint must contain a state dict: {checkpoint_path}")
    if "state_dict" in state and isinstance(state["state_dict"], dict):
        state = state["state_dict"]
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def _make_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize(INPUT_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


class PretrainedEyeMouthClassifier:
    """Load validated checkpoints and classify BGR eye and mouth crops."""

    def __init__(
        self,
        eye_weights: PathLike,
        mouth_weights: PathLike,
        device: str = "cpu",
        verify_hashes: bool = True,
        eye_sha256: str = EXPECTED_EYE_SHA256,
        mouth_sha256: str = EXPECTED_MOUTH_SHA256,
    ) -> None:
        self.device = torch.device(device)
        self.eye_weights = self._prepare_checkpoint(
            eye_weights,
            eye_sha256,
            verify_hashes,
        )
        self.mouth_weights = self._prepare_checkpoint(
            mouth_weights,
            mouth_sha256,
            verify_hashes,
        )
        self.eye_model = _load_model(self.eye_weights, self.device)
        self.mouth_model = _load_model(self.mouth_weights, self.device)
        self.transform = _make_transform()

    @staticmethod
    def _prepare_checkpoint(
        path: PathLike,
        expected_sha256: str,
        verify_hashes: bool,
    ) -> Path:
        checkpoint_path = Path(path).expanduser().resolve()
        if verify_hashes:
            return verify_checkpoint(checkpoint_path, expected_sha256)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(checkpoint_path)
        return checkpoint_path

    def _prepare_batch(self, crops: Sequence[np.ndarray]) -> torch.Tensor:
        if not crops:
            raise ValueError("At least one crop is required")

        tensors = []
        for index, crop in enumerate(crops):
            if not isinstance(crop, np.ndarray) or crop.size == 0:
                raise ValueError(f"Crop {index} must be a non-empty numpy array")
            if crop.ndim != 3 or crop.shape[2] != 3:
                raise ValueError(f"Crop {index} must be a three-channel BGR image")
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            tensors.append(self.transform(Image.fromarray(rgb)))
        return torch.stack(tensors).to(self.device)

    def _predict(
        self,
        model: nn.Module,
        crops: Sequence[np.ndarray],
    ) -> Tuple[np.ndarray, float]:
        batch = self._prepare_batch(crops)
        started = time.perf_counter()
        with torch.inference_mode():
            probabilities = torch.softmax(model(batch), dim=-1).cpu().numpy()
        latency_ms = (time.perf_counter() - started) * 1000.0
        return probabilities, latency_ms

    def predict(
        self,
        left_eye_crop: np.ndarray,
        right_eye_crop: np.ndarray,
        mouth_crop: np.ndarray,
    ) -> EyeMouthProbabilities:
        """Return closed-eye and open-mouth probabilities for one frame."""
        eye_probs, eye_latency_ms = self._predict(
            self.eye_model,
            (left_eye_crop, right_eye_crop),
        )
        mouth_probs, mouth_latency_ms = self._predict(
            self.mouth_model,
            (mouth_crop,),
        )
        return EyeMouthProbabilities(
            left_eye_closed=float(eye_probs[0, 0]),
            right_eye_closed=float(eye_probs[1, 0]),
            mouth_open=float(mouth_probs[0, 1]),
            eye_latency_ms=eye_latency_ms,
            mouth_latency_ms=mouth_latency_ms,
        )
