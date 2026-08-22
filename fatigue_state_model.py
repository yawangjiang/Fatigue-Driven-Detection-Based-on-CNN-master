"""Stable input/output contract for fatigue-state classifiers.

The landmark tracker extracts temporal eye and mouth features. Classifiers
consume the validated feature vector defined here and return a prediction with
a fatigue probability. Training and real-time inference must share this exact
feature order.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
import math
from typing import Mapping, Tuple


NORMAL_LABEL = "normal"
FATIGUE_LABEL = "fatigue"

FEATURE_NAMES = (
    "ear_mean",
    "ear_min",
    "mar_mean",
    "mar_max",
    "perclos",
    "blink_count",
    "blink_rate",
    "yawn_count",
    "longest_eye_closure",
    "max_yawn_duration",
)


@dataclass(frozen=True)
class FatigueFeatureVector:
    """Eye and mouth temporal statistics for one observation window."""

    ear_mean: float
    ear_min: float
    mar_mean: float
    mar_max: float
    perclos: float
    blink_count: float
    blink_rate: float
    yawn_count: float
    longest_eye_closure: float
    max_yawn_duration: float

    @classmethod
    def from_window_features(cls, values: Mapping[str, object]):
        """Build and validate a feature vector from tracker output."""
        missing = [name for name in FEATURE_NAMES if values.get(name) is None]
        if missing:
            raise ValueError(f"Missing fatigue features: {', '.join(missing)}")

        feature = cls(**{name: float(values[name]) for name in FEATURE_NAMES})
        feature.validate()
        return feature

    def validate(self):
        """Reject non-finite values and physically invalid measurements."""
        values = self.as_tuple()
        if not all(math.isfinite(value) for value in values):
            raise ValueError("Fatigue features must contain only finite numbers")
        if not 0.0 <= self.perclos <= 1.0:
            raise ValueError("perclos must be between 0 and 1")

        non_negative = (
            self.ear_mean,
            self.ear_min,
            self.mar_mean,
            self.mar_max,
            self.blink_count,
            self.blink_rate,
            self.yawn_count,
            self.longest_eye_closure,
            self.max_yawn_duration,
        )
        if any(value < 0.0 for value in non_negative):
            raise ValueError("Fatigue features cannot be negative")
        if self.ear_min > self.ear_mean:
            raise ValueError("ear_min cannot exceed ear_mean")
        if self.mar_mean > self.mar_max:
            raise ValueError("mar_mean cannot exceed mar_max")

    def as_tuple(self) -> Tuple[float, ...]:
        """Return values in the canonical training and inference order."""
        return tuple(getattr(self, name) for name in FEATURE_NAMES)

    def as_dict(self):
        """Return a serializable mapping for CSV and diagnostic output."""
        return {name: getattr(self, name) for name in FEATURE_NAMES}


@dataclass(frozen=True)
class FatiguePrediction:
    """Binary fatigue-state prediction returned by every classifier."""

    fatigue: bool
    confidence: float
    model_name: str

    def __post_init__(self):
        if not math.isfinite(self.confidence) or not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")
        if not self.model_name.strip():
            raise ValueError("model_name cannot be empty")

    @property
    def state(self):
        return FATIGUE_LABEL if self.fatigue else NORMAL_LABEL

    def as_dict(self):
        """Return the stable payload expected by the runtime and UI."""
        return {
            "fatigue": self.fatigue,
            "state": self.state,
            "confidence": self.confidence,
            "model_name": self.model_name,
        }


class FatigueStateModel(ABC):
    """Interface implemented by trained and fallback fatigue classifiers."""

    @abstractmethod
    def predict(self, features: FatigueFeatureVector) -> FatiguePrediction:
        """Classify one validated eye/mouth observation window."""
        raise NotImplementedError
