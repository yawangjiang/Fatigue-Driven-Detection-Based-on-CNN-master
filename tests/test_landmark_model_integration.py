import unittest

import numpy as np

from fatigue_landmark_detector import LandmarkFatigueConfig, LandmarkFatigueTracker
from pretrained_eye_mouth import EyeMouthProbabilities


class FakeClassifier:
    def __init__(self, result=None, error=None):
        self.result = result
        self.error = error
        self.calls = 0

    def predict(self, left_eye, right_eye, mouth):
        self.calls += 1
        if self.error is not None:
            raise self.error
        return self.result


def make_tracker(classifier):
    tracker = LandmarkFatigueTracker.__new__(LandmarkFatigueTracker)
    tracker.config = LandmarkFatigueConfig()
    tracker.state_classifier = classifier
    tracker.classifier_error = None
    crop = np.zeros((64, 64, 3), dtype=np.uint8)
    tracker._extract_model_crops = lambda frame, landmarks, width, height: (
        crop,
        crop,
        crop,
    )
    return tracker


class LandmarkModelIntegrationTest(unittest.TestCase):
    def test_model_probabilities_override_landmark_ratios(self):
        classifier = FakeClassifier(
            EyeMouthProbabilities(
                left_eye_closed=0.90,
                right_eye_closed=0.80,
                mouth_open=0.75,
                eye_latency_ms=12.0,
                mouth_latency_ms=4.0,
            )
        )
        tracker = make_tracker(classifier)

        state = tracker._resolve_eye_mouth_state(
            np.zeros((100, 100, 3), dtype=np.uint8),
            landmarks=[],
            width=100,
            height=100,
            ear=0.40,
            mar=0.20,
        )

        self.assertTrue(state["eye_closed"])
        self.assertTrue(state["mouth_open"])
        self.assertEqual(state["detector"], "pretrained_cnn")
        self.assertAlmostEqual(state["eye_closed_probability"], 0.85)
        self.assertEqual(state["model_latency_ms"], 16.0)

    def test_missing_classifier_uses_landmark_thresholds(self):
        tracker = make_tracker(None)

        state = tracker._resolve_eye_mouth_state(
            np.zeros((100, 100, 3), dtype=np.uint8),
            landmarks=[],
            width=100,
            height=100,
            ear=0.20,
            mar=0.60,
        )

        self.assertTrue(state["eye_closed"])
        self.assertTrue(state["mouth_open"])
        self.assertEqual(state["detector"], "landmark_rule")
        self.assertIsNone(state["eye_closed_probability"])

    def test_classifier_error_disables_model_and_uses_fallback(self):
        classifier = FakeClassifier(error=RuntimeError("inference failed"))
        tracker = make_tracker(classifier)

        state = tracker._resolve_eye_mouth_state(
            np.zeros((100, 100, 3), dtype=np.uint8),
            landmarks=[],
            width=100,
            height=100,
            ear=0.30,
            mar=0.20,
        )

        self.assertFalse(state["eye_closed"])
        self.assertFalse(state["mouth_open"])
        self.assertEqual(state["detector"], "landmark_rule")
        self.assertIsNone(tracker.state_classifier)
        self.assertEqual(tracker.classifier_error, "inference failed")


if __name__ == "__main__":
    unittest.main()
