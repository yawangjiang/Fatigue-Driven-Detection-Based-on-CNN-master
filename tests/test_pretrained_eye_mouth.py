import hashlib
from pathlib import Path
import tempfile
import unittest

import torch

from pretrained_eye_mouth import (
    BinaryROIClassifier,
    EyeMouthProbabilities,
    sha256_file,
    verify_checkpoint,
)


class EyeMouthProbabilitiesTest(unittest.TestCase):
    def test_exposes_mean_eye_probability_and_dictionary(self):
        result = EyeMouthProbabilities(
            left_eye_closed=0.2,
            right_eye_closed=0.6,
            mouth_open=0.7,
            eye_latency_ms=10.0,
            mouth_latency_ms=5.0,
        )

        self.assertAlmostEqual(result.mean_eye_closed, 0.4)
        self.assertEqual(result.as_dict()["mouth_open"], 0.7)

    def test_rejects_invalid_probability(self):
        with self.assertRaisesRegex(ValueError, "left_eye_closed"):
            EyeMouthProbabilities(
                left_eye_closed=1.1,
                right_eye_closed=0.5,
                mouth_open=0.5,
                eye_latency_ms=1.0,
                mouth_latency_ms=1.0,
            )


class CheckpointVerificationTest(unittest.TestCase):
    def test_accepts_expected_digest_and_rejects_wrong_digest(self):
        payload = b"known checkpoint content"
        expected = hashlib.sha256(payload).hexdigest()
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "model.pt"
            checkpoint.write_bytes(payload)

            self.assertEqual(sha256_file(checkpoint), expected)
            self.assertEqual(verify_checkpoint(checkpoint, expected), checkpoint)
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                verify_checkpoint(checkpoint, "0" * 64)


class BinaryROIClassifierTest(unittest.TestCase):
    def test_outputs_two_logits_per_crop(self):
        model = BinaryROIClassifier().eval()
        with torch.inference_mode():
            output = model(torch.zeros(2, 3, 64, 64))

        self.assertEqual(tuple(output.shape), (2, 2))


if __name__ == "__main__":
    unittest.main()
