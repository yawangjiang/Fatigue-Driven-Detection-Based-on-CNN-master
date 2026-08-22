# Pretrained Fatigue Model Adoption

## Decision

The first pretrained model candidate is the eye and mouth classification stage
from FatigueSense:

- Source: https://github.com/Fatigue-Sense/fatigue-sense
- License: MIT
- Eye checkpoint: `FatigueSense/eye_classifier/best_eye_classifier.pt`
- Mouth checkpoint: `FatigueSense/mouth_classifier/best_mouth_classifier.pt`
- Model family: MobileNetV3 binary classifiers

The initial adoption scope is limited to eye-state and mouth-state inference.
The FatigueSense pose model and BiGRU temporal model are not integrated during
the first validation stage.

## Rationale

This candidate matches the current project architecture closely:

1. MediaPipe landmarks identify eye and mouth regions.
2. Separate pretrained classifiers produce closed-eye and open-mouth
   probabilities.
3. The existing sliding-window tracker can aggregate those probabilities into
   temporal fatigue features.
4. PyTorch is already part of the current runtime.

The model remains an external pretrained component. Project documentation and
runtime diagnostics must retain its source and license attribution.

## Stage 1 Acceptance Criteria

The isolated validation stage is complete only when all criteria pass:

1. Both public checkpoints download without authentication.
2. Checkpoint files load with their published model architecture.
3. One eye crop produces finite `open` and `closed` probabilities.
4. One mouth crop produces finite `open` and `closed` probabilities.
5. Each probability pair is in `[0, 1]` and sums to approximately `1.0`.
6. CPU inference completes without modifying the current application runtime.
7. Exact dependency versions and validation commands are documented.

Failure of checkpoint loading, missing architecture metadata, incompatible
dependencies, or unclear output labels stops the integration for review.

## Integration Boundary

The later adapter will convert pretrained model output into the project's
stable feature contract:

- Eye output: `P(eye_closed)`
- Mouth output: `P(mouth_open)`
- Temporal output: existing window features in `LandmarkFatigueTracker`
- Final output: `FatiguePrediction` from `fatigue_state_model.py`

The existing landmark-rule evaluator remains available as a runtime fallback
until the pretrained path passes local evaluation.

## Known Risks

- FatigueSense officially targets Python 3.10 or newer.
- Its published full pipeline includes pose features and a long warm-up period.
- Eye and mouth crops in this project may differ from the model's training
  distribution.
- Published performance does not replace evaluation on this project's camera
  and users.
