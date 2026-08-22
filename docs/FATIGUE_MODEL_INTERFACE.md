# Fatigue State Model Interface

## Purpose

The fatigue-state classifier is separated from facial landmark extraction.
`LandmarkFatigueTracker` produces eye and mouth temporal statistics, while a
`FatigueStateModel` consumes those statistics and predicts `normal` or
`fatigue`.

This contract is shared by dataset collection, model training, offline
evaluation, and real-time inference.

## Input

Each sample represents one observation window and uses this fixed order:

1. `ear_mean` - mean eye aspect ratio.
2. `ear_min` - minimum eye aspect ratio.
3. `mar_mean` - mean mouth aspect ratio.
4. `mar_max` - maximum mouth aspect ratio.
5. `perclos` - proportion of frames with closed eyes, in `[0, 1]`.
6. `blink_count` - blink events in the window.
7. `blink_rate` - blink events per second.
8. `yawn_count` - yawn events in the window.
9. `longest_eye_closure` - longest eye closure in seconds.
10. `max_yawn_duration` - longest mouth-open event in seconds.

The canonical definition is `FEATURE_NAMES` in `fatigue_state_model.py`.
Training and inference must not reorder or omit these fields.

## Output

Every classifier returns a `FatiguePrediction` containing:

- `fatigue`: boolean decision.
- `state`: `normal` or `fatigue`.
- `confidence`: probability-like score in `[0, 1]`.
- `model_name`: identifier of the classifier producing the result.

## Boundary

MediaPipe and SSD remain feature detectors. The later RandomForest classifier
implements `FatigueStateModel` and owns the final fatigue-state decision. The
existing weighted rule can implement the same interface as a runtime fallback.
