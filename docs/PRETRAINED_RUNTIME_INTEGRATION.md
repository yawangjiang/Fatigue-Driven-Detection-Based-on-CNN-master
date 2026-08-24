# Pretrained Runtime Integration

## Runtime Flow

The live fatigue tracker now uses this order:

1. MediaPipe FaceMesh detects one face and provides facial landmarks.
2. The tracker extracts aligned `64x64` left-eye, right-eye, and mouth crops.
3. The pretrained classifiers return closed-eye and open-mouth probabilities.
4. The bilateral mean eye probability is compared with `0.80`.
5. The mouth-open probability is compared with `0.70`.
6. The existing sliding window calculates PERCLOS, blink rate, closure duration,
   mouth-open duration, and yawn count.
7. `FatigueEvaluator` applies the existing temporal fatigue rules.

The pretrained classifiers replace single-frame EAR/MAR state decisions. They
do not yet replace the final temporal fatigue evaluator.

## Configuration

`core_engine.py` reads these environment variables:

- `FATIGUE_EYE_WEIGHTS`
- `FATIGUE_MOUTH_WEIGHTS`
- `FATIGUE_MODEL_DEVICE` (`cpu` by default)

Without environment variables, it looks for:

- `weights/best_eye_classifier.pt`
- `weights/best_mouth_classifier.pt`

`start_ui.bat` loads `fatigue_models.local.bat` when present. That local file is
ignored by Git. `fatigue_models.example.bat` documents the portable format.

## Fallback Behavior

- Missing or invalid checkpoints: application startup continues with EAR/MAR.
- Empty eye or mouth crop: only that frame uses EAR/MAR.
- Classifier inference exception: the classifier is disabled for that tracker
  session and later frames use EAR/MAR.
- MediaPipe tracker exception: the existing SSD detector remains the final
  runtime fallback.

Every successful face result identifies the active state detector as either
`pretrained_cnn` or `landmark_rule`. CNN results also include
`eye_closed_probability`, `mouth_open_probability`, and `model_latency_ms` for
later performance evaluation.

## Acceptance Result

Date: 2026-08-24

- All seven unit tests passed.
- `core_engine.init_eye_mouth_classifier()` returned
  `PretrainedEyeMouthClassifier` with the configured external checkpoints.
- The production `LandmarkFatigueTracker.process_frame()` path returned
  `detector: pretrained_cnn` on the project sample image.
- Mean closed-eye probability: `0.005919`.
- Mouth-open probability: `0.163995`.
- Both states were correctly classified as open eyes and closed mouth.
- EAR and MAR remained available for diagnostics and fallback.

MediaPipe initialization warnings were non-blocking. Accuracy and sustained
camera FPS comparison are intentionally deferred to the evaluation stage.
