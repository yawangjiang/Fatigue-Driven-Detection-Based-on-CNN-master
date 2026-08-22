# Pretrained Eye/Mouth Model Validation

## Result

Status: **passed**

Date: 2026-08-23

The FatigueSense eye and mouth MobileNetV3-Small checkpoints load with an exact
state-dict match and produce valid two-class probabilities on CPU. MediaPipe
successfully extracts both eye regions and the mouth region from an existing
project registration image.

## Environment

- Python: `3.9.25`
- PyTorch: `2.8.0+cu126`
- Torchvision: `0.23.0+cu126`
- MediaPipe: `0.10.14`
- OpenCV: `4.10.0`
- NumPy: `1.26.4`
- Pillow: `11.2.1`
- Validation device: CPU

The upstream package officially targets Python 3.10+. The validation script
therefore reproduces only the published MobileNetV3-Small architecture,
preprocessing, class indices, and crop geometry using Python 3.9-compatible
syntax.

## Checkpoints

The checkpoint binaries are external pretrained assets and are not committed
to this repository.

- Eye SHA-256: `D62B57DA4500A7748571E8C3ADC378990DC99FD3C28C48CEF97BD97BEF9940F3`
- Mouth SHA-256: `104492BDE316D9B4BCEF1EFAFCB8E1374798BAFC1B85D3E6F914427E3F42AF84`
- Eye classes: index `0` closed, index `1` open
- Mouth classes: index `0` closed, index `1` open
- Input: grayscale converted to three channels, `64x64`, ImageNet normalization

The public files were downloaded manually because automated access to
Hugging Face was unavailable on the validation machine. No account token is
required by the published repositories.

## Command

```powershell
D:\Miniconda3\python.exe tools\validate_pretrained_eye_mouth.py `
  --eye-weights E:\best_eye_classifier.pt `
  --mouth-weights E:\best_mouth_classifier.pt `
  --image dataset\data\guozishuo_1.png `
  --device cpu
```

## Output

| Region | Closed probability | Open probability |
| --- | ---: | ---: |
| Left eye | 0.003393 | 0.996607 |
| Right eye | 0.008445 | 0.991555 |
| Mouth | 0.836005 | 0.163995 |

Measured inference latency after model loading:

- Two-eye batch: `55.12 ms`
- Mouth: `10.63 ms`

These values are consistent with the visible sample state: both eyes are open
and the mouth is closed. The test proves checkpoint compatibility and valid
inference, not general accuracy on the target population.

## Acceptance Checklist

- [x] Both public checkpoint files are available.
- [x] Both checkpoints load with strict architecture matching.
- [x] MediaPipe produces two eye crops and one mouth crop.
- [x] Eye outputs are finite probabilities in `[0, 1]`.
- [x] Mouth outputs are finite probabilities in `[0, 1]`.
- [x] Every probability pair sums to approximately `1.0`.
- [x] CPU inference completes in the current Python 3.9 runtime.
- [x] Dependencies, hashes, command, and outputs are recorded.

## Non-Blocking Runtime Messages

MediaPipe emitted XNNPACK initialization messages, feedback-tensor warnings,
and a protobuf deprecation warning. They did not affect crop extraction or
model inference. Integration should keep these warnings out of the UI log when
possible.
