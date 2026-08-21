# Project Structure

This project is the "Juancha Guard" fine-work fatigue check-in system.

## Run

Use the existing launcher:

```bat
start_ui.bat
```

The launcher uses:

```text
D:\Anaconda\envs\fatigue_cpu\python.exe
```

## Main Runtime Files

- `fatigue_ui.py` - PyQt5 dashboard and page switching.
- `core_engine.py` - camera threads, face authentication, and fatigue workflow.
- `fatigue_landmark_detector.py` - MediaPipe landmark fatigue detector, EAR/MAR/PERCLOS.
- `start_ui.bat` - Windows launcher.
- `requirements.txt` - Python dependency snapshot for reproduction.

## Core Algorithm Modules

- `model_v2.py` - MobileNetV2 feature extractor.
- `ssd_net_vgg.py` - legacy SSD detector.
- `detection.py` - SSD post-processing.
- `utils.py` - SSD helper functions.
- `voc0712.py` - SSD class labels and dataset helpers.
- `Config.py` - SSD configuration.
- `l2norm.py` - SSD layer helper.
- `sigjiansuobasic.py` - face feature retrieval helper.

## Models and Data

- `weights/` - SSD detector weights.
- `dataset/` - face feature data and sample registration images.
- `known_faces/` - reserved face data folder.
- `faceopenset_mobilenet123.pth` - current face feature checkpoint loaded by the main runtime.
- `deploy.prototxt` and `res10_300x300_ssd_iter_140000.caffemodel` - OpenCV DNN face detector.
- `traditional_features.pkl` - traditional face feature database.

## Legacy Archive

Files under `legacy/` are preserved for reference but are not part of the current main startup path.

- `legacy/old_ui/` - previous UI prototypes and merged experiment files.
- `legacy/tests/` - one-off test scripts and old camera/video demos.
- `legacy/training/` - training and preprocessing scripts.
- `legacy/training/helpers/` - helper modules used by old training scripts.
- `legacy/checkpoints/` - older checkpoints not loaded by the current main runtime.

If you need to run a legacy script, check its hard-coded paths first. Some older training scripts refer to local paths such as `E:/...`.
