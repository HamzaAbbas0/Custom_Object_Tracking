# Custom object tracking and counting with YOLO and DeepSORT

Detects a custom object class with a trained **YOLO** model, follows each object across frames with **DeepSORT**, and keeps a running count of unique objects seen. The included model is trained on pigeons.

## How it works

1. Each frame is resized to 320 × 240 for fast inference and passed through the custom YOLO model (`best.pt`).
2. Detections below 0.5 confidence are dropped.
3. DeepSORT assigns every object a stable ID and keeps it for up to 50 frames if the object is briefly hidden.
4. Each confirmed ID is stored in a set, so the on-screen total counts every object once, even if it leaves and re-enters detection.

## Stack

Python · Ultralytics YOLO · deep-sort-realtime · OpenCV

## Run it

```bash
pip install ultralytics deep-sort-realtime opencv-python
python object_tracking.py
```

The script opens `pigeon_vid.mp4`. The sample video in this repository is named `pigeon_vid1.mp4`, so rename the file or change the path in `cv2.VideoCapture(...)`. Use `0` for a webcam. Press `Esc` to quit.

## Use it for another object

1. Train a YOLO model on your own class.
2. Replace `best.pt` with your weights.
3. Change the label string `"pigeon"` in `object_tracking.py`.

## Author

Hamza Abbas · [LinkedIn](https://www.linkedin.com/in/hamza-abbas-851864229/) · [GitHub](https://github.com/HamzaAbbas0)
