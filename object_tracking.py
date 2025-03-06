from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import cv2

# Load YOLO model (Replace 'best.pt' with your trained model)
model = YOLO("best.pt")

# Initialize DeepSORT tracker
tracker = DeepSort(max_age=50)  # Prevents losing tracking too soon

# Load video (Use 0 for webcam)
cap = cv2.VideoCapture("pigeon_vid.mp4")

# Resize frame dimensions for faster processing
frame_width, frame_height = 320, 240

# Dictionary to store unique pigeon IDs
unique_pigeons = set()
confidence_threshold = 0.5  # Set confidence threshold to 70%

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Resize frame for fast inference
    frame = cv2.resize(frame, (frame_width, frame_height))

    # Run YOLO detection
    results = model(frame)

    detections = []
    for result in results:
        for box in result.boxes:
            conf = float(box.conf[0])  # Get confidence score

            if conf < confidence_threshold:  # Ignore low-confidence detections
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Append detections (format: [bbox, confidence, class_name])
            detections.append(([x1, y1, x2 - x1, y2 - y1], conf, "pigeon"))

    # Update tracker
    tracks = tracker.update_tracks(detections, frame=frame)

    # Draw tracking results
    for track in tracks:
        if not track.is_confirmed():
            continue

        # Get bounding box coordinates
        x1, y1, w, h = map(int, track.to_ltwh())
        track_id = track.track_id

        # Store unique pigeon IDs
        unique_pigeons.add(track_id)

        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), (0, 255, 0), 2)
        cv2.putText(frame, f"Pigeon {track_id} ({conf:.2f})", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Display total pigeon count
    total_pigeons = len(unique_pigeons)
    cv2.putText(frame, f"Total Pigeons: {total_pigeons}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Show frame
    cv2.imshow("Pigeon Tracking", frame)

    if cv2.waitKey(30) & 0xFF == 27:  # Press 'Esc' to exit
        break

cap.release()
cv2.destroyAllWindows()
