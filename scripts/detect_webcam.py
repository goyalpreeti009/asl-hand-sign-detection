import cv2
from ultralytics import YOLO

# Load your custom-trained ASL weights (not default yolov8n.pt)
model = YOLO("models/best.pt")

# Open webcam (0 is default, change to 1 if using an external USB cam)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not access the webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # conf=0.5 ignores predictions below 50% confidence
    results = model(frame, conf=0.5, verbose=False)

    # Draw boxes, labels, and confidence scores
    annotated = results[0].plot()

    # Display window
    cv2.imshow("ASL Hand Sign Detection", annotated)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
