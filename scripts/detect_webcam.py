from ultralytics import YOLO
import cv2

# Load trained model
model = YOLO("models/yolov8n.pt")
 # replace if path differs

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detect ASL hand signs
    results = model(frame)
    annotated = results[0].plot()  # draw bounding boxes

    # Display
    cv2.imshow("ASL Hand Sign Detection", annotated)

    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press q to quit
        break

cap.release()
cv2.destroyAllWindows()
