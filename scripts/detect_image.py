from ultralytics import YOLO
import cv2

# Load trained model
model = YOLO("models/yolov8n.pt")
  # replace if path differs

# Load image
img = cv2.imread("samples/input.jpg")

# Detect hand signs
results = model(img)
annotated = results[0].plot()

# Show result
cv2.imshow("ASL Detection Result", annotated)
cv2.waitKey(0)
cv2.destroyAllWindows()
