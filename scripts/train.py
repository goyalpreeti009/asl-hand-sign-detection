from ultralytics import YOLO

# Load pre-trained YOLOv8n model
model = YOLO("models/yolov8n.pt")

# Train on your custom ASL dataset
model.train(
    data="data/data.yaml",  # path to data.yaml
    epochs=30,
    imgsz=640,
    batch=8
)
