from ultralytics import YOLO

# Load base model
model = YOLO("models/yolov8n.pt")

# Train the model (adjust datasets/data.yaml to your exact path)
model.train(
    data="datasets/data.yaml", 
    epochs=10,        # Lower to 5-10 if testing locally on CPU
    imgsz=416,        # 416 runs significantly faster than 640 on CPU
    batch=4,          # Lower batch size prevents CPU memory choke
    workers=2
)