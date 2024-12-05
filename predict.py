from ultralytics import YOLO

# Load a pretrained YOLO11n model
model = YOLO("runs/detect/train/weights/best.pt")

source = "test/"

# Run inference on 'bus.jpg' with arguments
model.predict(source, 
              save=True,
              conf=0.01,
              iou=0.5)