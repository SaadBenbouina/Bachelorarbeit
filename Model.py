from ultralytics import YOLO

# Load a model
model = YOLO("yolov8n.yaml")  # build a new model from scratch

# Use the model
results =model.train(
data="config.yaml",epochs=10,
imgsz=640,
name="boat_detection_model",
project="/var/folders/3m/k2m2bg694w15lfb_1kz6blvh0000gn/T/wzQL.Cf1otW/Bachelorarbeit",
cache=True,
save_period=1  # Saves the model after each epoch
)