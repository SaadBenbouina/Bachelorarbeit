from ultralytics import YOLO

# Load a model
model = YOLO("boat_detection_yolo_model_new2/weights/best.pt")  # build a new model from scratch
# Use the model

results =model.train(
data="config.yaml",epochs=10,
imgsz=640,
name="boat_detection_yolo_model_new3",
project="/var/folders/3m/k2m2bg694w15lfb_1kz6blvh0000gn/T/wzQL.Cf1otW/Bachelorarbeit",
cache=True,
save_period=1,  # Saves the model after each epoch
val=True
)
"""
results = model.val(data="config.yaml", split="test")  # split="test" specifies the test dataset
print(model.names)
"""""
