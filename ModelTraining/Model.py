from ultralytics import YOLO

# Load a model
model = YOLO("../YoloModel/boat_detection_yolo_model_new2/weights/best.pt")  # build a new model from scratch
# Use the model

results =model.train(
data="config.yaml",epochs=40,
imgsz=640,
name="boat_detection_yolo_model_new3",
project="/var/folders/3m/k2m2bg694w15lfb_1kz6blvh0000gn/T/wzQL.Cf1otW/Bachelorarbeit",
cache=True,
save_period=4,  # Saves the model after 3 epoch
val=True
)
"""
results = model.val(data="config.yaml", split="test")  # split="test" specifies the test dataset
print(model.names)
"""""
