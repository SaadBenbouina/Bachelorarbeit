from ultralytics import YOLO

# Load a model
model = YOLO("/private/var/folders/3m/k2m2bg694w15lfb_1kz6blvh0000gn/T/wzQL.Cf1otW/Bachelorarbeit/YoloModel/boat_detection_yolo_model_new4/weights/best.pt")  # build a new model from scratch
# Use the model
hyp = {
    'lr0': 0.005,
    'weight_decay': 0.0005,
    'momentum': 0.937,
    'scale': 0.5,
    'translate': 0.1,
    'fliplr': 0.5,
}
results =model.train(
data="config.yaml",epochs=10,
imgsz=640,
name="boat_detection_yolo_model_new6",
project="/var/folders/3m/k2m2bg694w15lfb_1kz6blvh0000gn/T/wzQL.Cf1otW/Bachelorarbeit",
cache=True,
save_period=3,  # Saves the model after 3 epoch
val=True,
hyp= hyp
)

