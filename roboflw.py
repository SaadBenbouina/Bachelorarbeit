from roboflow import Roboflow
import supervision as sv
import cv2

# Define the image file path
image_file = "/Users/saadbenboujina/Desktop/Projects/bachelor arbeit/TrainDataSegmentation/FromShipScout/image_1134857_0.jpg"
image = cv2.imread(image_file)

# Replace this with your regenerated API key if necessary
rf = Roboflow(api_key="rf_YOUR_NEW_API_KEY")

# Correct project endpoint (Check the project name from Roboflow)
project = rf.workspace().project("segmentation2-kcltv")
model = project.version(1).model  # Adjust the version number accordingly

# Run inference on the chosen image
results = model.predict(image, confidence=40, overlap=30).json()

# Load the results into the supervision Detections API
detections = sv.Detections.from_inference(results)

# Create supervision annotators
box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

# Annotate the image with inference results
annotated_image = box_annotator.annotate(scene=image, detections=detections)
annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections)

# Display the annotated image
sv.plot_image(annotated_image)
