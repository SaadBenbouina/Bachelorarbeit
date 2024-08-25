import cv2
import os
from ultralytics import YOLO

def process_image(image_path, model):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image {image_path}")
        return
    print(f"Image loaded: {image.shape}")

    # Inference with YOLO model
    result = model.predict(image)
    detections = result[0]

    print(f"Detected objects: {len(detections.boxes)}")

    # Manually draw bounding boxes
    for box in detections.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
        confidence = box.conf[0]  # Confidence score
        label = f'{box.cls[0]} {confidence:.2f}'  # Label with class and confidence

        # Draw the bounding box on the image
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green box with thickness 2
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)  # Green text

    # Show the image with annotations
    cv2.imshow("Annotated Image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    output_path = os.path.join('output', os.path.basename(image_path))
    cv2.imwrite(output_path, image)
    print(f"Processed and saved: {output_path}")

def main():
    # Setze hier den Pfad zum hochgeladenen Bild
    image_path = "/var/folders/3m/k2m2bg694w15lfb_1kz6blvh0000gn/T/wzQL.Cf1otW/Bachelorarbeit/Data/Test/0000c035a08c3770.jpg"

    if not os.path.exists('output'):
        os.makedirs('output')

    model = YOLO("yolov8n.pt")

    process_image(image_path, model)

if __name__ == '__main__':
    main()
