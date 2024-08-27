import cv2
import os
from ultralytics import YOLO
# Programm macht segmentation für Boat in ein Img
def process_image_for_segmentation(image_path, model, output_dir='output', labels=None):
    if labels is None:
        labels = []

    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image {image_path}")
        return
    print(f"Image loaded: {image.shape}")

    # Perform inference with YOLO model
    results = model(image)
    detections = results[0]

    print(f"Detected objects: {len(detections.masks.data)}")

    # Draw segmentation masks
    if detections.masks:
        for i, mask in enumerate(detections.masks.data):
            class_id = int(detections.boxes.cls[i])
            label = model.names[class_id]

            if label in labels:
                mask = mask.cpu().numpy()  # Convert mask to a NumPy array

                # Resize the mask to match the image dimensions
                mask_resized = cv2.resize(mask, (image.shape[1], image.shape[0]))

                # Create a violet color mask
                violet_color = (238, 130, 238)  # RGB color for violet
                colored_mask = cv2.merge([mask_resized * violet_color[0],
                                          mask_resized * violet_color[1],
                                          mask_resized * violet_color[2]])

                # Convert mask to uint8 and ensure it's 3-channel
                colored_mask = colored_mask.astype('uint8')

                # Blend the image with the colored mask
                image = cv2.addWeighted(image, 1, colored_mask, 0.5, 0)

    # Show the image with segmentation overlay
    cv2.imshow("Segmented Image", image)
    cv2.waitKey(0)  # Wait for a key press to close the image window
    cv2.destroyAllWindows()

    # Save the processed image
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_image_path = os.path.join(output_dir, os.path.basename(image_path))
    cv2.imwrite(output_image_path, image)
    print(f"Processed and saved image: {output_image_path}")

def main():
    # Set the path to your image file
    image_path = "/Users/saadbenboujina/Downloads/1/620_Fishfisher_24_2048x2048.jpg"

    # Load your YOLO model (for segmentation)
    model = YOLO("yolov8s-seg.pt")  # Use the segmentation model variant

    # Define the labels you want to segment
    labels = ["boat"]  # Replace with your desired labels

    # Process the image and generate segmentation masks
    process_image_for_segmentation(image_path, model, labels=labels)

if __name__ == '__main__':
    main()
