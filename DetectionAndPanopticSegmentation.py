import cv2
import numpy as np
import requests
from ultralytics import YOLO
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
import torch
from bs4 import BeautifulSoup
import io
from PIL import Image
import xml.etree.ElementTree as ET
from ShipLabelFilter import ShipLabelFilter
import os
import pycocotools.mask as mask_util

# Setup Panoptic Segmentation Model
def setup_panoptic_model():
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml"))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml")
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    return DefaultPredictor(cfg)

# Perform YOLO object detection
def draw_yolo_detections(frame, yolo_result, yolo_model, detection_labels):
    detected = False
    detections = yolo_result[0]

    for i, box in enumerate(detections.boxes):
        class_id = int(box.cls[0])
        label = yolo_model.names[class_id]
        if label in detection_labels:
            detected = True
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = box.conf[0]
            label_text = f'{label} {confidence:.2f}'

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # Put label and confidence
            cv2.putText(frame, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            print(f"Detected {label} with confidence {confidence:.2f} at coordinates: ({x1}, {y1}), ({x2}, {y2})")
    return detected

# Panoptic segmentation on the frame
def apply_panoptic_segmentation(frame, panoptic_result):
    if panoptic_result is None:
        print("Panoptic segmentation result is None")
        return

    panoptic_seg = panoptic_result["panoptic_seg"][0].cpu().numpy()
    segments_info = panoptic_result["instances"]

    # Iterate through segments to apply the mask and assign random colors
    for idx in range(len(segments_info)):
        mask = panoptic_seg == idx
        color = np.random.randint(0, 255, (1, 3), dtype=np.uint8).tolist()[0]
        frame[mask] = frame[mask] * 0.5 + np.array(color) * 0.5  # Blend the mask with color

# Convert mask to RLE format for XML storage
def mask_to_rle(mask):
    rle = mask_util.encode(np.asfortranarray(mask.astype(np.uint8)))
    return rle

# Save processed image
def save_image(image_np, path, filename):
    if not os.path.exists(path):
        os.makedirs(path)
    image_path = os.path.join(path, filename)
    cv2.imwrite(image_path, image_np)  # Save the image
    return image_path

# Process images for detection, segmentation, and classification
def process_image(image, yolo_model, panoptic_model, detection_labels, url, photo_label, process_id, debug=True):
    image_np = np.array(image)
    width, height = image.size

    if debug:
        print(f"Processing image from URL: {url}")

    # Perform YOLO detection
    yolo_result = yolo_model.predict(image_np)
    panoptic_result = panoptic_model(image_np)

    # Check if panoptic_result contains the required keys
    if "panoptic_seg" not in panoptic_result or "instances" not in panoptic_result:
        print(f"Panoptic segmentation did not return the expected result: {panoptic_result.keys()}")
        return None

    # Draw YOLO detections and apply panoptic segmentation
    draw_yolo_detections(image_np, yolo_result, yolo_model, detection_labels)
    apply_panoptic_segmentation(image_np, panoptic_result)

    # Display the image in a window using OpenCV
    cv2.imshow(f"Processed Image: {photo_label}", image_np)
    cv2.waitKey(0)  # Wait for a key press to close the window
    cv2.destroyAllWindows()  # Close the window

    # Save the processed image
    image_path = save_image(image_np, "processed_images", f"{process_id}_processed.jpg")

    # Create XML structure
    image_metadata = ET.Element("image", id=str(process_id), name=f"{photo_label}.jpg", width=str(width), height=str(height))
    for idx in range(len(panoptic_result["instances"])):
        mask = panoptic_result["panoptic_seg"][0].cpu().numpy() == idx
        mask = mask.astype(np.uint8)
        rle = mask_to_rle(mask)

        mask_element = ET.SubElement(image_metadata, "mask", label="segment", source="auto")
        rle_str = ', '.join(str(count) for count in rle["counts"])
        mask_element.set("rle", rle_str)
        ET.SubElement(mask_element, "attribute", name="type", source="auto").text = photo_label

    return image_metadata, image_path

# Scrape images from ShipSpotting and process them
def scrape_and_process_ship_images(process_id, yolo_model, panoptic_model, detection_labels):
    url_prefix = 'https://www.shipspotting.com/photos/'
    url = f"{url_prefix}{str(process_id).zfill(7)}"

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Referer': 'https://www.shipspotting.com',
        'Accept-Language': 'en-US,en;q=0.9',
    }

    try:
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.text, 'html.parser')

        # Use the same image scraping logic from the original code
        divs = soup.find_all('div', class_='summary-photo__image-row__image')
        image_urls = [div.find('img')['src'] for div in divs if div.find('img')]

        if not image_urls:
            print(f"No image found for process_id: {process_id}")
            return None

        image_url = image_urls[0]
        if image_url.startswith('/'):
            image_url = 'https://www.shipspotting.com' + image_url

        print(f"Downloading image from: {image_url}")
        image_response = requests.get(image_url, headers=headers)
        image = Image.open(io.BytesIO(image_response.content))

        label = ShipLabelFilter.filter_label("ship")
        print(f"Processing ship image with label: {label[0]}")

        xml_data = process_image(image, yolo_model, panoptic_model, detection_labels, image_url, label[0], process_id, debug=True)

        return xml_data
    except Exception as e:
        print(f"Error processing image: {e}")
        return None

# Save XML file with detection, segmentation, and classification data
def save_xml(xml_data, file_name="output.xml", path=""):
    # If the path is not empty, ensure the directory exists
    if path:
        if not os.path.exists(path):
            os.makedirs(path)
        file_path = os.path.join(path, file_name)
    else:
        file_path = file_name  # Save to the current directory if no path is specified

    # Save the XML data to the specified path and file name
    tree = ET.ElementTree(xml_data)
    tree.write(file_path)

# Main function to execute the scraping and processing
def main():
    yolo_model = YOLO("boat_detection_yolo_model_new/weights/best.pt")
    panoptic_model = setup_panoptic_model()
    detection_labels = ["boat"]

    process_id = 1334001  # Example ShipSpotting image ID
    xml_data, image_path = scrape_and_process_ship_images(process_id, yolo_model, panoptic_model, detection_labels)

    if xml_data:
        save_xml(xml_data, f"ship_annotation_{process_id}.xml", "processed_images")

if __name__ == "__main__":
    main()
