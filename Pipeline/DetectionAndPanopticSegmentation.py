
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
from detectron2.data import MetadataCatalog


# Setup Panoptic Segmentation Model
def setup_panoptic_model():
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml"))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml")
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    predictor = DefaultPredictor(cfg)

    # Load the metadata for COCO Panoptic Segmentation
    metadata = MetadataCatalog.get("coco_2017_val_panoptic_separated")

    return predictor, metadata


# Perform YOLO object detection and ensure number of bounding boxes matches segmentations
def draw_yolo_detections(frame, yolo_result, yolo_model, detection_labels):
    detected_boxes = 0
    detections = yolo_result[0]
    boxes_data = []

    # Filter detections for the specified labels
    for i, box in enumerate(detections.boxes):
        class_id = int(box.cls[0])
        label = yolo_model.names[class_id]
        confidence = box.conf[0]
        # Only consider bounding boxes with confidence > 0.5 for the relevant labels
        if confidence > 0.5 and label in detection_labels:
            detected_boxes += 1
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = box.conf[0]
            label_text = f'{label} {confidence:.2f}'
            # Draw the bounding box and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # Put label and confidence
            cv2.putText(frame, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            print(f"Selected {label} with confidence {confidence:.2f} at coordinates: ({x1}, {y1}), ({x2}, {y2})")
            boxes_data.append({
                'label': label,
                'confidence': confidence,
                'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2
            })

    return detected_boxes, boxes_data


# Apply segmentation for the same number of boats as detected by YOLO
def apply_panoptic_segmentation(frame, panoptic_result, metadata, confidence_threshold=0.7, max_instances=0):
    class_colors = {
        "boat": np.array([255, 0, 0], dtype=np.uint8),
        "sky": np.array([0, 255, 0], dtype=np.uint8),
        "sea": np.array([255, 255, 255], dtype=np.uint8)
    }

    # Extrahieren und Konvertieren der Panoptic Segmentierungsdaten
    panoptic_seg, segments_info = panoptic_result["panoptic_seg"]  # Tuple aus Segmentierungsdaten und Segmentinfo
    panoptic_seg = panoptic_seg.cpu().numpy()

    # Extrahieren und Konvertieren der Instanzdaten
    instances = panoptic_result["instances"]
    pred_classes = instances.pred_classes.cpu().numpy()
    pred_masks = instances.pred_masks.cpu().numpy()
    pred_scores = instances.scores.cpu().numpy()

    # Filtern der validen "boat" Segmentierungen
    boat_mask_indices = np.where(
        (pred_scores > confidence_threshold) &
        (pred_classes < len(metadata.thing_classes)) &
        (np.array([metadata.thing_classes[cls] for cls in pred_classes]) == "boat")
    )[0]

    # Begrenzen der Anzahl der Instanzen basierend auf max_instances
    if max_instances > 0:
        boat_mask_indices = boat_mask_indices[:max_instances]

    # Anwenden der Masken auf das Frame
    for idx in boat_mask_indices:
        label = "boat"
        mask = pred_masks[idx]
        color = class_colors[label]
        # Blending der Farbe mit dem Originalbild
        frame[mask] = (frame[mask] * 0.5 + color * 0.5).astype(np.uint8)

    drawn_masks = [
        (pred_scores[idx], "boat", pred_masks[idx]) for idx in boat_mask_indices
    ]

    # Verarbeitung der "stuff" Klassen (sky, sea)
    # Extrahieren der IDs der relevanten "stuff" Klassen
    stuff_labels = ["sky", "sea"]
    stuff_category_ids = [metadata.stuff_classes.index(label) for label in stuff_labels if label in metadata.stuff_classes]

    # Filtern der Segmente für "stuff" Klassen
    stuff_segments = [
        seg for seg in segments_info
        if seg["category_id"] in stuff_category_ids
    ]

    for seg in stuff_segments:
        label = metadata.stuff_classes[seg["category_id"]]
        mask = panoptic_seg == seg["id"]
        color = class_colors[label]
        frame[mask] = (frame[mask] * 0.5 + color * 0.5).astype(np.uint8)
        drawn_masks.append((1.0, label, mask))  # Verwenden eines Standardwertes für die Konfidenz

    return drawn_masks

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
def process_image(image, yolo_model, panoptic_model, metadata, detection_labels, url, photo_label, taking_time,
                  process_id, confidence_threshold=0.7, debug=True):
    image_np = np.array(image)
    width, height = image.size

    if debug:
        print(f"Processing image from URL: {url}")

    # Perform YOLO detection
    yolo_result = yolo_model.predict(image_np)
    panoptic_result = panoptic_model(image_np)

    # Draw YOLO detections and apply panoptic segmentation
    detected_boxes, boxes_data = draw_yolo_detections(image_np, yolo_result, yolo_model, detection_labels)

    # Get the drawn masks from panoptic segmentation
    drawn_masks = apply_panoptic_segmentation(image_np, panoptic_result, metadata,
                                              confidence_threshold=confidence_threshold,
                                              max_instances=detected_boxes)

    # Save the processed image
    image_path = save_image(image_np, "/private/var/folders/3m/k2m2bg694w15lfb_1kz6blvh0000gn/T/wzQL.Cf1otW/Bachelorarbeit/Pipeline/output", f"{process_id}_processed.jpg")

    # Create XML structure for storing image metadata
    image_metadata = ET.Element("image", id=str(process_id), category=f"{photo_label}", date=f"{taking_time}",
                                width=str(width), height=str(height))

    # Store the bounding boxes in the XML
    for box in boxes_data:
        box_element = ET.SubElement(image_metadata, "Bounding_box", label=box['label'], confidence=f"{box['confidence']:.2f}")
        box_element.set("x1", str(box['x1']))
        box_element.set("y1", str(box['y1']))
        box_element.set("x2", str(box['x2']))
        box_element.set("y2", str(box['y2']))
        print(f"Added bounding box for {box['label']} with confidence {box['confidence']:.2f}")

    # Store the drawn masks in the XML
    for score, label, mask in drawn_masks:
        rle = mask_to_rle(mask)  # Convert the mask to RLE format
        mask_element = ET.SubElement(image_metadata, "mask", label=label, source="auto")
        rle_str = ', '.join(str(count) for count in rle["counts"])
        mask_element.set("rle", rle_str)
        print(f"Added mask for {label} with confidence {score:.2f}")

    # Display the image in a window using OpenCV
    cv2.imshow(f"Processed Image: {photo_label}", image_np)
    cv2.waitKey(0)  # Wait for a key press to close the window
    cv2.destroyAllWindows()  # Close the window

    return image_metadata, image_path

# Scrape images from ShipSpotting and process them
def scrape_and_process_ship_images(process_id, yolo_model, panoptic_model, metadata, detection_labels):
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
        label_divs = soup.find_all('div',
                                   class_='InformationItem__InfoItemStyle-sc-r4h3tv-0 hfSVPp information-item summary-photo__card-general__label')
        label_text = ""
        taken_time = ""
        for div in label_divs:
            information_title = div.find('span', class_='information-item__title')
            if information_title and information_title.text.strip() == "Captured:":
                taken_time_value = div.find('span', class_='information-item__value')
                if taken_time_value:
                    taken_time = taken_time_value.text.strip()
            if information_title and information_title.text.strip() == "Photo Category:":
                label_value = div.find('span', class_='information-item__value')
                if label_value:
                    # Extract the category text
                    label_text = label_value.text.strip()
                    break

        if not image_urls:
            print(f"No image found for process_id: {process_id}")
            return None, None

        image_url = image_urls[0]
        if image_url.startswith('/'):
            image_url = 'https://www.shipspotting.com' + image_url

        print(f"Downloading image from: {image_url}")
        image_response = requests.get(image_url, headers=headers)
        image = Image.open(io.BytesIO(image_response.content))

        label = ShipLabelFilter.filter_label(label_text)
        print(f"Processing ship image with label: {label[0]}")

        xml_data, image_path = process_image(image, yolo_model, panoptic_model, metadata, detection_labels, image_url,
                                             label[0], taken_time, process_id, debug=True)

        return xml_data, image_path
    except Exception as e:
        print(f"Error processing image: {e}")
        return None, None


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
    # Indent the XML structure for pretty printing
    ET.indent(tree, space="  ", level=0)  # Add indentation and newlines
    tree.write(file_path)


# Main function to execute the scraping and processing
def main():
    yolo_model = YOLO("../YoloModel/boat_detection_yolo_model_new3/weights/best.pt")
    panoptic_model, metadata = setup_panoptic_model()
    detection_labels = ["boat"]

    process_id = 1735020  # random.randint(5000, 2000000)  # Example ShipSpotting image ID
    xml_data, image_path = scrape_and_process_ship_images(process_id, yolo_model, panoptic_model, metadata,
                                                          detection_labels)

    if xml_data:
        save_xml(xml_data, f"{process_id}_processed.xml", "/private/var/folders/3m/k2m2bg694w15lfb_1kz6blvh0000gn/T/wzQL.Cf1otW/Bachelorarbeit/Pipeline/output")


if __name__ == "__main__":
    main()
