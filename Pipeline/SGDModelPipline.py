import numpy as np
import xml.etree.ElementTree as ET
import pycocotools.mask as mask_util
import random
import cv2
import requests
import torchvision.transforms as transforms
import torch
from torchvision import models
import torch.nn as nn
from bs4 import BeautifulSoup
import io
from PIL import Image
import os
import logging
import multiprocessing

# Import SAM modules
from segment_anything import sam_model_registry, SamPredictor
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights

from Pipeline.Mapper import map_number_to_ship

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # Log to console
        logging.FileHandler('pipeline.log')  # Log to file
    ]
)
logger = logging.getLogger(__name__)

def setup_sam_model():
    """
    Sets up the SAM model.

    Returns:
        SamPredictor: A SAM predictor object.
    """
    # Path to the pre-trained SAM model (ensure this is correct)
    sam_checkpoint = "/Users/saadbenboujina/Desktop/Projects/bachelor arbeit/sam_checkpoint/sam_vit_h_4b8939.pth"
    model_type = "vit_h"

    # Load the SAM model without weights_only
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to('cuda' if torch.cuda.is_available() else 'cpu')
    predictor = SamPredictor(sam)

    return predictor


def load_faster_rcnn_model(num_classes, model_path, device):
    """
    Loads the trained Faster R-CNN model.

    Args:
        num_classes (int): Number of classes (including background).
        model_path (str): Path to the saved model weights.
        device (torch.device): Device to load the model on.

    Returns:
        torch.nn.Module: The loaded Faster R-CNN model.
    """
    # Initialize the model with pre-trained weights
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn(weights=weights)

    # Get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    # Replace the pre-trained head with a new one (for your specific number of classes)
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # Load the trained weights with weights_only=True
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))

    # Move the model to the appropriate device
    model.to(device)
    model.eval()  # Set to evaluation mode

    return model

def load_classification_model(model_path, num_classes, device):
    """
    Loads the trained classification model.

    Args:
        model_path (str): Path to the saved classification model weights.
        num_classes (int): Number of classes for classification.
        device (torch.device): Device to load the model on.

    Returns:
        torch.nn.Module: The loaded classification model.
    """
    # Initialize the model (example with ResNet-50)
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    # Load the trained weights with weights_only=True
    if not os.path.exists(model_path):
        logger.error(f"Classification model file not found at {model_path}")
        return None
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))

    # Move the model to the appropriate device
    model.to(device)
    model.eval()  # Set to evaluation mode

    return model

def perform_faster_rcnn_detection(image, model, device, detection_labels, confidence_threshold=0.7):
    """
    Performs object detection using Faster R-CNN.

    Args:
        image (PIL.Image): The input image.
        model (torch.nn.Module): The Faster R-CNN model.
        device (torch.device): The device (CPU/GPU).
        detection_labels (list): List of labels to detect.
        confidence_threshold (float): Confidence threshold for detections.

    Returns:
        tuple: Number of detected boxes and list of box data.
    """
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    image_tensor = transform(image).to(device)

    with torch.no_grad():
        outputs = model([image_tensor])

    outputs = [{k: v.to('cpu') for k, v in t.items()} for t in outputs]
    detections = outputs[0]

    detected_boxes = 0
    boxes_data = []

    for box, label, score in zip(detections['boxes'], detections['labels'], detections['scores']):
        # Adjust label indexing based on your training (assuming labels start from 1)
        label_index = label.item() - 1
        if label_index < 0 or label_index >= len(detection_labels):
            continue  # Skip if label index is out of range
        label_name = detection_labels[label_index]
        if score >= confidence_threshold and label_name in detection_labels:
            detected_boxes += 1
            x1, y1, x2, y2 = box.int().tolist()
            confidence = score.item()
            label_text = f'{label_name} {confidence:.2f}'
            boxes_data.append({
                'label': label_name,
                'confidence': confidence,
                'x1': x1,
                'y1': y1,
                'x2': x2,
                'y2': y2
            })
            logger.info(
                f"Selected: {label_name} with Confidence {confidence:.2f} at coordinates: ({x1}, {y1}), ({x2}, {y2})")

    return detected_boxes, boxes_data

def generate_unique_color():
    """
    Generates a unique color for each object.

    Returns:
        np.ndarray: An array representing a color in RGB format.
    """
    return np.array([random.randint(0, 255), random.randint(
        0, 255), random.randint(0, 255)], dtype=np.uint8)

def mask_to_rle(mask):
    """
    Converts a mask to RLE format.

    Args:
        mask (np.ndarray): The mask to convert.

    Returns:
        dict: The RLE encoded mask.
    """
    rle = mask_util.encode(np.asfortranarray(mask.astype(np.uint8)))
    return rle

def save_image(image_np, path, filename):
    """
    Saves the processed image to the specified path.

    Args:
        image_np (np.ndarray): The image array.
        path (str): The directory path to save the image.
        filename (str): The filename for the image.

    Returns:
        str: The full path to the saved image.
    """
    if not os.path.exists(path):
        os.makedirs(path)
    image_path = os.path.join(path, filename)
    cv2.imwrite(image_path, image_np)  # Save the image
    return image_path

def apply_sam_segmentation(frame, predictor, boxes_data):
    """
    Applies SAM segmentation to the bounding boxes using point prompts and draws bounding boxes.

    Args:
        frame (np.ndarray): The original image.
        predictor (SamPredictor): The SAM predictor object.
        boxes_data (list): List of bounding box data.

    Returns:
        list: List of drawn masks with scores and labels.
    """
    drawn_masks = []

    # Prepare the image for SAM
    predictor.set_image(frame)

    for box in boxes_data:
        x1, y1, x2, y2 = box['x1'], box['y1'], box['x2'], box['y2']

        # SAM expects the input in the form [x0, y0, x1, y1]
        input_box = np.array([x1, y1, x2, y2])

        # Generate point prompts
        # Positive point at the center of the lower half of the bounding box
        pos_x = x1 + (x2 - x1) / 2
        pos_y = y1 + 3 * (y2 - y1) / 4  # Lower half
        positive_point = np.array([[pos_x, pos_y]])

        # Negative point at the center of the top edge of the bounding box
        neg_x = x1 + (x2 - x1) / 2
        neg_y = y1 + (y2 - y1) / 4  # Upper quarter
        negative_point = np.array([[neg_x, neg_y]])

        # Combine points
        point_coords = np.vstack([positive_point, negative_point])
        point_labels = np.array([1, 0])  # 1 for positive, 0 for negative

        # Generate the mask with SAM
        masks, scores, logits = predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            box=input_box[None, :],
            multimask_output=False
        )

        mask = masks[0]
        score = scores[0]

        # Draw the mask on the image
        color = generate_unique_color()
        mask_bool = mask.astype(bool)
        frame[mask_bool] = (frame[mask_bool] * 0.5 + color * 0.5).astype(np.uint8)

        drawn_masks.append((score, box['label'], mask_bool))

        logger.info(f"Mask added for {box['label']} with SAM-Score {score:.2f}")

        # Convert RGB to BGR for OpenCV
        color_bgr = color[::-1]

        # Draw bounding box using OpenCV
        cv2.rectangle(frame, (x1, y1), (x2, y2), color_bgr.tolist(), 2)
        label_text = f"{box['label']} {box['confidence']:.2f}"
        cv2.putText(frame, label_text, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_bgr.tolist(), 2)

    return drawn_masks

def process_image(image, faster_rcnn_model, sam_predictor, detection_labels, photo_label,
                  process_id, model_classification, device):
    """
    Processes a single image for detection, segmentation, and classification.

    Args:
        image (PIL.Image): The image to process.
        faster_rcnn_model (torch.nn.Module): The Faster R-CNN model.
        sam_predictor (SamPredictor): The SAM predictor.
        detection_labels (list): Labels to detect.
        photo_label (str): Label of the photo.
        process_id (int): Process ID for tracking.
        model_classification (nn.Module): The classification model.
        device (torch.device): The device (CPU/GPU).

    Returns:
        tuple: XML element with metadata and path to the saved image.
    """
    image_np = np.array(image)
    width, height = image.size

    # Perform object detection using Faster R-CNN
    detected_boxes, boxes_data = perform_faster_rcnn_detection(image, faster_rcnn_model, device, detection_labels, confidence_threshold=0.7)

    if detected_boxes == 0:
        logger.warning("Keine Objekte in Faster R-CNN-Detektion gefunden.")
        return None, None

    # Apply SAM segmentation to the bounding boxes
    drawn_masks = apply_sam_segmentation(image_np, sam_predictor, boxes_data)

    # Save the processed image
    image_path = save_image(image_np, "output",
                            f"{process_id}_processed.jpg")

    # Create XML structure to store image metadata
    image_metadata = ET.Element("image", id=str(process_id), category=f"{photo_label}",
                                width=str(width), height=str(height))

    # Save the bounding boxes in XML
    for box in boxes_data:
        box_element = ET.SubElement(
            image_metadata, "Bounding_box", label=box['label'], confidence=f"{box['confidence']:.2f}")
        box_element.set("x1", str(box['x1']))
        box_element.set("y1", str(box['y1']))
        box_element.set("x2", str(box['x2']))
        box_element.set("y2", str(box['y2']))

    # Save the drawn masks in XML
    for score, label, mask in drawn_masks:
        rle = mask_to_rle(mask)  # Convert mask to RLE
        mask_element = ET.SubElement(
            image_metadata, "mask", label=label, source="SAM")
        rle_str = ','.join([str(c) for c in rle['counts']])
        mask_element.set("rle", rle_str)

    return image_metadata, image_path

def classify_image(image, model_classification, device):
    """
    Classifies an image using the given classification model.

    Args:
        image (PIL.Image): The image to classify.
        model_classification (nn.Module): The classification model.
        device (torch.device): The device (CPU/GPU).

    Returns:
        int: The predicted class ID.
    """
    # Define the transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    # Apply transformations to the image
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model_classification(image_tensor)
        _, preds = torch.max(outputs, 1)

    return preds.item()

def scrape_and_process_ship_images(process_id, faster_rcnn_model, sam_predictor, detection_labels, model_classification, device):
    """
    Scrapes images from ShipSpotting.com and processes them.

    Args:
        process_id (int): The ID of the image to scrape and process.
        faster_rcnn_model (torch.nn.Module): The Faster R-CNN model for object detection.
        sam_predictor (SamPredictor): The SAM predictor.
        detection_labels (list): Labels to detect.
        model_classification (nn.Module): The classification model.
        device (torch.device): The device (CPU/GPU).

    Returns:
        tuple: XML data and path to the processed image.
    """
    url_prefix = 'https://www.shipspotting.com/photos/'
    url = f"{url_prefix}{str(process_id).zfill(7)}"

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko)'
                      ' Chrome/91.0.4472.124 Safari/537.36',
        'Referer': 'https://www.shipspotting.com',
        'Accept-Language': 'en-US,en;q=0.9',
    }

    # Use a Session to persist connections
    session = requests.Session()
    try:
        response = session.get(url, headers=headers)
        soup = BeautifulSoup(response.text, 'html.parser')

        # Scrape image URLs logic
        divs = soup.find_all('div', class_='summary-photo__image-row__image')
        image_urls = [div.find('img')['src']
                      for div in divs if div.find('img')]

        if not image_urls:
            logger.warning(f"Kein Bild gefunden für process_id: {process_id}")
            return None, None

        image_url = image_urls[0]
        if image_url.startswith('/'):
            image_url = 'https://www.shipspotting.com' + image_url

        image_response = session.get(image_url, headers=headers)
        image = Image.open(io.BytesIO(image_response.content)).convert("RGB")
        label_pred = map_number_to_ship(classify_image(image, model_classification, device))
        logger.info(f"Ship category is: {label_pred}")

        # Perform object detection using Faster R-CNN with confidence_threshold=0.7
        detected_boxes, boxes_data = perform_faster_rcnn_detection(
            image, faster_rcnn_model, device, detection_labels, confidence_threshold=0.7)

        if detected_boxes == 0:
            logger.warning("Keine Objekte in Faster R-CNN-Detektion gefunden.")
            return None, None

        # Apply SAM segmentation to the bounding boxes
        image_np = np.array(image)  # Convert PIL image to NumPy array
        drawn_masks = apply_sam_segmentation(
            image_np, sam_predictor, boxes_data)

        # Save the processed image
        image_path = save_image(image_np, "output",
                                f"{process_id}_processed.jpg")

        # Create XML structure to store image metadata
        width, height = image.size
        image_metadata = ET.Element("image", id=str(process_id), category=f"{label_pred}",
                                    width=str(width), height=str(height))

        # Save the bounding boxes in XML
        for box in boxes_data:
            box_element = ET.SubElement(
                image_metadata, "Bounding_box", label=box['label'], confidence=f"{box['confidence']:.2f}")
            box_element.set("x1", str(box['x1']))
            box_element.set("y1", str(box['y1']))
            box_element.set("x2", str(box['x2']))
            box_element.set("y2", str(box['y2']))

        # Save the drawn masks in XML
        for score, label, mask in drawn_masks:
            rle = mask_to_rle(mask)  # Convert mask to RLE
            mask_element = ET.SubElement(
                image_metadata, "mask", label=label, source="SAM")
            rle_str = ','.join([str(c) for c in rle['counts']])
            mask_element.set("rle", rle_str)

        return image_metadata, image_path
    except requests.exceptions.RequestException as e:
        logger.error(f"Request error for process_id {process_id}: {e}")
        return None, None
    except IOError as e:
        logger.error(f"I/O error for process_id {process_id}: {e}")
        return None, None
    except Exception as e:
        logger.error(f"Unexpected error for process_id {process_id}: {e}")
        return None, None
    finally:
        session.close()

def save_xml(xml_data, file_name="output.xml", path=""):
    """
    Saves XML data with detection, segmentation, and classification data.

    Args:
        xml_data (ET.Element): The XML data to save.
        file_name (str): The filename for the XML file.
        path (str): The directory path to save the XML file.

    Returns:
        str: The full path to the saved XML file.
    """
    # If path is not empty, ensure the directory exists
    if path:
        if not os.path.exists(path):
            os.makedirs(path)
        file_path = os.path.join(path, file_name)
    else:
        file_path = file_name  # Save in the current directory if no path is provided

    # Save the XML data to the specified path and filename
    tree = ET.ElementTree(xml_data)
    # Indent the XML structure for better readability
    ET.indent(tree, space="  ", level=0)  # Add indentation and line breaks
    tree.write(file_path)
    return file_path

def process_single_image(process_id, faster_rcnn_model, sam_predictor, detection_labels, model_classification, device):
    """
    Wrapper function to process a single image by its process_id. Used in multiprocessing.

    Args:
        process_id (int): The process ID of the image to scrape and process.
        faster_rcnn_model (torch.nn.Module): The Faster R-CNN model for object detection.
        sam_predictor (SamPredictor): The SAM predictor model.
        detection_labels (list): List of detection labels for Faster R-CNN.
        model_classification (nn.Module): The classification model.
        device (torch.device): The device (CPU/GPU).

    Returns:
        tuple: XML data and path to the processed image.
    """
    return scrape_and_process_ship_images(process_id, faster_rcnn_model, sam_predictor, detection_labels, model_classification, device)

def main():
    """
    Main function to execute the scraping and processing pipeline.
    """
    # Define the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Number of classes (including background)
    num_classes = 2  # Example: 1 class + background

    # Path to the saved Faster R-CNN weights
    faster_rcnn_model_path = "/Users/saadbenboujina/Desktop/Projects/bachelor arbeit/fasterrcnn_final.pth"

    # Load the Faster R-CNN model
    faster_rcnn_model = load_faster_rcnn_model(num_classes, faster_rcnn_model_path, device)

    # Initialize the SAM model
    sam_predictor = setup_sam_model()
    detection_labels = ["boat"]  # Adjust based on your classes

    # Define the list of process IDs (e.g., a list of ship spotting image IDs)
    process_ids = [954844, 933366]  # Example of multiple IDs

    # Path to the saved Classification model
    classification_model_path = "/Users/saadbenboujina/Desktop/Projects/bachelor arbeit/ship_classification_resnet50.pth"  # Update this path

    # Number of classes for classification
    num_classification_classes = 16  # Update based on your model

    # Load the Classification model
    model_classification = load_classification_model(classification_model_path, num_classification_classes, device)

    # Verify that the classification model is loaded
    if model_classification is None:
        logger.error("Classification model is not loaded. Please load your classification model.")
        return

    # Use a multiprocessing pool to process the images in parallel
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())

    # Create a list of arguments for each process
    args = [(process_id, faster_rcnn_model, sam_predictor, detection_labels, model_classification, device) for process_id in process_ids]

    # Run the processing in parallel
    results = pool.starmap(process_single_image, args)

    # Save results
    for xml_data, image_path in results:
        if xml_data:
            process_id = xml_data.attrib['id']
            save_xml(xml_data, f"{process_id}_processed.xml", "output")

    pool.close()
    pool.join()

if __name__ == "__main__":
    main()
