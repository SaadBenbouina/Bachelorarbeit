#!/usr/bin/env python3
"""
    TU Dortmund University
    Computer Science Department
    Chair for Modeling and Simulation
    Author: Alexander Puzicha <alexander.puzicha@cs.tu-dortmund.de>
    Year: 2024
    License: MIT
"""
from itertools import groupby
import random
import requests
import tqdm
from bs4 import BeautifulSoup
from PIL import Image
import io
import xml.etree.ElementTree as ET
import os
import concurrent.futures
import shutil
import datetime
import numpy as np
import cv2
import torch
from segment_anything import sam_model_registry, SamPredictor
import pycocotools.mask as mask_util
from ShipLabelFilter import ShipLabelFilter
from ultralytics import YOLO
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Load a pretrained YOLO model (recommended for training)
model = YOLO("yolov8n.pt")


def binary_mask_to_rle(binary_mask):
    rle = {'counts': [], 'size': list(binary_mask.shape)}
    counts = rle.get('counts')
    for i, (value, elements) in enumerate(groupby(binary_mask.ravel(order='F'))):
        if i == 0 and value == 1:
            counts.append(0)
        counts.append(len(list(elements)))
    return rle


# Initialize SAM model
def initialize_sam():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    sam = sam_model_registry["default"](checkpoint="sam_checkpoint/sam_vit_h_4b8939.pth")
    sam.to(device)
    return SamPredictor(sam)


predictor = initialize_sam()


def mask_to_rle(mask):
    """Convert a binary mask to RLE format."""
    rle = mask_util.encode(np.asfortranarray(mask))
    uncompressed_rle = mask_util.frPyObjects(rle, mask.shape[0], mask.shape[1])
    return uncompressed_rle


def mask_to_rle2(mask):
    """
    Convert a binary mask into RLE (Run Length Encoding) format.
    :param mask: numpy array of shape (height, width), 1 - mask, 0 - background
    :returns: String of run length as string formatted
    """
    pixels = mask.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    runs = runs.astype(int)
    return ' '.join(str(x) for x in runs)


def mask_to_compressed_rle(mask):
    """
    Convert a binary mask into compressed RLE format.
    :param mask: numpy array of shape (height, width), 1 - mask, 0 - background
    :returns: Compressed RLE
    """
    rle = mask_util.encode(np.asfortranarray(mask.astype(np.uint8)))
    #rle['counts'] = rle['counts'].decode('utf-8')  # Ensure the RLE counts are string
    return rle


def mask_to_uncompressed_rle(mask):
    """
    Convert a binary mask into uncompressed RLE (Run Length Encoding) format.
    :param mask: numpy array of shape (height, width), 1 - mask, 0 - background
    :returns: String of run length as string formatted
    """
    height, width = mask.shape
    y_indices, x_indices = np.where(mask == 1)
    rle_pairs = np.zeros((len(y_indices), 2), dtype=int)
    rle_pairs[:, 0] = x_indices
    rle_pairs[:, 1] = y_indices
    rle_str = ' '.join(map(str, rle_pairs.flatten()))
    return rle_str


def show_results(image_np, boxes, masks):
    # Display the image with YOLO bounding boxes and SAM masks
    fig, ax = plt.subplots(1, figsize=(12, 12))
    ax.imshow(image_np)
    for box in boxes:
        xyxy, conf, cls = box.xyxy, box.conf, box.cls
        xyxy_list = xyxy.tolist()[0]
        x1, y1, x2, y2 = xyxy_list[0], xyxy_list[1], xyxy_list[2], xyxy_list[3]
        if cls == 8 and conf > 0.3:  # Class 'boat' in YOLO
            rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor='r', facecolor='none')
            ax.add_patch(rect)

    ax.contour(masks[0], colors='yellow', linewidths=2)
    ax.contour(masks[1], colors='red', linewidths=2)
    ax.contour(masks[2], colors='blue', linewidths=2)
    plt.title(f"Image ID: {process_id}")
    plt.axis('off')
    plt.show()


def save_image(image, path, url):
    try:
        if not os.path.exists(path):
            os.makedirs(path)
    except OSError:
        print(f"Creation of the directory {path} failed")
        pass

    # Convert the image to a byte stream
    with io.BytesIO() as byte_stream:
        image.save(byte_stream, format='JPEG')  # Adjust the format as needed (e.g., 'PNG', 'JPEG')
        image_bytes = byte_stream.getvalue()

    # Speichern Sie das Bild im Ordner mit dem Namen des Foto-Labels
    image_name = url.split('/')[-1].split('?')[0]
    with open(path + image_name, 'wb') as f:
        f.write(image_bytes)
    return image_name


def process_image(image, url, photo_label, photo_area, process_id, labels, use_yolo=True, debug=True):
    """
    Process the given image and extract the necessary information.

    Args:
        image (PIL.Image): The image to be processed.
        url (str): The URL of the image.
        photo_label (str): The label of the photo.
        photo_area (str): The area of the photo.
        process_id (int): The process ID of the image.
        use_yolo (bool): Whether to use YOLO for object detection.
        debug (bool): Whether to display debug information.

    Returns:
        list or None: A list containing the image data and labels if successful,
                      None if an error occurred or no data found.
    """
    if image is None:
        return None

    width, height = image.size
    # Remove the last 20 pixels from the bottom up over the entire image width to remove the watermark, it would be bad for learning
    image = image.crop((0, 0, width, height - 20))

    if photo_label != "validation":
        # Convert image to numpy array
        image_np = np.array(image)

        boxes = []
        if use_yolo:
            # Perform YOLO detection
            results = model(image_np)

            # Extract bounding boxes for detected objects
            boxes = results[0].boxes  # x1, y1, x2, y2, confidence, class

        # Find the largest bounding box (assuming it's the ship)
        largest_box = None
        largest_area = 0
        for box in boxes:
            xyxy, conf, cls = box.xyxy, box.conf, box.cls
            xyxy_list = xyxy.tolist()[0]
            x1, y1, x2, y2 = xyxy_list[0], xyxy_list[1], xyxy_list[2], xyxy_list[3]
            if cls == 8 and conf > 0.3:  # Class 'boat' in YOLO (adjust as needed)
                area = (x2 - x1) * (y2 - y1)
                if area > largest_area:
                    largest_area = area
                    largest_box = box

        if largest_box is not None:
            xyxy, conf, cls = largest_box.xyxy, largest_box.conf, largest_box.cls
            xyxy_list = xyxy.tolist()[0]
            x1, y1, x2, y2 = xyxy_list[0], xyxy_list[1], xyxy_list[2], xyxy_list[3]

            midpoint = [(x1 + x2) / 2, y2 - (y2 - y1) * 0.25]
            # Define relative coordinates and labels
            relative_points = np.array([
                [0.5, 0.05],  # sky
                [midpoint[0] / width, midpoint[1] / height],  # boat
                [0.7, 0.95]   # water
            ])
        else:

            # Define relative coordinates and labels
            relative_points = np.array([
                [0.5, 0.05],  # sky
                [0.5, 0.55],   # boat
                [0.7, 0.95]   # water
            ])
        mask_labels = ["sky", "boat", "water"]

        # Convert relative coordinates to absolute coordinates
        absolute_points = np.round(relative_points * [width, height]).astype(int)

        # Predict masks using SAM
        predictor.set_image(image_np)
        masks, _, _ = predictor.predict(absolute_points, point_labels=np.array([1, 1, 1]))

        if debug:
            show_results(image_np, boxes, masks)

        # Erstellen Sie einen Ordner mit dem Namen des Foto-Labels, wenn er noch nicht existiert
        path_suffix = f'{photo_area}/{photo_label}/'
        path = f'export/{path_suffix}'

        image_name = save_image(image, path, url)

        # Fügen Sie die Metadaten für jedes Bild hinzu
        image_metadata = ET.Element("image", id=str(process_id), name=path_suffix+image_name, width=str(width), height=str(height))

        # Process each mask and save it as a CVAT annotation
        for mask, label, (x, y) in zip(masks, mask_labels, absolute_points):
            # Convert the mask to RLE format
            rle = mask_util.encode(np.asfortranarray(mask.astype(np.uint8)))


            #mask = mask.astype(np.uint8)
            #rle = mask_to_rle2(mask)
            #compressed_rle = mask_to_compressed_rle(mask)
            left, top, width, height = mask_util.toBbox(rle)
            rle_str = ', '.join(str(count) for count in rle["counts"])
            mask_element = ET.SubElement(image_metadata, "mask", label=label, source="semi-auto", occluded="0", rle=rle_str, left=str(int(left)), top=str(int(top)), width=str(int(width)), height=str(int(height)), z_order="0")

            if label == "boat":
                sublabel = ET.SubElement(mask_element, "attribute", name="type", source="auto")
                sublabel.text = photo_label
            if label == "sky":
                sublabel = ET.SubElement(mask_element, "attribute", name="type", source="auto")
                sublabel.text = "partially"

        # Fügen Sie das Foto-Label hinzu
        tag = ET.SubElement(image_metadata, "tag", label="boat", source="auto")
        sublabel = ET.SubElement(tag, "attribute", name="type", source="auto")
        sublabel.text = photo_label
        tag = ET.SubElement(image_metadata, "tag", label=f'boat_{photo_area}', source="auto")
        tag = ET.SubElement(image_metadata, "tag", label="daytime", source="auto")
        sublabel = ET.SubElement(tag, "attribute", name="type", source="auto")
        sublabel.text = "day"
        tag = ET.SubElement(image_metadata, "tag", label="weather", source="auto")
        sublabel = ET.SubElement(tag, "attribute", name="type", source="auto")
        sublabel.text = "cloudy"
        tag = ET.SubElement(image_metadata, "tag", label="season", source="auto")
        sublabel = ET.SubElement(tag, "attribute", name="type", source="auto")
        sublabel.text = "summer"
        tag = ET.SubElement(image_metadata, "tag", label="visibility", source="auto")
        sublabel = ET.SubElement(tag, "attribute", name="type", source="auto")
        sublabel.text = "normal"
        # Wenn das Label existiert, fügen Sie es hinzu
        # if label is not None and label != photo_label:
        # box = ET.SubElement(image_metadata, "box", label=label, occluded="0", xtl="0", ytl="0", xbr=str(width), ybr=str(height))
        # tag = ET.SubElement(image_metadata, "tag", label=label, source="auto")
        labels.append(mask_labels)
        return [image_metadata, labels]

    else:
        save_image(image, 'export/images/validation/', url)
        return None


def process_id(process_id, use_yolo=True, debug=True):
    """
    Process the given process_id and retrieve image data from a website.

    Args:
        process_id (int): The process ID to be processed.

    Returns:
        list or None: A list containing the image data and labels if successful,
                      None if an error occurred or no data found.
    """
    use_vessel_type = False

    # URL der Webseite, die Sie crawlen möchten
    url_prefix = 'https://www.shipspotting.com/photos/'

    label = None
    image = None
    labels = []

    # Convert the counter to a seven-digit number
    photo_id = "{:07d}".format(process_id)

    url = url_prefix + photo_id
    try:

        # Senden Sie eine HTTP-Anfrage an die URL und speichern Sie die Antwort
        response = requests.get(url)

        # Analysieren Sie die Antwort mit BeautifulSoup
        soup = BeautifulSoup(response.text, 'html.parser')

        # Finden Sie das div-Element mit der Klasse 'summary-photo__image-row__image'
        divs = soup.find_all('div', class_='summary-photo__image-row__image')

        # Extrahieren Sie die Bild-URLs aus den img-Elementen innerhalb der div-Elemente
        urls = [div.find('img')['src'] for div in divs]

        # Finden Sie das div-Element mit der Klasse 'InformationItem__InfoItemStyle-sc-r4h3tv-0 hfSVPp information-item summary-photo__card-general__label'
        divs = soup.find_all('div', class_='InformationItem__InfoItemStyle-sc-r4h3tv-0 hfSVPp information-item summary-photo__card-general__label')

        # Wenn das div-Element existiert, extrahieren Sie den Schiffstyp und verwenden Sie ihn als Label
        for div in divs:
            information_title = div.find('span', class_='information-item__title').text
            if information_title == "Photo Category:":
                link = None
                link = div.find('a', href=lambda href: href and href.startswith("/photos/gallery?category"))
                if link:
                    photo_label, photo_area = ShipLabelFilter.filter_label(link.text)
                    if photo_label is None:
                        return None

                    if photo_label not in labels:
                        labels.append(photo_label)

            if use_vessel_type and information_title == "Vessel Type:":  # oder verwenden Sie ein anderes Kriterium
                label = div.find('span', class_='information-item__value').text
                label = ShipLabelFilter.filter_label(label)
                if label not in labels:
                    labels.append(label)

        # Laden Sie jedes Bild herunter und speichern Sie es in einem lokalen Verzeichnis
        for url in urls:
            response = requests.get(url)
            image = Image.open(io.BytesIO(response.content))
            return process_image(image, url, photo_label, photo_area, process_id, labels, use_yolo, debug)

    except requests.exceptions.RequestException:
        return None
    return None

def main(number_of_images=1,parallel=False):
    """
    Main function for generating annotations and exporting images.

    Args:
        number_of_images (int): The number of images to generate annotations for and export.

    Returns:
        None
    """
    # Clear the export folder before starting
    if os.path.exists("export"):
        shutil.rmtree("export")
    os.makedirs("export")

    # Create the root element once, outside the loop
    annotations = ET.Element("annotations")
    version = ET.SubElement(annotations, "version").text = "1.1"

    images_found = 0
    tries = 0
    all_found_labels = []
    if parallel:
        while images_found < number_of_images and tries < 10:
            # Use multiple threads to speed up the process
            with concurrent.futures.ThreadPoolExecutor() as executor:
                results = executor.map(process_id, range(1000000, 9999999-number_of_images))
            has_results = False
            # Write the results to the file
            for result in tqdm.tqdm(results):
                # Add each image as a child of the root element
                if result is not None:
                    [xml_image_node, labels] = result
                    annotations.append(xml_image_node)
                    for label in labels:
                        if label not in all_found_labels:
                            all_found_labels.append(label)
                    has_results = True
                    images_found += 1
        tries += 1
    else:
        # Use a single thread
        while images_found < number_of_images and tries < 10:
            #random.seed(42)
            random_id = random.randint(1000000, 9999999-number_of_images)
            has_results = False
            # Write the results to the file
            for id in tqdm.tqdm(range(random_id, random_id+number_of_images)):
                # Add each image as a child of the root element
                result = process_id(id)
                if result is not None:
                    [xml_image_node, labels] = result
                    annotations.append(xml_image_node)
                    for label in labels:
                        if label not in all_found_labels:
                            all_found_labels.append(label)
                    has_results = True
                    images_found += 1
        tries += 1

    if has_results:
        # Save the XML file
        tree = ET.ElementTree(annotations)
        tree.write("export/annotations.xml")

        # Get the current date and time
        now = datetime.datetime.now()
        date_time = now.strftime("%Y-%m-%d_%H-%M-%S")

        # Create the archive name with the date and time
        archive_name = f"export_{date_time}.zip"

        # Zip the export folder
        shutil.make_archive("export", "zip", "export")

        # Rename the zip file
        os.rename("export.zip", archive_name)
        print(f"Exported {images_found} images with {len(all_found_labels)} labels to {archive_name}")
        print(f"Labels found: {all_found_labels}")
    else:
        print("No results found")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='download ship images and create annotations')
    parser.add_argument('--number_of_images', metavar='number', required=False, default=1,
                        help='number of images to download')
    args = parser.parse_args()
    main(number_of_images=args.number_of_images)
