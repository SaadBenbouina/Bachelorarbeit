import os
import requests
from bs4 import BeautifulSoup
from ultralytics import YOLO
from PIL import Image
import logging

from Pipeline.ShipLabelFilter import ShipLabelFilter

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define headers for HTTP requests
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Referer': 'https://www.shipspotting.com',
    'Accept-Language': 'en-US,en;q=0.9',
}

# Load the YOLO model
yolo_model = YOLO("../YoloModel/boat_detection_yolo_model/weights/best.pt")

def download_images_from_shipspotting(process_id, url_prefix='https://www.shipspotting.com/photos/'):
    image_urls = []
    labels = []
    photo_id = "{:07d}".format(process_id)
    url = url_prefix + photo_id

    logger.info(f"Fetching URL: {url}")

    try:
        response = requests.get(url, headers=HEADERS)
        if response.status_code != 200:
            logger.error(f"Error: Received status code {response.status_code} from {url}")
            return image_urls, labels

        soup = BeautifulSoup(response.text, 'html.parser')

        # Find all image containers on the page
        divs = soup.findAll('div', class_='summary-photo__image-row__image')
        for div in divs:
            img_tag = div.find('img')
            if img_tag and 'src' in img_tag.attrs:
                image_urls.append(img_tag['src'])

        # Find the "Photo Category" labels
        label_divs = soup.find_all('div',
                                   class_='InformationItem__InfoItemStyle-sc-r4h3tv-0 hfSVPp information-item summary-photo__card-general__label')
        for div in label_divs:
            information_title = div.find('span', class_='information-item__title')
            if information_title and information_title.text.strip() == "Photo Category:":
                label_value = div.find('span', class_='information-item__value')
                if label_value:
                    label = ShipLabelFilter.filter_label(label_value.text.strip())
                    logger.info(f"Filtered label: {label}")
                    labels.append(label)

    except Exception as e:
        logger.error(f"Error retrieving images for process_id {process_id}: {e}")

    return image_urls, labels

def save_images_into_categories(image_urls, labels, output_dir='downloaded_images', process_id=None):
    """
    Saves images into category-specific folders.

    Args:
        image_urls (list): List of image URLs to download.
        labels (list): Corresponding list of category labels.
        output_dir (str): Base directory to save images.
        process_id (int): ID of the current processing batch.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    saved_paths = []
    try:
        for index, image_url in enumerate(image_urls):
            if image_url:
                logger.info(f"Downloading image from {image_url}")
                image_data = requests.get(image_url, headers=HEADERS).content

                if index < len(labels):
                    category = labels[index]
                else:
                    category = "Unknown"

                # Define the category directory
                category_dir = os.path.join(output_dir, category if category else "Unknown")
                os.makedirs(category_dir, exist_ok=True)

                # Save the image in the category directory
                image_filename = f'image_{process_id}_{index}.jpg'
                image_path = os.path.join(category_dir, image_filename)
                with open(image_path, 'wb') as image_file:
                    image_file.write(image_data)
                logger.info(f"Image saved to {image_path}")
                saved_paths.append(image_path)
            else:
                logger.warning(f"No valid image URL provided for process_id {process_id}")
    except Exception as e:
        logger.error(f"Error saving image for {process_id}: {e}")

    return saved_paths

def process_image_with_yolo(image_path, category, output_base_dir='processed_boats'):
    """
    Processes an image with YOLO, crops detected boats, and saves them into category-specific folders.

    Args:
        image_path (str): Path to the image to process.
        category (str): Category label of the image.
        output_base_dir (str): Base directory to save processed images.
    """
    try:
        # Load the image
        image = Image.open(image_path).convert("RGB")

        # Perform YOLO detection
        results = yolo_model(image_path)

        for result in results:
            boxes = result.boxes  # Assuming result.boxes contains the bounding boxes
            for idx, box in enumerate(boxes):
                # Convert box.cls from Tensor to Python scalar
                cls = box.cls.item() if hasattr(box.cls, 'item') else box.cls

                # Check confidence
                confidence = box.conf.item() if hasattr(box.conf, 'item') else box.conf
                if confidence < 0.7:
                    logger.info(f"Detection {idx} in {image_path} skipped due to low confidence: {confidence}")
                    continue  # Skip this box

                if cls == 0:  # Assuming class 0 is 'Boat'
                    xyxy = box.xyxy.tolist()
                    logger.debug(f"Original box.xyxy.tolist(): {xyxy}")

                    # Flatten if necessary
                    if isinstance(xyxy[0], list):
                        xyxy = xyxy[0]
                        logger.debug(f"Flattened box.xyxy.tolist(): {xyxy}")

                    try:
                        x1, y1, x2, y2 = [int(coord) for coord in xyxy]
                    except Exception as conv_e:
                        logger.error(f"Error converting coordinates to integers: {conv_e}")
                        continue  # Skip this box

                    # Optional: Check image boundaries
                    width, height = image.size
                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = min(width, x2)
                    y2 = min(height, y2)

                    cropped_image = image.crop((x1, y1, x2, y2))

                    # Define the category directory within the processed base directory
                    category_dir = os.path.join(output_base_dir, category if category else "Unknown")
                    os.makedirs(category_dir, exist_ok=True)

                    # Save the cropped image in the category directory
                    base_filename = os.path.splitext(os.path.basename(image_path))[0]
                    boat_filename = f"{base_filename}_boat_{idx}.jpg"  # Use index as unique identifier
                    boat_path = os.path.join(category_dir, boat_filename)
                    cropped_image.save(boat_path)
                    logger.info(f"Cropped boat image saved to {boat_path}")

    except Exception as e:
        logger.error(f"Error processing image {image_path} with YOLO: {e}")

def main():
    start_id = 1134410
    end_id = 1134414
    download_folder = "/Users/saadbenboujina/Desktop/Projects/bachelor arbeit/ForCategory/RawData"
    processed_boats_dir = "/Users/saadbenboujina/Desktop/Projects/bachelor arbeit/ForCategory/ProcessedBoats"

    for process_id in range(start_id, end_id):
        logger.info(f"Processing ID: {process_id}")

        # Step 1: Download images and labels
        image_urls, labels = download_images_from_shipspotting(process_id)
        logger.info(f"Image URLs: {image_urls}")
        logger.info(f"Labels: {labels}")

        # Step 2: Save images into category-specific folders
        saved_image_paths = save_images_into_categories(image_urls, labels, download_folder, process_id)
        logger.info(f"Saved Images: {saved_image_paths}")

        # Step 3: Process images with YOLO and save cropped boats
        for idx, image_path in enumerate(saved_image_paths):
            # Determine the category for the current image
            if idx < len(labels):
                category = labels[idx]
            else:
                category = "Unknown"

            process_image_with_yolo(image_path, category, output_base_dir=processed_boats_dir)

if __name__ == '__main__':
    main()
