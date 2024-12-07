import os
import time

import requests
from bs4 import BeautifulSoup
from ultralytics import YOLO
from PIL import Image
import logging
from selenium import webdriver
from selenium.webdriver.chrome.options import Options

from Pipeline.ShipLabelFilter import ShipLabelFilter

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define headers for HTTP requests (used when downloading images)
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Referer': 'https://www.shipspotting.com',
    'Accept-Language': 'en-US,en;q=0.9',
}

# Load the YOLO model
yolo_model = YOLO("/Users/saadbenboujina/Downloads/optuna_trial_2/weights/best.pt")

def scrape_page_selenium(process_id):
    """
    Scrapes the ShipSpotting.com page using Selenium and returns the page source.
    """
    url = f"https://www.shipspotting.com/photos/{process_id}"
    logger.info(f"Opening URL with Selenium: {url}")

    chrome_options = Options()
    chrome_options.add_argument("--headless")  # Run in headless mode
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
                                "Chrome/91.0.4472.124 Safari/537.36")

    # Initialize the WebDriver
    driver = webdriver.Chrome(options=chrome_options)

    try:
        driver.get(url)
        # Wait for dynamic content to load
        time.sleep(0.5)  # Adjust sleep time as necessary

        page_source = driver.page_source
        return page_source

    except Exception as e:
        logger.error(f"Error scraping page for process_id {process_id} with Selenium: {e}")
        return None
    finally:
        driver.quit()

def extract_image_urls(soup):
    """
    Extracts image URLs from the BeautifulSoup-parsed HTML.
    """
    image_urls = []
    divs = soup.findAll('div', class_='summary-photo__image-row__image')
    for div in divs:
        img_tag = div.find('img')
        if img_tag and 'src' in img_tag.attrs:
            image_urls.append(img_tag['src'])
    return image_urls

def extract_labels(soup):
    """
    Extrahiert Labels aus dem mit BeautifulSoup geparsten HTML.
    """
    labels = []
    label_divs = soup.find_all('div',
                               class_='InformationItem__InfoItemStyle-sc-r4h3tv-0 hfSVPp information-item summary-photo__card-general__label')
    for div in label_divs:
        information_title = div.find('span', class_='information-item__title')
        if information_title and information_title.text.strip() == "Photo Category:":
            label_value = div.find('span', class_='information-item__value')
            if label_value:
                label_text = label_value.text.strip()
                if label_text:  # Sicherstellen, dass das Label nicht leer ist
                    label = ShipLabelFilter.filter_label(label_text)
                    if label:  # Sicherstellen, dass filter_label kein None zurückgibt
                        logger.info(f"Gefiltertes Label: {label}")
                        labels.append(label)
                    else:
                        logger.warning(f"filter_label gab None zurück für label_text: '{label_text}'")
                else:
                    logger.warning(f"Leeres label_text gefunden für process_id.")
    return labels


def download_images_from_shipspotting(process_id, url_prefix='https://www.shipspotting.com/photos/'):
    """
    Downloads image URLs and labels using Selenium.
    """
    image_urls = []
    labels = []
    photo_id = "{:07d}".format(process_id)
    url = url_prefix + photo_id

    logger.info(f"Fetching URL: {url} using Selenium")

    page_source = scrape_page_selenium(process_id)
    if not page_source:
        logger.error(f"Failed to retrieve page source for process_id {process_id}")
        return image_urls, labels

    soup = BeautifulSoup(page_source, 'html.parser')

    image_urls = extract_image_urls(soup)
    labels = extract_labels(soup)

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
            # Check if a corresponding label exists
            if index < len(labels):
                label = labels[index]
            else:
                label = None

            if label is None:
                logger.info(f"Skipping image at index {index} for process_id {process_id} due to excluded or unmapped label.")
                continue  # Skip saving this image

            if image_url:
                # Handle relative URLs
                if image_url.startswith('/'):
                    image_url = 'https://www.shipspotting.com' + image_url

                logger.info(f"Downloading image from {image_url}")
                try:
                    response = requests.get(image_url, headers=HEADERS, timeout=10)
                    response.raise_for_status()
                    image_data = response.content
                except requests.RequestException as req_e:
                    logger.error(f"Failed to download image from {image_url}: {req_e}")
                    continue  # Skip this image due to download error

                # Define the category directory
                category_dir = os.path.join(output_dir, label[0])
                os.makedirs(category_dir, exist_ok=True)

                # Save the image in the category directory
                image_filename = f'image_{process_id}_{index}.jpg'
                image_path = os.path.join(category_dir, image_filename)
                try:
                    with open(image_path, 'wb') as image_file:
                        image_file.write(image_data)
                    logger.info(f"Image saved to {image_path}")
                    saved_paths.append(image_path)
                except IOError as io_e:
                    logger.error(f"Failed to save image to {image_path}: {io_e}")
            else:
                logger.warning(f"No valid image URL provided for process_id {process_id}, index {index}")
    except Exception as e:
        logger.error(f"Error saving image for process_id {process_id}: {e}")

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
        detections_above_threshold = 0  # Counter for detections

        for result in results:
            boxes = result.boxes  # Assuming result.boxes contains the bounding boxes
            for idx, box in enumerate(boxes):
                # Convert box.cls from Tensor to Python scalar
                cls = box.cls.item() if hasattr(box.cls, 'item') else box.cls

                # Check confidence
                confidence = box.conf.item() if hasattr(box.conf, 'item') else box.conf
                if cls == 0 and confidence > 0.6:  # Assuming class 0 is 'Boat'
                    detections_above_threshold += 1

        # Skip image if more than 1 boat is detected above threshold
        if detections_above_threshold > 1:
            logger.info(f"Skipping image {image_path}: {detections_above_threshold} boats detected above confidence 0.6")
            return

        for result in results:
            boxes = result.boxes
            for idx, box in enumerate(boxes):
                cls = box.cls.item() if hasattr(box.cls, 'item') else box.cls
                confidence = box.conf.item() if hasattr(box.conf, 'item') else box.conf
                if cls == 0 and confidence > 0.6:  # Assuming class 0 is 'Boat'
                    xyxy = box.xyxy.tolist()
                    if isinstance(xyxy[0], list):
                        xyxy = xyxy[0]

                    try:
                        x1, y1, x2, y2 = [int(coord) for coord in xyxy]
                    except Exception as conv_e:
                        logger.error(f"Error converting coordinates to integers: {conv_e}")
                        continue  # Skip this box

                    # Check image boundaries
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
                    boat_filename = f"{base_filename}_boat_{idx}.jpg"
                    boat_path = os.path.join(category_dir, boat_filename)
                    try:
                        cropped_image.save(boat_path)
                        logger.info(f"Cropped boat image saved to {boat_path}")
                    except IOError as io_e:
                        logger.error(f"Failed to save cropped image to {boat_path}: {io_e}")

    except Exception as e:
        logger.error(f"Error processing image {image_path} with YOLO: {e}")

def main():
    start_id = 3774385
    end_id = 3774387
    download_folder = "/Users/saadbenboujina/Desktop/Projects/bachelor arbeit/RawData"
    processed_boats_dir = "/Users/saadbenboujina/Desktop/Projects/bachelor arbeit/ForCategory/train22"

    for process_id in range(start_id, end_id):
        logger.info(f"Processing ID: {process_id}")

        # Step 1: Download images and labels using Selenium
        image_urls, labels = download_images_from_shipspotting(process_id)
        logger.info(f"Image URLs: {image_urls}")
        logger.info(f"Labels: {labels}")

        if not image_urls or not labels:
            logger.warning(f"No images or labels found for process_id {process_id}. Skipping.")
            continue

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
