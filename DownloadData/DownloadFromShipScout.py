import os
import time
import random
import logging
import requests
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup

# Konfiguration des Loggings
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Liste der User-Agent-Strings
USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0.3 Safari/605.1.15',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36',
]

def get_random_user_agent():
    return random.choice(USER_AGENTS)

def get_selenium_driver():
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--window-size=1920,1080")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    # Füge einen zufälligen User-Agent hinzu
    chrome_options.add_argument(f"user-agent={get_random_user_agent()}")
    # Optional: Weitere Optionen, um den Browser noch authentischer zu machen
    driver = webdriver.Chrome(options=chrome_options)
    return driver

# Funktion zum Herunterladen von Bildern mit Selenium
def download_images_with_selenium(process_id, driver, url_prefix='https://www.shipspotting.com/photos/'):
    image_urls = []
    labels = []
    photo_id = "{:07d}".format(process_id)
    url = url_prefix + photo_id
    logger.info(f"Fetching URL with Selenium: {url}")

    try:
        driver.get(url)
        time.sleep(random.uniform(3, 5))  # Wartezeit, bis die Seite vollständig geladen ist

        # Extrahiere den Seitenquellcode nach dem Laden von JavaScript
        page_source = driver.page_source
        soup = BeautifulSoup(page_source, 'html.parser')

        # Finde alle Bildcontainer
        divs = soup.findAll('div', class_='summary-photo__image-row__image')
        for div in divs:
            img_tag = div.find('img')
            if img_tag and 'src' in img_tag.attrs:
                image_urls.append(img_tag['src'])

        # Finde die "Photo Category" Labels
        label_divs = soup.find_all('div', class_='InformationItem__InfoItemStyle-sc-r4h3tv-0 hfSVPp information-item summary-photo__card-general__label')
        for div in label_divs:
            information_title = div.find('span', class_='information-item__title')
            if information_title and information_title.text.strip() == "Photo Category:":
                label_value = div.find('span', class_='information-item__value')
                if label_value:
                    label = label_value.text.strip()
                    labels.append(label)

    except Exception as e:
        logger.error(f"Error retrieving images for process_id {process_id} with Selenium: {e}")

    return image_urls, labels

# Funktion zum Speichern der Bilder und Labels
def save_images_and_labels(image_urls, labels, output_dir='downloaded_images', process_id=None):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    saved_paths = []
    try:
        for index, image_url in enumerate(image_urls):
            if image_url:
                logger.info(f"Downloading image from {image_url}")
                try:
                    # Verwende eine separate Session für das Herunterladen der Bilder
                    with requests.Session() as session:
                        # Setze einen zufälligen User-Agent für jede Anfrage
                        headers = {
                            'User-Agent': get_random_user_agent(),
                            'Referer': 'https://www.shipspotting.com',
                            'Accept-Language': 'en-US,en;q=0.9',
                        }
                        response = session.get(image_url, headers=headers, timeout=10)
                        if response.status_code == 200:
                            image_data = response.content
                            image_filename = f'image_{process_id}_{index}.jpg'
                            image_path = os.path.join(output_dir, image_filename)
                            with open(image_path, 'wb') as image_file:
                                image_file.write(image_data)
                            logger.info(f"Image saved to {image_path}")
                            saved_paths.append(image_path)
                        else:
                            logger.warning(f"Failed to download image {image_url}: Status code {response.status_code}")
                except requests.exceptions.RequestException as e:
                    logger.error(f"Request error while downloading image {image_url}: {e}")
            else:
                logger.warning(f"No valid image URL provided for process_id {process_id}")
    except Exception as e:
        logger.error(f"Error saving image or label for {process_id}: {e}")

    return saved_paths

def main():
    # Definiere spezifische process IDs
    process_ids = [59692, 63412, 87571, 94582, 14764, 56746]
    output_folder = "/Users/saadbenboujina/Desktop/Projects/bachelor arbeit/TrainDataYolo/FromShipScout23"

    # Erstelle einen Selenium WebDriver
    driver = get_selenium_driver()

    for process_id in process_ids:
        try:
            # Schritt 1: Download der Bilder und Labels mit Selenium
            image_urls, labels = download_images_with_selenium(process_id, driver)
            logger.info(f"Image URLs: {image_urls}")
            logger.info(f"Labels: {labels}")

            # Schritt 2: Speichere die Bilder und Labels lokal
            saved_paths = save_images_and_labels(image_urls, labels, output_folder, process_id)
            logger.info(f"Saved Images and Labels: {saved_paths}")

            # Füge eine zufällige Pause zwischen 3 und 6 Sekunden hinzu
            time.sleep(random.uniform(3, 6))
        except Exception as e:
            logger.error(f"Error processing process_id {process_id}: {e}")

    # Schließe den WebDriver nach der Verarbeitung
    driver.quit()

if __name__ == '__main__':
    main()
