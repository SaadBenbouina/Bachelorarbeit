import os
import time
import requests
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import urllib.parse  # Wichtig für urljoin
import re  # Für reguläre Ausdrücke

from ultralytics import YOLO
from PIL import Image
import logging

# Logging konfigurieren
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# YOLO-Modell laden (Pfad anpassen)
yolo_model = YOLO("/Users/saadbenboujina/Downloads/optuna_trial_2/weights/best.pt")

def scrape_and_download_images(url,
                               download_dir="/Users/saadbenboujina/Desktop/Projects/bachelor arbeit/ForCategory/dow",
                               processed_dir="/Users/saadbenboujina/Desktop/Projects/bachelor arbeit/ForCategory/train223/rescue_ship22"):
    chrome_options = Options()
    chrome_options.add_argument("--headless")  # Headless-Modus aktivieren
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--window-size=1920,1080")  # Fenstergröße setzen
    chrome_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                                "AppleWebKit/537.36 (KHTML, like Gecko) "
                                "Chrome/112.0.0.0 Safari/537.36")  # Füge einen User-Agent hinzu

    driver = webdriver.Chrome(options=chrome_options)
    try:
        logger.info(f"Rufe Seite auf: {url}")
        driver.get(url)

        # **Warte bis mindestens ein Bild geladen ist**
        wait = WebDriverWait(driver, 20)  # Warte bis zu 20 Sekunden
        wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "img.photo-card__ship-photo")))

        # **Scrollen bis zum Ende der Seite**
        scroll_pause_time = 2.0
        last_height = driver.execute_script("return document.body.scrollHeight")

        while True:
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(scroll_pause_time)
            new_height = driver.execute_script("return document.body.scrollHeight")
            if new_height == last_height:
                break
            last_height = new_height

        # **Zusätzliche Wartezeit, falls Bilder nachgeladen werden**
        time.sleep(5)

        # **Option: Speichere den Seitenquelltext zur Überprüfung**
        # with open("page_source.html", "w", encoding="utf-8") as f:
        #     f.write(driver.page_source)
        # logger.info("Seitenquelltext in 'page_source.html' gespeichert zur weiteren Analyse.")

        # **Option: Screenshot zur Überprüfung**
        # driver.save_screenshot("screenshot.png")
        # logger.info("Screenshot der Seite gespeichert als 'screenshot.png'.")

        # **Verwende Selenium, um die img-Tags zu finden**
        img_elements = driver.find_elements(By.CSS_SELECTOR, "img.photo-card__ship-photo")
        logger.info(f"Anzahl gefundener img-Tags mit class 'photo-card__ship-photo': {len(img_elements)}")

        if not img_elements:
            logger.warning("Keine img-Tags mit der Klasse 'photo-card__ship-photo' gefunden.")
            return

        os.makedirs(download_dir, exist_ok=True)
        os.makedirs(processed_dir, exist_ok=True)

        img_counter = 0

        for img_element in img_elements:
            img_url = img_element.get_attribute("src") or img_element.get_attribute("data-src")
            if not img_url:
                logger.info(f"img-Tag ohne src oder data-src gefunden: {img_element}")
                continue

            # Absolut-URL erstellen, falls das Bild eine relative URL hat
            absolute_img_url = urllib.parse.urljoin(url, img_url)
            logger.info(f"Bild gefunden (rel='{img_url}'; abs='{absolute_img_url}')")

            try:
                response = requests.get(absolute_img_url, timeout=10)
                response.raise_for_status()

                img_counter += 1
                filename = f"image_{img_counter}.jpg"  # Nummerierung ab 1
                filepath = os.path.join(download_dir, filename)

                with open(filepath, 'wb') as f:
                    f.write(response.content)

                logger.info(f"Bild gespeichert unter: {filepath}")

                # YOLO-Inferenz
                process_image_with_yolo(filepath, processed_dir)

            except requests.RequestException as req_err:
                logger.error(f"Fehler beim Download von {absolute_img_url}: {req_err}")

    except Exception as e:
        logger.error(f"Ein Fehler ist aufgetreten: {e}")
    finally:
        driver.quit()

def process_image_with_yolo(image_path, processed_boats_dir):
    try:
        original_image = Image.open(image_path).convert("RGB")
        results = yolo_model(image_path)
        detections_above_threshold = 0

        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls = box.cls.item() if hasattr(box.cls, 'item') else box.cls
                confidence = box.conf.item() if hasattr(box.conf, 'item') else box.conf
                if cls == 0 and confidence > 0.6:  # Annahme: class 0 = 'Boat'
                    detections_above_threshold += 1

        if detections_above_threshold > 1:
            logger.info(f"Überspringe {image_path}: {detections_above_threshold} Boote erkannt (> 0.6).")
            return

        for result in results:
            boxes = result.boxes
            for idx, box in enumerate(boxes):
                cls = box.cls.item() if hasattr(box.cls, 'item') else box.cls
                confidence = box.conf.item() if hasattr(box.conf, 'item') else box.conf
                if cls == 0 and confidence > 0.6:
                    xyxy = box.xyxy.tolist()
                    if isinstance(xyxy[0], list):
                        xyxy = xyxy[0]
                    try:
                        x1, y1, x2, y2 = [int(coord) for coord in xyxy]
                    except ValueError as conv_e:
                        logger.error(f"Fehler beim Konvertieren der Koordinaten: {conv_e}")
                        continue

                    width, height = original_image.size
                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = min(width, x2)
                    y2 = min(height, y2)

                    cropped_image = original_image.crop((x1, y1, x2, y2))

                    base_filename = os.path.splitext(os.path.basename(image_path))[0]
                    boat_filename = f"{base_filename}_boat_{idx}.jpg"
                    boat_path = os.path.join(processed_boats_dir, boat_filename)

                    try:
                        cropped_image.save(boat_path)
                        logger.info(f"Boot ausgeschnitten und gespeichert: {boat_path}")
                    except IOError as io_e:
                        logger.error(f"Fehler beim Speichern von {boat_path}: {io_e}")

    except Exception as e:
        logger.error(f"Fehler bei YOLO-Verarbeitung von {image_path}: {e}")

if __name__ == "__main__":
    # Aktualisierte URL entsprechend deiner neuen Link
    test_url = "https://www.shipspotting.com/photos/gallery?category=50"
    scrape_and_download_images(test_url)
