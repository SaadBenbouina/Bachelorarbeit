import os
import requests
from bs4 import BeautifulSoup

def download_images_from_shipspotting(process_id, url_prefix='https://www.shipspotting.com/photos/'):
    image_urls = []
    labels = []

    photo_id = "{:07d}".format(process_id)
    url = url_prefix + photo_id

    print(f"Fetching URL: {url}")

    try:
        response = requests.get(url)
        if response.status_code != 200:
            print(f"Error: Received status code {response.status_code} from {url}")
            return image_urls, labels

        soup = BeautifulSoup(response.text, 'html.parser')

        # Extract image URLs using the updated class name
        divs = soup.find_all('div', class_='summary-photo__image-row__image')
        print(f"Found {len(divs)} image containers")

        # Extract the image URLs from the 'img' tags within the 'div'
        image_urls = [div.find('img')['src'] for div in divs]

    except Exception as e:
        print(f"Error retrieving images for process_id {process_id}: {e}")

    return image_urls, labels

def save_images(image_urls, output_dir='downloaded_images', process_id=None):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    image_paths = []
    for i, url in enumerate(image_urls):
        try:
            print(f"Downloading image from {url}")
            image_data = requests.get(url).content
            # Use process_id and index i to create a unique filename
            image_filename = f'image_{process_id}_{i}.jpg'
            image_path = os.path.join(output_dir, image_filename)
            with open(image_path, 'wb') as image_file:
                image_file.write(image_data)
            image_paths.append(image_path)
            print(f"Image saved to {image_path}")
        except Exception as e:
            print(f"Error saving image {url}: {e}")

    return image_paths

def main():
    for i in range(1234560, 1234567):
        process_id = i
        output_folder = "/Users/saadbenboujina/Downloads/1/2"

        # Step 1: Download images from shipspotting.com
        image_urls, labels = download_images_from_shipspotting(process_id)
        print(f"Image URLs: {image_urls}")

        # Step 2: Save images locally with a unique name
        save_images(image_urls, output_folder, process_id)

if __name__ == '__main__':
    main()
