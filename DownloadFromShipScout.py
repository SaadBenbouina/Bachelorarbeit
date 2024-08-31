import os
import requests
from bs4 import BeautifulSoup

def download_images_from_shipspotting(process_id, url_prefix='https://www.shipspotting.com/photos/'):
    image_url = ""
    labels = []

    photo_id = "{:07d}".format(process_id)
    url = url_prefix + photo_id

    print(f"Fetching URL: {url}")

    try:
        response = requests.get(url)
        if response.status_code != 200:
            print(f"Error: Received status code {response.status_code} from {url}")
            return image_url, labels

        soup = BeautifulSoup(response.text, 'html.parser')

        # Extract image URL using the updated class name
        div = soup.find('div', class_='summary-photo__image-row__image')
        if div is not None:
            img_tag = div.find('img')
            if img_tag and 'src' in img_tag.attrs:
                image_url = img_tag['src']
        else:
            print(f"No image container found for process_id {process_id}")

    except Exception as e:
        print(f"Error retrieving images for process_id {process_id}: {e}")

    return image_url, labels

def save_images(image_url, output_dir='downloaded_images', process_id=None):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    image_path = ""
    try:
        if image_url:
            print(f"Downloading image from {image_url}")
            image_data = requests.get(image_url).content
            # Use process_id to create a unique filename
            image_filename = f'image_{process_id}.jpg'
            image_path = os.path.join(output_dir, image_filename)
            with open(image_path, 'wb') as image_file:
                image_file.write(image_data)
            print(f"Image saved to {image_path}")
        else:
            print(f"No valid image URL provided for process_id {process_id}")
    except Exception as e:
        print(f"Error saving image {image_url}: {e}")

    return image_path

def main():
    for i in range(1234560, 1234567):
        process_id = i
        output_folder = "/Users/saadbenboujina/Downloads/1/2"

        # Step 1: Download images from shipspotting.com
        image_url, labels = download_images_from_shipspotting(process_id)
        print(f"Image URL: {image_url}")

        # Step 2: Save images locally with a unique name
        save_images(image_url, output_folder, process_id)

if __name__ == '__main__':
    main()
