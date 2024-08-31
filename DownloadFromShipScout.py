import os
import requests
from bs4 import BeautifulSoup

# Function to download images from shipspotting.com based on a process_id
def download_images_from_shipspotting(process_id, url_prefix='https://www.shipspotting.com/photos/'):
    image_url = ""
    labels = []
    # Format the process_id into a seven-digit number (e.g., 123 -> 0000123)
    photo_id = "{:07d}".format(process_id)
    # Construct the full URL by appending the photo_id to the base URL
    url = url_prefix + photo_id

    print(f"Fetching URL: {url}")

    try:
        # Send an HTTP GET request to the URL to retrieve the page
        response = requests.get(url)
        # Check if the request was successful (status code 200)
        if response.status_code != 200:
            print(f"Error: Received status code {response.status_code} from {url}")
            return image_url, labels  # Return if the request failed

        # Parse the HTML content of the page
        soup = BeautifulSoup(response.text, 'html.parser')

        # Find the first image container on the HTML page
        div = soup.find('div', class_='summary-photo__image-row__image')
        # Check if the container was found
        if div is not None:
            # Find the 'img' tag within the container
            img_tag = div.find('img')
            # Check if the 'src' attribute exists
            if img_tag and 'src' in img_tag.attrs:
                # Extract the image URL
                image_url = img_tag['src']
        else:
            print(f"No image container found for process_id {process_id}")

    except Exception as e:
        # Catch and report errors that occur during execution
        print(f"Error retrieving images for process_id {process_id}: {e}")

    # Return the found image URL and the labels (currently unused)
    return image_url, labels

# Function to save the downloaded images to disk
def save_images(image_url, output_dir='downloaded_images', process_id=None):
    # Check if the output directory exists; if not, create it
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    image_path = ""
    try:
        # Check if a valid image URL is provided
        if image_url:
            print(f"Downloading image from {image_url}")
            # Download the image data
            image_data = requests.get(image_url).content

            # Create a unique filename using the process_id
            image_filename = f'image_{process_id}.jpg'
            # Combine the output directory with the filename
            image_path = os.path.join(output_dir, image_filename)
            # Open the file in write-binary mode and save the image data
            with open(image_path, 'wb') as image_file:
                image_file.write(image_data)
            print(f"Image saved to {image_path}")
        else:
            print(f"No valid image URL provided for process_id {process_id}")
    except Exception as e:
        # Catch and report errors that occur during the save operation
        print(f"Error saving image {image_url}: {e}")

    return image_path

def main():
    # Iterate over a range of process_ids
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
