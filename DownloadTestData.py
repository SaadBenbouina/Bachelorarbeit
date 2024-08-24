import os
import shutil
import fiftyone.zoo as foz

# Define the label and number of images to download
label_name = "Boat"
num_images = 10

# Load the Open Images dataset with the specific label
dataset = foz.load_zoo_dataset(
    "open-images-v7",
    split="train",
    label_types=["detections"],
    classes=[label_name],
    max_samples=num_images,
)

# Set the directory where images will be saved
output_dir = "/var/folders/3m/k2m2bg694w15lfb_1kz6blvh0000gn/T/wzQL.Cf1otW/Bachelorarbeit/Data/Test"

# Ensure the output directory exists
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Iterate over the samples and copy the images
for sample in dataset:
    print("Available fields:", sample.field_names)
    if "ground_truth" in sample.field_names:
        # Copy the image only, skip label creation
        image_path = sample.filepath
        image_output_path = os.path.join(output_dir, os.path.basename(image_path))
        shutil.copy(image_path, image_output_path)
    else:
        print(f"No ground_truth field found in sample {sample.id}")

print(f"Downloaded {num_images} images to the '{output_dir}' directory.")
