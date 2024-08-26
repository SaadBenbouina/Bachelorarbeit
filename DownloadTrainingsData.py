import os
import shutil
import fiftyone.zoo as foz

# Define the label and number of images to download
label_name = "Boat"
num_images = 2000

# Load the Open Images dataset with the specific label
dataset = foz.load_zoo_dataset(
    "open-images-v7",
    split="train",
    label_types=["detections"],
    classes=[label_name],
    max_samples=num_images,
)

# Set the directory where images and labels will be saved
images_dir = "/var/folders/3m/k2m2bg694w15lfb_1kz6blvh0000gn/T/wzQL.Cf1otW/Bachelorarbeit/Data/Train/images"  # Pfad für Bilder
labels_dir = "/var/folders/3m/k2m2bg694w15lfb_1kz6blvh0000gn/T/wzQL.Cf1otW/Bachelorarbeit/Data/Train/labels"  # Pfad für Labels

# Ensure the output directories exist
if not os.path.exists(images_dir):
    os.makedirs(images_dir)

if not os.path.exists(labels_dir):
    os.makedirs(labels_dir)

# Iterate over the samples and check available fields
for sample in dataset:
    print("Available fields:", sample.field_names)
    if "ground_truth" in sample.field_names:
        # Copy the image
        image_path = sample.filepath
        image_output_path = os.path.join(images_dir, os.path.basename(image_path))
        shutil.copy(image_path, image_output_path)

        # Save the label information in a compact format
        # Change the extension of the image file to .txt for the label file
        label_filename = os.path.splitext(os.path.basename(image_path))[0] + ".txt"
        labels_output_path = os.path.join(labels_dir, label_filename)

        # Ensure the file is opened and written to correctly
        with open(labels_output_path, 'w') as f:
            detections = sample.ground_truth.detections
            for detection in detections:
                if detection.label == label_name:
                    # Bounding box format: [xmin, ymin, width, height]
                    bbox = detection.bounding_box
                    bbox_str = f"{bbox[0]:.4f} {bbox[1]:.4f} {bbox[2]:.4f} {bbox[3]:.4f}"
                    label_str = f"0 {bbox_str}\n"  # Replace detection.label with '0' as numeric ID
                    f.write(label_str)
    else:
        print(f"No ground_truth field found in sample {sample.id}")

print(f"Downloaded {num_images} images to '{images_dir}' and labels to '{labels_dir}' directory.")
